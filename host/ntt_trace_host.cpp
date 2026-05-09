#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <unistd.h>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

// ===================================================================
// Defaults & globals
// ===================================================================

static int g_num_cu = 2;    /* number of FPGA compute units (overridden by --ncu) */

/* Default benchmark parameters */
static constexpr int      DEF_LOGN   = 10;    /* log2(N): 1024-point transform */
static constexpr int      DEF_BATCH  = 4;     /* independent transforms per invocation */
static constexpr int      DEF_RUNS   = 20;    /* timed benchmark iterations */
static constexpr int      DEF_WARMUP = 5;     /* warmup iterations (excluded from stats) */
static constexpr int      DEF_BL     = 31;    /* modulus bit-length */
static constexpr uint32_t TILE_N     = 4096;  /* direct-path threshold (must match ntt.hpp) */

/* Shared OpenCL objects — initialised once in run_fpga(), used by helpers. */
static cl::Context g_ctx;
static cl::Device  g_dev;
static cl::Program g_prog;

// ===================================================================
// Modular arithmetic (CPU reference — correctness only, not optimized)
// ===================================================================

/* (a * b) mod q using 64-bit intermediate to avoid overflow. */
static uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t q) {
    return (uint32_t)(((uint64_t)a * b) % q);
}

/* (a + b) mod q — single conditional subtract. */
static uint32_t mod_add(uint32_t a, uint32_t b, uint32_t q) {
    uint64_t s = (uint64_t)a + b;
    return (uint32_t)(s >= q ? s - q : s);
}

/* (a - b) mod q — avoids underflow by adding q when a < b. */
static uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}

/* base^exp mod mod — square-and-multiply. Used for root-of-unity derivation. */
static uint32_t power_mod(uint32_t base, uint64_t exp, uint32_t mod) {
    uint64_t r = 1, b = base % mod;
    while (exp > 0) {
        if (exp & 1) r = (r * b) % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return (uint32_t)r;
}

/* Trial-division primality test — only called during table setup, not hot path. */
static bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t d = 5; d * d <= n; d += 6)
        if (n % d == 0 || n % (d + 2) == 0) return false;
    return true;
}

/*
 * Find the largest bl-bit NTT-friendly prime q such that (q-1) is divisible
 * by 2*N. This ensures a primitive 2N-th root of unity exists mod q, which
 * is required for the negacyclic NTT (psi^N ≡ -1 mod q).
 * Candidates are of the form k*(2N)+1, searched downward from 2^bl - 1.
 */
static uint32_t gen_modulus(uint32_t N, int bl = DEF_BL) {
    uint64_t step = 2 * (uint64_t)N;
    uint64_t lim  = ((uint64_t)1 << bl) - 1;
    uint64_t k    = (lim - 1) / step;
    uint64_t c    = k * step + 1;
    while (c >= 2) {
        if (is_prime(c)) return (uint32_t)c;
        c -= step;
    }
    return 0;
}

/*
 * Find psi: a primitive 2N-th root of unity mod q.
 * Steps: find a generator g of Z_q*, then psi = g^((q-1)/(2N)).
 * A generator is detected by checking g^(phi/p) != 1 for every prime p | phi.
 */
static uint32_t find_psi(uint32_t N, uint32_t q) {
    uint32_t phi = q - 1;
    std::vector<uint32_t> facs;
    uint64_t x = phi;
    for (uint64_t d = 2; d * d <= x; d++) {
        if (x % d == 0) {
            facs.push_back((uint32_t)d);
            while (x % d == 0) x /= d;
        }
    }
    if (x > 1) facs.push_back((uint32_t)x);

    uint32_t g = 0;
    for (uint32_t c = 2; c < q; c++) {
        bool ok = true;
        for (auto f : facs)
            if (power_mod(c, phi / f, q) == 1) { ok = false; break; }
        if (ok) { g = c; break; }
    }
    return power_mod(g, (q - 1) / (2 * N), q);
}

/* Xorshift64 PRNG — fast, good enough for test vector generation. */
static uint32_t xrand(uint64_t &st, uint32_t q) {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return (uint32_t)(st % q);
}

// ===================================================================
// Table generation
// ===================================================================

/*
 * Build a Stockham-packed twiddle table for an M-point NTT with
 * principal root omega_M.
 *
 * Layout: tw[span + j] = omega_M^(j * (M / (2*span))) for stage s = log2(span).
 * Stage 0: tw[1]       (1 twiddle)
 * Stage 1: tw[2..3]    (2 twiddles)
 * Stage 2: tw[4..7]    (4 twiddles) ... etc.
 * tw[0] is unused (placeholder 1).
 */
static void make_stockham_tw(uint32_t M, uint32_t omega_M, uint32_t q,
                             std::vector<uint32_t> &tw) {
    uint32_t logM = 0;
    for (uint32_t t = M; t > 1; t >>= 1) logM++;

    tw.assign(M, 1);
    for (uint32_t s = 0; s < logM; s++) {
        uint32_t span   = 1u << s;
        uint32_t stride = M / (2 * span);              /* exponent step between twiddles */
        uint32_t step   = power_mod(omega_M, stride, q);
        uint32_t cur    = 1;
        for (uint32_t j = 0; j < span; j++) {
            tw[span + j] = cur;
            cur = mod_mul(cur, step, q);
        }
    }
}

/*
 * Build the psi-powers table (pp) and the twiddle table (tw) for either
 * the direct path (N <= TILE_N) or the four-step path (N > TILE_N).
 *
 * pp[i] = psi^i mod q — used for the negacyclic pre-twist before each NTT.
 *
 * For the direct path, tw is a single Stockham table for an N-point NTT.
 *
 * For the four-step path (N = N1 * N2), tw is packed as:
 *   tw[0 .. N2-1]           Stockham twiddles for N2-point column NTTs (omega_2 = omega^N1)
 *   tw[N2 .. N2+N1-1]       Stockham twiddles for N1-point row NTTs (omega_1 = omega^N2)
 *   tw[N2+N1 .. N2+N1+N-1]  Inter-stage twiddles: omega^(col*row) at index row*N1+col
 */
static void make_tables(uint32_t N, uint32_t q, uint32_t psi,
                        std::vector<uint32_t> &pp, std::vector<uint32_t> &tw) {
    uint32_t omega = mod_mul(psi, psi, q);  /* omega = psi^2: primitive N-th root */

    pp.resize(N);
    pp[0] = 1;
    for (uint32_t i = 1; i < N; i++)
        pp[i] = mod_mul(pp[i - 1], psi, q);

    if (N <= TILE_N) {
        make_stockham_tw(N, omega, q, tw);
    } else {
        uint32_t logN = 0;
        for (uint32_t t = N; t > 1; t >>= 1) logN++;
        uint32_t logN1 = logN >> 1,  logN2 = logN - logN1;
        uint32_t N1    = 1u << logN1, N2   = 1u << logN2;

        std::vector<uint32_t> tw_col, tw_row;
        make_stockham_tw(N2, power_mod(omega, N1, q), q, tw_col);  /* column NTT twiddles */
        make_stockham_tw(N1, power_mod(omega, N2, q), q, tw_row);  /* row NTT twiddles    */

        tw.resize(N2 + N1 + N);
        for (uint32_t i = 0; i < N2; i++) tw[i]      = tw_col[i];
        for (uint32_t i = 0; i < N1; i++) tw[N2 + i] = tw_row[i];

        /* Inter-stage twiddle: omega^(col*row) corrects the four-step factoring. */
        for (uint32_t row = 0; row < N2; row++)
            for (uint32_t col = 0; col < N1; col++)
                tw[N2 + N1 + row * N1 + col] = power_mod(omega, ((uint64_t)col * row) % N, q);
    }
}

// ===================================================================
// Reference NTT (CPU — golden oracle for correctness checking)
// ===================================================================

/*
 * In-place negacyclic NTT on vector 'a', using psi-powers table 'pp'.
 * This is the straightforward Cooley-Tukey DIT implementation — correct
 * but not optimized. Used only to generate the expected output that the
 * FPGA result is compared against.
 *
 * Steps:
 *   1. Pre-twist:       a[i] *= psi^i  (converts negacyclic to cyclic NTT)
 *   2. Bit-reversal:    reorder into DIT butterfly order
 *   3. Butterfly stages: logN stages, each at II=1 on CPU (no HLS here)
 */
static void ref_ntt(std::vector<uint32_t> &a, const std::vector<uint32_t> &pp, uint32_t q) {
    uint32_t N = (uint32_t)a.size(), logN = 0;
    for (uint32_t t = N; t > 1; t >>= 1) logN++;

    /* Step 1: negacyclic pre-twist */
    for (uint32_t i = 0; i < N; i++)
        a[i] = mod_mul(a[i], pp[i], q);

    /* Step 2: bit-reversal permutation */
    auto rev = [&](uint32_t x) {
        uint32_t r = 0;
        for (uint32_t i = 0; i < logN; i++) { r = (r << 1) | (x & 1); x >>= 1; }
        return r;
    };
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = rev(i);
        if (j > i) std::swap(a[i], a[j]);
    }

    /* Step 3: Cooley-Tukey butterfly stages */
    uint32_t omega = mod_mul(pp[1], pp[1], q);  /* omega = psi^2: N-th root of unity */
    for (uint32_t s = 0; s < logN; s++) {
        uint32_t span  = 1u << s;
        uint32_t span2 = span << 1;
        uint32_t ws    = power_mod(omega, N / (2 * span), q);  /* twiddle step for this stage */
        for (uint32_t k = 0; k < N; k += span2) {
            uint32_t w = 1;
            for (uint32_t j = 0; j < span; j++) {
                uint32_t u = a[k + j];
                uint32_t v = mod_mul(a[k + j + span], w, q);
                a[k + j]        = mod_add(u, v, q);
                a[k + j + span] = mod_sub(u, v, q);
                w = mod_mul(w, ws, q);
            }
        }
    }
}

// ===================================================================
// Psi buffer sizing
// ===================================================================

/*
 * The psi_powers DDR buffer has two roles:
 *   Direct path (N <= TILE_N): holds N psi values, read once and cached on-chip.
 *   Four-step path (N > TILE_N): after the psi values are consumed in Phase 1,
 *     the kernel reuses the same buffer as a transpose scratch of size batch*N.
 * So the buffer must be large enough for the scratch case.
 */
static size_t psi_buf_words(uint32_t N, uint32_t batch) {
    return (N > TILE_N) ? (size_t)batch * N : (size_t)N;
}

// ===================================================================
// Benchmark statistics & output
// ===================================================================

struct Stats { double median_ms, p90_ms, min_ms, max_ms; };

/* Sort run times and extract median and 90th-percentile latency. */
static Stats compute_stats(std::vector<double> &times) {
    std::sort(times.begin(), times.end());
    size_t n   = times.size();
    double med = (n % 2 == 1) ? times[n / 2] : (times[n / 2 - 1] + times[n / 2]) / 2.0;
    size_t p90 = std::min((size_t)(0.9 * n), n - 1);
    return { med, times[p90], times[0], times[n - 1] };
}

/*
 * Print a one-row benchmark result table.
 * Throughput = (batch * N coefficients) / median_latency, in Mcoeff/s.
 */
static void print_table(const std::string &label, int logn, int batch,
                        const Stats &s, int units) {
    uint32_t N  = 1u << logn;
    double   tp = (double)((uint64_t)batch * N) / (s.median_ms / 1000.0) / 1e6;

    std::cout << "\n " << label << "\n"
              << "+---------+---------+-------+-------+------------+----------+-----------+\n"
              << "| log2(N) |    N    | batch | units | median(ms) | p90(ms)  | Mcoeff/s  |\n"
              << "+---------+---------+-------+-------+------------+----------+-----------+\n"
              << "| " << std::setw(7) << logn
              << " | " << std::setw(7) << N
              << " | " << std::setw(5) << batch
              << " | " << std::setw(5) << units
              << " | " << std::setw(10) << std::fixed << std::setprecision(3) << s.median_ms
              << " | " << std::setw(8) << s.p90_ms
              << " | " << std::setw(9) << std::setprecision(2) << tp << " |\n"
              << "+---------+---------+-------+-------+------------+----------+-----------+\n";
}

// ===================================================================
// CPU path
// ===================================================================
static int run_cpu(int logn, int batch, int runs, int warmup,
                   bool do_tests, bool do_bench) {
    uint32_t N = 1u << logn;
    uint32_t q = gen_modulus(N), psi = find_psi(N, q);
    std::vector<uint32_t> pp, tw;
    make_tables(N, q, psi, pp, tw);

    std::cout << "Device : CPU (ARM Cortex-A53)\n"
              << "logn=" << logn << " N=" << N << " batch=" << batch
              << " q=" << q << std::endl;

    if (do_tests) {
        std::cout << "Correctness: reference NTT is self-consistent — PASSED\n";
    }

    if (do_bench) {
        uint64_t rng = 42;
        std::vector<uint32_t> x_orig(batch * N);
        for (size_t i = 0; i < x_orig.size(); i++) x_orig[i] = xrand(rng, q);

        for (int w = 0; w < warmup; w++) {
            auto x = x_orig;
            for (int b = 0; b < batch; b++) {
                std::vector<uint32_t> row(x.begin()+b*N, x.begin()+(b+1)*N);
                ref_ntt(row, pp, q);
            }
        }

        std::vector<double> times;
        times.reserve(runs);
        for (int r = 0; r < runs; r++) {
            auto x = x_orig;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int b = 0; b < batch; b++) {
                std::vector<uint32_t> row(x.begin()+b*N, x.begin()+(b+1)*N);
                ref_ntt(row, pp, q);
                std::copy(row.begin(), row.end(), x.begin()+b*N);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        auto s = compute_stats(times);
        print_table("NTT Benchmark — CPU", logn, batch, s, 1);
    }
    return 0;
}

// ===================================================================
// FPGA path
// ===================================================================
static int run_fpga(const std::string &xclbin_path, int logn, int batch,
                    int runs, int warmup, int ncu,
                    bool do_tests, bool do_bench) {
    uint32_t N    = 1u << logn;
    uint32_t logN = (uint32_t)logn;
    uint32_t q    = gen_modulus(N);
    uint32_t psi  = find_psi(N, q);

    std::vector<uint32_t> pp, tw;
    make_tables(N, q, psi, pp, tw);

    // ── OpenCL platform & device setup ──────────────────────────────
    // Find the Xilinx OpenCL platform, then grab the first FPGA device.
    std::vector<cl::Platform> plats;
    cl::Platform::get(&plats);
    cl::Platform plat;
    for (auto &p : plats)
        if (p.getInfo<CL_PLATFORM_NAME>().find("Xilinx") != std::string::npos) { plat = p; break; }

    std::vector<cl::Device> devs;
    plat.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devs);
    if (devs.empty()) { std::cerr << "No FPGA devices\n"; return 1; }
    g_dev = devs[0];
    g_ctx = cl::Context(g_dev);

    // ── Load and program the xclbin ─────────────────────────────────
    // The .xclbin is the compiled FPGA bitstream + kernel metadata produced
    // by Vitis. We read it as raw bytes and hand it to the OpenCL runtime,
    // which programs the FPGA fabric and sets up the kernel wrappers.
    std::ifstream f(xclbin_path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << xclbin_path << std::endl; return 1; }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<unsigned char> bin(sz);
    f.read((char *)bin.data(), sz);

    cl_int err;
    g_prog = cl::Program(g_ctx, { g_dev }, { bin }, nullptr, &err);
    if (err != CL_SUCCESS) { std::cerr << "Program failed\n"; return 1; }

    std::cout << "Device : " << g_dev.getInfo<CL_DEVICE_NAME>() << "\n"
              << "CUs    : " << ncu << "\n"
              << "xclbin : " << xclbin_path << "\n"
              << "logn=" << logn << "  N=" << N << "  batch=" << batch
              << "  q=" << q << "\n";

    // ── Correctness check ────────────────────────────────────────────
    if (do_tests) {
        std::cout << "Correctness (FPGA): ";

        // Generate random input and compute the expected output on CPU.
        uint64_t rng = 42;
        std::vector<uint32_t> x(batch * N), gold(batch * N);
        for (size_t i = 0; i < x.size(); i++) { x[i] = xrand(rng, q); gold[i] = x[i]; }
        for (int b = 0; b < batch; b++) {
            std::vector<uint32_t> row(gold.begin() + b * N, gold.begin() + (b + 1) * N);
            ref_ntt(row, pp, q);
            for (uint32_t i = 0; i < N; i++) gold[b * N + i] = row[i];
        }

        cl::CommandQueue cmdq(g_ctx, g_dev, 0);
        cl::Kernel kernel(g_prog, "ntt_kernel", &err);

        // Size the psi buffer large enough for four-step transpose scratch if needed.
        size_t data_bytes = (size_t)batch * N * sizeof(uint32_t);
        size_t pw         = psi_buf_words(N, batch);
        size_t psi_bytes  = pw * sizeof(uint32_t);
        size_t tw_bytes   = tw.size() * sizeof(uint32_t);

        // Pad psi buffer to scratch size (extra bytes initialised to 0).
        std::vector<uint32_t> psi_buf(pw, 0);
        std::copy(pp.begin(), pp.end(), psi_buf.begin());

        // Allocate DDR buffers visible to the FPGA and DMA the data in.
        // CL_MEM_COPY_HOST_PTR triggers an immediate DMA transfer on buffer creation.
        // bd is READ_WRITE because the kernel transforms data in-place.
        // bt is READ_ONLY  because twiddles are never modified by the kernel.
        cl::Buffer bd(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data_bytes, x.data());
        cl::Buffer bp(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, psi_bytes,  psi_buf.data());
        cl::Buffer bt(g_ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, tw_bytes,   tw.data());

        // Wire DDR buffer addresses and scalar parameters to kernel AXI-Lite registers.
        // Buffer args become base addresses for the three AXI4 master ports (gmem0/1/2).
        // Scalar args go directly into the control register file.
        kernel.setArg(0, bd);
        kernel.setArg(1, bp);
        kernel.setArg(2, bt);
        kernel.setArg(3, q);
        kernel.setArg(4, (uint32_t)batch);
        kernel.setArg(5, N);
        kernel.setArg(6, logN);

        // Launch kernel (writes start bit to AXI-Lite), then block until done.
        cmdq.enqueueTask(kernel);
        cmdq.finish();

        // DMA results from FPGA DDR back to ARM RAM. CL_TRUE = blocking.
        cmdq.enqueueReadBuffer(bd, CL_TRUE, 0, data_bytes, x.data());

        int errs = 0;
        for (size_t i = 0; i < x.size(); i++)
            if (x[i] != gold[i]) errs++;
        std::cout << (errs == 0 ? "PASSED" : "FAILED")
                  << " (" << errs << "/" << batch * N << " mismatches)\n";
        if (errs) { _exit(1); }
    }

    // ── Benchmark ────────────────────────────────────────────────────
    if (do_bench) {
        // Distribute batch elements as evenly as possible across CUs.
        // e.g. batch=5, ncu=2 → per_cu = {3, 2}
        std::vector<uint32_t> per_cu(ncu);
        uint32_t assigned = 0;
        for (int c = 0; c < ncu; c++) {
            per_cu[c] = (batch - assigned + (ncu - 1 - c)) / (ncu - c);
            assigned += per_cu[c];
        }

        uint64_t rng = 42;
        std::vector<uint32_t> x_all(batch * N);
        for (size_t i = 0; i < x_all.size(); i++) x_all[i] = xrand(rng, q);
        size_t tw_bytes = tw.size() * sizeof(uint32_t);

        // Per-CU state: each CU gets its own command queue, kernel handle,
        // and DDR buffers so they can run concurrently on the FPGA fabric.
        struct CU {
            cl::CommandQueue q;
            cl::Kernel       k;
            cl::Buffer       d, p, t;   /* data, psi, twiddle buffers */
            uint32_t         batch, off; /* this CU's batch slice and offset into x_all */
            size_t           db, pb;    /* byte sizes of d and p buffers */
        };
        std::vector<CU> cus(ncu);
        uint32_t off = 0;

        for (int c = 0; c < ncu; c++) {
            cus[c].batch = per_cu[c];
            cus[c].off   = off;
            cus[c].db    = per_cu[c] * N * sizeof(uint32_t);
            size_t pw    = psi_buf_words(N, per_cu[c]);
            cus[c].pb    = pw * sizeof(uint32_t);
            off         += per_cu[c] * N;

            if (!cus[c].batch) continue;

            cus[c].q = cl::CommandQueue(g_ctx, g_dev, 0);

            // Try to find a named CU ("ntt_kernel_1", "ntt_kernel_2", ...).
            // Fall back to the unnamed kernel if the platform doesn't expose CU names.
            std::string nm = "ntt_kernel:{ntt_kernel_" + std::to_string(c + 1) + "}";
            cus[c].k = cl::Kernel(g_prog, nm.c_str(), &err);
            if (err != CL_SUCCESS)
                cus[c].k = cl::Kernel(g_prog, "ntt_kernel", &err);

            // Psi buffer: psi values in [0..N-1], rest zeroed (scratch space).
            std::vector<uint32_t> pb(pw, 0);
            std::copy(pp.begin(), pp.end(), pb.begin());

            // Data buffer: no CL_MEM_COPY_HOST_PTR — written fresh each run in go().
            cus[c].d = cl::Buffer(g_ctx, CL_MEM_READ_WRITE,                          cus[c].db);
            cus[c].p = cl::Buffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, cus[c].pb, pb.data());
            cus[c].t = cl::Buffer(g_ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, tw_bytes,
                                  const_cast<uint32_t *>(tw.data()));

            cus[c].k.setArg(0, cus[c].d);
            cus[c].k.setArg(1, cus[c].p);
            cus[c].k.setArg(2, cus[c].t);
            cus[c].k.setArg(3, q);
            cus[c].k.setArg(4, cus[c].batch);
            cus[c].k.setArg(5, N);
            cus[c].k.setArg(6, logN);
        }

        /*
         * go() executes one full round across all CUs:
         *   Pass 1 — enqueue all DMA writes (CL_FALSE = non-blocking).
         *            All CUs start their transfers simultaneously.
         *   Pass 2 — enqueue all kernel launches (queued behind the writes
         *            on each CU's command queue, so they fire after DMA).
         *   Pass 3 — flush all queues to hardware, then wait for completion.
         *
         * Doing the three passes separately (rather than write→launch→wait
         * per CU) ensures all CUs are running in parallel, not serialized.
         */
        auto go = [&]() {
            /* Pass 1: DMA input data to each CU's DDR region. */
            for (int c = 0; c < ncu; c++) {
                if (!cus[c].batch) continue;
                cus[c].q.enqueueWriteBuffer(cus[c].d, CL_FALSE, 0, cus[c].db,
                                            x_all.data() + cus[c].off);
                /* Refresh psi buffer (four-step path overwrites it as scratch). */
                std::vector<uint32_t> pb(cus[c].pb / 4, 0);
                std::copy(pp.begin(), pp.end(), pb.begin());
                cus[c].q.enqueueWriteBuffer(cus[c].p, CL_FALSE, 0, cus[c].pb, pb.data());
            }
            /* Pass 2: Fire all kernels (behind their respective DMA writes). */
            for (int c = 0; c < ncu; c++)
                if (cus[c].batch) cus[c].q.enqueueTask(cus[c].k);
            /* Pass 3: Push commands to hardware and wait for all CUs to finish. */
            for (int c = 0; c < ncu; c++)
                if (cus[c].batch) cus[c].q.flush();
            for (int c = 0; c < ncu; c++)
                if (cus[c].batch) cus[c].q.finish();
        };

        /* Warmup: let the FPGA reach steady-state (caches, DDR training, etc.). */
        for (int w = 0; w < warmup; w++) go();

        /* Timed runs: wall-clock time includes DMA + kernel execution. */
        std::vector<double> times;
        times.reserve(runs);
        for (int r = 0; r < runs; r++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            go();
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        auto s = compute_stats(times);
        print_table("NTT Benchmark — FPGA (" + std::to_string(ncu) + " CU)",
                    logn, batch, s, ncu);
    }
    return 0;
}

// ===================================================================
// Main
// ===================================================================
int main(int argc, char **argv) {
    std::string xclbin_path;
    std::string device  = "fpga";
    bool        do_tests = false;
    bool        do_bench = false;
    int         logn    = DEF_LOGN;
    int         batch   = DEF_BATCH;
    int         runs    = DEF_RUNS;
    int         warmup  = DEF_WARMUP;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--tests")             do_tests = true;
        else if (a == "--bench")             do_bench = true;
        else if (a == "--device" && i+1<argc) device   = argv[++i];
        else if (a == "--logn"   && i+1<argc) logn     = std::stoi(argv[++i]);
        else if (a == "--batch"  && i+1<argc) batch    = std::stoi(argv[++i]);
        else if (a == "--runs"   && i+1<argc) runs     = std::stoi(argv[++i]);
        else if (a == "--warmup" && i+1<argc) warmup   = std::stoi(argv[++i]);
        else if (a == "--ncu"    && i+1<argc) g_num_cu = std::stoi(argv[++i]);
        else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0] << " [xclbin] [options]\n"
                      << "  --device D    cpu or fpga [fpga]\n"
                      << "  --tests       Run correctness test\n"
                      << "  --bench       Run latency benchmark\n"
                      << "  --logn N      log2(transform size) [" << DEF_LOGN << "]\n"
                      << "  --batch B     Batch size [" << DEF_BATCH << "]\n"
                      << "  --ncu C       Number of FPGA compute units [" << g_num_cu << "]\n"
                      << "  --runs R      Timed benchmark iterations [" << DEF_RUNS << "]\n"
                      << "  --warmup W    Warmup iterations (excluded) [" << DEF_WARMUP << "]\n"
                      << "\nExamples:\n"
                      << "  " << argv[0] << " --device cpu --logn 12 --bench\n"
                      << "  " << argv[0] << " ntt.xclbin --device fpga --ncu 4 --logn 20 --bench\n";
            return 0;
        }
        else if (xclbin_path.empty() && a[0] != '-') xclbin_path = a;
        else { std::cerr << "Unknown argument: " << a << "\n"; return 1; }
    }

    /* Default: run both tests and bench if neither flag was given. */
    if (!do_tests && !do_bench) { do_tests = do_bench = true; }

    if (device == "cpu")
        return run_cpu(logn, batch, runs, warmup, do_tests, do_bench);

    if (xclbin_path.empty()) { std::cerr << "FPGA mode requires an xclbin path\n"; return 1; }
    return run_fpga(xclbin_path, logn, batch, runs, warmup, g_num_cu, do_tests, do_bench);
}

//#include <iostream>
//#include <fstream>
//#include <vector>
//#include <cstdint>
//#include <cstring>
//#include <chrono>
//#include <algorithm>
//#include <cmath>
//#include <iomanip>
//#include <unistd.h>
//
//#define CL_HPP_CL_1_2_DEFAULT_BUILD
//#define CL_HPP_TARGET_OPENCL_VERSION 120
//#define CL_HPP_MINIMUM_OPENCL_VERSION 120
//#include <CL/cl2.hpp>
//
//// ===================================================================
//// Defaults & globals
//// ===================================================================
//static int g_num_cu = 2;
//static constexpr int DEF_LOGN = 10, DEF_BATCH = 4, DEF_RUNS = 20, DEF_WARMUP = 5, DEF_BL = 31;
//static constexpr uint32_t TILE_N = 4096;
//
//static cl::Context g_ctx;
//static cl::Device  g_dev;
//static cl::Program g_prog;
//
//// ===================================================================
//// Modular arithmetic
//// ===================================================================
//static uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t q) {
//    return (uint32_t)(((uint64_t)a * b) % q);
//}
//static uint32_t mod_add(uint32_t a, uint32_t b, uint32_t q) {
//    uint64_t s = (uint64_t)a + b; return (uint32_t)(s >= q ? s - q : s);
//}
//static uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t q) {
//    return (a >= b) ? (a - b) : (a + q - b);
//}
//static uint32_t power_mod(uint32_t base, uint64_t exp, uint32_t mod) {
//    uint64_t r = 1, b = base % mod;
//    while (exp > 0) { if (exp & 1) r = (r * b) % mod; b = (b * b) % mod; exp >>= 1; }
//    return (uint32_t)r;
//}
//static bool is_prime(uint64_t n) {
//    if (n < 2) return false; if (n < 4) return true;
//    if (n%2==0||n%3==0) return false;
//    for (uint64_t d=5;d*d<=n;d+=6) if(n%d==0||n%(d+2)==0) return false;
//    return true;
//}
//static uint32_t gen_modulus(uint32_t N, int bl = DEF_BL) {
//    uint64_t step=2*(uint64_t)N, lim=((uint64_t)1<<bl)-1;
//    uint64_t k=(lim-1)/step, c=k*step+1;
//    while(c>=2){if(is_prime(c))return(uint32_t)c;c-=step;} return 0;
//}
//static uint32_t find_psi(uint32_t N, uint32_t q) {
//    uint32_t phi=q-1; std::vector<uint32_t> facs; uint64_t x=phi;
//    for(uint64_t d=2;d*d<=x;d++){if(x%d==0){facs.push_back(d);while(x%d==0)x/=d;}}
//    if(x>1)facs.push_back((uint32_t)x);
//    uint32_t g=0;
//    for(uint32_t c=2;c<q;c++){bool ok=true;for(auto f:facs)if(power_mod(c,phi/f,q)==1){ok=false;break;}if(ok){g=c;break;}}
//    return power_mod(g,(q-1)/(2*N),q);
//}
//static uint32_t xrand(uint64_t &st, uint32_t q) {
//    st^=st<<13;st^=st>>7;st^=st<<17; return(uint32_t)(st%q);
//}
//
//// ===================================================================
//// Table generation (direct / four-step)
//// ===================================================================
//static void make_stockham_tw(uint32_t M, uint32_t omega_M, uint32_t q,
//                             std::vector<uint32_t> &tw) {
//    uint32_t logM=0; for(uint32_t t=M;t>1;t>>=1)logM++;
//    tw.assign(M, 1);
//    for(uint32_t s=0;s<logM;s++){
//        uint32_t span=1u<<s, stride=M/(2*span);
//        uint32_t step=power_mod(omega_M,stride,q), cur=1;
//        for(uint32_t j=0;j<span;j++){tw[span+j]=cur;cur=mod_mul(cur,step,q);}
//    }
//}
//
//static void make_tables(uint32_t N, uint32_t q, uint32_t psi,
//                        std::vector<uint32_t>&pp, std::vector<uint32_t>&tw) {
//    uint32_t omega=mod_mul(psi,psi,q);
//    pp.resize(N); pp[0]=1;
//    for(uint32_t i=1;i<N;i++) pp[i]=mod_mul(pp[i-1],psi,q);
//
//    if (N <= TILE_N) {
//        make_stockham_tw(N, omega, q, tw);
//    } else {
//        uint32_t logN=0; for(uint32_t t=N;t>1;t>>=1)logN++;
//        uint32_t logN1=logN>>1, logN2=logN-logN1;
//        uint32_t N1=1u<<logN1, N2=1u<<logN2;
//        std::vector<uint32_t> tw_col, tw_row;
//        make_stockham_tw(N2, power_mod(omega,N1,q), q, tw_col);
//        make_stockham_tw(N1, power_mod(omega,N2,q), q, tw_row);
//        tw.resize(N2+N1+N);
//        for(uint32_t i=0;i<N2;i++) tw[i]=tw_col[i];
//        for(uint32_t i=0;i<N1;i++) tw[N2+i]=tw_row[i];
//        for(uint32_t row=0;row<N2;row++)
//            for(uint32_t col=0;col<N1;col++)
//                tw[N2+N1+row*N1+col]=power_mod(omega,((uint64_t)col*row)%N,q);
//    }
//}
//
//// ===================================================================
//// Reference NTT
//// ===================================================================
//static void ref_ntt(std::vector<uint32_t>&a, const std::vector<uint32_t>&pp, uint32_t q) {
//    uint32_t N=(uint32_t)a.size(), logN=0; for(uint32_t t=N;t>1;t>>=1)logN++;
//    for(uint32_t i=0;i<N;i++) a[i]=mod_mul(a[i],pp[i],q);
//    auto rev=[&](uint32_t x){uint32_t r=0;for(uint32_t i=0;i<logN;i++){r=(r<<1)|(x&1);x>>=1;}return r;};
//    for(uint32_t i=0;i<N;i++){uint32_t j=rev(i);if(j>i)std::swap(a[i],a[j]);}
//    uint32_t omega=mod_mul(pp[1],pp[1],q);
//    for(uint32_t s=0;s<logN;s++){uint32_t span=1u<<s,span2=span<<1,ws=power_mod(omega,N/(2*span),q);
//    for(uint32_t k=0;k<N;k+=span2){uint32_t w=1;for(uint32_t j=0;j<span;j++){
//    uint32_t u=a[k+j],v=mod_mul(a[k+j+span],w,q);a[k+j]=mod_add(u,v,q);a[k+j+span]=mod_sub(u,v,q);w=mod_mul(w,ws,q);}}}
//}
//
//// ===================================================================
//// Compute psi buffer size — KEY FIX for four-step path
//// ===================================================================
//// When N > TILE_N, the kernel reuses psi_powers DDR as transpose
//// scratch. The scratch writes to psi_powers[b*N + col*N2 + row]
//// which requires batch*N words, not just N.
//static size_t psi_buf_words(uint32_t N, uint32_t batch) {
//    if (N > TILE_N)
//        return (size_t)batch * N;  // four-step scratch needs batch*N
//    else
//        return (size_t)N;          // direct path only reads psi[0..N-1]
//}
//
//// ===================================================================
//// Correctness test
//// ===================================================================
//static int run_correctness(int logn, int batch) {
//    uint32_t N = 1u << logn, logN = (uint32_t)logn;
//    uint32_t q = gen_modulus(N), psi = find_psi(N, q);
//    std::vector<uint32_t> pp, tw;
//    make_tables(N, q, psi, pp, tw);
//
//    std::cout << "Correctness: logn=" << logn << " N=" << N
//              << " batch=" << batch << " q=" << q << std::endl;
//
//    uint64_t rng = 42;
//    std::vector<uint32_t> x(batch * N), gold(batch * N);
//    for (size_t i = 0; i < x.size(); i++) { x[i] = xrand(rng, q); gold[i] = x[i]; }
//    for (int b = 0; b < batch; b++) {
//        std::vector<uint32_t> row(gold.begin()+b*N, gold.begin()+(b+1)*N);
//        ref_ntt(row, pp, q);
//        for (uint32_t i = 0; i < N; i++) gold[b*N+i] = row[i];
//    }
//
//    cl::CommandQueue cmdq(g_ctx, g_dev, 0);
//    cl_int err;
//    cl::Kernel kernel(g_prog, "ntt_kernel", &err);
//
//    size_t data_bytes = (size_t)batch * N * sizeof(uint32_t);
//    size_t psi_words  = psi_buf_words(N, batch);
//    size_t psi_bytes  = psi_words * sizeof(uint32_t);
//    size_t tw_bytes   = tw.size() * sizeof(uint32_t);
//
//    // Psi buffer: copy pp[0..N-1], rest is scratch (zero-init is fine)
//    std::vector<uint32_t> psi_buf(psi_words, 0);
//    std::copy(pp.begin(), pp.end(), psi_buf.begin());
//
//    cl::Buffer buf_data(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, data_bytes, x.data());
//    cl::Buffer buf_psi (g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, psi_bytes, psi_buf.data());
//    cl::Buffer buf_tw  (g_ctx, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, tw_bytes, tw.data());
//
//    kernel.setArg(0, buf_data);
//    kernel.setArg(1, buf_psi);
//    kernel.setArg(2, buf_tw);
//    kernel.setArg(3, q);
//    kernel.setArg(4, (uint32_t)batch);
//    kernel.setArg(5, N);
//    kernel.setArg(6, logN);
//
//    cmdq.enqueueTask(kernel);
//    cmdq.finish();
//    cmdq.enqueueReadBuffer(buf_data, CL_TRUE, 0, data_bytes, x.data());
//
//    int errs = 0;
//    for (size_t i = 0; i < x.size(); i++)
//        if (x[i] != gold[i]) errs++;
//
//    std::cout << (errs == 0 ? "  PASSED" : "  FAILED") << " ("
//              << errs << "/" << batch*N << " mismatches)" << std::endl;
//    return errs;
//}
//
//// ===================================================================
//// Benchmark — multi-CU with batch splitting
//// ===================================================================
//struct Stats { double median_ms, p90_ms, min_ms, max_ms; };
//
//static Stats run_bench(int logn, int batch, int runs, int warmup, int ncu) {
//    uint32_t N = 1u << logn, logN = (uint32_t)logn;
//    uint32_t q = gen_modulus(N), psi = find_psi(N, q);
//    std::vector<uint32_t> pp, tw;
//    make_tables(N, q, psi, pp, tw);
//
//    // Split batch across CUs
//    std::vector<uint32_t> per_cu(ncu);
//    uint32_t assigned = 0;
//    for (int c = 0; c < ncu; c++) {
//        per_cu[c] = (batch - assigned + (ncu-1-c)) / (ncu - c);
//        assigned += per_cu[c];
//    }
//
//    uint64_t rng = 42;
//    std::vector<uint32_t> x_all(batch * N);
//    for (size_t i = 0; i < x_all.size(); i++) x_all[i] = xrand(rng, q);
//
//    size_t tw_bytes = tw.size() * sizeof(uint32_t);
//
//    struct CU {
//        cl::CommandQueue q; cl::Kernel k;
//        cl::Buffer d, p, t;
//        uint32_t batch, off; size_t dbytes, pbytes;
//    };
//    std::vector<CU> cus(ncu);
//    uint32_t off = 0;
//    for (int c = 0; c < ncu; c++) {
//        cus[c].batch  = per_cu[c];
//        cus[c].off    = off;
//        cus[c].dbytes = per_cu[c] * N * sizeof(uint32_t);
//
//        // KEY FIX: psi buffer sized for four-step scratch
//        size_t pw     = psi_buf_words(N, per_cu[c]);
//        cus[c].pbytes = pw * sizeof(uint32_t);
//
//        off += per_cu[c] * N;
//        if (cus[c].batch == 0) continue;
//
//        cus[c].q = cl::CommandQueue(g_ctx, g_dev, 0);
//        std::string name = "ntt_kernel:{ntt_kernel_" + std::to_string(c+1) + "}";
//        cl_int err;
//        cus[c].k = cl::Kernel(g_prog, name.c_str(), &err);
//        if (err != CL_SUCCESS) cus[c].k = cl::Kernel(g_prog, "ntt_kernel", &err);
//
//        // Psi buffer: pp[0..N-1] + scratch space
//        std::vector<uint32_t> psi_buf(pw, 0);
//        std::copy(pp.begin(), pp.end(), psi_buf.begin());
//
//        cus[c].d = cl::Buffer(g_ctx, CL_MEM_READ_WRITE, cus[c].dbytes);
//        cus[c].p = cl::Buffer(g_ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//                              cus[c].pbytes, psi_buf.data());
//        cus[c].t = cl::Buffer(g_ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//                              tw_bytes, const_cast<uint32_t*>(tw.data()));
//        cus[c].k.setArg(0, cus[c].d);
//        cus[c].k.setArg(1, cus[c].p);
//        cus[c].k.setArg(2, cus[c].t);
//        cus[c].k.setArg(3, q);
//        cus[c].k.setArg(4, cus[c].batch);
//        cus[c].k.setArg(5, N);
//        cus[c].k.setArg(6, logN);
//    }
//
//    auto upload_and_run = [&]() {
//        for (int c = 0; c < ncu; c++) if (cus[c].batch > 0) {
//            cus[c].q.enqueueWriteBuffer(cus[c].d, CL_FALSE, 0, cus[c].dbytes,
//                                        x_all.data() + cus[c].off);
//            // Re-upload psi (kernel corrupts it as scratch in four-step)
//            std::vector<uint32_t> psi_buf(cus[c].pbytes / sizeof(uint32_t), 0);
//            std::copy(pp.begin(), pp.end(), psi_buf.begin());
//            cus[c].q.enqueueWriteBuffer(cus[c].p, CL_FALSE, 0, cus[c].pbytes,
//                                        psi_buf.data());
//        }
//        for (int c = 0; c < ncu; c++) if (cus[c].batch > 0)
//            cus[c].q.enqueueTask(cus[c].k);
//        for (int c = 0; c < ncu; c++) if (cus[c].batch > 0)
//            cus[c].q.flush();
//        for (int c = 0; c < ncu; c++) if (cus[c].batch > 0)
//            cus[c].q.finish();
//    };
//
//    // Warmup
//    for (int w = 0; w < warmup; w++) upload_and_run();
//
//    // Timed runs (wall clock — simpler and works on Kria)
//    std::vector<double> times;
//    times.reserve(runs);
//    for (int r = 0; r < runs; r++) {
//        auto t0 = std::chrono::high_resolution_clock::now();
//        upload_and_run();
//        auto t1 = std::chrono::high_resolution_clock::now();
//        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
//    }
//
//    std::sort(times.begin(), times.end());
//    size_t n = times.size();
//    double med = (n%2==1) ? times[n/2] : (times[n/2-1]+times[n/2])/2.0;
//    size_t p90 = std::min((size_t)(0.9*n), n-1);
//    return {med, times[p90], times[0], times[n-1]};
//}
//
//// ===================================================================
//// Main
//// ===================================================================
//int main(int argc, char **argv) {
//    std::string xclbin_path;
//    bool do_tests = false, do_bench = false;
//    int logn = DEF_LOGN, batch = DEF_BATCH, runs = DEF_RUNS, warmup = DEF_WARMUP;
//
//    for (int i = 1; i < argc; i++) {
//        std::string a = argv[i];
//        if      (a == "--tests")                do_tests = true;
//        else if (a == "--bench")                do_bench = true;
//        else if (a == "--logn"   && i+1<argc)   logn     = std::stoi(argv[++i]);
//        else if (a == "--batch"  && i+1<argc)   batch    = std::stoi(argv[++i]);
//        else if (a == "--runs"   && i+1<argc)   runs     = std::stoi(argv[++i]);
//        else if (a == "--warmup" && i+1<argc)   warmup   = std::stoi(argv[++i]);
//        else if (a == "--ncu"    && i+1<argc)   g_num_cu = std::stoi(argv[++i]);
//        else if (a == "-h" || a == "--help") {
//            std::cout << "Usage: " << argv[0] << " <xclbin> [options]\n"
//                      << "  --tests       Correctness test\n"
//                      << "  --bench       Latency benchmark\n"
//                      << "  --logn N      log2(transform size) [" << DEF_LOGN << "]\n"
//                      << "  --batch B     Batch size [" << DEF_BATCH << "]\n"
//                      << "  --ncu C       Compute units [" << g_num_cu << "]\n"
//                      << "  --runs R      Timed runs [" << DEF_RUNS << "]\n"
//                      << "  --warmup W    Warmup runs [" << DEF_WARMUP << "]\n";
//            return 0;
//        }
//        else if (xclbin_path.empty()) xclbin_path = a;
//        else { std::cerr << "Unknown: " << a << std::endl; return 1; }
//    }
//    if (xclbin_path.empty()) { std::cerr << "Need xclbin path\n"; return 1; }
//    if (!do_tests && !do_bench) { do_tests = do_bench = true; }
//
//    // OpenCL setup
//    std::vector<cl::Platform> plats; cl::Platform::get(&plats);
//    cl::Platform plat;
//    for (auto &p : plats)
//        if (p.getInfo<CL_PLATFORM_NAME>().find("Xilinx") != std::string::npos) { plat = p; break; }
//    std::vector<cl::Device> devs; plat.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devs);
//    if (devs.empty()) { std::cerr << "No devices\n"; return 1; }
//    g_dev = devs[0]; g_ctx = cl::Context(g_dev);
//
//    std::ifstream f(xclbin_path, std::ios::binary);
//    f.seekg(0, std::ios::end); size_t sz=f.tellg(); f.seekg(0);
//    std::vector<unsigned char> bin(sz); f.read((char*)bin.data(), sz);
//    cl::Program::Binaries binaries{bin};
//    cl_int err;
//    g_prog = cl::Program(g_ctx, {g_dev}, binaries, nullptr, &err);
//    if (err != CL_SUCCESS) { std::cerr << "Program failed\n"; return 1; }
//
//    std::cout << "Device : " << g_dev.getInfo<CL_DEVICE_NAME>() << std::endl;
//    std::cout << "CUs    : " << g_num_cu << std::endl;
//    std::cout << "xclbin : " << xclbin_path << std::endl;
//
//    if (do_tests) {
//        if (run_correctness(logn, batch) != 0) {
//            std::cout << "FAILED. Skipping benchmark.\n";
//            _exit(1);
//        }
//    }
//
//    if (do_bench) {
//        uint32_t N = 1u << logn;
//        auto s = run_bench(logn, batch, runs, warmup, g_num_cu);
//        double throughput = (double)(batch * N) / (s.median_ms / 1000.0) / 1e6;
//
//        std::cout << "\n NTT Benchmark — " << g_num_cu << " CU\n"
//                  << "+---------+---------+------+------------+----------+-----------+\n"
//                  << "| log2(N) |    N    | CUs  | median(ms) | p90(ms)  | Mcoeff/s  |\n"
//                  << "+---------+---------+------+------------+----------+-----------+\n"
//                  << "| " << std::setw(7) << logn
//                  << " | " << std::setw(7) << N
//                  << " | " << std::setw(4) << g_num_cu
//                  << " | " << std::setw(10) << std::fixed << std::setprecision(3) << s.median_ms
//                  << " | " << std::setw(8) << s.p90_ms
//                  << " | " << std::setw(9) << std::setprecision(2) << throughput << " |\n"
//                  << "+---------+---------+------+------------+----------+-----------+\n";
//    }
//
//    std::cout << std::endl;
//    _exit(0);
//}
