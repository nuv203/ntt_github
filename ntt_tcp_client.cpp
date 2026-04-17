/*
 * ntt_tcp_client.cpp
 *
 * Benchmarking client for the NTT FPGA accelerator server.
 *
 * Runs three benchmark sections:
 *   1. NTT throughput (existing) — FPGA, various logN and batch sizes
 *   2. CPU vs FPGA comparison   — same cases, ARM CPU on Kria vs FPGA PL
 *   3. HE primitive demo        — BFV-style ring operations using NTT
 *
 * Usage: ./ntt_tcp_client <server-ip> [port]
 *
 * Protocol (per connection):
 *   Client → Server : uint32_t logN, batch, q, psi_words, tw_words, num_runs, mode
 *                       mode 0 = FPGA accelerator
 *                       mode 1 = Kria ARM CPU (software NTT)
 *   Client → Server : psi_words × uint32_t  (pre-built psi_powers table)
 *   Client → Server : tw_words  × uint32_t  (pre-built twiddle table)
 *   [loop num_runs:]
 *     Client → Server : batch*N × uint32_t  (input coefficients)
 *     Server → Client : batch*N × uint32_t  (NTT results)
 *     Server → Client : uint64_t proc_us    (server-side processing time)
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cstdint>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// ===================================================================
// Benchmark configuration
// ===================================================================

static constexpr int DEF_PORT   = 54321;
static constexpr int DEF_BL     = 31;
static constexpr int NUM_RUNS   = 10;   /* timed runs per case   */
static constexpr int NUM_WARMUP = 2;    /* warm-up runs per case */

struct BenchCase { uint32_t logN; uint32_t batch; };

static const BenchCase BENCH_CASES[] = {
    { 8,  1}, { 9,  1},
    {10,  1}, {10,  4}, {10,  8},
    {11,  1},
    {12,  1}, {12,  4},
    {13,  1}, {14,  1}, {15,  1}, {16,  1},
};
static constexpr int NUM_CASES = sizeof(BENCH_CASES) / sizeof(BENCH_CASES[0]);

static constexpr uint32_t TILE_N = 4096;

// ===================================================================
// Modular arithmetic (CPU reference)
// ===================================================================

static uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t q) {
    return (uint32_t)(((uint64_t)a * b) % q);
}
static uint32_t mod_add(uint32_t a, uint32_t b, uint32_t q) {
    uint64_t s = (uint64_t)a + b;
    return (uint32_t)(s >= q ? s - q : s);
}
static uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}
static uint32_t power_mod(uint32_t base, uint64_t exp, uint32_t mod) {
    uint64_t r = 1, b = base % mod;
    while (exp > 0) {
        if (exp & 1) r = (r * b) % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return (uint32_t)r;
}
static uint32_t xrand(uint64_t &st, uint32_t q) {
    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
    return (uint32_t)(st % q);
}
static bool is_prime(uint64_t n) {
    if (n < 2) return false; if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t d = 5; d * d <= n; d += 6)
        if (n % d == 0 || n % (d + 2) == 0) return false;
    return true;
}
static uint32_t gen_modulus(uint32_t N, int bl = DEF_BL) {
    uint64_t step = 2 * (uint64_t)N, lim = ((uint64_t)1 << bl) - 1;
    uint64_t k = (lim - 1) / step, c = k * step + 1;
    while (c >= 2) { if (is_prime(c)) return (uint32_t)c; c -= step; }
    return 0;
}
static uint32_t find_psi(uint32_t N, uint32_t q) {
    uint32_t phi = q - 1;
    std::vector<uint32_t> facs; uint64_t x = phi;
    for (uint64_t d = 2; d * d <= x; d++)
        if (x % d == 0) { facs.push_back((uint32_t)d); while (x % d == 0) x /= d; }
    if (x > 1) facs.push_back((uint32_t)x);
    uint32_t g = 0;
    for (uint32_t c = 2; c < q; c++) {
        bool ok = true;
        for (auto f : facs) if (power_mod(c, phi / f, q) == 1) { ok = false; break; }
        if (ok) { g = c; break; }
    }
    return power_mod(g, (q - 1) / (2 * N), q);
}

// ===================================================================
// Twiddle table generation (sent to server — not computed there)
// ===================================================================

static void make_stockham_tw(uint32_t M, uint32_t omega_M, uint32_t q,
                             std::vector<uint32_t> &tw) {
    uint32_t logM = 0;
    for (uint32_t t = M; t > 1; t >>= 1) logM++;
    tw.assign(M, 1);
    for (uint32_t s = 0; s < logM; s++) {
        uint32_t span   = 1u << s;
        uint32_t stride = M / (2 * span);
        uint32_t step   = power_mod(omega_M, stride, q), cur = 1;
        for (uint32_t j = 0; j < span; j++) { tw[span + j] = cur; cur = mod_mul(cur, step, q); }
    }
}

static void make_tables(uint32_t N, uint32_t q, uint32_t psi,
                        std::vector<uint32_t> &pp, std::vector<uint32_t> &tw) {
    uint32_t omega = mod_mul(psi, psi, q);
    pp.resize(N); pp[0] = 1;
    for (uint32_t i = 1; i < N; i++) pp[i] = mod_mul(pp[i-1], psi, q);

    if (N <= TILE_N) {
        make_stockham_tw(N, omega, q, tw);
    } else {
        uint32_t logN = 0; for (uint32_t t = N; t > 1; t >>= 1) logN++;
        uint32_t logN1 = logN >> 1, logN2 = logN - logN1;
        uint32_t N1 = 1u << logN1, N2 = 1u << logN2;
        std::vector<uint32_t> tw_col, tw_row;
        make_stockham_tw(N2, power_mod(omega, N1, q), q, tw_col);
        make_stockham_tw(N1, power_mod(omega, N2, q), q, tw_row);
        tw.resize(N2 + N1 + N);
        for (uint32_t i = 0; i < N2; i++) tw[i]      = tw_col[i];
        for (uint32_t i = 0; i < N1; i++) tw[N2 + i] = tw_row[i];
        for (uint32_t row = 0; row < N2; row++)
            for (uint32_t col = 0; col < N1; col++)
                tw[N2 + N1 + row*N1 + col] = power_mod(omega, ((uint64_t)col * row) % N, q);
    }
}

static size_t psi_buf_words(uint32_t N, uint32_t batch) {
    return (N > TILE_N) ? (size_t)batch * N : (size_t)N;
}

// ===================================================================
// Reference NTT (CPU golden oracle — forward)
// ===================================================================
static void ref_ntt(std::vector<uint32_t> &a,
                    const std::vector<uint32_t> &pp, uint32_t q) {
    uint32_t N = (uint32_t)a.size(), logN = 0;
    for (uint32_t t = N; t > 1; t >>= 1) logN++;
    for (uint32_t i = 0; i < N; i++) a[i] = mod_mul(a[i], pp[i], q);
    auto rev = [&](uint32_t x) {
        uint32_t r = 0;
        for (uint32_t i = 0; i < logN; i++) { r = (r << 1) | (x & 1); x >>= 1; }
        return r;
    };
    for (uint32_t i = 0; i < N; i++) { uint32_t j = rev(i); if (j > i) std::swap(a[i], a[j]); }
    uint32_t omega = mod_mul(pp[1], pp[1], q);
    for (uint32_t s = 0; s < logN; s++) {
        uint32_t span = 1u << s, span2 = span << 1;
        uint32_t ws = power_mod(omega, N / (2 * span), q);
        for (uint32_t k = 0; k < N; k += span2) {
            uint32_t w = 1;
            for (uint32_t j = 0; j < span; j++) {
                uint32_t u = a[k+j], v = mod_mul(a[k+j+span], w, q);
                a[k+j] = mod_add(u, v, q); a[k+j+span] = mod_sub(u, v, q);
                w = mod_mul(w, ws, q);
            }
        }
    }
}

// ===================================================================
// Inverse NTT (CPU golden oracle — inverse)
//
// Implements the exact inverse of ref_ntt:
//   ref_intt(ref_ntt(a)) == a
//
// Algorithm: DIF butterfly (Gentleman-Sande) with omega_inv in reverse
// stage order → bit-reversal → scale by 1/N → un-twist by psi_inv^i.
// ===================================================================
static void ref_intt(std::vector<uint32_t> &a,
                     const std::vector<uint32_t> &pp, uint32_t q) {
    uint32_t N = (uint32_t)a.size(), logN = 0;
    for (uint32_t t = N; t > 1; t >>= 1) logN++;

    uint32_t psi       = pp[1];
    uint32_t omega     = mod_mul(psi, psi, q);
    uint32_t omega_inv = power_mod(omega, q - 2, q);
    uint32_t N_inv     = power_mod(N, q - 2, q);
    uint32_t psi_inv   = power_mod(psi, q - 2, q);

    // DIF butterfly stages with omega_inv, in reverse stage order.
    for (int s = (int)logN - 1; s >= 0; s--) {
        uint32_t span  = 1u << s;
        uint32_t span2 = span << 1;
        uint32_t ws    = power_mod(omega_inv, N / span2, q);
        for (uint32_t k = 0; k < N; k += span2) {
            uint32_t w = 1;
            for (uint32_t j = 0; j < span; j++) {
                uint32_t u = a[k + j], v = a[k + j + span];
                a[k + j]        = mod_add(u, v, q);
                a[k + j + span] = mod_mul(mod_sub(u, v, q), w, q);
                w = mod_mul(w, ws, q);
            }
        }
    }

    // Bit-reversal (same as forward NTT — undoes the DIF output permutation).
    auto rev = [&](uint32_t x) {
        uint32_t r = 0;
        for (uint32_t i = 0; i < logN; i++) { r = (r << 1) | (x & 1); x >>= 1; }
        return r;
    };
    for (uint32_t i = 0; i < N; i++) { uint32_t j = rev(i); if (j > i) std::swap(a[i], a[j]); }

    // Scale by 1/N and undo psi twist: a[i] *= N_inv * psi_inv^i.
    uint32_t cur_pi = 1;   // psi_inv^0
    for (uint32_t i = 0; i < N; i++) {
        a[i]   = mod_mul(mod_mul(a[i], N_inv, q), cur_pi, q);
        cur_pi = mod_mul(cur_pi, psi_inv, q);
    }
}

// ===================================================================
// Ring polynomial helpers (for HE demo)
// ===================================================================

/*
 * Schoolbook negacyclic polynomial multiplication in Z_q[x]/(x^N+1).
 * O(N^2) — used only for correctness verification on small N.
 * The ring relation x^N ≡ -1 means coefficients that wrap past index N
 * are subtracted rather than added.
 */
static std::vector<uint32_t> poly_mul_ref(
        const std::vector<uint32_t> &a, const std::vector<uint32_t> &b, uint32_t q) {
    uint32_t N = (uint32_t)a.size();
    std::vector<uint32_t> c(N, 0);
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            uint32_t k    = (i + j) % N;
            uint32_t prod = mod_mul(a[i], b[j], q);
            if (i + j >= N)
                c[k] = mod_sub(c[k], prod, q);   // negacyclic sign flip
            else
                c[k] = mod_add(c[k], prod, q);
        }
    }
    return c;
}

/* Coefficient-wise addition mod q (no NTT needed — models HE ciphertext add). */
static std::vector<uint32_t> he_poly_add(
        const std::vector<uint32_t> &a, const std::vector<uint32_t> &b, uint32_t q) {
    std::vector<uint32_t> c(a.size());
    for (size_t i = 0; i < a.size(); i++) c[i] = mod_add(a[i], b[i], q);
    return c;
}

/* Pointwise multiplication in NTT domain (models HE ciphertext multiplication
 * after both ciphertexts have been NTT-transformed). */
static std::vector<uint32_t> he_poly_mul_ntt(
        const uint32_t *ntt_a, const uint32_t *ntt_b, uint32_t N, uint32_t q) {
    std::vector<uint32_t> c(N);
    for (uint32_t i = 0; i < N; i++) c[i] = mod_mul(ntt_a[i], ntt_b[i], q);
    return c;
}

// ===================================================================
// Socket helpers
// ===================================================================
static bool recv_all(int fd, void *buf, size_t n) {
    char *p = static_cast<char *>(buf);
    while (n > 0) { ssize_t r = recv(fd, p, n, 0); if (r <= 0) return false; p += r; n -= r; }
    return true;
}
static bool send_all(int fd, const void *buf, size_t n) {
    const char *p = static_cast<const char *>(buf);
    while (n > 0) { ssize_t s = send(fd, p, n, 0); if (s <= 0) return false; p += s; n -= s; }
    return true;
}

// ===================================================================
// Connect helper
// ===================================================================
static int connect_to(const char *ip, int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;
    sockaddr_in srv{};
    srv.sin_family = AF_INET;
    srv.sin_port   = htons(port);
    inet_pton(AF_INET, ip, &srv.sin_addr);
    if (connect(sock, reinterpret_cast<sockaddr *>(&srv), sizeof(srv)) < 0) {
        close(sock); return -1;
    }
    return sock;
}

/*
 * Send connection handshake: 7 scalar fields then the two pre-built tables.
 * mode: 0 = FPGA accelerator, 1 = Kria ARM CPU.
 */
static bool send_handshake(int sock, uint32_t logN, uint32_t batch,
                           uint32_t num_runs, uint32_t mode,
                           const std::vector<uint32_t> &pp,
                           const std::vector<uint32_t> &tw,
                           uint32_t q) {
    uint32_t N         = 1u << logN;
    uint32_t psi_words = (uint32_t)psi_buf_words(N, batch);
    uint32_t tw_words  = (uint32_t)tw.size();

    std::vector<uint32_t> psi_buf(psi_words, 0);
    std::copy(pp.begin(), pp.end(), psi_buf.begin());

    return send_all(sock, &logN,      sizeof(logN))      &&
           send_all(sock, &batch,     sizeof(batch))     &&
           send_all(sock, &q,         sizeof(q))         &&
           send_all(sock, &psi_words, sizeof(psi_words)) &&
           send_all(sock, &tw_words,  sizeof(tw_words))  &&
           send_all(sock, &num_runs,  sizeof(num_runs))  &&
           send_all(sock, &mode,      sizeof(mode))      &&
           send_all(sock, psi_buf.data(), psi_words * sizeof(uint32_t)) &&
           send_all(sock, tw.data(),      tw_words  * sizeof(uint32_t));
}

// ===================================================================
// Stats
// ===================================================================
struct Stats { double median_us, p90_us, min_us, max_us; };

static Stats compute_stats(std::vector<double> &v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double med = (n % 2 == 1) ? v[n/2] : (v[n/2-1] + v[n/2]) / 2.0;
    return { med, v[std::min((size_t)(0.9*n), n-1)], v[0], v[n-1] };
}

// ===================================================================
// Result storage (for FPGA vs CPU comparison table)
// ===================================================================
struct CaseResult {
    double proc_ms;  // server-side processing time (median)
    double rtt_ms;   // round-trip time (median)
};

// ===================================================================
// Helper: run one set of timed runs, return stats
// ===================================================================
static bool run_timed(int sock, uint32_t batch, uint32_t N,
                      const std::vector<uint32_t> &input,
                      uint32_t total_runs, uint32_t warmup,
                      std::vector<double> &rtt_us_out,
                      std::vector<double> &proc_us_out) {
    size_t data_bytes = (size_t)batch * N * sizeof(uint32_t);
    std::vector<uint32_t> result(batch * N);
    rtt_us_out.clear();  proc_us_out.clear();
    rtt_us_out.reserve(total_runs - warmup);
    proc_us_out.reserve(total_runs - warmup);

    for (uint32_t r = 0; r < total_runs; r++) {
        auto t0 = std::chrono::steady_clock::now();
        if (!send_all(sock, input.data(), data_bytes)) return false;
        uint64_t proc_us = 0;
        if (!recv_all(sock, result.data(), data_bytes)) return false;
        if (!recv_all(sock, &proc_us, sizeof(proc_us))) return false;
        auto t1 = std::chrono::steady_clock::now();
        double rtt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (r >= warmup) {
            rtt_us_out.push_back(rtt);
            proc_us_out.push_back((double)proc_us);
        }
    }
    return true;
}

// ===================================================================
// CPU vs FPGA comparison benchmark
// ===================================================================
static void run_cpu_comparison(const char *server_ip, int port,
                               const std::vector<CaseResult> &fpga_res) {
    std::cout << "\n=== CPU vs FPGA Comparison (Kria ARM A53 vs PL @ 200 MHz) ===\n\n";
    std::cout << std::left
              << std::setw(7)  << "logN"
              << std::setw(9)  << "N"
              << std::setw(7)  << "batch"
              << std::setw(14) << "FPGA(ms)"
              << std::setw(14) << "ARM(ms)"
              << std::setw(10) << "Speedup"
              << "\n"
              << std::string(61, '-') << "\n";

    for (int ci = 0; ci < NUM_CASES; ci++) {
        uint32_t logN  = BENCH_CASES[ci].logN;
        uint32_t batch = BENCH_CASES[ci].batch;
        uint32_t N     = 1u << logN;
        uint32_t total = NUM_WARMUP + NUM_RUNS;

        int sock = connect_to(server_ip, port);
        if (sock < 0) {
            std::cerr << "connect() failed for logN=" << logN << " (CPU)\n"; continue;
        }

        uint32_t q   = gen_modulus(N);
        uint32_t psi = find_psi(N, q);
        std::vector<uint32_t> pp, tw;
        make_tables(N, q, psi, pp, tw);

        // mode=1: ARM CPU NTT on Kria
        if (!send_handshake(sock, logN, batch, total, /*mode=*/1, pp, tw, q)) {
            std::cerr << "Handshake failed (CPU, logN=" << logN << ")\n";
            close(sock); continue;
        }

        uint64_t rng = 77;
        std::vector<uint32_t> input(batch * N);
        for (auto &v : input) v = xrand(rng, q);

        std::vector<double> rtt_us, proc_us;
        if (!run_timed(sock, batch, N, input, total, NUM_WARMUP, rtt_us, proc_us)) {
            close(sock); continue;
        }
        close(sock);

        auto arm_s  = compute_stats(proc_us);
        double fpga_ms = fpga_res[ci].proc_ms;
        double arm_ms  = arm_s.median_us / 1000.0;
        double speedup = (arm_ms > 0) ? arm_ms / fpga_ms : 0.0;

        std::cout << std::fixed
                  << std::left  << std::setw(7)  << logN
                  << std::left  << std::setw(9)  << N
                  << std::left  << std::setw(7)  << batch
                  << std::right << std::setw(10) << std::setprecision(3) << fpga_ms << "   "
                  << std::right << std::setw(10) << arm_ms  << "   "
                  << std::right << std::setw(6)  << std::setprecision(1) << speedup << "x"
                  << "\n";
    }
}

// ===================================================================
// Homomorphic Encryption primitive demo
// ===================================================================
/*
 * Demonstrates BFV-style ring arithmetic where NTT is the bottleneck:
 *
 *   HE-Add:  (p1 + p2) in Z_q[x]/(x^N+1) — coefficient-wise, no NTT
 *   HE-Mul:  INTT(NTT(p1) ⊙ NTT(p2))     — NTT on server, INTT on client
 *
 * Two server modes are benchmarked: FPGA PL (mode=0) and Kria ARM (mode=1),
 * giving a direct comparison of FPGA acceleration for the NTT bottleneck.
 *
 * The scalar demo uses constant polynomials:
 *   p = [m, 0, 0, ..., 0]  — convolution gives [m1*m2, 0, ..., 0]
 *
 * The ring multiply correctness check uses random polynomials verified
 * against a schoolbook O(N^2) reference.
 */
static void run_he_benchmark(const char *server_ip, int port) {
    std::cout << "\n=== Homomorphic Encryption Primitive Demo ===\n";
    std::cout << "Ring: Z_q[x] / (x^N + 1)   (negacyclic ring used in Kyber / Dilithium)\n\n";

    // ── Parameters ────────────────────────────────────────────────────────
    // N=512 for correctness demos (schoolbook O(N^2) is ~1ms)
    // N=4096 for throughput (standard BFV security parameter)
    static constexpr uint32_t DEMO_LOGN =  9;   // N=512 for scalar/correctness demos
    static constexpr uint32_t DEMO_N    = 1u << DEMO_LOGN;
    static constexpr uint32_t HE_BATCH  =  2;   // two polynomials per NTT call

    // Build tables for DEMO_N
    uint32_t q_d   = gen_modulus(DEMO_N);
    uint32_t psi_d = find_psi(DEMO_N, q_d);
    std::vector<uint32_t> pp_d, tw_d;
    make_tables(DEMO_N, q_d, psi_d, pp_d, tw_d);

    // ── Scalar encoding demo ──────────────────────────────────────────────
    // Encode integers as constant polynomials: [m, 0, ..., 0].
    // Ring convolution of two constant polys gives [m1*m2 mod q, 0, ..., 0].
    struct ScalarPair { uint32_t a, b; };
    static const ScalarPair ADD_PAIRS[] = {{42, 17}, {100, 200}, {999, 1}, {1234, 5678}};
    static const ScalarPair MUL_PAIRS[] = {{42, 17}, {100, 100}, {37, 99}, {255, 255}};

    std::cout << "--- Scalar operations (N=" << DEMO_N << ", q=" << q_d << ") ---\n\n";

    // Helper lambda: send two polys as batch=2, return product poly via INTT
    auto he_mul_via_server = [&](uint32_t mode_val,
                                 const std::vector<uint32_t> &pa,
                                 const std::vector<uint32_t> &pb,
                                 uint32_t N, uint32_t q,
                                 const std::vector<uint32_t> &pp,
                                 const std::vector<uint32_t> &tw,
                                 uint32_t logN) -> std::vector<uint32_t> {
        int sock = connect_to(server_ip, port);
        if (sock < 0) return {};

        // Pack [pa | pb] into single batch=2 buffer.
        std::vector<uint32_t> data_in(HE_BATCH * N);
        std::copy(pa.begin(), pa.end(), data_in.begin());
        std::copy(pb.begin(), pb.end(), data_in.begin() + N);

        if (!send_handshake(sock, logN, HE_BATCH, /*runs=*/1, mode_val, pp, tw, q)) {
            close(sock); return {};
        }

        size_t data_bytes = HE_BATCH * N * sizeof(uint32_t);
        std::vector<uint32_t> ntt_out(HE_BATCH * N);
        uint64_t proc_us = 0;

        if (!send_all(sock, data_in.data(), data_bytes)  ||
            !recv_all(sock, ntt_out.data(), data_bytes)  ||
            !recv_all(sock, &proc_us, sizeof(proc_us))) {
            close(sock); return {};
        }
        close(sock);

        auto ntt_prod = he_poly_mul_ntt(ntt_out.data(), ntt_out.data() + N, N, q);
        ref_intt(ntt_prod, pp, q);
        return ntt_prod;
    };

    // HE-Add: coefficient-wise, no NTT
    std::cout << "  HE-Add  (client only, no NTT):\n";
    for (auto &pr : ADD_PAIRS) {
        std::vector<uint32_t> pa(DEMO_N, 0), pb(DEMO_N, 0);
        pa[0] = pr.a; pb[0] = pr.b;
        auto p_sum = he_poly_add(pa, pb, q_d);
        uint32_t expected = (pr.a + pr.b) % q_d;
        bool ok = (p_sum[0] == expected);
        std::cout << "    " << std::setw(5) << pr.a << "  +  " << std::setw(5) << pr.b
                  << "  =  " << std::setw(6) << p_sum[0]
                  << "   [" << (ok ? "PASSED" : "FAILED") << "]\n";
    }

    // HE-Mul via FPGA (mode=0)
    std::cout << "\n  HE-Mul  (FPGA NTT + client INTT):\n";
    for (auto &pr : MUL_PAIRS) {
        std::vector<uint32_t> pa(DEMO_N, 0), pb(DEMO_N, 0);
        pa[0] = pr.a; pb[0] = pr.b;
        auto prod = he_mul_via_server(0, pa, pb, DEMO_N, q_d, pp_d, tw_d, DEMO_LOGN);
        if (prod.empty()) {
            std::cout << "    " << pr.a << " × " << pr.b << "  connection failed\n";
        } else {
            uint32_t expected = (uint32_t)((uint64_t)pr.a * pr.b % q_d);
            bool ok = (prod[0] == expected);
            std::cout << "    " << std::setw(5) << pr.a << "  x  " << std::setw(5) << pr.b
                      << "  =  " << std::setw(6) << prod[0]
                      << "   [" << (ok ? "PASSED" : "FAILED") << "]\n";
        }
    }

    // HE-Mul via ARM (mode=1)
    std::cout << "\n  HE-Mul  (Kria ARM NTT + client INTT):\n";
    for (auto &pr : MUL_PAIRS) {
        std::vector<uint32_t> pa(DEMO_N, 0), pb(DEMO_N, 0);
        pa[0] = pr.a; pb[0] = pr.b;
        auto prod = he_mul_via_server(1, pa, pb, DEMO_N, q_d, pp_d, tw_d, DEMO_LOGN);
        if (prod.empty()) {
            std::cout << "    " << pr.a << " × " << pr.b << "  connection failed\n";
        } else {
            uint32_t expected = (uint32_t)((uint64_t)pr.a * pr.b % q_d);
            bool ok = (prod[0] == expected);
            std::cout << "    " << std::setw(5) << pr.a << "  x  " << std::setw(5) << pr.b
                      << "  =  " << std::setw(6) << prod[0]
                      << "   [" << (ok ? "PASSED" : "FAILED") << "]\n";
        }
    }

    // ── Ring multiplication correctness (random polynomials) ─────────────
    std::cout << "\n--- Ring multiply correctness (N=" << DEMO_N
              << ", random polys) ---\n";
    {
        uint64_t rng = 0xDEADBEEF42ULL;
        std::vector<uint32_t> ra(DEMO_N), rb(DEMO_N);
        for (auto &v : ra) v = xrand(rng, q_d);
        for (auto &v : rb) v = xrand(rng, q_d);

        // CPU schoolbook reference (O(N^2))
        auto ref = poly_mul_ref(ra, rb, q_d);

        // FPGA NTT path
        auto fpga_prod = he_mul_via_server(0, ra, rb, DEMO_N, q_d, pp_d, tw_d, DEMO_LOGN);
        if (fpga_prod.empty()) {
            std::cout << "  FPGA ring mul: connection failed\n";
        } else {
            bool ok = (fpga_prod == ref);
            std::cout << "  FPGA NTT product vs schoolbook: [" << (ok ? "PASSED" : "FAILED") << "]\n";
        }

        // ARM NTT path
        auto arm_prod = he_mul_via_server(1, ra, rb, DEMO_N, q_d, pp_d, tw_d, DEMO_LOGN);
        if (arm_prod.empty()) {
            std::cout << "  ARM  ring mul: connection failed\n";
        } else {
            bool ok = (arm_prod == ref);
            std::cout << "  ARM  NTT product vs schoolbook: [" << (ok ? "PASSED" : "FAILED") << "]\n";
        }
    }

    // ── HE-Mul throughput (FPGA vs ARM, varying N and batch) ─────────────
    std::cout << "\n--- HE-Mul throughput (" << NUM_RUNS << " runs, "
              << NUM_WARMUP << " warmup) ---\n\n";

    // Cases: {logN, batch} — batch is always even (two polys per HE-Mul)
    struct HeCase { uint32_t logN; uint32_t batch; };
    static const HeCase HE_CASES[] = {
        { 9,  2}, {10,  2}, {11,  2},
        {12,  2}, {12,  4}, {12,  8},
        {13,  2}, {14,  2},
    };
    static constexpr int NUM_HE_CASES = (int)(sizeof(HE_CASES) / sizeof(HE_CASES[0]));
    static constexpr uint32_t HE_TOTAL = NUM_WARMUP + NUM_RUNS;

    // Print header
    std::cout << std::left
              << std::setw(6)  << " logN"
              << std::setw(8)  << "      N"
              << std::setw(7)  << "  batch"
              << std::setw(14) << "  FPGA NTT(ms)"
              << std::setw(13) << "  ARM NTT(ms)"
              << std::setw(10) << "  Speedup"
              << std::setw(17) << "  INTT+ptwise(ms)"
              << std::setw(15) << "  Full FPGA(ms)"
              << std::setw(14) << "  Full ARM(ms)"
              << "\n"
              << std::string(104, '-') << "\n";

    for (int ci = 0; ci < NUM_HE_CASES; ci++) {
        uint32_t logN  = HE_CASES[ci].logN;
        uint32_t batch = HE_CASES[ci].batch;
        uint32_t N     = 1u << logN;

        // Build tables for this N
        uint32_t q   = gen_modulus(N);
        uint32_t psi = find_psi(N, q);
        std::vector<uint32_t> pp, tw;
        make_tables(N, q, psi, pp, tw);

        uint64_t rng2 = 0xABCD1234ULL + ci;
        std::vector<uint32_t> he_input(batch * N);
        for (auto &v : he_input) v = xrand(rng2, q);
        size_t he_data_bytes = batch * N * sizeof(uint32_t);
        std::vector<uint32_t> he_result(batch * N);

        std::vector<double> fpga_proc_us, arm_proc_us;

        // FPGA throughput (mode=0)
        {
            int sock = connect_to(server_ip, port);
            if (sock >= 0) {
                send_handshake(sock, logN, batch, HE_TOTAL, 0, pp, tw, q);
                uint64_t proc_us = 0;
                for (uint32_t r = 0; r < HE_TOTAL; r++) {
                    send_all(sock, he_input.data(), he_data_bytes);
                    recv_all(sock, he_result.data(), he_data_bytes);
                    recv_all(sock, &proc_us, sizeof(proc_us));
                    if (r >= NUM_WARMUP) fpga_proc_us.push_back((double)proc_us);
                }
                close(sock);
            }
        }

        // ARM throughput (mode=1)
        {
            int sock = connect_to(server_ip, port);
            if (sock >= 0) {
                send_handshake(sock, logN, batch, HE_TOTAL, 1, pp, tw, q);
                uint64_t proc_us = 0;
                for (uint32_t r = 0; r < HE_TOTAL; r++) {
                    send_all(sock, he_input.data(), he_data_bytes);
                    recv_all(sock, he_result.data(), he_data_bytes);
                    recv_all(sock, &proc_us, sizeof(proc_us));
                    if (r >= NUM_WARMUP) arm_proc_us.push_back((double)proc_us);
                }
                close(sock);
            }
        }

        if (fpga_proc_us.empty() || arm_proc_us.empty()) {
            std::cout << "  logN=" << logN << " batch=" << batch << "  connection failed\n";
            continue;
        }

        // Client-side INTT + pointwise timing (same for both paths)
        std::vector<double> client_us_v;
        for (int r = 0; r < NUM_RUNS; r++) {
            // he_result holds the last NTT output from ARM run — good enough for timing
            auto tc0 = std::chrono::steady_clock::now();
            auto ntt_prod = he_poly_mul_ntt(he_result.data(), he_result.data() + N, N, q);
            ref_intt(ntt_prod, pp, q);
            double ct = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - tc0).count();
            client_us_v.push_back(ct);
        }

        auto fpga_s   = compute_stats(fpga_proc_us);
        auto arm_s    = compute_stats(arm_proc_us);
        auto client_s = compute_stats(client_us_v);

        double fpga_full = fpga_s.median_us + client_s.median_us;
        double arm_full  = arm_s.median_us  + client_s.median_us;
        double speedup   = (fpga_s.median_us > 0) ? arm_s.median_us / fpga_s.median_us : 0.0;

        std::cout << std::fixed
                  << std::right << std::setw(5)  << logN  << " "
                  << std::right << std::setw(7)  << N     << " "
                  << std::right << std::setw(6)  << batch << " "
                  << std::right << std::setw(13) << std::setprecision(3) << fpga_s.median_us / 1000.0 << " "
                  << std::right << std::setw(12) << arm_s.median_us  / 1000.0 << " "
                  << std::right << std::setw(8)  << std::setprecision(1) << speedup << "x "
                  << std::right << std::setw(16) << std::setprecision(3) << client_s.median_us / 1000.0 << " "
                  << std::right << std::setw(14) << fpga_full / 1000.0 << " "
                  << std::right << std::setw(13) << arm_full  / 1000.0
                  << "\n";
    }
}

// ===================================================================
// Main
// ===================================================================
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <server-ip> [port]\n"; return 1;
    }
    const char *server_ip = argv[1];
    int         port      = (argc >= 3) ? std::stoi(argv[2]) : DEF_PORT;

    // ── Correctness check (logN=10, batch=1, 1 run) ──────────────────
    std::cout << "=== Correctness check (logN=10, batch=1) ===\n";
    {
        int sock = connect_to(server_ip, port);
        if (sock < 0) { std::cerr << "connect() failed\n"; return 1; }

        uint32_t logN = 10, batch = 1, num_runs = 1, N = 1u << logN;

        uint32_t q   = gen_modulus(N);
        uint32_t psi = find_psi(N, q);
        std::vector<uint32_t> pp, tw;
        make_tables(N, q, psi, pp, tw);

        if (!send_handshake(sock, logN, batch, num_runs, /*mode=*/0, pp, tw, q)) {
            std::cerr << "Handshake failed\n"; return 1;
        }

        uint64_t rng = 42;
        std::vector<uint32_t> input(N);
        for (auto &v : input) v = xrand(rng, q);
        std::vector<uint32_t> gold(input);
        ref_ntt(gold, pp, q);

        send_all(sock, input.data(), N * sizeof(uint32_t));
        std::vector<uint32_t> result(N);
        uint64_t fpga_us = 0;
        recv_all(sock, result.data(), N * sizeof(uint32_t));
        recv_all(sock, &fpga_us, sizeof(fpga_us));
        close(sock);

        int mismatches = 0;
        for (uint32_t i = 0; i < N; i++) if (result[i] != gold[i]) mismatches++;
        std::cout << (mismatches == 0 ? "PASSED" : "FAILED")
                  << " (" << mismatches << " mismatches, FPGA took "
                  << fpga_us << " us)\n\n";
        if (mismatches) return 1;
    }

    // ── INTT self-check (client-side only, no server needed) ─────────
    {
        uint32_t N = 512;
        uint32_t q = gen_modulus(N), psi = find_psi(N, q);
        std::vector<uint32_t> pp, tw;
        make_tables(N, q, psi, pp, tw);
        uint64_t rng = 12345;
        std::vector<uint32_t> orig(N);
        for (auto &v : orig) v = xrand(rng, q);
        auto a = orig;
        ref_ntt(a, pp, q);
        ref_intt(a, pp, q);
        bool intt_ok = (a == orig);
        std::cout << "[self-check] INTT(NTT(x)) == x for N=512: ["
                  << (intt_ok ? "PASSED" : "FAILED") << "]\n\n";
    }

    // ── FPGA Benchmark loop ───────────────────────────────────────────
    std::cout << "=== FPGA Benchmark (" << NUM_RUNS << " runs, "
              << NUM_WARMUP << " warmup) ===\n\n";

    std::cout
        << std::left
        << std::setw(7)  << "logN"
        << std::setw(9)  << "N"
        << std::setw(7)  << "batch"
        << std::setw(16) << "round_trip(ms)"
        << std::setw(14) << "fpga(ms)"
        << std::setw(14) << "net_io(ms)"
        << std::setw(12) << "Mcoeff/s"
        << "\n"
        << std::string(79, '-') << "\n";

    std::vector<CaseResult> fpga_results(NUM_CASES, {0.0, 0.0});

    for (int ci = 0; ci < NUM_CASES; ci++) {
        uint32_t logN      = BENCH_CASES[ci].logN;
        uint32_t batch     = BENCH_CASES[ci].batch;
        uint32_t N         = 1u << logN;
        uint32_t total_runs = NUM_WARMUP + NUM_RUNS;

        int sock = connect_to(server_ip, port);
        if (sock < 0) {
            std::cerr << "connect() failed for logN=" << logN << "\n"; continue;
        }

        uint32_t q   = gen_modulus(N);
        uint32_t psi = find_psi(N, q);
        std::vector<uint32_t> pp, tw;
        make_tables(N, q, psi, pp, tw);

        if (!send_handshake(sock, logN, batch, total_runs, /*mode=*/0, pp, tw, q)) {
            std::cerr << "Handshake failed for logN=" << logN << "\n";
            close(sock); continue;
        }

        uint64_t rng = 99;
        std::vector<uint32_t> input(batch * N);
        for (auto &v : input) v = xrand(rng, q);

        std::vector<double> rtt_us, proc_us;
        if (!run_timed(sock, batch, N, input, total_runs, NUM_WARMUP, rtt_us, proc_us)) {
            close(sock); continue;
        }
        close(sock);

        auto rtt_s  = compute_stats(rtt_us);
        auto proc_s = compute_stats(proc_us);
        auto net_v  = rtt_us;
        for (size_t i = 0; i < net_v.size(); i++) net_v[i] -= proc_us[i];
        auto net_s  = compute_stats(net_v);
        double tp   = (double)((uint64_t)batch * N) / (rtt_s.median_us / 1e6) / 1e6;

        fpga_results[ci] = { proc_s.median_us / 1000.0, rtt_s.median_us / 1000.0 };

        std::cout << std::fixed
                  << std::left  << std::setw(7)  << logN
                  << std::left  << std::setw(9)  << N
                  << std::left  << std::setw(7)  << batch
                  << std::right << std::setw(12) << std::setprecision(3) << rtt_s.median_us  / 1000.0 << "   "
                  << std::right << std::setw(10) << proc_s.median_us / 1000.0 << "   "
                  << std::right << std::setw(10) << net_s.median_us  / 1000.0 << "   "
                  << std::right << std::setw(9)  << std::setprecision(2) << tp
                  << "\n";
    }

    // ── CPU vs FPGA comparison ────────────────────────────────────────
    run_cpu_comparison(server_ip, port, fpga_results);

    // ── HE primitive demo ─────────────────────────────────────────────
    run_he_benchmark(server_ip, port);

    std::cout << "\nDone.\n";
    return 0;
}
