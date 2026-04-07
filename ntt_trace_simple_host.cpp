/*
 * ntt_trace_simple_host.cpp
 *
 * Minimal XRT host for the negacyclic NTT kernel.
 * Modeled after the Xilinx vadd "hello world" example — no OpenCL,
 * no benchmarking, no multi-CU logic. Just allocate, run, verify.
 *
 * Usage: ./ntt_trace_simple_host <path/to/ntt.xclbin>
 */

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <chrono>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

// XRT native API — replaces OpenCL entirely
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// ===================================================================
// Parameters
// ===================================================================
static constexpr int      DEF_BL     = 31;    /* modulus bit-length              */
static constexpr uint32_t TILE_N     = 4096;  /* direct/four-step threshold      */
static constexpr int      SERVER_PORT = 54321; /* TCP port to listen on           */

// ===================================================================
// Modular arithmetic (CPU reference — not optimized)
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

static bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t d = 5; d * d <= n; d += 6)
        if (n % d == 0 || n % (d + 2) == 0) return false;
    return true;
}

/* Largest bl-bit prime q where (q-1) % (2*N) == 0 — required for negacyclic NTT. */
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

/* psi = primitive 2N-th root of unity mod q (psi^N ≡ -1 mod q). */
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

/* Xorshift64 PRNG for test vector generation. */
static uint32_t xrand(uint64_t &st, uint32_t q) {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return (uint32_t)(st % q);
}

// ===================================================================
// Twiddle table generation
// ===================================================================

/*
 * Stockham-packed twiddle table for an M-point NTT.
 * tw[span + j] = omega_M^(j * M/(2*span)) for stage s = log2(span).
 */
static void make_stockham_tw(uint32_t M, uint32_t omega_M, uint32_t q,
                             std::vector<uint32_t> &tw) {
    uint32_t logM = 0;
    for (uint32_t t = M; t > 1; t >>= 1) logM++;

    tw.assign(M, 1);
    for (uint32_t s = 0; s < logM; s++) {
        uint32_t span   = 1u << s;
        uint32_t stride = M / (2 * span);
        uint32_t step   = power_mod(omega_M, stride, q);
        uint32_t cur    = 1;
        for (uint32_t j = 0; j < span; j++) {
            tw[span + j] = cur;
            cur = mod_mul(cur, step, q);
        }
    }
}

/*
 * Build psi-powers table (pp) and twiddle table (tw).
 * Handles both direct path (N <= TILE_N) and four-step path (N > TILE_N).
 */
static void make_tables(uint32_t N, uint32_t q, uint32_t psi,
                        std::vector<uint32_t> &pp, std::vector<uint32_t> &tw) {
    uint32_t omega = mod_mul(psi, psi, q);

    pp.resize(N);
    pp[0] = 1;
    for (uint32_t i = 1; i < N; i++)
        pp[i] = mod_mul(pp[i - 1], psi, q);

    if (N <= TILE_N) {
        make_stockham_tw(N, omega, q, tw);
    } else {
        uint32_t logN = 0;
        for (uint32_t t = N; t > 1; t >>= 1) logN++;
        uint32_t logN1 = logN >> 1,   logN2 = logN - logN1;
        uint32_t N1    = 1u << logN1, N2    = 1u << logN2;

        std::vector<uint32_t> tw_col, tw_row;
        make_stockham_tw(N2, power_mod(omega, N1, q), q, tw_col);
        make_stockham_tw(N1, power_mod(omega, N2, q), q, tw_row);

        tw.resize(N2 + N1 + N);
        for (uint32_t i = 0; i < N2; i++) tw[i]      = tw_col[i];
        for (uint32_t i = 0; i < N1; i++) tw[N2 + i] = tw_row[i];
        for (uint32_t row = 0; row < N2; row++)
            for (uint32_t col = 0; col < N1; col++)
                tw[N2 + N1 + row * N1 + col] = power_mod(omega, ((uint64_t)col * row) % N, q);
    }
}

/*
 * Psi buffer must be large enough for the four-step transpose scratch.
 * Direct path: N words. Four-step path: batch*N words.
 */
static size_t psi_buf_words(uint32_t N, uint32_t batch) {
    return (N > TILE_N) ? (size_t)batch * N : (size_t)N;
}

// ===================================================================
// Reference NTT (CPU golden oracle for correctness check)
// ===================================================================
static void ref_ntt(std::vector<uint32_t> &a, const std::vector<uint32_t> &pp, uint32_t q) {
    uint32_t N = (uint32_t)a.size(), logN = 0;
    for (uint32_t t = N; t > 1; t >>= 1) logN++;

    for (uint32_t i = 0; i < N; i++)
        a[i] = mod_mul(a[i], pp[i], q);

    auto rev = [&](uint32_t x) {
        uint32_t r = 0;
        for (uint32_t i = 0; i < logN; i++) { r = (r << 1) | (x & 1); x >>= 1; }
        return r;
    };
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = rev(i);
        if (j > i) std::swap(a[i], a[j]);
    }

    uint32_t omega = mod_mul(pp[1], pp[1], q);
    for (uint32_t s = 0; s < logN; s++) {
        uint32_t span  = 1u << s;
        uint32_t span2 = span << 1;
        uint32_t ws    = power_mod(omega, N / (2 * span), q);
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
// Socket helpers
// ===================================================================

/* Receive exactly n bytes — keeps calling recv() until the buffer is full.
 * TCP is a stream; a single recv() may return fewer bytes than requested. */
static bool recv_all(int fd, void *buf, size_t n) {
    char *p = static_cast<char *>(buf);
    while (n > 0) {
        ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) return false;   /* connection closed or error */
        p += r;
        n -= r;
    }
    return true;
}

/* Send exactly n bytes — loops until all bytes are flushed into the socket. */
static bool send_all(int fd, const void *buf, size_t n) {
    const char *p = static_cast<const char *>(buf);
    while (n > 0) {
        ssize_t s = send(fd, p, n, 0);
        if (s <= 0) return false;
        p += s;
        n -= s;
    }
    return true;
}

// ===================================================================
// Main — TCP server
// ===================================================================
//
// Usage: ./ntt_trace_simple_host <ntt.xclbin> [port]
//
// Listens for TCP connections. Each connection follows this protocol:
//   Client → Server : uint32_t logN, uint32_t batch
//   Server → Client : uint32_t q            (modulus, so client can verify)
//   Client → Server : batch*N × uint32_t    (input coefficients)
//   Server → Client : batch*N × uint32_t    (NTT results)
//
// The XRT device and kernel are opened once at startup and reused across
// all connections — no reload per request.

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <ntt.xclbin> [port]\n";
        return 1;
    }
    std::string xclbin_path = argv[1];
    int port = (argc >= 3) ? std::stoi(argv[2]) : SERVER_PORT;

    // ── Open XRT device and load xclbin (once) ────────────────────────
    std::cout << "Open device 0\n";
    auto device = xrt::device(0);

    std::cout << "Load xclbin: " << xclbin_path << "\n";
    auto uuid = device.load_xclbin(xclbin_path);

    auto krnl = xrt::kernel(device, uuid, "ntt_kernel");
    std::cout << "Kernel ready\n";

    // ── Create server socket ──────────────────────────────────────────
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) { std::cerr << "socket() failed\n"; return 1; }

    /* Allow immediate restart after crash without waiting for TIME_WAIT. */
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);

    if (bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        std::cerr << "bind() failed\n"; return 1;
    }
    listen(listen_fd, 1);
    std::cout << "Listening on port " << port << "\n";

    // ── Accept loop ───────────────────────────────────────────────────
    while (true) {
        sockaddr_in client_addr{};
        socklen_t   client_len = sizeof(client_addr);
        int conn_fd = accept(listen_fd, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
        if (conn_fd < 0) { std::cerr << "accept() failed\n"; continue; }
        std::cout << "Client connected\n";

        // ── Receive transform parameters and run count ────────────────
        uint32_t logN, batch, num_runs;
        if (!recv_all(conn_fd, &logN,     sizeof(logN))     ||
            !recv_all(conn_fd, &batch,    sizeof(batch))    ||
            !recv_all(conn_fd, &num_runs, sizeof(num_runs))) {
            std::cerr << "Failed to receive parameters\n";
            close(conn_fd); continue;
        }
        uint32_t N = 1u << logN;
        std::cout << "logN=" << logN << "  N=" << N
                  << "  batch=" << batch << "  runs=" << num_runs << "\n";

        // ── Build NTT tables once for this connection ─────────────────
        uint32_t q   = gen_modulus(N);
        uint32_t psi = find_psi(N, q);
        std::vector<uint32_t> pp, tw;
        make_tables(N, q, psi, pp, tw);

        // ── Send q so client can generate valid coefficients ──────────
        if (!send_all(conn_fd, &q, sizeof(q))) {
            std::cerr << "Failed to send q\n";
            close(conn_fd); continue;
        }

        // ── Allocate FPGA buffers once for this (N, batch) ───────────
        size_t data_bytes = (size_t)batch * N * sizeof(uint32_t);
        size_t pw         = psi_buf_words(N, batch);
        size_t psi_bytes  = pw * sizeof(uint32_t);
        size_t tw_bytes   = tw.size() * sizeof(uint32_t);

        auto bo_data = xrt::bo(device, data_bytes, krnl.group_id(0));
        auto bo_psi  = xrt::bo(device, psi_bytes,  krnl.group_id(1));
        auto bo_tw   = xrt::bo(device, tw_bytes,   krnl.group_id(2));

        auto data_map = bo_data.map<uint32_t *>();
        auto psi_map  = bo_psi.map<uint32_t *>();
        auto tw_map   = bo_tw.map<uint32_t *>();

        // Psi and twiddle tables are identical for every run — fill once.
        std::fill(psi_map, psi_map + pw, 0u);
        std::memcpy(psi_map, pp.data(), N * sizeof(uint32_t));
        std::memcpy(tw_map,  tw.data(), tw_bytes);
        bo_psi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_tw.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ── Run loop — one iteration per benchmark run ────────────────
        for (uint32_t run_i = 0; run_i < num_runs; run_i++) {

            // Receive fresh input coefficients for this run.
            if (!recv_all(conn_fd, data_map, data_bytes)) {
                std::cerr << "Failed to receive coefficients (run " << run_i << ")\n";
                break;
            }

            // Time from DMA-to-device through DMA-from-device.
            // This captures the full FPGA-side latency seen by the server,
            // which the client subtracts from its round-trip to isolate
            // the network transfer overhead.
            auto t0 = std::chrono::steady_clock::now();

            bo_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto kern_run = krnl(bo_data, bo_psi, bo_tw, q, batch, N, logN);
            kern_run.wait();
            bo_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

            auto t1  = std::chrono::steady_clock::now();
            uint64_t fpga_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

            // Send results, then send FPGA processing time.
            if (!send_all(conn_fd, data_map, data_bytes) ||
                !send_all(conn_fd, &fpga_us, sizeof(fpga_us))) {
                std::cerr << "Failed to send results (run " << run_i << ")\n";
                break;
            }
        }

        close(conn_fd);
        std::cout << "Connection complete, waiting for next client\n";
    }
}
