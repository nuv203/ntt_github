/*
 * ntt_trace_tcp_host.cpp — NTT FPGA TCP server
 *
 * Receives pre-built tables from the client and processes each run either via
 * the FPGA accelerator (mode=0) or the Kria ARM CPU (mode=1). The CPU path
 * enables a direct FPGA-vs-ARM comparison with identical data and network costs.
 *
 * Usage: ./ntt_trace_tcp_host <ntt.xclbin> [port]
 *
 * Protocol (per connection):
 *   Client → Server : uint32_t logN, batch, q, psi_words, tw_words, num_runs, mode
 *                       mode 0 = FPGA accelerator
 *                       mode 1 = Kria ARM CPU (software NTT)
 *   Client → Server : psi_words × uint32_t   (psi_powers table)
 *   Client → Server : tw_words  × uint32_t   (twiddle table)
 *   [loop num_runs:]
 *     Client → Server : batch*N × uint32_t   (input coefficients)
 *     Server → Client : batch*N × uint32_t   (NTT results)
 *     Server → Client : uint64_t             (processing time in µs)
 */

#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <cstdint>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

static constexpr int SERVER_PORT = 54321;

// ── CPU NTT helpers (used only in mode=1) ────────────────────────────────────

static inline uint32_t cpu_mod_mul(uint32_t a, uint32_t b, uint32_t q) {
    return (uint32_t)(((uint64_t)a * b) % q);
}
static inline uint32_t cpu_mod_add(uint32_t a, uint32_t b, uint32_t q) {
    uint32_t s = a + b; return s >= q ? s - q : s;
}
static inline uint32_t cpu_mod_sub(uint32_t a, uint32_t b, uint32_t q) {
    return a >= b ? a - b : a + q - b;
}
static uint32_t cpu_power_mod(uint32_t base, uint32_t exp, uint32_t mod) {
    uint64_t r = 1, b = base;
    while (exp > 0) {
        if (exp & 1) r = (r * b) % mod;
        b = (b * b) % mod;
        exp >>= 1;
    }
    return (uint32_t)r;
}

/*
 * Software negacyclic NTT — matches ref_ntt on the client exactly.
 *
 * Uses the received psi_map for the psi-powers twist (psi_map[i] = psi^i).
 * Derives omega = psi^2 from psi_map[1] and computes twiddles on the fly so
 * it works for all N, including four-step sizes (N > TILE_N).
 *
 * Algorithm: psi twist → bit-reversal → Cooley-Tukey DIT butterfly.
 */
static void cpu_ntt_forward(uint32_t *data, const uint32_t *psi_map,
                            uint32_t q, uint32_t batch,
                            uint32_t N, uint32_t logN) {
    uint32_t omega = cpu_mod_mul(psi_map[1], psi_map[1], q);  // omega = psi^2

    for (uint32_t b = 0; b < batch; b++) {
        uint32_t *a = data + (size_t)b * N;

        // Step 1: psi twist — a[i] *= psi^i
        for (uint32_t i = 0; i < N; i++)
            a[i] = cpu_mod_mul(a[i], psi_map[i], q);

        // Step 2: bit-reversal permutation
        for (uint32_t i = 0; i < N; i++) {
            uint32_t j = 0, x = i;
            for (uint32_t k = 0; k < logN; k++) { j = (j << 1) | (x & 1); x >>= 1; }
            if (j > i) { uint32_t tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
        }

        // Step 3: Cooley-Tukey DIT butterfly stages
        for (uint32_t s = 0; s < logN; s++) {
            uint32_t span  = 1u << s;
            uint32_t span2 = span << 1;
            // Root for this stage: omega^(N / (2*span))
            uint32_t ws = cpu_power_mod(omega, N / span2, q);
            for (uint32_t k = 0; k < N; k += span2) {
                uint32_t w = 1;
                for (uint32_t j = 0; j < span; j++) {
                    uint32_t u = a[k + j];
                    uint32_t v = cpu_mod_mul(a[k + j + span], w, q);
                    a[k + j]        = cpu_mod_add(u, v, q);
                    a[k + j + span] = cpu_mod_sub(u, v, q);
                    w = cpu_mod_mul(w, ws, q);
                }
            }
        }
    }
}

// ── Socket helpers ────────────────────────────────────────────────────────────

static bool recv_all(int fd, void *buf, size_t n) {
    char *p = static_cast<char *>(buf);
    while (n > 0) {
        ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}

static bool send_all(int fd, const void *buf, size_t n) {
    const char *p = static_cast<const char *>(buf);
    while (n > 0) {
        ssize_t s = send(fd, p, n, 0);
        if (s <= 0) return false;
        p += s; n -= s;
    }
    return true;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <ntt.xclbin> [port]\n";
        return 1;
    }
    int port = (argc >= 3) ? std::stoi(argv[2]) : SERVER_PORT;

    // Open XRT device and program the FPGA once — reused for every connection.
    std::cout << "Opening device and loading " << argv[1] << "\n";
    auto device = xrt::device(0);
    auto uuid   = device.load_xclbin(argv[1]);
    auto krnl   = xrt::kernel(device, uuid, "ntt_kernel");
    std::cout << "Kernel ready. Listening on port " << port << "\n";

    // Server socket setup.
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));
    listen(listen_fd, 1);

    // Accept loop — one client at a time.
    while (true) {
        int conn_fd = accept(listen_fd, nullptr, nullptr);
        if (conn_fd < 0) { std::cerr << "accept() failed\n"; continue; }
        std::cout << "Client connected\n";

        // ── Receive connection parameters ─────────────────────────────
        uint32_t logN, batch, q, psi_words, tw_words, num_runs, mode;
        if (!recv_all(conn_fd, &logN,      sizeof(logN))      ||
            !recv_all(conn_fd, &batch,     sizeof(batch))     ||
            !recv_all(conn_fd, &q,         sizeof(q))         ||
            !recv_all(conn_fd, &psi_words, sizeof(psi_words)) ||
            !recv_all(conn_fd, &tw_words,  sizeof(tw_words))  ||
            !recv_all(conn_fd, &num_runs,  sizeof(num_runs))  ||
            !recv_all(conn_fd, &mode,      sizeof(mode))) {
            std::cerr << "Failed to receive parameters\n";
            close(conn_fd); continue;
        }
        uint32_t N = 1u << logN;
        std::cout << "logN=" << logN << " batch=" << batch
                  << " q=" << q << " psi_words=" << psi_words
                  << " tw_words=" << tw_words << " runs=" << num_runs
                  << " mode=" << (mode == 0 ? "FPGA" : "ARM-CPU") << std::endl;

        // ── Allocate XRT buffers (used for FPGA mode; host-accessible for CPU mode) ──
        // group_id maps each BO to the correct AXI master port:
        //   group_id(0) → gmem0 → data       (in/out)
        //   group_id(1) → gmem1 → psi_powers
        //   group_id(2) → gmem2 → twiddles
        size_t data_bytes = (size_t)batch * N * sizeof(uint32_t);
        size_t psi_bytes  = (size_t)psi_words * sizeof(uint32_t);
        size_t tw_bytes   = (size_t)tw_words  * sizeof(uint32_t);

        auto bo_data = xrt::bo(device, data_bytes, krnl.group_id(0));
        auto bo_psi  = xrt::bo(device, psi_bytes,  krnl.group_id(1));
        auto bo_tw   = xrt::bo(device, tw_bytes,   krnl.group_id(2));

        auto data_map = bo_data.map<uint32_t *>();
        auto psi_map  = bo_psi.map<uint32_t *>();
        auto tw_map   = bo_tw.map<uint32_t *>();

        // ── Receive psi and twiddle tables (once per connection) ──────
        if (!recv_all(conn_fd, psi_map, psi_bytes) ||
            !recv_all(conn_fd, tw_map,  tw_bytes)) {
            std::cerr << "Failed to receive tables\n";
            close(conn_fd); continue;
        }

        // Sync tables to FPGA only when needed.
        if (mode == 0) {
            bo_psi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_tw.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        std::cout << "Tables ready, entering run loop\n";

        // ── Run loop ──────────────────────────────────────────────────
        uint64_t total_proc_us = 0;
        uint32_t completed = 0;

        for (uint32_t i = 0; i < num_runs; i++) {
            if (!recv_all(conn_fd, data_map, data_bytes)) {
                std::cerr << "recv coefficients failed (run " << i << ")\n"; break;
            }

            auto t0 = std::chrono::steady_clock::now();

            if (mode == 0) {
                // FPGA path: DMA → kernel → DMA
                bo_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                krnl(bo_data, bo_psi, bo_tw, q, batch, N, logN).wait();
                bo_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            } else {
                // CPU path: software NTT on Kria ARM, operates directly on data_map
                cpu_ntt_forward(data_map, psi_map, q, batch, N, logN);
            }

            auto t1 = std::chrono::steady_clock::now();
            uint64_t proc_us = std::chrono::duration_cast<
                std::chrono::microseconds>(t1 - t0).count();

            if (!send_all(conn_fd, data_map, data_bytes) ||
                !send_all(conn_fd, &proc_us, sizeof(proc_us))) {
                std::cerr << "send results failed (run " << i << ")\n"; break;
            }

            total_proc_us += proc_us;
            completed++;
        }

        close(conn_fd);
        if (completed > 0) {
            std::cout << "Done: " << completed << " runs, avg "
                      << (mode == 0 ? "fpga=" : "arm=")
                      << total_proc_us / completed << "us\n";
        }
    }
}
