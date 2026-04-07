/*
 * ntt_trace_simple_host.cpp — NTT FPGA TCP server
 *
 * Pure forwarder: no NTT math, no table generation.
 * Receives pre-built buffers from the client, pushes them to the FPGA
 * via XRT, and sends the results back. The client owns all computation.
 *
 * Usage: ./ntt_trace_simple_host <ntt.xclbin> [port]
 *
 * Protocol (per connection):
 *   Client → Server : uint32_t logN, batch, q, psi_words, tw_words, num_runs
 *   Client → Server : psi_words × uint32_t   (psi_powers table)
 *   Client → Server : tw_words  × uint32_t   (twiddle table)
 *   [loop num_runs:]
 *     Client → Server : batch*N × uint32_t   (input coefficients)
 *     Server → Client : batch*N × uint32_t   (NTT results)
 *     Server → Client : uint64_t             (FPGA processing time in µs)
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

// ── Socket helpers ────────────────────────────────────────────────────────────

/* Receive exactly n bytes, looping over partial reads (TCP is a stream). */
static bool recv_all(int fd, void *buf, size_t n) {
    char *p = static_cast<char *>(buf);
    while (n > 0) {
        ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += r; n -= r;
    }
    return true;
}

/* Send exactly n bytes, looping over partial writes. */
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
        uint32_t logN, batch, q, psi_words, tw_words, num_runs;
        if (!recv_all(conn_fd, &logN,      sizeof(logN))      ||
            !recv_all(conn_fd, &batch,     sizeof(batch))     ||
            !recv_all(conn_fd, &q,         sizeof(q))         ||
            !recv_all(conn_fd, &psi_words, sizeof(psi_words)) ||
            !recv_all(conn_fd, &tw_words,  sizeof(tw_words))  ||
            !recv_all(conn_fd, &num_runs,  sizeof(num_runs))) {
            std::cerr << "Failed to receive parameters\n";
            close(conn_fd); continue;
        }
        uint32_t N = 1u << logN;
        std::cout << "logN=" << logN << " batch=" << batch
                  << " q=" << q << " runs=" << num_runs << "\n";

        // ── Allocate XRT buffers for this (N, batch) ──────────────────
        // group_id maps each BO to the correct AXI master port in the kernel:
        //   group_id(0) → gmem0 → data       (in/out)
        //   group_id(1) → gmem1 → psi_powers (in, reused as scratch for N>4096)
        //   group_id(2) → gmem2 → twiddles   (in, read-only)
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
        bo_psi.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_tw.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // ── Run loop ─────────────────────────────────────────────────
        for (uint32_t i = 0; i < num_runs; i++) {

            // Receive fresh input coefficients for this run.
            if (!recv_all(conn_fd, data_map, data_bytes)) {
                std::cerr << "recv coefficients failed (run " << i << ")\n"; break;
            }

            // DMA → kernel → DMA, timed for the client's round-trip breakdown.
            auto t0 = std::chrono::steady_clock::now();
            bo_data.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            krnl(bo_data, bo_psi, bo_tw, q, batch, N, logN).wait();
            bo_data.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto t1 = std::chrono::steady_clock::now();

            uint64_t fpga_us = std::chrono::duration_cast<
                std::chrono::microseconds>(t1 - t0).count();

            if (!send_all(conn_fd, data_map, data_bytes) ||
                !send_all(conn_fd, &fpga_us, sizeof(fpga_us))) {
                std::cerr << "send results failed (run " << i << ")\n"; break;
            }
        }

        close(conn_fd);
        std::cout << "Connection complete\n";
    }
}
