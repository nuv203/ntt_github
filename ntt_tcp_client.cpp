/*
 * ntt_tcp_client.cpp
 *
 * Benchmarking client for the NTT FPGA accelerator server.
 *
 * Runs multiple (logN, batch) combinations, timing each round-trip.
 * The server also reports how long the FPGA took on its side, so we can
 * isolate:
 *   round_trip = network_send + fpga_processing + network_recv
 *   net_io     = round_trip - fpga_processing
 *
 * Usage: ./ntt_tcp_client <server-ip> [port]
 *
 * Protocol (per connection):
 *   Client → Server : uint32_t logN, batch, q, psi_words, tw_words, num_runs
 *   Client → Server : psi_words × uint32_t  (pre-built psi_powers table)
 *   Client → Server : tw_words  × uint32_t  (pre-built twiddle table)
 *   [loop num_runs:]
 *     Client → Server : batch*N × uint32_t  (input coefficients)
 *     Server → Client : batch*N × uint32_t  (NTT results)
 *     Server → Client : uint64_t fpga_us    (FPGA-side processing time in µs)
 *
 * The server does no NTT math — all tables are computed here and forwarded.
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

// ===================================================================
// Benchmark configuration
// ===================================================================

static constexpr int DEF_PORT = 54321;
static constexpr int DEF_BL   = 31;

struct BenchCase { uint32_t logN; uint32_t batch; };

/* Default full case list — used when no --suite / --case flags are given. */
static const BenchCase DEFAULT_CASES[] = {
    { 8,  1}, { 9,  1}, {10,  1}, {11,  1}, {12,  1},   /* direct  */
    {13,  1}, {14,  1}, {15,  1}, {16,  1},               /* 4-step  */
    {10,  4}, {10,  8}, {12,  4},                         /* batched */
};

/* Predefined suites. */
static std::vector<BenchCase> suite_cases(const std::string &name) {
    if (name == "quick")    return {{10,1},{12,1},{14,1}};
    if (name == "full")     return {{8,1},{9,1},{10,1},{11,1},{12,1},
                                    {13,1},{14,1},{15,1},{16,1},
                                    {10,4},{10,8},{12,4}};
    if (name == "direct")   return {{8,1},{9,1},{10,1},{11,1},{12,1}};
    if (name == "fourstep") return {{13,1},{14,1},{15,1},{16,1}};
    if (name == "batched")  return {{10,4},{10,8},{12,4},{14,2}};
    return {}; /* empty = unknown */
}

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

/* Build psi-powers (pp) and packed twiddle table (tw) for this N.
 * Four-step path (N > TILE_N): tw = [tw_col(N2) | tw_row(N1) | inter(N)]. */
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

/* Psi buffer size: direct path needs N words; four-step needs batch*N for scratch. */
static size_t psi_buf_words(uint32_t N, uint32_t batch) {
    return (N > TILE_N) ? (size_t)batch * N : (size_t)N;
}

// ===================================================================
// Reference NTT (CPU golden oracle)
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

/* Helper: build psi_buf and tw, then send all headers + tables to server. */
static bool send_handshake(int sock, uint32_t logN, uint32_t batch,
                           uint32_t num_runs,
                           const std::vector<uint32_t> &pp,
                           const std::vector<uint32_t> &tw,
                           uint32_t q) {
    uint32_t N         = 1u << logN;
    uint32_t psi_words = (uint32_t)psi_buf_words(N, batch);
    uint32_t tw_words  = (uint32_t)tw.size();

    // Psi buffer: pp values in [0..N-1], rest zeroed (scratch for four-step).
    std::vector<uint32_t> psi_buf(psi_words, 0);
    std::copy(pp.begin(), pp.end(), psi_buf.begin());

    return send_all(sock, &logN,      sizeof(logN))      &&
           send_all(sock, &batch,     sizeof(batch))     &&
           send_all(sock, &q,         sizeof(q))         &&
           send_all(sock, &psi_words, sizeof(psi_words)) &&
           send_all(sock, &tw_words,  sizeof(tw_words))  &&
           send_all(sock, &num_runs,  sizeof(num_runs))  &&
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
// Main
// ===================================================================
static void print_usage(const char *prog) {
    std::cerr
        << "Usage: " << prog << " <server-ip> [port]\n"
        << "           [--suite <name>]        predefined case list (repeatable)\n"
        << "           [--case <logN>:<batch>] individual case (repeatable)\n"
        << "           [--runs  <N>]           timed iterations (default 10)\n"
        << "           [--warmup <N>]          warmup iterations (default 2)\n"
        << "           [--serial]              disable pipelined sends (default: pipelined)\n"
        << "\n"
        << "Suites: quick, full, direct, fourstep, batched\n";
}

int main(int argc, char **argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    const char *server_ip = nullptr;
    int         port       = DEF_PORT;
    int         num_runs   = 10;
    int         num_warmup = 2;
    bool        serial     = false;
    std::vector<BenchCase> bench_cases;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--suite") {
            if (i + 1 >= argc) { std::cerr << "--suite requires a name\n"; return 1; }
            std::string name = argv[++i];
            auto cases = suite_cases(name);
            if (cases.empty()) {
                std::cerr << "Unknown suite '" << name << "'. Valid: quick, full, direct, fourstep, batched\n";
                return 1;
            }
            bench_cases.insert(bench_cases.end(), cases.begin(), cases.end());
        } else if (arg == "--case") {
            if (i + 1 >= argc) { std::cerr << "--case requires logN:batch\n"; return 1; }
            uint32_t ln, bt;
            if (sscanf(argv[++i], "%u:%u", &ln, &bt) != 2) {
                std::cerr << "Invalid --case format '" << argv[i] << "', expected logN:batch\n"; return 1;
            }
            bench_cases.push_back({ln, bt});
        } else if (arg == "--runs") {
            if (i + 1 >= argc) { std::cerr << "--runs requires a number\n"; return 1; }
            num_runs = std::stoi(argv[++i]);
        } else if (arg == "--warmup") {
            if (i + 1 >= argc) { std::cerr << "--warmup requires a number\n"; return 1; }
            num_warmup = std::stoi(argv[++i]);
        } else if (arg == "--serial") {
            serial = true;
        } else if (arg[0] == '-') {
            std::cerr << "Unknown flag '" << arg << "'\n"; print_usage(argv[0]); return 1;
        } else if (!server_ip) {
            server_ip = argv[i];
        } else {
            port = std::stoi(argv[i]);
        }
    }

    if (!server_ip) { print_usage(argv[0]); return 1; }

    /* Default: run the full case list if nothing was specified. */
    if (bench_cases.empty()) {
        bench_cases.assign(std::begin(DEFAULT_CASES), std::end(DEFAULT_CASES));
    }

    // ── Correctness check (logN=10, batch=1, 1 run) ──────────────────
    std::cout << "=== Correctness check (logN=10, batch=1) ===\n";
    {
        int sock = connect_to(server_ip, port);
        if (sock < 0) { std::cerr << "connect() failed\n"; return 1; }

        uint32_t logN = 10, batch = 1, N = 1u << logN;
        uint32_t check_runs = 1;

        // Build all NTT tables locally — server receives them, no computation there.
        std::cout << "[dbg] building tables N=" << N << std::flush;
        uint32_t q   = gen_modulus(N);
        uint32_t psi = find_psi(N, q);
        std::vector<uint32_t> pp, tw;
        make_tables(N, q, psi, pp, tw);
        std::cout << " q=" << q << " psi_words=" << psi_buf_words(N,batch) << " tw_words=" << tw.size() << std::endl;

        std::cout << "[dbg] sending handshake..." << std::flush;
        if (!send_handshake(sock, logN, batch, check_runs, pp, tw, q)) {
            std::cerr << "Handshake failed\n"; return 1;
        }
        std::cout << " done" << std::endl;

        // Generate input and compute CPU reference.
        uint64_t rng = 42;
        std::vector<uint32_t> input(N);
        for (auto &v : input) v = xrand(rng, q);
        std::vector<uint32_t> gold(input);
        ref_ntt(gold, pp, q);

        // Send coefficients, receive results — measure full round-trip.
        std::cout << "[dbg] sending " << N << " coefficients..." << std::flush;
        auto t0 = std::chrono::steady_clock::now();
        send_all(sock, input.data(), N * sizeof(uint32_t));
        std::cout << " waiting for results..." << std::flush;
        std::vector<uint32_t> result(N);
        uint64_t fpga_us = 0;
        recv_all(sock, result.data(), N * sizeof(uint32_t));
        recv_all(sock, &fpga_us, sizeof(fpga_us));
        auto t1 = std::chrono::steady_clock::now();
        close(sock);

        double rtt_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        double net_us = rtt_us - (double)fpga_us;
        std::cout << " done\n"
                  << "  round_trip=" << rtt_us/1000.0 << "ms"
                  << "  fpga="       << fpga_us/1000.0 << "ms"
                  << "  net_io="     << net_us/1000.0  << "ms\n";

        int mismatches = 0;
        for (uint32_t i = 0; i < N; i++) if (result[i] != gold[i]) mismatches++;
        std::cout << (mismatches == 0 ? "PASSED" : "FAILED")
                  << " (" << mismatches << " mismatches)\n\n";
        if (mismatches) return 1;
    }

    // ── Benchmark loop ────────────────────────────────────────────────
    std::cout << "=== Benchmark (" << num_runs << " runs, "
              << num_warmup << " warmup, "
              << (serial ? "serial" : "pipelined") << ") ===\n\n";

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

    for (const auto &bc : bench_cases) {
        uint32_t logN      = bc.logN;
        uint32_t batch     = bc.batch;
        uint32_t N         = 1u << logN;
        uint32_t total_runs = (uint32_t)(num_warmup + num_runs);

        // One connection per benchmark case — all runs share the connection.
        int sock = connect_to(server_ip, port);
        if (sock < 0) {
            std::cerr << "connect() failed for logN=" << logN << "\n"; continue;
        }

        // Build all tables locally — server receives them, does no NTT math.
        uint32_t q   = gen_modulus(N);
        uint32_t psi = find_psi(N, q);
        std::vector<uint32_t> pp, tw;
        make_tables(N, q, psi, pp, tw);

        if (!send_handshake(sock, logN, batch, total_runs, pp, tw, q)) {
            std::cerr << "Handshake failed for logN=" << logN << "\n";
            close(sock); continue;
        }

        // Build input data (same seed every case for reproducibility).
        uint64_t rng = 99;
        std::vector<uint32_t> input(batch * N);
        for (auto &v : input) v = xrand(rng, q);
        size_t data_bytes = (size_t)batch * N * sizeof(uint32_t);
        std::vector<uint32_t> result(batch * N);

        std::vector<double> rtt_us, fpga_us_v, net_us_v;
        rtt_us.reserve(num_runs);
        fpga_us_v.reserve(num_runs);
        net_us_v.reserve(num_runs);

        /* Helper: send one run's input and receive result + fpga_us. */
        auto do_recv = [&](uint64_t &fpga_us_out) -> bool {
            fpga_us_out = 0;
            return recv_all(sock, result.data(), data_bytes) &&
                   recv_all(sock, &fpga_us_out, sizeof(fpga_us_out));
        };

        if (serial) {
            /* Serial: one full round-trip per run — baseline mode. */
            for (uint32_t r = 0; r < total_runs; r++) {
                auto t0 = std::chrono::steady_clock::now();
                send_all(sock, input.data(), data_bytes);
                uint64_t fpga_us = 0;
                do_recv(fpga_us);
                auto t1 = std::chrono::steady_clock::now();
                double rtt = std::chrono::duration_cast<
                    std::chrono::microseconds>(t1 - t0).count();
                if (r >= (uint32_t)num_warmup) {
                    rtt_us.push_back(rtt);
                    fpga_us_v.push_back((double)fpga_us);
                    net_us_v.push_back(rtt - (double)fpga_us);
                }
            }
        } else {
            /*
             * Pipelined: send run N+1 while waiting for run N's result.
             * Depth-1 pipeline eliminates the idle gap between each kernel
             * call without risking TCP buffer deadlock (at most 2 runs'
             * worth of data in flight at any time).
             *
             * Timing: the timed phase is bracketed by t0/t1 so we get
             * total elapsed time for all num_runs timed iterations.
             * Per-run fpga_us is still reported by the server each run.
             */

            /* Warmup phase — send, pipeline recv, discard. */
            if (num_warmup > 0) {
                send_all(sock, input.data(), data_bytes);
                for (int r = 1; r < num_warmup; r++) {
                    send_all(sock, input.data(), data_bytes); /* pre-send next */
                    uint64_t discard = 0; do_recv(discard);
                }
                uint64_t discard = 0; do_recv(discard);      /* drain last warmup */
            }

            /* Timed phase. */
            auto t0 = std::chrono::steady_clock::now();
            send_all(sock, input.data(), data_bytes);         /* prime the pipeline */
            for (int r = 1; r < num_runs; r++) {
                send_all(sock, input.data(), data_bytes);     /* pre-send run r */
                uint64_t fpga_us = 0; do_recv(fpga_us);      /* recv run r-1 */
                fpga_us_v.push_back((double)fpga_us);
            }
            uint64_t fpga_us = 0; do_recv(fpga_us);          /* drain last run */
            auto t1 = std::chrono::steady_clock::now();
            fpga_us_v.push_back((double)fpga_us);

            /* Distribute total elapsed time evenly across timed runs. */
            double avg_rtt = std::chrono::duration_cast<std::chrono::microseconds>(
                t1 - t0).count() / (double)num_runs;
            for (int r = 0; r < num_runs; r++) {
                rtt_us.push_back(avg_rtt);
                net_us_v.push_back(avg_rtt - fpga_us_v[r]);
            }
        }
        close(sock);

        auto rtt_s  = compute_stats(rtt_us);
        auto fpga_s = compute_stats(fpga_us_v);
        auto net_s  = compute_stats(net_us_v);
        double tp   = (double)((uint64_t)batch * N) / (rtt_s.median_us / 1e6) / 1e6;

        std::cout << std::fixed
                  << std::left  << std::setw(7)  << logN
                  << std::left  << std::setw(9)  << N
                  << std::left  << std::setw(7)  << batch
                  << std::right << std::setw(12) << std::setprecision(3) << rtt_s.median_us  / 1000.0 << "   "
                  << std::right << std::setw(10) << fpga_s.median_us / 1000.0 << "   "
                  << std::right << std::setw(10) << net_s.median_us  / 1000.0 << "   "
                  << std::right << std::setw(9)  << std::setprecision(2) << tp
                  << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
