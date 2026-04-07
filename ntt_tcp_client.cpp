/*
 * ntt_tcp_client.cpp
 *
 * TCP client for the NTT FPGA accelerator server.
 *
 * Connects to ntt_trace_simple_host, sends a batch of random polynomial
 * coefficients, receives the NTT results, and verifies them against a
 * local CPU reference. Prints PASS or FAIL.
 *
 * Usage: ./ntt_tcp_client <server-ip> [port] [logN] [batch]
 *
 * Protocol (per connection):
 *   Client → Server : uint32_t logN
 *   Client → Server : uint32_t batch
 *   Server → Client : uint32_t q       (modulus — client uses this for ref_ntt)
 *   Client → Server : batch*N × uint32_t  (input coefficients in [0, q))
 *   Server → Client : batch*N × uint32_t  (NTT results)
 */

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// ===================================================================
// Defaults
// ===================================================================
static constexpr int DEF_LOGN    = 10;
static constexpr int DEF_BATCH   = 4;
static constexpr int DEF_PORT    = 54321;
static constexpr int DEF_BL      = 31;
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

/* Xorshift64 PRNG — used to generate random test coefficients. */
static uint32_t xrand(uint64_t &st, uint32_t q) {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return (uint32_t)(st % q);
}

/* find_psi: primitive 2N-th root of unity mod q — needed to run ref_ntt. */
static bool is_prime(uint64_t n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (uint64_t d = 5; d * d <= n; d += 6)
        if (n % d == 0 || n % (d + 2) == 0) return false;
    return true;
}

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

// ===================================================================
// Reference NTT (CPU golden oracle)
// ===================================================================

/*
 * In-place negacyclic NTT. pp[i] = psi^i mod q (pre-computed by caller).
 * Result is the expected output that the FPGA result is compared against.
 */
static void ref_ntt(std::vector<uint32_t> &a,
                    const std::vector<uint32_t> &pp, uint32_t q) {
    uint32_t N = (uint32_t)a.size(), logN = 0;
    for (uint32_t t = N; t > 1; t >>= 1) logN++;

    /* Pre-twist: a[i] *= psi^i */
    for (uint32_t i = 0; i < N; i++)
        a[i] = mod_mul(a[i], pp[i], q);

    /* Bit-reversal permutation */
    auto rev = [&](uint32_t x) {
        uint32_t r = 0;
        for (uint32_t i = 0; i < logN; i++) { r = (r << 1) | (x & 1); x >>= 1; }
        return r;
    };
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = rev(i);
        if (j > i) std::swap(a[i], a[j]);
    }

    /* Cooley-Tukey butterfly stages */
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

// ===================================================================
// Main
// ===================================================================
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <server-ip> [port] [logN] [batch]\n"
                  << "  Defaults: port=" << DEF_PORT
                  << "  logN=" << DEF_LOGN
                  << "  batch=" << DEF_BATCH << "\n";
        return 1;
    }

    const char *server_ip = argv[1];
    int         port      = (argc >= 3) ? std::stoi(argv[2]) : DEF_PORT;
    uint32_t    logN      = (argc >= 4) ? std::stoi(argv[3]) : DEF_LOGN;
    uint32_t    batch     = (argc >= 5) ? std::stoi(argv[4]) : DEF_BATCH;
    uint32_t    N         = 1u << logN;

    // ── Connect to server ─────────────────────────────────────────────
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { std::cerr << "socket() failed\n"; return 1; }

    sockaddr_in srv{};
    srv.sin_family = AF_INET;
    srv.sin_port   = htons(port);
    if (inet_pton(AF_INET, server_ip, &srv.sin_addr) <= 0) {
        std::cerr << "Invalid server IP: " << server_ip << "\n"; return 1;
    }
    if (connect(sock, reinterpret_cast<sockaddr *>(&srv), sizeof(srv)) < 0) {
        std::cerr << "connect() failed — is the server running?\n"; return 1;
    }
    std::cout << "Connected to " << server_ip << ":" << port << "\n";

    // ── Send transform parameters ─────────────────────────────────────
    if (!send_all(sock, &logN,  sizeof(logN))  ||
        !send_all(sock, &batch, sizeof(batch))) {
        std::cerr << "Failed to send parameters\n"; return 1;
    }

    // ── Receive q from server ─────────────────────────────────────────
    // The server computes q = gen_modulus(N) and sends it back so we use
    // the exact same modulus for local verification.
    uint32_t q = 0;
    if (!recv_all(sock, &q, sizeof(q))) {
        std::cerr << "Failed to receive q\n"; return 1;
    }
    std::cout << "logN=" << logN << "  N=" << N
              << "  batch=" << batch << "  q=" << q << "\n";

    // ── Build psi-powers table for ref_ntt ────────────────────────────
    uint32_t psi = find_psi(N, q);
    std::vector<uint32_t> pp(N);
    pp[0] = 1;
    for (uint32_t i = 1; i < N; i++)
        pp[i] = (uint32_t)(((uint64_t)pp[i - 1] * psi) % q);

    // ── Generate random input coefficients ───────────────────────────
    uint64_t rng = 42;
    std::vector<uint32_t> input(batch * N);
    for (auto &v : input) v = xrand(rng, q);

    // ── Compute expected output locally (CPU reference) ───────────────
    std::vector<uint32_t> gold(input);
    for (uint32_t b = 0; b < batch; b++) {
        std::vector<uint32_t> row(gold.begin() + b * N, gold.begin() + (b + 1) * N);
        ref_ntt(row, pp, q);
        for (uint32_t i = 0; i < N; i++) gold[b * N + i] = row[i];
    }

    // ── Send coefficients to server ───────────────────────────────────
    size_t data_bytes = (size_t)batch * N * sizeof(uint32_t);
    std::cout << "Sending " << batch * N << " coefficients...\n";
    if (!send_all(sock, input.data(), data_bytes)) {
        std::cerr << "Failed to send coefficients\n"; return 1;
    }

    // ── Receive NTT results ───────────────────────────────────────────
    std::vector<uint32_t> result(batch * N);
    std::cout << "Waiting for results...\n";
    if (!recv_all(sock, result.data(), data_bytes)) {
        std::cerr << "Failed to receive results\n"; return 1;
    }

    close(sock);

    // ── Verify against local CPU reference ───────────────────────────
    int mismatches = 0;
    for (size_t i = 0; i < result.size(); i++)
        if (result[i] != gold[i]) mismatches++;

    if (mismatches == 0) {
        std::cout << "TEST PASSED — all " << batch * N
                  << " coefficients match\n";
        return 0;
    } else {
        std::cerr << "TEST FAILED — " << mismatches << "/" << batch * N
                  << " mismatches\n";
        return 1;
    }
}
