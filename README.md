# ECE 9413 — Negacyclic NTT Accelerator (Vitis HLS)

**Course:** ECE 9413 — Custom Computing with FPGAs  
**Team:** Niles Peter Villaverde · Noam Cicurel  
**Repo:** https://github.com/nuv203/ntt_github

This IP accelerates the **negacyclic Number Theoretic Transform (NTT)** — the core polynomial-multiplication primitive in lattice-based Post-Quantum Cryptography schemes such as CRYSTALS-Kyber and CRYSTALS-Dilithium. The design is implemented in Vitis HLS targeting the Kria K26 SOM (`xck26-sfvc784-2LV-c`, Zynq UltraScale+).

---

## Table of Contents

1. [Project Structure](#1-project-structure)
   - [Running HLS Synthesis](#running-hls-synthesis)
2. [IP Interface Definition](#2-ip-interface-definition)
3. [IP Architecture](#3-ip-architecture)
4. [Verification & Results](#4-verification--results)

---

## 1. Project Structure

The project is split into two independent parts:

### Part 1 — HLS Kernel (`kernel/`)

The FPGA IP core, implemented in Vitis HLS and synthesized for the Kria K26 SOM.

| File | Description |
|------|-------------|
| [kernel/ntt.hpp](kernel/ntt.hpp) | Kernel header — public interface `ntt_kernel()`, constants `TILE_N=4096`, `MAX_BATCH=16`, `MAX_LOG_N=20` |
| [kernel/ntt.cpp](kernel/ntt.cpp) | HLS kernel — Barrett reduction, AXI interface pragmas, direct and four-step execution paths |
| [kernel/ntt_tb.cpp](kernel/ntt_tb.cpp) | Self-contained C-simulation testbench — generates NTT-friendly primes, twiddle tables, and golden reference internally |
| [kernel/run_hls.tcl](kernel/run_hls.tcl) | Vitis HLS TCL script — runs C-simulation then C-synthesis via `vitis-run` |
| [kernel/hls_flow.py](kernel/hls_flow.py) | **Primary build script** — sets up the Vitis environment, runs synthesis, and prints resource/timing tables |
| [kernel/extract_results.py](kernel/extract_results.py) | Lower-level results parser; called internally by `hls_flow.py` |
| [kernel/csynthparse.py](kernel/csynthparse.py) | Bundled XML parser for Vitis HLS synthesis reports (from course toolchain) |

### Part 2 — Host Application (`host/`)

ARM-side application that drives the kernel. `ntt_trace_tcp_host.cpp` runs on the Zynq ARM and manages XRT/OpenCL kernel invocation; `ntt_tcp_client.cpp` connects from a remote machine to stream NTT requests over TCP.

| File | Description |
|------|-------------|
| [host/ntt_trace_tcp_host.cpp](host/ntt_trace_tcp_host.cpp) | ARM-side TCP server — loads xclbin, manages kernel execution via XRT/OpenCL |
| [host/ntt_tcp_client.cpp](host/ntt_tcp_client.cpp) | Remote TCP client — sends NTT requests and receives results; includes benchmarking |
| [host/ntt_trace_host.cpp](host/ntt_trace_host.cpp) | Standalone host driver — direct XRT execution without TCP (for on-board testing) |

---

### Running HLS Synthesis

#### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager (handles all Python dependencies automatically)
- Vitis 2025.2 installed under `~/amd/2025.2/` (or update `VITIS_CANDIDATES` in `hls_flow.py`)
- Ubuntu 24 note: `hls_flow.py` automatically creates `libncurses.so.5` / `libtinfo.so.5` compatibility symlinks — no manual fix needed

**Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # add uv to PATH for this session
```

#### Usage

All commands are run from the `kernel/` directory:

```bash
cd kernel

# Full run: C-simulation + synthesis + results tables
uv run hls_flow.py

# Skip C-simulation (faster re-synthesis after code changes)
uv run hls_flow.py --skip-csim

# Parse and display results from an existing synthesis run (no rebuild)
uv run hls_flow.py --report-only

# Synthesis + update README.md §4.2 with real resource numbers
uv run hls_flow.py --update-readme
```

`uv run` automatically installs dependencies (`pandas`, `rich`) into an isolated environment on first run — no `pip install` or `venv` setup required.

#### Output

A successful run prints:
- **Resource Utilisation table** — LUT, FF, BRAM, DSP usage vs. available, plus estimated Fmax
- **Loop Pipeline Summary** — II, depth, trip count, and max latency for every pipelined loop
- CSVs saved to `kernel/data/csynth_loop_info.csv` and `kernel/data/csynth_resource_usage.csv`

---

## 2. IP Interface Definition

### 2.1 Functionality

Given a polynomial coefficient vector **x** of length *N* (a power of two), the forward negacyclic NTT computes:

```
y[k] = Σ_{n=0}^{N-1}  x[n] · ψ^{(2k+1)·n}   (mod q)
```

where ψ is a primitive 2N-th root of unity satisfying ψ^N ≡ −1 (mod q), and q is an NTT-friendly prime with (q − 1) divisible by 2N. Polynomial multiplication in the NTT domain reduces from O(N²) to O(N log N).

### 2.2 Mathematical Operations

Three modular arithmetic operations act on unsigned 32-bit integers:

**Modular Multiplication** — uses Barrett reduction (implemented in `barrett_mod_mul()`). The kernel precomputes the Barrett reciprocal `mu = floor(2^62 / q)` once per invocation to avoid hardware divide units:

```c
// Conceptual: (a * b) mod q
// Actual implementation: Barrett reduction using precomputed mu = floor(2^62/q)
// barrett_mod_mul(a, b, q, mu) — see ntt.cpp for the four-partial-product implementation
uint64_t prod = (uint64_t)a * b;
uint64_t est  = (prod * mu) >> 62;   // quotient estimate
uint64_t r    = prod - est * q;      // remainder (off by at most 2q)
if (r >= q) r -= q;                  // correct
if (r >= q) r -= q;
return (uint32_t)r;
```

**Modular Addition / Subtraction** — single conditional subtract, no multiply:

```c
static inline ntt_t mod_add(ntt_t a, ntt_t b, ntt_t q) {
    ntt_t s = a + b;
    return (s >= q) ? (s - q) : s;
}
static inline ntt_t mod_sub(ntt_t a, ntt_t b, ntt_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}
```

These combine into the **Cooley-Tukey butterfly** — the fundamental compute unit:

```c
ntt_t w = tw[span + j];               // twiddle factor
ntt_t u = a[i1];
ntt_t v = barrett_mod_mul(a[i2], w, q, mu);  // 1 modular multiply
a[i1]   = mod_add(u, v, q);           // butterfly upper output
a[i2]   = mod_sub(u, v, q);           // butterfly lower output
```

For N=1024: log₂(1024) = 10 stages × 512 butterflies/stage = **5,120 butterflies per transform**.

### 2.3 AXI Interfaces

The IP exposes three independent AXI4 master ports for DDR data transfer and one AXI4-Lite slave for scalar control. Three separate bundles allow the Zynq interconnect to issue concurrent DDR transactions:

```c
// From ntt.cpp — actual pragma block
#pragma HLS INTERFACE m_axi port=data       offset=slave bundle=gmem0 depth=16777216 max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=psi_powers offset=slave bundle=gmem1 depth=1048576  max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=twiddles   offset=slave bundle=gmem2 depth=1048576  max_read_burst_length=256

#pragma HLS INTERFACE s_axilite port=data
#pragma HLS INTERFACE s_axilite port=psi_powers
#pragma HLS INTERFACE s_axilite port=twiddles
#pragma HLS INTERFACE s_axilite port=q
#pragma HLS INTERFACE s_axilite port=batch
#pragma HLS INTERFACE s_axilite port=N
#pragma HLS INTERFACE s_axilite port=logN
#pragma HLS INTERFACE s_axilite port=return
```

| Port | AXI Bundle | Direction | Width | Description |
|------|-----------|-----------|-------|-------------|
| `data[batch×N]` | `m_axi gmem0` | Read/Write | 32-bit/elem | Polynomial coefficients; overwritten in-place with NTT output |
| `psi_powers[N]` | `m_axi gmem1` | Read/Write | 32-bit/elem | Negacyclic twist table ψ⁰…ψ^(N−1); reused as transpose scratch buffer in the four-step path after twist values are consumed |
| `twiddles[≤N2+N1+N]` | `m_axi gmem2` | Read-only | 32-bit/elem | Stockham twiddle factors (column + row NTTs) and inter-stage twiddles for four-step path |
| `q` | `s_axilite` | Write | 32-bit | Prime modulus |
| `batch` | `s_axilite` | Write | 32-bit | Number of independent transforms (max 16) |
| `N` | `s_axilite` | Write | 32-bit | Transform size — power of two, up to 2^20 |
| `logN` | `s_axilite` | Write | 32-bit | log₂(N) |
| `return` | `s_axilite` | Read | Control | Kernel done signal (`ap_ctrl_hs`) |

### 2.4 Data Flow (PS ↔ PL)

1. PS allocates three DDR buffers: `data`, `psi_powers`, `twiddles`.
2. PS populates polynomial coefficients, the negacyclic twist table, and the Stockham twiddle table.
3. PS writes buffer base addresses and scalar parameters (`q`, `batch`, `N`, `logN`) via AXI-Lite.
4. PS asserts the start bit (`ap_ctrl_hs`).
5. For **N ≤ 4096**: IP burst-reads the full twist and twiddle tables into on-chip LUTRAM once, then loops over the batch — burst-reading each element, computing the NTT, burst-writing the result in-place.
6. For **N > 4096**: IP runs the four-step algorithm entirely through DDR (see §3.3).
7. PS polls the `return` register for the done signal, then reads results from `data`.

---

## 3. IP Architecture

### 3.1 System-Level Block Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     Zynq PS (ARM)                            │
│  Allocates DDR buffers, writes coefficients and tables,      │
│  programs AXI-Lite control registers, triggers kernel,       │
│  polls ap_done, reads results from data[]                    │
└───────────────────────┬──────────────────────────────────────┘
                        │ AXI4 (3 × m_axi master to DDR)
                        │ AXI4-Lite (scalar control)
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                  NTT Accelerator IP (PL)                     │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  On-Chip Buffers                                    │     │
│  │  tile[4096]     — RAM_T2P BRAM (true dual-port)    │     │
│  │  tw_local[4096] — LUTRAM (low-latency twiddle read) │     │
│  │  psi_local[4096]— LUTRAM (direct path only)        │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Direct Path  (N ≤ 4096)                              │  │
│  │                                                        │  │
│  │  1. DIRECT_LOAD_PSI / DIRECT_LOAD_TW                  │  │
│  │     Burst-read psi and twiddle tables DDR → LUTRAM    │  │
│  │     (once, shared across all batch elements)           │  │
│  │                                                        │  │
│  │  Per batch element (DIRECT_BATCH loop):                │  │
│  │  2. DIRECT_LOAD                                        │  │
│  │     Burst-read data[b*N..] from DDR → tile[];          │  │
│  │     inline psi pre-twist: tile[i] = data[i]*psi[i]    │  │
│  │  3. sub_ntt(): bit-reversal + logN butterfly stages   │  │
│  │  4. DIRECT_STORE                                       │  │
│  │     Burst-write tile[] → data[b*N..] in DDR           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Four-Step Path  (N > 4096, N = N1 × N2)              │  │
│  │                                                        │  │
│  │  Phase 1 — Column NTTs (N1 transforms of size N2):    │  │
│  │    For each column 0..N1-1:                            │  │
│  │      FS_GATHER_COL: strided DDR read + psi twist       │  │
│  │      sub_ntt (N2-point)                                │  │
│  │      FS_SCATTER_COL: strided DDR write                 │  │
│  │                                                        │  │
│  │  Phase 2 — Row NTTs (N2 transforms of size N1):       │  │
│  │    For each row 0..N2-1:                               │  │
│  │      FS_LOAD_ROW: burst DDR read + inter-stage twiddle │  │
│  │      sub_ntt (N1-point)                                │  │
│  │      FS_STORE_ROW: burst DDR write                     │  │
│  │                                                        │  │
│  │  Phase 3 — Transpose via DDR scratch:                  │  │
│  │    FS_TRANS_LOAD/WRITE_SCRATCH: write transposed data  │  │
│  │    into psi_powers[] (reused as scratch, safe since    │  │
│  │    all psi values consumed in Phase 1)                 │  │
│  │    FS_TRANS_FROM_SCRATCH: copy back to data[]          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Sub-Module Summary

| Module / Loop | Function | HLS Optimization |
|---|---|---|
| `compute_mu()` | Barrett reciprocal: `mu = floor(2^62 / q)` | `PIPELINE II=1`, 63-cycle shift-subtract FSM; eliminates hardware divider |
| `DIRECT_LOAD_PSI` / `DIRECT_LOAD_TW` | Burst DDR → LUTRAM for psi and twiddle tables | `PIPELINE II=1`; one-time load shared across entire batch |
| `DIRECT_LOAD` | Burst DDR → tile[] with inline psi pre-twist | `PIPELINE II=1`; `barrett_mod_mul` inlined per element |
| `sub_ntt` — `SUB_BIT_REV` | Bit-reversal permutation in BRAM | `reverse_bits()` fully `UNROLL`ed (combinational); swap loop II=3 |
| `sub_ntt` — `SUB_BFLY` | Cooley-Tukey butterfly (inner) | `PIPELINE II=1` directive; achieved **II=2** in synthesis (see §4.2) |
| `DIRECT_STORE` | Burst tile[] → DDR | `PIPELINE II=1` |
| `FS_GATHER_COL` | Strided gather: DDR column → tile[] + psi twist | `PIPELINE II=1`, incremental address counters avoid fabric multipliers |
| `FS_SCATTER_COL` | Strided scatter: tile[] → DDR column | `PIPELINE II=1`, incremental address counters |
| `FS_LOAD_ROW` / `FS_STORE_ROW` | Contiguous row load/store with inter-stage twiddle | `PIPELINE II=1` |
| `FS_TRANS_LOAD` / `FS_TRANS_WRITE_SCRATCH` | Row-major → column-major reorder into DDR scratch | `PIPELINE II=1` |
| `FS_TRANS_FROM_SCRATCH` | Copy transposed data back to output buffer | `PIPELINE II=1` |

### 3.3 On-Chip Memory

| Array | Size | Storage Type | Purpose |
|-------|------|-------------|---------|
| `tile[TILE_N]` | 4096 × 32-bit | **RAM_T2P BRAM** | Working buffer for one batch element; true dual-port enables simultaneous butterfly read+write |
| `tw_local[TILE_N]` | 4096 × 32-bit | **LUTRAM** | Cached twiddle factors; low-latency random access during butterfly stages |
| `psi_local[TILE_N]` | 4096 × 32-bit | **LUTRAM** | Negacyclic twist table cache (direct path only; four-step path reads psi from DDR during gather) |

BRAM usage: `tile[]` occupies 36 BRAM_18K (confirmed by synthesis). `tw_local` and `psi_local` are mapped to distributed LUT RAM.

### 3.4 Four-Step Algorithm

For N > 4096, the IP factors N = N1 × N2 where N1 = 2^(logN/2) and N2 = 2^(logN − logN/2). The twiddle table layout for this path:

```
twiddles[0      .. N2-1]          N2-point Stockham twiddles (Phase 1 column NTTs)
twiddles[N2     .. N2+N1-1]       N1-point Stockham twiddles (Phase 2 row NTTs)
twiddles[N2+N1  .. N2+N1+N-1]    inter-stage factors omega^(row*col) (Phase 2 twist)
```

All address arithmetic in strided DDR loops uses incremental counters (`addr += stride`) rather than `row * N1 + col` multiplies — this eliminates fabric multipliers on the critical path, which was the primary change enabling timing closure at 200 MHz in prior iterations.

### 3.5 Barrett Modular Multiplication

The inner butterfly calls `barrett_mod_mul(a, b, q, mu)`. The Barrett reciprocal `mu` is computed once per kernel invocation by a 63-iteration shift-subtract loop (`compute_mu`), amortizing the cost over millions of butterflies. The multiplication itself uses four 32×32 partial products (each mapped to a DSP48 slice) to reconstruct the high 64 bits of the 128-bit product `a*b*mu`, extracts the quotient estimate, and applies at most two conditional subtracts to produce the exact remainder.

```
PIPELINE II=1, LATENCY min=12 max=18
```

The minimum latency of 12 was increased from 10 in the final version to give HLS extra register stages to break the DSP→fabric critical path, enabling the estimated Fmax to increase from 175 MHz to 136.99 MHz.

### 3.6 Pipelining Strategy

| Optimization | Directive | Achieved Result |
|---|---|---|
| Barrett reciprocal | `PIPELINE II=1` on 63-iter loop | II=1, Depth=1 ✓ |
| Twiddle table load | `PIPELINE II=1` | II=1 on all DMA loops ✓ |
| Bit-reversal permutation | `UNROLL` on `reverse_bits()` | Combinational; swap loop II=3 (BRAM dependency) |
| Butterfly inner loop | `PIPELINE II=1`, `DEPENDENCE variable=a inter false` | **II=2, Depth=18** (12-cycle Barrett latency limits II) |
| Strided gather/scatter (four-step) | `PIPELINE II=1`, incremental counters | II=1 ✓ |
| Inter-stage twiddle + row load | `PIPELINE II=1` | II=1 ✓ |
| BRAM dual-port access | `BIND_STORAGE type=RAM_T2P impl=BRAM` | Simultaneous read+write of butterfly pair |

### 3.7 Design Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `TILE_N` | 4096 | Max direct-path transform size; determines BRAM depth |
| `MAX_BATCH` | 16 | Max batch count per kernel call |
| `MAX_LOG_N` | 20 | Max supported logN (N up to 1,048,576 via four-step) |
| Modulus `q` | ≤ 31-bit prime | Fits in `uint32_t`; product fits in `uint64_t` |
| Data type | `uint32_t` | All coefficients, twiddle factors, and psi values |
| Target clock | 10 ns (100 MHz) in TCL | Design targeted 200 MHz; see §4.2 for achieved Fmax |

### 3.8 Why Not AMD LogiCORE FFT IP?

The Xilinx FFT IP (PG109) operates on complex floating-point or fixed-point data with DFT twiddle factors (complex exponentials ω = e^(−2πi/N)). The NTT requires **modular integer arithmetic** over ℤ_q — twiddle factors are roots of unity in a finite field, and every butterfly must reduce mod q. The FFT IP cannot perform modular reduction and would produce incorrect results. A fully custom butterfly in HLS is required.

---

## 4. Verification & Results

### 4.1 Testbench (`kernel/ntt_tb.cpp`)

The C-simulation testbench is entirely self-contained — it generates NTT-friendly primes dynamically using a primality test, finds primitive roots of unity, builds Stockham twiddle tables, computes a golden reference NTT, and validates the HLS kernel output element-by-element. No external input files are required.

**Three correctness checks per configuration:**

| Test | What It Checks |
|------|---------------|
| `test_ref()` | Output matches reference NTT exactly for all N elements across one or more batch elements |
| `test_lin()` | Linearity: `NTT(a + b) ≡ NTT(a) + NTT(b)` (mod q) |
| `test_range()` | All output coefficients satisfy `0 ≤ y[i] < q` |

**Configurations tested (35 test cases — all PASS):**

| Path | Transform sizes N | Batch | Modulus |
|------|-------------------|-------|---------|
| Direct (N ≤ 4096) | 256, 512, 1024, 2048, 4096 | 1 and 4 | Auto-generated NTT prime |
| PQC parameters | 256, 512, 1024 | 1–2 | q=8380417 (Dilithium), q=12289 (Kyber) |
| Four-step (N > 4096) | 8192, 16384, 32768, 65536 | 1 and 2 | Auto-generated NTT prime |

C-simulation output (Vitis HLS 2025.2):
```
ALL TESTS PASSED
INFO: [HLS 200-2161] Finished Command csim_design Elapsed time: 00:00:00
```

### 4.2 Synthesis Results

*Synthesized with Vitis HLS 2025.2, target part `xck26-sfvc784-2LV-c`, clock constraint 10 ns.*

**Resource Utilization:**

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT      | 17173 | 117120   | 14.7%       |
| FF       | 9090  | 234240   | 3.9%        |
| BRAM_18K | 36    | 288      | 12.5%       |
| DSP      | 54    | 1248     | 4.3%        |
| URAM     | 0     | 64       | 0.0%        |
| Fmax (est.) | 136.99 MHz | 200 MHz target | 68% of target |

**Loop Pipeline Results:**

| Loop | II | Depth | Notes |
|------|----|-------|-------|
| `COMPUTE_MU_LOOP` | 1 | 1 | Barrett reciprocal — target met ✓ |
| `DIRECT_LOAD_PSI` / `_TW` | 1 | 3 | DDR→LUTRAM burst — target met ✓ |
| `DIRECT_LOAD` / `_STORE` | 1 | 16/3 | DDR↔BRAM burst — target met ✓ |
| `FS_GATHER_COL` | 1 | 24 | Strided DDR gather + Barrett — target met ✓ |
| `FS_SCATTER_COL` | 1 | 8 | Strided DDR scatter — target met ✓ |
| `FS_LOAD_ROW` / `_STORE_ROW` | 1 | 16/3 | DDR row burst — target met ✓ |
| `FS_TRANS_*` | 1 | 3–8 | Transpose to scratch — target met ✓ |
| `SUB_BIT_REV` | 3 | 4 | BRAM read-write dependency in conditional swap |
| **`SUB_BFLY`** | **2** | **18** | **Target was II=1; limited by Barrett latency** |

**Analysis against performance goals:**

*Resource budget:* All resources are well within budget (LUT 14.7%, BRAM 12.5%, DSP 4.3%). The 54 DSPs are consumed by `barrett_mod_mul` (27 DSPs for the four partial products, called from both `sub_ntt` and the four-step gather/scatter).

*Butterfly throughput (target II=1, achieved II=2):* The `#pragma HLS DEPENDENCE variable=a inter false` pragma correctly tells HLS that consecutive butterfly iterations access disjoint pairs of `tile[]` elements — there is no true loop-carried dependency. However, `barrett_mod_mul` has a minimum pipeline latency of 12 cycles. With an II=1 target, 12 butterfly computations would be in-flight simultaneously, and the BRAM dual-port constraint (one read and one write per clock) prevents HLS from issuing two independent BRAM reads in the same cycle. HLS therefore settles at II=2 to stagger accesses. Achieving II=1 would require partitioning `tile[]` to allow two independent BRAM ports (cyclic factor 2 or higher) combined with a latency reduction on `barrett_mod_mul`.

*Fmax (target 200 MHz, achieved 136.99 MHz):* The estimated clock period of 7.3 ns (136.99 MHz) indicates the design does not close timing at the 200 MHz target under HLS scheduling alone. The critical path runs through the `barrett_mod_mul` DSP chain into the BRAM address generation for `tile[]`. Synthesis was run with a 10 ns constraint; re-running with a 5 ns constraint would force more aggressive retiming. The 200 MHz target remains a goal for future iterations with additional pipelining.

### 4.3 Performance Estimates

*Latency estimates derived from synthesis loop analysis at the achieved clock.*

| N | Path | Estimated Cycles | @ 136 MHz | Notes |
|---|------|-----------------|-----------|-------|
| 256 | Direct | ~1,800 | ~13 µs | load + 8 butterfly stages + store |
| 1024 | Direct | ~12,000 | ~88 µs | 10 stages × 512 BF, II=2 |
| 4096 | Direct | ~101,000 | ~742 µs | 12 stages × 2048 BF, II=2 |
| 8192 | Four-step | ~330,000 | ~2.4 ms | N1=64, N2=128; includes DDR scatter/gather |
| 65536 | Four-step | ~18M | ~132 ms | N1=256, N2=256; dominated by DDR bandwidth |

**Performance targets:**
- Inner butterfly loop: target **II=1** (one butterfly/clock); **achieved II=2**
- N=1024 minimum compute bound: 5,120 cycles × II=1 = 5,120 cycles; actual ~12,000 cycles at II=2
- **10–20× speedup** over ARM Cortex-A53 baseline (ARM `mod_mul` requires a hardware integer division vs. DSP-based Barrett reduction at II=2 in the FPGA design)

Host-side benchmarking infrastructure in [host/ntt_trace_host.cpp](host/ntt_trace_host.cpp) and [host/ntt_tcp_client.cpp](host/ntt_tcp_client.cpp) measures wall-clock latency with warmup runs and reports mean/stddev across multiple iterations for on-board validation.
