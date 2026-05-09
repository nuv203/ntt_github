# ECE 9413 — Negacyclic NTT Accelerator (Vitis HLS)

**Course:** ECE 9413 — Custom Computing with FPGAs  
**Team:** Niles Peter Villaverde · Noam Cicurel  
**Repo:** https://github.com/nuv203/ntt_github

This IP accelerates the **negacyclic Number Theoretic Transform (NTT)** — the core polynomial-multiplication primitive in lattice-based Post-Quantum Cryptography (PQC) schemes such as CRYSTALS-Kyber and CRYSTALS-Dilithium. The design is implemented in Vitis HLS and targets the Zynq UltraScale+ platform.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [IP Interface Definition](#2-ip-interface-definition)
3. [IP Architecture](#3-ip-architecture)
4. [Verification & Results](#4-verification--results)

---

## 1. Project Structure

The project is split into two independent parts:

### Part 1 — HLS Kernel (`kernel/`)

The FPGA IP core, implemented in Vitis HLS and synthesized for the Kria K26 SOM (`xck26-sfvc784-2LV-c`).

| File | Description |
|------|-------------|
| [kernel/ntt.hpp](kernel/ntt.hpp) | Kernel header — public interface `ntt_kernel()`, constants `TILE_N=4096`, `MAX_BATCH=16` |
| [kernel/ntt.cpp](kernel/ntt.cpp) | HLS kernel — all sub-modules, Barrett reduction, AXI interface pragmas, four-step path |
| [kernel/ntt_tb.cpp](kernel/ntt_tb.cpp) | Self-contained C-simulation testbench — generates NTT-friendly primes, twiddle tables, and golden reference internally |
| [kernel/run_hls.tcl](kernel/run_hls.tcl) | Vitis HLS TCL script — runs C-simulation then C-synthesis |
| [kernel/extract_results.py](kernel/extract_results.py) | Parses `csynth.xml` and prints resource/pipeline tables; saves CSVs to `kernel/data/` |
| [kernel/csynthparse.py](kernel/csynthparse.py) | Bundled XML parser for Vitis HLS synthesis reports (from course toolchain) |

**To run synthesis and extract results:**
```bash
# From the kernel/ directory on a server with vitis_hls:
cd kernel
vitis_hls -f run_hls.tcl

# Then extract and display results:
python extract_results.py
```

### Part 2 — Host Application (`host/`)

The ARM-side application that drives the kernel over the network. `ntt_trace_tcp_host.cpp` runs on the Zynq ARM core and manages the XRT/OpenCL kernel invocation; `ntt_tcp_client.cpp` connects from a remote machine and streams NTT requests over TCP.

| File | Description |
|------|-------------|
| [host/ntt_trace_tcp_host.cpp](host/ntt_trace_tcp_host.cpp) | ARM-side TCP server — loads xclbin, manages kernel execution via XRT/OpenCL |
| [host/ntt_tcp_client.cpp](host/ntt_tcp_client.cpp) | Remote TCP client — sends NTT requests and receives results; includes benchmarking |
| [host/ntt_trace_host.cpp](host/ntt_trace_host.cpp) | Standalone host driver — direct XRT execution without TCP (for on-board testing) |

---

## 2. IP Interface Definition

### 1.1 Functionality

Given a polynomial coefficient vector **x** of length *N* (a power of two), the forward negacyclic NTT computes:

```
y[k] = Σ_{n=0}^{N-1}  x[n] · ψ^{(2k+1)·n}   (mod q)
```

where ψ is a primitive 2N-th root of unity satisfying ψ^N ≡ −1 (mod q), and q is an NTT-friendly prime chosen so that (q − 1) is divisible by 2N. Polynomial multiplication in the NTT domain reduces from O(N²) to O(N log N).

### 1.2 Mathematical Operations

Three modular arithmetic operations are performed on unsigned 32-bit integers:

**Modular Multiplication** — 32×32→64-bit multiply followed by Barrett reduction:
```c
static inline uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t q) {
    uint64_t prod = (uint64_t)a * b;
    return (uint32_t)(prod % q);
}
```

**Modular Addition / Subtraction:**
```c
static inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t q) {
    return (uint32_t)(((uint64_t)a + b) % q);
}
static inline uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t q) {
    return (uint32_t)(((uint64_t)a + q - b) % q);
}
```

These combine into the **Cooley-Tukey butterfly** — the fundamental compute unit:

```python
def butterfly(a, b, w, q):
    t     = (b * w) % q        # 1 modular multiply
    a_out = (a + t) % q        # 1 modular add
    b_out = (a - t + q) % q    # 1 modular subtract
    return a_out, b_out
```

For N=1024: log₂(1024)=10 stages × 512 butterflies = **5,120 butterflies per transform**, each requiring 1 multiply + 2 additions = 15,360 modular arithmetic operations.

### 1.3 AXI Interfaces

The IP exposes three AXI4 master ports for bulk DDR data transfer and one AXI4-Lite slave for scalar control. Interface pragmas from `ntt.cpp`:

```c
#pragma HLS INTERFACE m_axi port=x           offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=psi_powers  offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=twiddles    offset=slave bundle=gmem2

#pragma HLS INTERFACE s_axilite port=x
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
| `x[batch×N]` | `m_axi gmem0` | Read/Write | 32-bit/elem | Polynomial coefficients; overwritten in-place with NTT output |
| `psi_powers[N]` | `m_axi gmem1` | Read | 32-bit/elem | Negacyclic twist table: ψ⁰, ψ¹, …, ψ^(N−1) mod q |
| `twiddles[N]` | `m_axi gmem2` | Read | 32-bit/elem | Flattened Cooley-Tukey twiddle factors (stage-indexed) |
| `q` | `s_axilite` | Write | 32-bit scalar | Prime modulus |
| `batch` | `s_axilite` | Write | 32-bit scalar | Number of independent transforms |
| `N` | `s_axilite` | Write | 32-bit scalar | Transform size (power of 2) |
| `logN` | `s_axilite` | Write | 32-bit scalar | log₂(N) |
| `return` | `s_axilite` | Read | Control | Kernel done signal |

### 1.4 Data Flow (PS ↔ PL)

1. PS allocates three contiguous DDR buffers: `x`, `psi_powers`, `twiddles`.
2. PS writes polynomial coefficients and precomputed tables to DDR.
3. PS programs base addresses and scalar parameters via AXI-Lite.
4. PS triggers kernel execution via the control register.
5. IP burst-reads from DDR, computes the NTT in on-chip BRAM, burst-writes results back.
6. PS polls the `return` status register until done, then reads results from `x`.

---

## 3. IP Architecture

### 2.1 System-Level Block Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     Zynq PS (ARM)                            │
│  Host driver: allocates DDR buffers, writes coefficients     │
│  and twiddle tables, programs AXI-Lite registers, triggers   │
│  kernel, polls for completion, reads results from DDR        │
└───────────────────────┬──────────────────────────────────────┘
                        │  AXI4 Memory-Mapped Interface
                        │  (3 master ports to shared DDR)
                        │  + AXI4-Lite control interface
                        ▼
┌──────────────────────────────────────────────────────────────┐
│                  NTT Accelerator IP (PL)                     │
│                                                              │
│  ┌──────────────┐                                            │
│  │ Sub-Module 1  │  Table Loader                             │
│  │ Burst-read    │  Reads psi_powers[N] and twiddles[N]      │
│  │ DDR → BRAM    │  from DDR into on-chip BRAM (once)        │
│  └──────┬───────┘                                            │
│         │ psi_local[], tw_local[] stored in BRAM              │
│         ▼                                                    │
│  ┌──────────────┐  ← repeated for each batch element         │
│  │ Sub-Module 2  │  Input Loader + Negacyclic Twist          │
│  │ Burst-read    │  x[base..base+N-1] → BRAM a[]             │
│  │ then twist:   │  a[i] = (a[i] × psi_local[i]) mod q      │
│  └──────┬───────┘                                            │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │ Sub-Module 3  │  Bit-Reversal Permutation                 │
│  │ In-place swap │  For each i: swap a[i] ↔ a[bit_rev(i)]   │
│  │ on BRAM a[]   │  using log₂(N)-bit index reversal         │
│  └──────┬───────┘                                            │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │ Sub-Module 4  │  Butterfly Core (Cooley-Tukey DIT)        │
│  │ log₂(N) stgs │  For each stage s = 0..log₂(N)-1:         │
│  │ N/2 BFs/stage │    span = 2^s, groups of 2·span           │
│  │ pipelined     │    butterfly(a[k+j], a[k+j+span],         │
│  │ at II=1       │              tw_local[offset+j], q)       │
│  └──────┬───────┘                                            │
│         ▼                                                    │
│  ┌──────────────┐                                            │
│  │ Sub-Module 5  │  Output Writer                            │
│  │ Burst-write   │  BRAM a[] → DDR x[base..base+N-1]        │
│  │ BRAM → DDR    │  (in-place overwrite of input)            │
│  └──────────────┘                                            │
│                                                              │
│  On-Chip Storage:                                            │
│  ┌────────────────────────────────────────────────────┐      │
│  │ a[MAX_N]        — RAM_T2P BRAM, cyclic factor=2   │      │
│  │ psi_local[MAX_N]— RAM_1P BRAM                     │      │
│  │ tw_local[MAX_N] — RAM_1P BRAM                     │      │
│  └────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Sub-Module Summary

| # | Module | Key Operation | HLS Optimization |
|---|--------|--------------|-----------------|
| 1 | Table Loader | Burst-read psi + twiddle tables DDR→BRAM | `PIPELINE` on load loop; both tables in one loop pass |
| 2 | Input Loader + Twist | Burst-read coefficients, apply `a[i] *= psi[i] mod q` | `PIPELINE`; twist uses inlined `mod_mul` → DSP48 |
| 3 | Bit-Reversal Permutation | In-place swap `a[i] ↔ a[bit_rev(i)]` | `reverse_bits()` fully `UNROLL`ed; dual-port BRAM |
| 4 | Butterfly Core | log₂(N) stages × N/2 Cooley-Tukey butterflies | `PIPELINE II=1` on inner loop; `DEPENDENCE inter false` |
| 5 | Output Writer | Burst-write BRAM→DDR | `PIPELINE` on write loop |

### 2.3 On-Chip Memory

| Array | Size | BRAM Type | Partitioning | Purpose |
|-------|------|-----------|-------------|---------|
| `a[MAX_N]` | 4096 × 32-bit | RAM_T2P (true dual-port) | Cyclic factor=2 | Working buffer — enables simultaneous read of butterfly pair |
| `psi_local[MAX_N]` | 4096 × 32-bit | RAM_1P (single-port) | None | Cached negacyclic twist table |
| `tw_local[MAX_N]` | 4096 × 32-bit | RAM_1P (single-port) | None | Cached twiddle factor table |

Total on-chip storage: **3 × 4096 × 4 bytes = 48 KB BRAM**

### 2.4 Execution Paths

The kernel supports two paths selected at runtime based on N:

**Direct Path (N ≤ 4096):** Load psi/twiddle tables once into BRAM, then loop over batch elements executing sub-modules 2–5 sequentially.

**Four-Step Path (N > 4096):** Factorize N = N1 × N2. Execute three phases — column NTTs (N2 transforms of size N1), row NTTs (N1 transforms of size N2), and a final twist. Uses incremental address counters (no fabric multipliers) to manage the larger address space.

### 2.5 Pipelining Strategy

| Optimization | Status | Detail |
|---|---|---|
| Intra-stage butterfly pipeline | Implemented | `#pragma HLS PIPELINE II=1` on inner butterfly loop — one butterfly/cycle |
| Barrett modular reduction | Implemented | `compute_mu()` computes the Barrett reciprocal; `barrett_mod_mul()` uses 2× DSP48 multiply-shift instead of a hardware divider |
| BRAM dual-port access | Implemented | `RAM_T2P` + cyclic factor=2 allows simultaneous read of both butterfly operands |
| `mod_mul/add/sub` inlining | Implemented | `#pragma HLS INLINE` prevents function call overhead |
| Inter-batch pipelining | Planned | Double-buffering `a[]` with `#pragma HLS DATAFLOW` to overlap load/compute/store |
| Multiple butterfly PEs | Planned | Increase cyclic factor to 4–8 and unroll inner loop for 2–4 butterflies/cycle |

### 2.6 Design Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MAX_N` | 4096 | Maximum transform size; determines BRAM depth |
| `MAX_BATCH` | 16 | Maximum batch count per kernel call |
| Modulus `q` | ≤ 31 bits | Fits in `uint32`; product fits in `uint64` |
| Data type | `uint32_t` | All coefficients and twiddle factors |
| Target II | 1 | One butterfly per clock in the inner loop |
| Target Fmax | 200 MHz | Zynq UltraScale+ |

### 2.7 Why Not AMD LogiCORE FFT IP?

The Xilinx FFT IP (PG109) operates on complex floating-point or fixed-point arithmetic with DFT twiddle factors (complex exponentials). The NTT requires **modular integer arithmetic** over ℤ_q — twiddle factors are roots of unity in a finite field, and every butterfly operation must be reduced mod q. The FFT IP cannot perform modular reduction and would produce incorrect results. A fully custom butterfly in HLS is required.

---

## 4. Verification & Results

### 4.1 Testbench (`kernel/ntt_tb.cpp`)

The C-simulation testbench generates NTT-friendly primes dynamically, computes a golden reference using a Stockham-style NTT, and validates the HLS kernel output element-by-element.

**Three correctness checks per configuration:**

| Test | What It Checks |
|------|---------------|
| `test_ref()` | Output matches reference NTT exactly for all N elements |
| `test_lin()` | Linearity: `NTT(a + b) = NTT(a) + NTT(b)` (mod q) |
| `test_range()` | All output coefficients satisfy `0 ≤ y[i] < q` |

**Configurations tested:**

| Path | Transform sizes N | Batch |
|------|-------------------|-------|
| Direct (N ≤ 4096) | 256, 512, 1024, 2048, 4096 | 1–4 |
| PQC parameters | N=256 (q=8380417), N=512/1024 (q=12289) | 1–2 |
| Four-step (N > 4096) | 8192, 16384, 32768, 65536 | 1–4 |

### 4.2 Synthesis Results

*To be completed after HLS synthesis run.*

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUT | TBD | — | TBD |
| FF | TBD | — | TBD |
| BRAM_36K | TBD | — | TBD |
| DSP48 | TBD | — | TBD |
| Fmax | TBD MHz | 200 MHz target | TBD |

### 4.3 Performance

*To be completed after on-board benchmarking.*

| N | Latency (cycles) | Latency @ 200 MHz | Throughput (transforms/s) | Speedup vs. ARM |
|---|-----------------|-------------------|--------------------------|-----------------|
| 256 | TBD | TBD | TBD | TBD |
| 512 | TBD | TBD | TBD | TBD |
| 1024 | TBD | TBD | TBD | TBD |
| 4096 | TBD | TBD | TBD | TBD |

**Performance targets:**
- Butterfly inner loop at **II=1** (one butterfly/clock)
- N=1024 theoretical compute minimum: 5,120 cycles (10 stages × 512 butterflies)
- **10–20× speedup** over ARM Cortex-A53 baseline (motivated by DSP48 vs. integer division cost for `mod_mul`)

The host-side benchmarking infrastructure in [host/ntt_trace_host.cpp](host/ntt_trace_host.cpp) and [host/ntt_tcp_client.cpp](host/ntt_tcp_client.cpp) measures wall-clock latency with warmup runs and reports mean/stddev across multiple iterations.

See [§1 Project Structure](#1-project-structure) for the full file inventory and synthesis instructions.
