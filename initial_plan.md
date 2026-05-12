# ECE 9413 — Custom Vitis HLS IP: 4-Stage Negacyclic NTT Accelerator

> **This is the initial planning document.** It was written before implementation began and contains several elements that changed during development (interface port names, memory types, Barrett reduction marked as "planned", single-path only). For the final design as actually built, see [detailed_plan.md](detailed_plan.md) and [README.md](README.md). For synthesis results, see [TESTBENCH_RESULTS.md](TESTBENCH_RESULTS.md).

## Project Plan

---

## 1. GitHub Repository

`https://github.com/nuv203/ntt_github`

---

## 2. Project Team

| Name | Role |
|------|------|
| Niles Peter Villaverde | RTL/HLS design, host-side driver, integration |
| Noam Cicurel | Architecture, verification, optimization |

---

## 3. IP Definition

### 3.1 Functionality

The IP accelerates the **negacyclic Number Theoretic Transform (NTT)**, the core polynomial-multiplication primitive used in lattice-based Post-Quantum Cryptography (PQC) schemes such as CRYSTALS-Kyber and CRYSTALS-Dilithium. Polynomial multiplication is the single most expensive operation in these cryptographic schemes, and the NTT reduces its complexity from O(N²) to O(N log N).

Given a polynomial represented as a coefficient vector **x** of length *N* (power of two), the forward negacyclic NTT computes:

```
y[k] = Σ_{n=0}^{N-1}  x[n] · ψ^{(2k+1)·n}   (mod q)
```

where ψ is a primitive 2N-th root of unity satisfying ψ^N ≡ −1 (mod q), and q is an NTT-friendly prime modulus chosen such that (q − 1) is divisible by 2N.

### 3.2 Specific Mathematical Operations

The IP performs three distinct classes of modular arithmetic on unsigned 32-bit integers, all reduced modulo a prime q:

**Modular Multiplication** (`mod_mul`): Computes `(a × b) mod q`. This is the most expensive operation. It requires a 32×32 → 64-bit unsigned multiply followed by a 64-bit modular reduction. In our HLS implementation:

```c
static inline uint32_t mod_mul(uint32_t a, uint32_t b, uint32_t q) {
    uint64_t prod = (uint64_t)a * b;
    return (uint32_t)(prod % q);
}
```

**Modular Addition** (`mod_add`): Computes `(a + b) mod q`. Requires a 33-bit addition (to handle overflow) and a modular reduction:

```c
static inline uint32_t mod_add(uint32_t a, uint32_t b, uint32_t q) {
    uint64_t sum = (uint64_t)a + b;
    return (uint32_t)(sum % q);
}
```

**Modular Subtraction** (`mod_sub`): Computes `(a - b) mod q`. Adds q before subtracting to avoid underflow:

```c
static inline uint32_t mod_sub(uint32_t a, uint32_t b, uint32_t q) {
    uint64_t diff = (uint64_t)a + q - b;
    return (uint32_t)(diff % q);
}
```

These three operations combine into the **Cooley-Tukey butterfly**, the fundamental compute unit of the NTT. Each butterfly takes two coefficients and a twiddle factor, and produces two outputs:

```python
def butterfly(a, b, w, q):
    t     = (b * w) % q        # 1 modular multiply
    a_out = (a + t) % q        # 1 modular add
    b_out = (a - t + q) % q    # 1 modular subtract
    return a_out, b_out
```

For an N-point NTT, log₂(N) stages each execute N/2 butterflies, totaling **(N/2)·log₂(N)** butterflies per transform. For N=1024, that is **5,120 butterflies**, each requiring 1 multiply + 2 additions = **15,360 modular arithmetic operations** per transform.

### 3.3 Why These Operations Are Well-Suited for Hardware Acceleration

**1. Computational intensity motivates offloading.** A single NTT of N=1024 requires 5,120 modular multiplications, each involving a 32×32→64-bit multiply and a 64-bit modular reduction. On a general-purpose ARM core, modular reduction via the `%` operator requires a full integer division — typically 20-40 cycles. On an FPGA, the DSP48E2 slice performs a 27×18-bit multiply in a single clock cycle, and modular reduction can be done via Barrett or Montgomery methods in 2-3 cycles using a second DSP48 slice. This represents a 10-20× reduction in per-operation latency.

**2. Massive data-level parallelism.** Within each butterfly stage, all N/2 butterflies are independent — they read and write disjoint pairs of coefficients. This means up to N/2 butterfly processing elements (BPEs) could operate simultaneously. Even with modest parallelism (2-4 BPEs), the FPGA can sustain much higher throughput than a scalar CPU core.

**3. Regular, predictable memory access patterns.** The butterfly access pattern follows a stride-based schedule: stage s accesses pairs separated by 2^s elements. This regularity maps cleanly to BRAM banking strategies (e.g., cyclic partitioning with factor 2), allowing conflict-free dual-port reads every cycle.

**4. Fixed-width integer arithmetic — no floating point needed.** All values are unsigned 32-bit integers with 64-bit intermediates. This maps directly to DSP48 slices on AMD/Xilinx FPGAs without any floating-point overhead, unlike the traditional FFT which requires complex floating-point arithmetic.

**5. Batch throughput.** PQC workloads (key generation, encapsulation, signing) require many independent NTT transforms. These can be pipelined batch-over-batch on the FPGA, achieving sustained throughput that amortizes the initial setup cost.

**6. Latency-sensitive use case.** Post-quantum TLS handshakes demand low-latency polynomial multiplication on edge/embedded devices where the ARM PS alone cannot meet real-time deadlines.

---

## 4. IP Architecture

### 4.1 System-Level Overview

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

### 4.2 Host-to-IP Interface: AXI4 Memory-Mapped (Shared DDR)

The IP uses **shared DDR memory via three AXI4 master ports** for bulk data transfer, plus an **AXI4-Lite slave** for scalar control registers. This shared-memory approach (rather than AXI4-Stream) is used because the NTT operates in-place on a coefficient array that the host must also read/write, and because Vitis HLS natively generates this interface from `#pragma HLS INTERFACE m_axi` directives.

The HLS interface pragmas from the actual implementation:

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
#pragma HLS INTERFACE s_axilite port=return
```

| Port | AXI Bundle | Direction | Width | Description |
|------|-----------|-----------|-------|-------------|
| `x[batch×N]` | `m_axi gmem0` | Read/Write | 32-bit per element | Input polynomial coefficients; overwritten in-place with NTT output |
| `psi_powers[N]` | `m_axi gmem1` | Read-only | 32-bit per element | Negacyclic twist table: ψ^0, ψ^1, ..., ψ^(N-1) mod q |
| `twiddles[N]` | `m_axi gmem2` | Read-only | 32-bit per element | Flattened Cooley-Tukey twiddle factors (stage-indexed) |
| `q` | `s_axilite` | Write | 32-bit scalar | Prime modulus |
| `batch` | `s_axilite` | Write | 32-bit scalar | Number of independent transforms |
| `N` | `s_axilite` | Write | 32-bit scalar | Transform size (power of 2, ≤ MAX_N) |
| `return` | `s_axilite` | Read | Control/status | Kernel done signal |

**Data flow between PS and PL:**

1. PS allocates three contiguous DDR buffers for `x`, `psi_powers`, and `twiddles`.
2. PS writes polynomial coefficients and precomputed tables to DDR.
3. PS programs the base addresses and scalar parameters via AXI-Lite.
4. PS triggers kernel execution by writing to the control register.
5. IP reads data from DDR via AXI4 burst reads, processes it, and writes results back via AXI4 burst writes.
6. PS polls the `return` status register until done, then reads results from the `x` buffer.

### 4.3 Sub-Module Descriptions

#### Sub-Module 1: Table Loader

- **Function:** Burst-reads the precomputed twiddle tables `psi_powers[N]` and `twiddles[N]` from DDR into two on-chip BRAM arrays (`psi_local[]` and `tw_local[]`). This is done **once per kernel invocation** and the tables are reused across all batch elements.
- **Interfaces:** Reads from AXI4 master ports `gmem1` and `gmem2`. Writes to local BRAMs `psi_local[MAX_N]` and `tw_local[MAX_N]`, both bound as `RAM_1P` BRAM.
- **HLS optimization:** The load loop is pipelined (`#pragma HLS PIPELINE`) for single-cycle-per-element throughput. Both tables are loaded in the same loop iteration to halve the total load time.
- **Resource cost:** 2 × MAX_N × 32-bit BRAM entries = 2 × 4096 × 4 bytes = 32 KB BRAM.

#### Sub-Module 2: Input Loader + Negacyclic Twist

- **Function:** For each batch element b, burst-reads `x[b×N .. b×N+N-1]` from DDR into the working BRAM array `a[]`, then applies the negacyclic twist: `a[i] = (a[i] × psi_local[i]) mod q` for i = 0..N-1. The twist converts the negacyclic NTT into a standard cyclic NTT by multiplying each coefficient by the appropriate power of ψ.
- **Interfaces:** Reads from AXI4 master port `gmem0`. Reads from BRAM `psi_local[]`. Writes to BRAM `a[MAX_N]`, bound as `RAM_T2P` (true dual-port) BRAM with `cyclic factor=2` array partitioning.
- **HLS optimization:** Both the load loop and the twist loop are pipelined. The twist requires one `mod_mul` per element (inlined, maps to DSP48 slice).
- **Mathematical operation:** N modular multiplications: `a[i] = (a[i] × ψ^i) mod q`.

#### Sub-Module 3: Bit-Reversal Permutation

- **Function:** Performs an in-place bit-reversal permutation on the BRAM array `a[]`. For each index i, computes j = bit_reverse(i, log₂(N)) and swaps a[i] ↔ a[j] when j > i. This reorders the coefficients into the sequence expected by the Cooley-Tukey DIT algorithm.
- **Interfaces:** Reads/writes BRAM `a[]` only (no external memory access).
- **HLS optimization:** The `reverse_bits` function is fully unrolled (`#pragma HLS UNROLL`). The cyclic factor=2 partitioning on `a[]` enables dual-port access for simultaneous read of both swap elements.
- **Mathematical operation:** Pure permutation — no arithmetic, just data movement.

#### Sub-Module 4: Butterfly Core (Cooley-Tukey DIT) — Main Compute Engine

- **Function:** Executes `log₂(N)` stages of radix-2 Cooley-Tukey decimation-in-time butterflies. Each stage s processes N/2 independent butterfly operations. The twiddle factors are read from `tw_local[]` using a stage-flattened offset scheme: stage s uses twiddles from `tw_local[tw_offset .. tw_offset + 2^s - 1]`, then `tw_offset += 2^s`.
- **Inner loop computation (the critical path):**

```c
// For each butterfly pair (idx1, idx2) with twiddle w:
uint32_t u = a[idx1];                        // BRAM read
uint32_t v = mod_mul(a[idx2], w, q);         // 32×32→64 multiply + mod q
a[idx1] = mod_add(u, v, q);                  // (u + v) mod q → BRAM write
a[idx2] = mod_sub(u, v, q);                  // (u - v + q) mod q → BRAM write
```

- **Interfaces:** Reads/writes BRAM `a[]` (dual-port via cyclic partitioning). Reads BRAM `tw_local[]`.
- **HLS optimization:** The innermost butterfly loop is pipelined at II=1 (`#pragma HLS PIPELINE II=1`), meaning one butterfly per clock cycle. The `mod_mul`, `mod_add`, and `mod_sub` functions are all inlined (`#pragma HLS INLINE`).
- **Resource mapping:** Each `mod_mul` maps to DSP48 slice(s) for the 32×32-bit multiply. The modular reduction (`% q`) is synthesized by Vitis HLS into a divider circuit; a future optimization would replace this with Barrett reduction to use only multiplies.
- **Throughput:** For N=1024, this module executes 10 stages × 512 butterflies = 5,120 butterflies. At II=1, this takes ~5,120 cycles per transform.

#### Sub-Module 5: Output Writer

- **Function:** Burst-writes the completed NTT result from BRAM `a[0..N-1]` back to DDR at `x[b×N .. b×N+N-1]`, overwriting the input in-place.
- **Interfaces:** Reads from BRAM `a[]`. Writes to AXI4 master port `gmem0`.
- **HLS optimization:** Pipelined burst write (`#pragma HLS PIPELINE`).

### 4.4 On-Chip Memory Architecture

| Array | Size | BRAM Type | Partitioning | Purpose |
|-------|------|-----------|-------------|---------|
| `a[MAX_N]` | 4096 × 32-bit | RAM_T2P (true dual-port) | Cyclic factor=2 | Working coefficient buffer — enables simultaneous read of butterfly pair (a[idx1], a[idx2]) |
| `psi_local[MAX_N]` | 4096 × 32-bit | RAM_1P (single-port) | None | Cached negacyclic twist table |
| `tw_local[MAX_N]` | 4096 × 32-bit | RAM_1P (single-port) | None | Cached twiddle factor table |

### 4.5 Consideration of Existing AMD/Xilinx IPs

We considered leveraging the following AMD/Xilinx IPs:

**AMD LogiCORE FFT IP (PG109):** The Xilinx FFT IP core provides a highly optimized, configurable FFT engine with AXI4-Stream interfaces, supporting pipelined streaming and burst I/O architectures. However, it operates on **complex floating-point or fixed-point** data with standard DFT twiddle factors (complex exponentials). The NTT differs fundamentally: it operates in **modular integer arithmetic** (mod q over Z_q), not complex arithmetic. The twiddle factors are powers of a root of unity in a finite field, not complex exponentials. Therefore, the FFT IP cannot be directly reused for NTT — it would produce incorrect results because it does not perform modular reduction. We instead implement a custom butterfly that performs modular arithmetic natively.

**AMD Multiplier IP (PG108):** The Xilinx Multiplier LogiCORE provides fine-grained control over DSP48 slice usage, pipelining depth, and symmetric rounding for fixed-point multiplication up to 64×64 bits. We plan to explore instantiating this IP (or using `#pragma HLS BIND_OP` directives) for the 32×32→64-bit modular multiply in the butterfly core to ensure optimal DSP48 utilization and achieve maximum clock frequency. Currently Vitis HLS infers the multiply automatically, but explicit binding could reduce the number of DSP48 slices consumed.

**AMD AXI Interconnect / SmartConnect:** The AXI interconnect infrastructure is already used implicitly by Vitis to connect the three `m_axi` master ports (`gmem0`, `gmem1`, `gmem2`) to the shared DDR memory controller. We rely on Vitis platform generation to configure this correctly.

**AMD AXI DMA IP:** Not needed for this design. Since the NTT operates in-place on DDR buffers via memory-mapped AXI4, we do not require a separate DMA engine. The HLS-generated AXI master ports handle burst reads/writes directly.

### 4.6 Dataflow and Pipelining Strategy

The current architecture processes batches **sequentially** — sub-modules 2 through 5 repeat for each batch element. Planned optimizations:

1. **Intra-stage pipelining (implemented):** The innermost butterfly loop targets II=1 via `#pragma HLS PIPELINE II=1`.
2. **Inter-batch pipelining (planned):** Double-buffer `a[]` so that while batch b is in the butterfly core, batch b+1's data is being loaded from DDR. This would use `#pragma HLS DATAFLOW` to overlap sub-modules 2 and 5 across consecutive batches.
3. **Multiple butterfly PEs (planned):** Increase BRAM cyclic partitioning factor from 2 to 4 or 8, and unroll the inner butterfly loop to process 2-4 butterflies per cycle. This trades BRAM and DSP48 resources for throughput.
4. **Barrett reduction (planned):** Replace the generic `% q` modular reduction with Barrett reduction, which uses only multiplies and shifts — eliminating the expensive divider circuit and reducing latency.

### 4.7 Design Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MAX_N` | 4096 | Maximum transform size; determines BRAM depth |
| `MAX_BATCH` | 16 | Maximum batch count per kernel call |
| Modulus `q` | ≤ 31 bits | Fits in uint32; product fits in uint64 |
| Data type | `uint32` | All coefficients and twiddles |
| Target II | 1 | One butterfly per clock in the inner loop |

### 4.8 Verification Strategy

- **C simulation:** The `ntt_tb_cpp.cpp` testbench loads golden vectors from binary files (`x.bin`, `ref.bin`, `psi_powers.bin`, `twiddles.bin`) generated by the Python/SymPy reference oracle, runs the HLS kernel, and compares every output element.
- **Python golden model:** `reference.py` implements `negacyclic_ntt_oracle()` using SymPy's trusted cyclic NTT with the standard twist trick. `export_vectors.py` generates binary test vectors from this oracle.
- **Co-simulation:** Vitis HLS C/RTL co-simulation will validate the synthesized RTL matches C-sim results cycle-accurately.
- **On-board test:** After bitstream generation, the host driver will run the same golden vectors through the accelerator on a Zynq board and compare results.

---

## 5. Summary and Next Steps

| Milestone | Description |
|-----------|-------------|
| Week 1 | Finalize architecture, complete project plan, create GitHub repo |
| Week 2 | Optimize HLS pragmas (Barrett reduction, BRAM partitioning), pass C-sim |
| Week 3 | Run Vitis HLS synthesis, analyze resource/timing/DSP48 reports, iterate |
| Week 4 | Integrate with Vitis platform, run co-simulation, explore dataflow pipelining |
| Week 5 | On-board validation, performance benchmarking, final report |
