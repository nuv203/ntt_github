# ECE 9413 — Custom Vitis HLS IP: 4-Stage Negacyclic NTT Accelerator

## Project Plan

---

## 1. GitHub Repository

https://github.com/nuv203/ntt_github

---

## 2. Project Team

| Name | Role |
|------|------|
| Niles Peter Villaverde | HLS design, testbench, host-side application |
| Noam Cicurel | performance optimization, TCP streaming, live-hardware demo |

---

## 3. IP Definition

### 3.1 Functionality

The IP accelerates the **negacyclic Number Theoretic Transform (NTT)**, the core polynomial-multiplication primitive used in lattice-based Post-Quantum Cryptography (PQC) schemes such as CRYSTALS-Kyber and CRYSTALS-Dilithium.

Given a polynomial represented as a coefficient vector **x** of length *N* (power of two), the forward negacyclic NTT computes:
```
y[k] = Σ_{n=0}^{N-1}  x[n] · ψ^{(2k+1)·n}   (mod q)
```

where ψ is a primitive 2N-th root of unity satisfying ψ^N ≡ −1 (mod q), and q is a prime modulus chosen such that (q − 1) is divisible by 2N (an "NTT-friendly" prime).

### 3.2 Mathematical Operations

The computation decomposes into four sequential stages using the Cooley-Tukey decimation-in-time (DIT) butterfly algorithm. Each stage performs log₂(N) layers of butterfly operations. A single butterfly consists of:
```python
# Cooley-Tukey butterfly (one butterfly unit)
def butterfly(a, b, w, q):
    """
    a, b : two input coefficients (uint32, values in [0, q))
    w    : twiddle factor ψ^k mod q
    q    : prime modulus
    Returns (a', b') where:
        a' = (a + b·w) mod q
        b' = (a - b·w) mod q
    """
    t  = (b * w) % q          # modular multiply  (64-bit intermediate)
    a_out = (a + t) % q       # modular add
    b_out = (a - t + q) % q   # modular subtract
    return a_out, b_out
```

The full 4-stage forward NTT pipeline is:
```python
def ntt_forward(x, psi_powers, twiddles, q, N):
    """
    Stage 0 — Negacyclic twist:
        x'[n] = x[n] · ψ^n  mod q      for n = 0..N-1

    Stage 1 — Bit-reversal permutation:
        Reorder x' by reversing log2(N)-bit indices.

    Stage 2 — Butterfly stages (log2(N) layers):
        For stage s = 0, 1, ..., log2(N)-1:
            span = 2^s
            For each group of 2·span elements:
                For j = 0..span-1:
                    butterfly(a[k+j], a[k+j+span], twiddle[s][j], q)

    Stage 3 — Write-back:
        Store results to output buffer.
    """
```

### 3.3 Why Hardware Acceleration?

The NTT is an excellent candidate for FPGA acceleration for several reasons:

- **Massive data parallelism:** Each butterfly is independent within a given layer, enabling spatial unrolling of multiple butterfly units operating in parallel.
- **Regular memory access patterns:** The stride-based access pattern in each stage is predictable and maps cleanly to BRAM banking strategies.
- **Modular arithmetic on fixed-width integers:** All operations are mod-q on 32-bit unsigned integers with 64-bit intermediates — this maps efficiently to DSP48 slices on Xilinx FPGAs (no floating point needed).
- **Compute-bound inner loop:** The core butterfly requires one 32×32→64-bit multiply and two additions per pair of elements per stage. For N=1024 with log₂(N)=10 stages, that is 5120 butterflies per transform — highly repetitive work ideal for pipelining.
- **Batch processing:** PQC workloads require many independent NTT transforms (e.g., key generation, encapsulation). These can be processed in parallel across multiple butterfly units or pipelined batch-over-batch.
- **Latency-sensitive cryptographic use case:** PQC operations on embedded/edge devices require low-latency transforms that a CPU cannot deliver at the throughput demanded by real-time TLS handshakes.

---

## 4. IP Architecture

### 4.1 System-Level Overview
```
┌──────────────────────────────────────────────────────────┐
│                    Zynq PS (ARM)                         │
│   Host driver: loads coefficients, twiddle tables,       │
│   triggers kernel, reads results                         │
└────────────────────┬─────────────────────────────────────┘
                     │  AXI4 Memory-Mapped (shared DDR)
                     ▼
┌──────────────────────────────────────────────────────────┐
│               NTT Accelerator IP (PL)                    │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐  ┌──────┐ │
│  │  Table     │  │ Twist      │  │ Butterfly│  │Write-│ │
│  │  Loader    │→ │ & Bit-Rev  │→ │ Core     │→ │back  │ │
│  │  (Stage 0) │  │ (Stage 1)  │  │ (Stage 2)│  │(St 3)│ │
│  └────────────┘  └────────────┘  └──────────┘  └──────┘ │
│       ↑               ↑              ↑                   │
│       └───── BRAM: psi_local, tw_local, coeff[] ─────────│
└──────────────────────────────────────────────────────────┘
```

### 4.2 Interface: AXI4 Memory-Mapped (Shared Memory)

The IP uses **shared DDR memory via AXI4 master ports** (not AXI4-Stream) for data transfer. This is the simpler integration path for Vitis and matches the existing HLS pragma structure:

| Port | AXI Bundle | Direction | Description |
|------|-----------|-----------|-------------|
| `x[batch*N]` | `gmem0` | Read/Write | Input coefficients; overwritten in-place with NTT output |
| `psi_powers[N]` | `gmem1` | Read | Negacyclic twist table: ψ^0, ψ^1, ..., ψ^{N-1} mod q |
| `twiddles[N]` | `gmem2` | Read | Flattened Cooley-Tukey twiddle factors per stage |
| `q, batch, N` | AXI-Lite | Write | Scalar control registers set by host before launch |

The host (PS) allocates contiguous buffers in DDR, writes coefficient data and precomputed tables, programs the scalar registers via AXI-Lite, and triggers execution. On completion, results are read back from the same `x` buffer.

### 4.3 Module Descriptions

#### Module 1: Table Loader

- **Function:** Burst-reads `psi_powers[N]` and `twiddles[N]` from DDR into local BRAM once per kernel invocation. These tables are reused across all batches.
- **Interface:** AXI4 master read → BRAM write. Pipelined with `#pragma HLS PIPELINE`.
- **Resources:** Two N-entry BRAM arrays (RAM_1P).

#### Module 2: Twist & Bit-Reversal (Stage 0 + Stage 1)

- **Function:** For each batch element, burst-reads `x[base..base+N-1]` into local BRAM `a[]`, then applies the negacyclic twist (`a[i] = a[i] · psi_local[i] mod q`) and performs in-place bit-reversal permutation.
- **Interface:** AXI4 master read → local BRAM `a[]`. The twist loop is pipelined. Bit-reversal uses conditional swaps.
- **Key optimization:** The BRAM array `a[]` is partitioned with `cyclic factor=2` to allow dual-port access for the swap operations.

#### Module 3: Butterfly Core (Stage 2 — the compute engine)

- **Function:** Executes `log₂(N)` layers of Cooley-Tukey DIT butterflies on the local BRAM array `a[]`. Each layer processes N/2 independent butterflies.
- **Inner loop (pipelined at II=1):**
```
u = a[idx1]
v = (a[idx2] * twiddle) mod q     // 32×32 → 64-bit, then mod q
a[idx1] = (u + v) mod q
a[idx2] = (u - v + q) mod q
```

- **Interface:** Reads/writes local BRAM `a[]` and reads `tw_local[]`.
- **Key optimization targets:** Pipeline the inner butterfly loop at II=1. BRAM partitioning (`cyclic factor=2`) enables simultaneous read of `a[idx1]` and `a[idx2]`. Potential future optimization: unroll butterfly pairs or use multiple butterfly processing elements (BPEs) for intra-stage parallelism.

#### Module 4: Write-Back (Stage 3)

- **Function:** Burst-writes the completed NTT result from local BRAM `a[]` back to DDR at `x[base..base+N-1]`.
- **Interface:** BRAM read → AXI4 master write. Pipelined burst.

### 4.4 Dataflow & Pipelining Strategy

The current architecture processes batches **sequentially** — each batch goes through all four stages before the next begins. A natural optimization path is:

1. **Intra-stage pipelining:** Already applied — inner butterfly loop targets II=1.
2. **Inter-batch pipelining (future):** Use double-buffering on `a[]` so that while one batch is in the butterfly core, the next batch's data is being loaded from DDR.
3. **Multiple butterfly PEs (future):** Instantiate 2 or 4 parallel butterfly units to process independent pairs within a stage simultaneously.

### 4.5 Design Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MAX_N` | 4096 | Maximum transform size; determines BRAM depth |
| `MAX_BATCH` | 16 | Maximum batch count per kernel call |
| Modulus `q` | ≤ 31 bits | Fits in uint32; product fits in uint64 |
| Data type | `uint32` | All coefficients and twiddles |

### 4.6 Verification Strategy

- **C simulation:** The existing `ntt_tb_cpp.cpp` testbench loads golden vectors (generated from the Python/SymPy reference) and compares against the HLS kernel output.
- **Python golden model:** `export_vectors.py` generates binary test vectors (`x.bin`, `ref.bin`, `psi_powers.bin`, `twiddles.bin`) from the trusted `negacyclic_ntt_oracle`.
- **Co-simulation:** Vitis HLS C/RTL co-simulation will validate the synthesized RTL matches C-sim results.
- **On-board test:** After bitstream generation, the host driver will run the same golden vectors through the accelerator on a Zynq/Alveo board and compare results.

---

## 5. Summary & Next Steps

| Milestone | Description |
|-----------|-------------|
| Week 1 | Finalize architecture, complete this project plan, create GitHub repo |
| Week 2 | Optimize HLS pragmas (pipelining, BRAM partitioning), pass C-sim |
| Week 3 | Run Vitis HLS synthesis, analyze resource/timing reports, iterate |
| Week 4 | Integrate with Vitis platform, run co-simulation |
| Week 5 | On-board validation, performance benchmarking, final report |
