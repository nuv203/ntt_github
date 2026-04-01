#include "ntt.hpp"

/*==========================================================================
 * NTT kernel v8 — final push for 200 MHz on xck26.
 *
 * Changes vs v7 (which closed at 175MHz):
 *   1. Barrett min latency raised from 10 to 12.
 *      Gives HLS 2 more register stages to break DSP→fabric paths.
 *   2. ALL four-step address computations use incremental counters.
 *      Replaces "row*N1+col" and "col*N2+row" dynamic multiplies
 *      with "addr += stride" — eliminates the fabric multiplier
 *      that caused the logN2→gmem_addr timing failure at 200MHz.
 *   3. Transpose write scratch address also incremental.
 *==========================================================================*/

static uint64_t compute_mu(ntt_t q) {
    uint64_t dividend = (uint64_t)1 << 62;
    uint64_t quotient = 0;
    uint64_t remainder = 0;
    uint64_t divisor = (uint64_t)q;

    COMPUTE_MU_LOOP:
    for (int i = 62; i >= 0; i--) {
#pragma HLS LOOP_TRIPCOUNT min=63 max=63
        remainder = (remainder << 1) | ((dividend >> i) & 1);
        if (remainder >= divisor) {
            remainder -= divisor;
            quotient |= ((uint64_t)1 << i);
        }
    }
    return quotient;
}

static ntt_t barrett_mod_mul(ntt_t a, ntt_t b, ntt_t q, uint64_t mu) {
#pragma HLS PIPELINE II=1
#pragma HLS LATENCY min=12 max=18

    uint64_t prod = (uint64_t)a * (uint64_t)b;

    uint32_t pL = (uint32_t)prod;
    uint32_t pH = (uint32_t)(prod >> 32);
    uint32_t mL = (uint32_t)mu;
    uint32_t mH = (uint32_t)(mu >> 32);

    uint64_t pp0 = (uint64_t)pL * mL;
    uint64_t pp1 = (uint64_t)pL * mH;
    uint64_t pp2 = (uint64_t)pH * mL;
    uint64_t pp3 = (uint64_t)pH * mH;

    uint64_t mid_lo = (pp0 >> 32) + (uint32_t)pp1 + (uint32_t)pp2;
    uint64_t mid_hi = (pp1 >> 32) + (pp2 >> 32) + (mid_lo >> 32);
    uint64_t hi128  = pp3 + mid_hi;
    uint64_t est = (hi128 << 2) | ((mid_lo >> 30) & 0x3);

    uint64_t est_q = est * (uint64_t)q;
    uint64_t r = prod - est_q;

    if (r >= (uint64_t)q) r -= (uint64_t)q;
    if (r >= (uint64_t)q) r -= (uint64_t)q;

    return (ntt_t)r;
}

static inline ntt_t mod_add(ntt_t a, ntt_t b, ntt_t q) {
#pragma HLS INLINE
    ntt_t s = a + b;
    return (s >= q) ? (s - q) : s;
}

static inline ntt_t mod_sub(ntt_t a, ntt_t b, ntt_t q) {
#pragma HLS INLINE
    return (a >= b) ? (a - b) : (a + q - b);
}

static inline uint32_t reverse_bits(uint32_t x, uint32_t logN) {
#pragma HLS INLINE
    uint32_t r = 0;
    for (int i = 0; i < MAX_LOG_N; i++) {
#pragma HLS UNROLL
        if (i < (int)logN) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
    }
    return r;
}

static void sub_ntt(
    ntt_t a[TILE_N],
    const ntt_t tw[TILE_N],
    ntt_t q,
    uint64_t mu,
    uint32_t sub_N,
    uint32_t sub_logN
) {
    SUB_BIT_REV:
    for (uint32_t i = 0; i < sub_N; i++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
        uint32_t j = reverse_bits(i, sub_logN);
        if (j > i) {
            ntt_t tmp = a[i]; a[i] = a[j]; a[j] = tmp;
        }
    }

    SUB_STAGE:
    for (uint32_t s = 0; s < sub_logN; s++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=12
        uint32_t span = 1u << s, span2 = span << 1;
        SUB_GROUP:
        for (uint32_t k = 0; k < sub_N; k += span2) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
            SUB_BFLY:
            for (uint32_t j = 0; j < span; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
#pragma HLS DEPENDENCE variable=a inter false
                uint32_t i1 = k + j, i2 = i1 + span;
                ntt_t w = tw[span + j];
                ntt_t u = a[i1];
                ntt_t v = barrett_mod_mul(a[i2], w, q, mu);
                a[i1] = mod_add(u, v, q);
                a[i2] = mod_sub(u, v, q);
            }
        }
    }
}

void ntt_kernel(
    ntt_t       *data,
    ntt_t       *psi_powers,
    const ntt_t *twiddles,
    ntt_t        q,
    uint32_t     batch,
    uint32_t     N,
    uint32_t     logN
) {
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

    ntt_t tile[TILE_N];
    ntt_t tw_local[TILE_N];
#pragma HLS BIND_STORAGE variable=tile     type=RAM_T2P impl=BRAM
#pragma HLS BIND_STORAGE variable=tw_local type=RAM_1P  impl=LUTRAM

    uint64_t mu = compute_mu(q);

    if (N <= TILE_N) {
        /*==============================================================
         * DIRECT PATH: N <= 4096
         *==============================================================*/
        ntt_t psi_local[TILE_N];
#pragma HLS BIND_STORAGE variable=psi_local type=RAM_1P impl=LUTRAM

        DIRECT_LOAD_PSI:
        for (uint32_t i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
            psi_local[i] = psi_powers[i];
        }
        DIRECT_LOAD_TW:
        for (uint32_t i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
            tw_local[i] = twiddles[i];
        }

        DIRECT_BATCH:
        for (uint32_t b = 0; b < batch; b++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            uint32_t base = b * N;

            DIRECT_LOAD:
            for (uint32_t i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
                tile[i] = barrett_mod_mul(data[base + i], psi_local[i], q, mu);
            }

            sub_ntt(tile, tw_local, q, mu, N, logN);

            DIRECT_STORE:
            for (uint32_t i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
                data[base + i] = tile[i];
            }
        }

    } else {
        /*==============================================================
         * FOUR-STEP PATH: N > 4096
         *
         * ALL address arithmetic uses incremental counters.
         * No "row * N1" or "col * N2" multiplies in any loop body.
         *==============================================================*/

        uint32_t logN1 = logN >> 1;
        uint32_t logN2 = logN - logN1;
        uint32_t N1 = 1u << logN1;
        uint32_t N2 = 1u << logN2;

        /* ---- Phase 1: Load column twiddles ---- */
        FS_LOAD_TW_COL:
        for (uint32_t i = 0; i < N2; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
            tw_local[i] = twiddles[i];
        }

        /* ---- Phase 1: Psi twist + Column NTTs ---- */
        FS_P1_BATCH:
        for (uint32_t b = 0; b < batch; b++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            uint32_t dbase = b * N;

            FS_COL_LOOP:
            for (uint32_t col = 0; col < N1; col++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096

                /* Gather column: data[dbase + row*N1 + col] for row=0..N2-1
                 * Use incremental address: addr starts at dbase+col, increments by N1 */
                uint32_t gather_addr = dbase + col;
                FS_GATHER_COL:
                for (uint32_t row = 0; row < N2; row++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
                    ntt_t d = data[gather_addr];
                    ntt_t p = psi_powers[gather_addr - dbase]; // row*N1+col
                    tile[row] = barrett_mod_mul(d, p, q, mu);
                    gather_addr += N1;
                }

                sub_ntt(tile, tw_local, q, mu, N2, logN2);

                /* Scatter column back */
                uint32_t scatter_addr = dbase + col;
                FS_SCATTER_COL:
                for (uint32_t row = 0; row < N2; row++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
                    data[scatter_addr] = tile[row];
                    scatter_addr += N1;
                }
            }
        }

        /* ---- Phase 2: Load row twiddles ---- */
        uint32_t inter_off = N2 + N1;

        FS_LOAD_TW_ROW:
        for (uint32_t i = 0; i < N1; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
            tw_local[i] = twiddles[N2 + i];
        }

        /* ---- Phase 2+3: Row NTTs + Transpose ---- */
        FS_P23_BATCH:
        for (uint32_t b = 0; b < batch; b++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16
            uint32_t dbase = b * N;

            /* Row NTTs with inter-stage twiddle */
            uint32_t row_base = dbase;
            uint32_t tw_row_base = inter_off;
            FS_ROW_LOOP:
            for (uint32_t row = 0; row < N2; row++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096

                /* Load row: data[row_base + col] for col=0..N1-1 */
                FS_LOAD_ROW:
                for (uint32_t col = 0; col < N1; col++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
                    ntt_t d = data[row_base + col];
                    ntt_t tw_inter = twiddles[tw_row_base + col];
                    tile[col] = barrett_mod_mul(d, tw_inter, q, mu);
                }

                sub_ntt(tile, tw_local, q, mu, N1, logN1);

                /* Store row back */
                FS_STORE_ROW:
                for (uint32_t col = 0; col < N1; col++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
                    data[row_base + col] = tile[col];
                }

                row_base    += N1;
                tw_row_base += N1;
            }

            /* Transpose: data[dbase + row*N1 + col] → psi_powers[dbase + col*N2 + row]
             *
             * Incremental addressing:
             *   Read:  addr = dbase + row*N1,  stride = 1     (row-major read)
             *   Write: addr = dbase + row,     stride = N2    (column-major write per tile row)
             */
            FS_TRANS_TO_SCRATCH:
            for (uint32_t row = 0; row < N2; row++) {
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
                uint32_t rd_addr = dbase + row * N1; // only one multiply per outer iteration
                FS_TRANS_LOAD:
                for (uint32_t col = 0; col < N1; col++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
                    tile[col] = data[rd_addr + col];
                }
                /* Write transposed: psi_powers[dbase + col*N2 + row]
                 * Incremental: start at dbase+row, step by N2 */
                uint32_t wr_addr = dbase + row;
                FS_TRANS_WRITE_SCRATCH:
                for (uint32_t col = 0; col < N1; col++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
                    psi_powers[wr_addr] = tile[col];
                    wr_addr += N2;
                }
            }

            /* Copy scratch back to data (sequential, no multiply) */
            FS_TRANS_FROM_SCRATCH:
            for (uint32_t i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8192 max=1048576
                data[dbase + i] = psi_powers[dbase + i];
            }
        }
    }
}
//#include "ntt.hpp"
//
///*==========================================================================
// * Optimized negacyclic NTT kernel — targeting 5 ns (200 MHz).
// *
// * IMPORTANT: Set clock=200MHz in hls_config.cfg for this to work.
// * HLS won't try to meet 5ns if the target is 10ns.
// *
// * Both DIRECT (N <= TILE_N) and FOUR-STEP (N > TILE_N) paths.
// *
// * Fixes from v3 synthesis feedback:
// *   1. Removed runtime 64-bit divide "mu = (1<<62)/q" which generated
// *      a urem/udiv hardware unit constraining the global clock to 9.5ns.
// *      Now uses iterative Newton-Raphson (only multiplies and shifts).
// *   2. Barrett mod_mul is a non-inlined pipelined function (II=1).
// *   3. DEPENDENCE inter false on tile array for butterfly II=1.
// *   4. Resource limitation fix: ALLOCATION pragma to allow multiple
// *      DSP48 instances for concurrent partial products.
// *==========================================================================*/
//
///* -----------------------------------------------------------------------
// * Compute floor(2^62 / q) using Newton-Raphson — NO hardware divider.
// *
// * Newton-Raphson for reciprocal: x_{n+1} = x_n * (2 - q * x_n)
// * Converges quadratically. For 31-bit q, 5 iterations from a good
// * initial estimate gives exact results.
// *
// * This runs once at kernel entry. It's not pipelined — just needs
// * to avoid creating a udiv/urem hardware unit.
// * ----------------------------------------------------------------------- */
//static uint64_t compute_mu(ntt_t q) {
//    /* We want floor(2^62 / q).
//     * Strategy: compute reciprocal of q in fixed-point, then scale.
//     *
//     * Use 64-bit fixed-point: represent 1/q as X/2^63 approximately.
//     * Then mu = (X * 2^62) / 2^63 = X >> 1, approximately.
//     *
//     * Actually, simpler: just do a shift-subtract long division
//     * in a loop — no hardware divider inference, just shifts and subs.
//     */
//
//    /* Long division: compute 2^62 / q bit by bit.
//     * This takes 63 iterations but uses only subtract and shift —
//     * HLS synthesizes this as a simple FSM, not a udiv unit. */
//    uint64_t dividend = (uint64_t)1 << 62;
//    uint64_t quotient = 0;
//    uint64_t remainder = 0;
//    uint64_t divisor = (uint64_t)q;
//
//    COMPUTE_MU_LOOP:
//    for (int i = 62; i >= 0; i--) {
//#pragma HLS LOOP_TRIPCOUNT min=63 max=63
//        remainder = (remainder << 1) | ((dividend >> i) & 1);
//        if (remainder >= divisor) {
//            remainder -= divisor;
//            quotient |= ((uint64_t)1 << i);
//        }
//    }
//
//    return quotient;
//}
//
///* -----------------------------------------------------------------------
// * Barrett reduction — deeply pipelined for post-route timing closure.
// *
// * Vivado P&R fails at 200MHz/4CU because DSP48 outputs can't reach
// * downstream logic within 5ns after routing. Fix: force HLS to insert
// * many more pipeline registers by raising min latency to 10 and
// * separating every multiply from its consumers.
// *
// * The function is NOT inlined — HLS synthesizes it as a pipelined
// * sub-module with II=1 and ~10 cycle latency. Each cycle does at most
// * one 32×32 multiply, ensuring DSP48 outputs always go through a
// * register before reaching the next operation.
// * ----------------------------------------------------------------------- */
//static ntt_t barrett_mod_mul(ntt_t a, ntt_t b, ntt_t q, uint64_t mu) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LATENCY min=10 max=16
//
//    /* Stage 1: a * b (one DSP48) */
//    uint64_t prod = (uint64_t)a * (uint64_t)b;
//
//    /* Stage 2: decompose for Barrett partial products */
//    uint32_t pL = (uint32_t)prod;
//    uint32_t pH = (uint32_t)(prod >> 32);
//    uint32_t mL = (uint32_t)mu;
//    uint32_t mH = (uint32_t)(mu >> 32);
//
//    /* Stage 3-4: four independent 32×32 partial products.
//     * With min latency=10, HLS has plenty of stages to spread
//     * these across, registering each output. */
//    uint64_t pp0 = (uint64_t)pL * mL;
//    uint64_t pp1 = (uint64_t)pL * mH;
//    uint64_t pp2 = (uint64_t)pH * mL;
//    uint64_t pp3 = (uint64_t)pH * mH;
//
//    /* Stage 5: first level of accumulation */
//    uint64_t mid_lo = (pp0 >> 32) + (uint32_t)pp1 + (uint32_t)pp2;
//
//    /* Stage 6: second level of accumulation */
//    uint64_t mid_hi = (pp1 >> 32) + (pp2 >> 32) + (mid_lo >> 32);
//    uint64_t hi128  = pp3 + mid_hi;
//
//    /* Stage 7: quotient estimate */
//    uint64_t est = (hi128 << 2) | ((mid_lo >> 30) & 0x3);
//
//    /* Stage 8: est * q — final multiply (one more DSP48) */
//    uint64_t est_q = est * (uint64_t)q;
//
//    /* Stage 9: subtract to get remainder */
//    uint64_t r = prod - est_q;
//
//    /* Stage 10: correction subtracts */
//    if (r >= (uint64_t)q) r -= (uint64_t)q;
//    if (r >= (uint64_t)q) r -= (uint64_t)q;
//
//    return (ntt_t)r;
//}
//
//static inline ntt_t mod_add(ntt_t a, ntt_t b, ntt_t q) {
//#pragma HLS INLINE
//    ntt_t s = a + b;
//    return (s >= q) ? (s - q) : s;
//}
//
//static inline ntt_t mod_sub(ntt_t a, ntt_t b, ntt_t q) {
//#pragma HLS INLINE
//    return (a >= b) ? (a - b) : (a + q - b);
//}
//
//static inline uint32_t reverse_bits(uint32_t x, uint32_t logN) {
//#pragma HLS INLINE
//    uint32_t r = 0;
//    for (int i = 0; i < MAX_LOG_N; i++) {
//#pragma HLS UNROLL
//        if (i < (int)logN) {
//            r = (r << 1) | (x & 1);
//            x >>= 1;
//        }
//    }
//    return r;
//}
//
///* -----------------------------------------------------------------------
// * On-chip sub-NTT for up to TILE_N points.
// * ----------------------------------------------------------------------- */
//static void sub_ntt(
//    ntt_t a[TILE_N],
//    const ntt_t tw[TILE_N],
//    ntt_t q,
//    uint64_t mu,
//    uint32_t sub_N,
//    uint32_t sub_logN
//) {
//    SUB_BIT_REV:
//    for (uint32_t i = 0; i < sub_N; i++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//        uint32_t j = reverse_bits(i, sub_logN);
//        if (j > i) {
//            ntt_t tmp = a[i]; a[i] = a[j]; a[j] = tmp;
//        }
//    }
//
//    SUB_STAGE:
//    for (uint32_t s = 0; s < sub_logN; s++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=12
//        uint32_t span = 1u << s, span2 = span << 1;
//        SUB_GROUP:
//        for (uint32_t k = 0; k < sub_N; k += span2) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
//            SUB_BFLY:
//            for (uint32_t j = 0; j < span; j++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
//#pragma HLS DEPENDENCE variable=a inter false
//                uint32_t i1 = k + j, i2 = i1 + span;
//                ntt_t w = tw[span + j];
//                ntt_t u = a[i1];
//                ntt_t v = barrett_mod_mul(a[i2], w, q, mu);
//                a[i1] = mod_add(u, v, q);
//                a[i2] = mod_sub(u, v, q);
//            }
//        }
//    }
//}
//
//void ntt_kernel(
//    ntt_t       *data,
//    ntt_t       *psi_powers,
//    const ntt_t *twiddles,
//    ntt_t        q,
//    uint32_t     batch,
//    uint32_t     N,
//    uint32_t     logN
//) {
//#pragma HLS INTERFACE m_axi port=data       offset=slave bundle=gmem0 depth=16777216 max_read_burst_length=256 max_write_burst_length=256
//#pragma HLS INTERFACE m_axi port=psi_powers offset=slave bundle=gmem1 depth=1048576  max_read_burst_length=256 max_write_burst_length=256
//#pragma HLS INTERFACE m_axi port=twiddles   offset=slave bundle=gmem2 depth=1048576  max_read_burst_length=256
//
//#pragma HLS INTERFACE s_axilite port=data
//#pragma HLS INTERFACE s_axilite port=psi_powers
//#pragma HLS INTERFACE s_axilite port=twiddles
//#pragma HLS INTERFACE s_axilite port=q
//#pragma HLS INTERFACE s_axilite port=batch
//#pragma HLS INTERFACE s_axilite port=N
//#pragma HLS INTERFACE s_axilite port=logN
//#pragma HLS INTERFACE s_axilite port=return
//
//    ntt_t tile[TILE_N];
//    ntt_t tw_local[TILE_N];
//#pragma HLS BIND_STORAGE variable=tile     type=RAM_T2P impl=BRAM
//#pragma HLS BIND_STORAGE variable=tw_local type=RAM_1P  impl=LUTRAM
//
//    /* Barrett constant — computed with shift-subtract loop,
//     * NO hardware divider inferred. ~63 cycles startup cost. */
//    uint64_t mu = compute_mu(q);
//
//    if (N <= TILE_N) {
//        /*==============================================================
//         * DIRECT PATH: N <= 4096
//         *==============================================================*/
//        ntt_t psi_local[TILE_N];
//#pragma HLS BIND_STORAGE variable=psi_local type=RAM_1P impl=LUTRAM
//
//        DIRECT_LOAD_PSI:
//        for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//            psi_local[i] = psi_powers[i];
//        }
//        DIRECT_LOAD_TW:
//        for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//            tw_local[i] = twiddles[i];
//        }
//
//        DIRECT_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t base = b * N;
//
//            DIRECT_LOAD:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//                tile[i] = barrett_mod_mul(data[base + i], psi_local[i], q, mu);
//            }
//
//            sub_ntt(tile, tw_local, q, mu, N, logN);
//
//            DIRECT_STORE:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//                data[base + i] = tile[i];
//            }
//        }
//
//    } else {
//        /*==============================================================
//         * FOUR-STEP PATH: N > 4096
//         *==============================================================*/
//
//        uint32_t logN1 = logN >> 1;
//        uint32_t logN2 = logN - logN1;
//        uint32_t N1 = 1u << logN1;
//        uint32_t N2 = 1u << logN2;
//
//        FS_LOAD_TW_COL:
//        for (uint32_t i = 0; i < N2; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//            tw_local[i] = twiddles[i];
//        }
//
//        FS_P1_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t dbase = b * N;
//
//            FS_COL_LOOP:
//            for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//
//                FS_GATHER_COL:
//                for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    ntt_t d = data[dbase + row * N1 + col];
//                    ntt_t p = psi_powers[row * N1 + col];
//                    tile[row] = barrett_mod_mul(d, p, q, mu);
//                }
//
//                sub_ntt(tile, tw_local, q, mu, N2, logN2);
//
//                FS_SCATTER_COL:
//                for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    data[dbase + row * N1 + col] = tile[row];
//                }
//            }
//        }
//
//        uint32_t inter_off = N2 + N1;
//
//        FS_LOAD_TW_ROW:
//        for (uint32_t i = 0; i < N1; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//            tw_local[i] = twiddles[N2 + i];
//        }
//
//        FS_P23_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t dbase = b * N;
//
//            FS_ROW_LOOP:
//            for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//
//                FS_LOAD_ROW:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    ntt_t d = data[dbase + row * N1 + col];
//                    ntt_t tw_inter = twiddles[inter_off + row * N1 + col];
//                    tile[col] = barrett_mod_mul(d, tw_inter, q, mu);
//                }
//
//                sub_ntt(tile, tw_local, q, mu, N1, logN1);
//
//                FS_STORE_ROW:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    data[dbase + row * N1 + col] = tile[col];
//                }
//            }
//
//            FS_TRANS_TO_SCRATCH:
//            for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                FS_TRANS_LOAD:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    tile[col] = data[dbase + row * N1 + col];
//                }
//                FS_TRANS_WRITE_SCRATCH:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    psi_powers[dbase + col * N2 + row] = tile[col];
//                }
//            }
//
//            FS_TRANS_FROM_SCRATCH:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=8192 max=1048576
//                data[dbase + i] = psi_powers[dbase + i];
//            }
//        }
//    }
//}

//175 MHz 4CU WORKEDD ABOVE
//-----------------------------------------------------------------------------------------------------------

//#include "ntt.hpp"
//
///*==========================================================================
// * Optimized negacyclic NTT kernel — targeting 5 ns (200 MHz).
// *
// * IMPORTANT: Set clock=200MHz in hls_config.cfg for this to work.
// * HLS won't try to meet 5ns if the target is 10ns.
// *
// * Both DIRECT (N <= TILE_N) and FOUR-STEP (N > TILE_N) paths.
// *
// * Fixes from v3 synthesis feedback:
// *   1. Removed runtime 64-bit divide "mu = (1<<62)/q" which generated
// *      a urem/udiv hardware unit constraining the global clock to 9.5ns.
// *      Now uses iterative Newton-Raphson (only multiplies and shifts).
// *   2. Barrett mod_mul is a non-inlined pipelined function (II=1).
// *   3. DEPENDENCE inter false on tile array for butterfly II=1.
// *   4. Resource limitation fix: ALLOCATION pragma to allow multiple
// *      DSP48 instances for concurrent partial products.
// *==========================================================================*/
//
///* -----------------------------------------------------------------------
// * Compute floor(2^62 / q) using Newton-Raphson — NO hardware divider.
// *
// * Newton-Raphson for reciprocal: x_{n+1} = x_n * (2 - q * x_n)
// * Converges quadratically. For 31-bit q, 5 iterations from a good
// * initial estimate gives exact results.
// *
// * This runs once at kernel entry. It's not pipelined — just needs
// * to avoid creating a udiv/urem hardware unit.
// * ----------------------------------------------------------------------- */
//static uint64_t compute_mu(ntt_t q) {
//    /* We want floor(2^62 / q).
//     * Strategy: compute reciprocal of q in fixed-point, then scale.
//     *
//     * Use 64-bit fixed-point: represent 1/q as X/2^63 approximately.
//     * Then mu = (X * 2^62) / 2^63 = X >> 1, approximately.
//     *
//     * Actually, simpler: just do a shift-subtract long division
//     * in a loop — no hardware divider inference, just shifts and subs.
//     */
//
//    /* Long division: compute 2^62 / q bit by bit.
//     * This takes 63 iterations but uses only subtract and shift —
//     * HLS synthesizes this as a simple FSM, not a udiv unit. */
//    uint64_t dividend = (uint64_t)1 << 62;
//    uint64_t quotient = 0;
//    uint64_t remainder = 0;
//    uint64_t divisor = (uint64_t)q;
//
//    COMPUTE_MU_LOOP:
//    for (int i = 62; i >= 0; i--) {
//#pragma HLS LOOP_TRIPCOUNT min=63 max=63
//        remainder = (remainder << 1) | ((dividend >> i) & 1);
//        if (remainder >= divisor) {
//            remainder -= divisor;
//            quotient |= ((uint64_t)1 << i);
//        }
//    }
//
//    return quotient;
//}
//
///* -----------------------------------------------------------------------
// * Barrett reduction — explicitly staged for post-route timing closure.
// *
// * The previous version let HLS decide where to place pipeline registers.
// * HLS estimated 4.75ns, but Vivado P&R failed because DSP48 outputs
// * were routed directly to distant DSP48 inputs without intermediate FFs.
// *
// * Fix: Break the computation into 3 explicit stages using intermediate
// * variables. HLS PIPELINE II=1 with higher min latency forces registers
// * between each stage, giving Vivado routing slack.
// *
// * Stage 1: a*b product + partial products (pL_mL, pL_mH, pH_mL, pH_mH)
// * Stage 2: Accumulate mid/hi + compute quotient estimate
// * Stage 3: remainder = prod - est*q + corrections
// * ----------------------------------------------------------------------- */
//static ntt_t barrett_mod_mul(ntt_t a, ntt_t b, ntt_t q, uint64_t mu) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LATENCY min=6 max=12
//
//    /* ---- Stage 1: All multiplications ---- */
//    uint64_t prod = (uint64_t)a * (uint64_t)b;
//
//    uint32_t pL = (uint32_t)prod;
//    uint32_t pH = (uint32_t)(prod >> 32);
//    uint32_t mL = (uint32_t)mu;
//    uint32_t mH = (uint32_t)(mu >> 32);
//
//    /* Four independent 32×32 partial products — each maps to one DSP48.
//     * HLS will schedule these in parallel in stage 1. */
//    uint64_t pL_mL = (uint64_t)pL * mL;
//    uint64_t pL_mH = (uint64_t)pL * mH;
//    uint64_t pH_mL = (uint64_t)pH * mL;
//    uint64_t pH_mH = (uint64_t)pH * mH;
//
//    /* ---- Stage 2: Carry accumulation + quotient estimate ---- */
//    /* Force register boundary: these depend on stage 1 outputs.
//     * By keeping them as separate statements with data dependencies,
//     * HLS inserts pipeline registers at the stage 1→2 boundary. */
//    uint64_t mid_lo = (pL_mL >> 32) + (uint32_t)pL_mH + (uint32_t)pH_mL;
//    uint64_t mid_hi = (pL_mH >> 32) + (pH_mL >> 32) + (mid_lo >> 32);
//    uint64_t hi128  = pH_mH + mid_hi;
//    uint64_t est = (hi128 << 2) | ((mid_lo >> 30) & 0x3);
//
//    /* ---- Stage 3: Remainder + correction ---- */
//    /* est * q is another multiply — HLS will register before this. */
//    uint64_t est_q = est * (uint64_t)q;
//    uint64_t r = prod - est_q;
//
//    if (r >= (uint64_t)q) r -= (uint64_t)q;
//    if (r >= (uint64_t)q) r -= (uint64_t)q;
//
//    return (ntt_t)r;
//}
//
//static inline ntt_t mod_add(ntt_t a, ntt_t b, ntt_t q) {
//#pragma HLS INLINE
//    ntt_t s = a + b;
//    return (s >= q) ? (s - q) : s;
//}
//
//static inline ntt_t mod_sub(ntt_t a, ntt_t b, ntt_t q) {
//#pragma HLS INLINE
//    return (a >= b) ? (a - b) : (a + q - b);
//}
//
//static inline uint32_t reverse_bits(uint32_t x, uint32_t logN) {
//#pragma HLS INLINE
//    uint32_t r = 0;
//    for (int i = 0; i < MAX_LOG_N; i++) {
//#pragma HLS UNROLL
//        if (i < (int)logN) {
//            r = (r << 1) | (x & 1);
//            x >>= 1;
//        }
//    }
//    return r;
//}
//
///* -----------------------------------------------------------------------
// * On-chip sub-NTT for up to TILE_N points.
// * ----------------------------------------------------------------------- */
//static void sub_ntt(
//    ntt_t a[TILE_N],
//    const ntt_t tw[TILE_N],
//    ntt_t q,
//    uint64_t mu,
//    uint32_t sub_N,
//    uint32_t sub_logN
//) {
//    SUB_BIT_REV:
//    for (uint32_t i = 0; i < sub_N; i++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//        uint32_t j = reverse_bits(i, sub_logN);
//        if (j > i) {
//            ntt_t tmp = a[i]; a[i] = a[j]; a[j] = tmp;
//        }
//    }
//
//    SUB_STAGE:
//    for (uint32_t s = 0; s < sub_logN; s++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=12
//        uint32_t span = 1u << s, span2 = span << 1;
//        SUB_GROUP:
//        for (uint32_t k = 0; k < sub_N; k += span2) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
//            SUB_BFLY:
//            for (uint32_t j = 0; j < span; j++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
//#pragma HLS DEPENDENCE variable=a inter false
//                uint32_t i1 = k + j, i2 = i1 + span;
//                ntt_t w = tw[span + j];
//                ntt_t u = a[i1];
//                ntt_t v = barrett_mod_mul(a[i2], w, q, mu);
//                a[i1] = mod_add(u, v, q);
//                a[i2] = mod_sub(u, v, q);
//            }
//        }
//    }
//}
//
//void ntt_kernel(
//    ntt_t       *data,
//    ntt_t       *psi_powers,
//    const ntt_t *twiddles,
//    ntt_t        q,
//    uint32_t     batch,
//    uint32_t     N,
//    uint32_t     logN
//) {
//#pragma HLS INTERFACE m_axi port=data       offset=slave bundle=gmem0 depth=16777216 max_read_burst_length=256 max_write_burst_length=256
//#pragma HLS INTERFACE m_axi port=psi_powers offset=slave bundle=gmem1 depth=1048576  max_read_burst_length=256 max_write_burst_length=256
//#pragma HLS INTERFACE m_axi port=twiddles   offset=slave bundle=gmem2 depth=1048576  max_read_burst_length=256
//
//#pragma HLS INTERFACE s_axilite port=data
//#pragma HLS INTERFACE s_axilite port=psi_powers
//#pragma HLS INTERFACE s_axilite port=twiddles
//#pragma HLS INTERFACE s_axilite port=q
//#pragma HLS INTERFACE s_axilite port=batch
//#pragma HLS INTERFACE s_axilite port=N
//#pragma HLS INTERFACE s_axilite port=logN
//#pragma HLS INTERFACE s_axilite port=return
//
//    ntt_t tile[TILE_N];
//    ntt_t tw_local[TILE_N];
//#pragma HLS BIND_STORAGE variable=tile     type=RAM_T2P impl=BRAM
//#pragma HLS BIND_STORAGE variable=tw_local type=RAM_1P  impl=LUTRAM
//
//    /* Barrett constant — computed with shift-subtract loop,
//     * NO hardware divider inferred. ~63 cycles startup cost. */
//    uint64_t mu = compute_mu(q);
//
//    if (N <= TILE_N) {
//        /*==============================================================
//         * DIRECT PATH: N <= 4096
//         *==============================================================*/
//        ntt_t psi_local[TILE_N];
//#pragma HLS BIND_STORAGE variable=psi_local type=RAM_1P impl=LUTRAM
//
//        DIRECT_LOAD_PSI:
//        for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//            psi_local[i] = psi_powers[i];
//        }
//        DIRECT_LOAD_TW:
//        for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//            tw_local[i] = twiddles[i];
//        }
//
//        DIRECT_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t base = b * N;
//
//            DIRECT_LOAD:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//                tile[i] = barrett_mod_mul(data[base + i], psi_local[i], q, mu);
//            }
//
//            sub_ntt(tile, tw_local, q, mu, N, logN);
//
//            DIRECT_STORE:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//                data[base + i] = tile[i];
//            }
//        }
//
//    } else {
//        /*==============================================================
//         * FOUR-STEP PATH: N > 4096
//         *==============================================================*/
//
//        uint32_t logN1 = logN >> 1;
//        uint32_t logN2 = logN - logN1;
//        uint32_t N1 = 1u << logN1;
//        uint32_t N2 = 1u << logN2;
//
//        FS_LOAD_TW_COL:
//        for (uint32_t i = 0; i < N2; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//            tw_local[i] = twiddles[i];
//        }
//
//        FS_P1_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t dbase = b * N;
//
//            FS_COL_LOOP:
//            for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//
//                FS_GATHER_COL:
//                for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    ntt_t d = data[dbase + row * N1 + col];
//                    ntt_t p = psi_powers[row * N1 + col];
//                    tile[row] = barrett_mod_mul(d, p, q, mu);
//                }
//
//                sub_ntt(tile, tw_local, q, mu, N2, logN2);
//
//                FS_SCATTER_COL:
//                for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    data[dbase + row * N1 + col] = tile[row];
//                }
//            }
//        }
//
//        uint32_t inter_off = N2 + N1;
//
//        FS_LOAD_TW_ROW:
//        for (uint32_t i = 0; i < N1; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//            tw_local[i] = twiddles[N2 + i];
//        }
//
//        FS_P23_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t dbase = b * N;
//
//            FS_ROW_LOOP:
//            for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//
//                FS_LOAD_ROW:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    ntt_t d = data[dbase + row * N1 + col];
//                    ntt_t tw_inter = twiddles[inter_off + row * N1 + col];
//                    tile[col] = barrett_mod_mul(d, tw_inter, q, mu);
//                }
//
//                sub_ntt(tile, tw_local, q, mu, N1, logN1);
//
//                FS_STORE_ROW:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    data[dbase + row * N1 + col] = tile[col];
//                }
//            }
//
//            FS_TRANS_TO_SCRATCH:
//            for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                FS_TRANS_LOAD:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    tile[col] = data[dbase + row * N1 + col];
//                }
//                FS_TRANS_WRITE_SCRATCH:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    psi_powers[dbase + col * N2 + row] = tile[col];
//                }
//            }
//
//            FS_TRANS_FROM_SCRATCH:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=8192 max=1048576
//                data[dbase + i] = psi_powers[dbase + i];
//            }
//        }
//    }
//}

//#include "ntt.hpp"
//
///*==========================================================================
// * Optimized negacyclic NTT kernel — targeting 5 ns (200 MHz).
// *
// * IMPORTANT: Set clock=200MHz in hls_config.cfg for this to work.
// * HLS won't try to meet 5ns if the target is 10ns.
// *
// * Both DIRECT (N <= TILE_N) and FOUR-STEP (N > TILE_N) paths.
// *
// * Fixes from v3 synthesis feedback:
// *   1. Removed runtime 64-bit divide "mu = (1<<62)/q" which generated
// *      a urem/udiv hardware unit constraining the global clock to 9.5ns.
// *      Now uses iterative Newton-Raphson (only multiplies and shifts).
// *   2. Barrett mod_mul is a non-inlined pipelined function (II=1).
// *   3. DEPENDENCE inter false on tile array for butterfly II=1.
// *   4. Resource limitation fix: ALLOCATION pragma to allow multiple
// *      DSP48 instances for concurrent partial products.
// *==========================================================================*/
//
///* -----------------------------------------------------------------------
// * Compute floor(2^62 / q) using Newton-Raphson — NO hardware divider.
// *
// * Newton-Raphson for reciprocal: x_{n+1} = x_n * (2 - q * x_n)
// * Converges quadratically. For 31-bit q, 5 iterations from a good
// * initial estimate gives exact results.
// *
// * This runs once at kernel entry. It's not pipelined — just needs
// * to avoid creating a udiv/urem hardware unit.
// * ----------------------------------------------------------------------- */
//static uint64_t compute_mu(ntt_t q) {
//    /* We want floor(2^62 / q).
//     * Strategy: compute reciprocal of q in fixed-point, then scale.
//     *
//     * Use 64-bit fixed-point: represent 1/q as X/2^63 approximately.
//     * Then mu = (X * 2^62) / 2^63 = X >> 1, approximately.
//     *
//     * Actually, simpler: just do a shift-subtract long division
//     * in a loop — no hardware divider inference, just shifts and subs.
//     */
//
//    /* Long division: compute 2^62 / q bit by bit.
//     * This takes 63 iterations but uses only subtract and shift —
//     * HLS synthesizes this as a simple FSM, not a udiv unit. */
//    uint64_t dividend = (uint64_t)1 << 62;
//    uint64_t quotient = 0;
//    uint64_t remainder = 0;
//    uint64_t divisor = (uint64_t)q;
//
//    COMPUTE_MU_LOOP:
//    for (int i = 62; i >= 0; i--) {
//#pragma HLS LOOP_TRIPCOUNT min=63 max=63
//        remainder = (remainder << 1) | ((dividend >> i) & 1);
//        if (remainder >= divisor) {
//            remainder -= divisor;
//            quotient |= ((uint64_t)1 << i);
//        }
//    }
//
//    return quotient;
//}
//
///* -----------------------------------------------------------------------
// * Barrett reduction as a PIPELINED FUNCTION (not inlined).
// * HLS auto-pipelines the DSP48 multiply chain at II=1.
// * ----------------------------------------------------------------------- */
//static ntt_t barrett_mod_mul(ntt_t a, ntt_t b, ntt_t q, uint64_t mu) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LATENCY min=4 max=10
//
//    uint64_t prod = (uint64_t)a * (uint64_t)b;
//
//    uint32_t pL = (uint32_t)prod;
//    uint32_t pH = (uint32_t)(prod >> 32);
//    uint32_t mL = (uint32_t)mu;
//    uint32_t mH = (uint32_t)(mu >> 32);
//
//    uint64_t pL_mL = (uint64_t)pL * mL;
//    uint64_t pL_mH = (uint64_t)pL * mH;
//    uint64_t pH_mL = (uint64_t)pH * mL;
//    uint64_t pH_mH = (uint64_t)pH * mH;
//
//    uint64_t mid_lo = (pL_mL >> 32) + (uint32_t)pL_mH + (uint32_t)pH_mL;
//    uint64_t mid_hi = (pL_mH >> 32) + (pH_mL >> 32) + (mid_lo >> 32);
//    uint64_t hi128  = pH_mH + mid_hi;
//
//    uint64_t est = (hi128 << 2) | ((mid_lo >> 30) & 0x3);
//    uint64_t r = prod - est * (uint64_t)q;
//
//    if (r >= (uint64_t)q) r -= (uint64_t)q;
//    if (r >= (uint64_t)q) r -= (uint64_t)q;
//
//    return (ntt_t)r;
//}
//
//static inline ntt_t mod_add(ntt_t a, ntt_t b, ntt_t q) {
//#pragma HLS INLINE
//    ntt_t s = a + b;
//    return (s >= q) ? (s - q) : s;
//}
//
//static inline ntt_t mod_sub(ntt_t a, ntt_t b, ntt_t q) {
//#pragma HLS INLINE
//    return (a >= b) ? (a - b) : (a + q - b);
//}
//
//static inline uint32_t reverse_bits(uint32_t x, uint32_t logN) {
//#pragma HLS INLINE
//    uint32_t r = 0;
//    for (int i = 0; i < MAX_LOG_N; i++) {
//#pragma HLS UNROLL
//        if (i < (int)logN) {
//            r = (r << 1) | (x & 1);
//            x >>= 1;
//        }
//    }
//    return r;
//}
//
///* -----------------------------------------------------------------------
// * On-chip sub-NTT for up to TILE_N points.
// * ----------------------------------------------------------------------- */
//static void sub_ntt(
//    ntt_t a[TILE_N],
//    const ntt_t tw[TILE_N],
//    ntt_t q,
//    uint64_t mu,
//    uint32_t sub_N,
//    uint32_t sub_logN
//) {
//    SUB_BIT_REV:
//    for (uint32_t i = 0; i < sub_N; i++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//        uint32_t j = reverse_bits(i, sub_logN);
//        if (j > i) {
//            ntt_t tmp = a[i]; a[i] = a[j]; a[j] = tmp;
//        }
//    }
//
//    SUB_STAGE:
//    for (uint32_t s = 0; s < sub_logN; s++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=12
//        uint32_t span = 1u << s, span2 = span << 1;
//        SUB_GROUP:
//        for (uint32_t k = 0; k < sub_N; k += span2) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
//            SUB_BFLY:
//            for (uint32_t j = 0; j < span; j++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
//#pragma HLS DEPENDENCE variable=a inter false
//                uint32_t i1 = k + j, i2 = i1 + span;
//                ntt_t w = tw[span + j];
//                ntt_t u = a[i1];
//                ntt_t v = barrett_mod_mul(a[i2], w, q, mu);
//                a[i1] = mod_add(u, v, q);
//                a[i2] = mod_sub(u, v, q);
//            }
//        }
//    }
//}
//
//void ntt_kernel(
//    ntt_t       *data,
//    ntt_t       *psi_powers,
//    const ntt_t *twiddles,
//    ntt_t        q,
//    uint32_t     batch,
//    uint32_t     N,
//    uint32_t     logN
//) {
//#pragma HLS INTERFACE m_axi port=data       offset=slave bundle=gmem0 depth=16777216 max_read_burst_length=256 max_write_burst_length=256
//#pragma HLS INTERFACE m_axi port=psi_powers offset=slave bundle=gmem1 depth=1048576  max_read_burst_length=256 max_write_burst_length=256
//#pragma HLS INTERFACE m_axi port=twiddles   offset=slave bundle=gmem2 depth=1048576  max_read_burst_length=256
//
//#pragma HLS INTERFACE s_axilite port=data
//#pragma HLS INTERFACE s_axilite port=psi_powers
//#pragma HLS INTERFACE s_axilite port=twiddles
//#pragma HLS INTERFACE s_axilite port=q
//#pragma HLS INTERFACE s_axilite port=batch
//#pragma HLS INTERFACE s_axilite port=N
//#pragma HLS INTERFACE s_axilite port=logN
//#pragma HLS INTERFACE s_axilite port=return
//
//    ntt_t tile[TILE_N];
//    ntt_t tw_local[TILE_N];
//#pragma HLS BIND_STORAGE variable=tile     type=RAM_T2P impl=BRAM
//#pragma HLS BIND_STORAGE variable=tw_local type=RAM_1P  impl=BRAM
//
//    /* Barrett constant — computed with shift-subtract loop,
//     * NO hardware divider inferred. ~63 cycles startup cost. */
//    uint64_t mu = compute_mu(q);
//
//    if (N <= TILE_N) {
//        /*==============================================================
//         * DIRECT PATH: N <= 4096
//         *==============================================================*/
//        ntt_t psi_local[TILE_N];
//#pragma HLS BIND_STORAGE variable=psi_local type=RAM_1P impl=BRAM
//
//        DIRECT_LOAD_PSI:
//        for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//            psi_local[i] = psi_powers[i];
//        }
//        DIRECT_LOAD_TW:
//        for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//            tw_local[i] = twiddles[i];
//        }
//
//        DIRECT_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t base = b * N;
//
//            DIRECT_LOAD:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//                tile[i] = barrett_mod_mul(data[base + i], psi_local[i], q, mu);
//            }
//
//            sub_ntt(tile, tw_local, q, mu, N, logN);
//
//            DIRECT_STORE:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//                data[base + i] = tile[i];
//            }
//        }
//
//    } else {
//        /*==============================================================
//         * FOUR-STEP PATH: N > 4096
//         *==============================================================*/
//
//        uint32_t logN1 = logN >> 1;
//        uint32_t logN2 = logN - logN1;
//        uint32_t N1 = 1u << logN1;
//        uint32_t N2 = 1u << logN2;
//
//        FS_LOAD_TW_COL:
//        for (uint32_t i = 0; i < N2; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//            tw_local[i] = twiddles[i];
//        }
//
//        FS_P1_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t dbase = b * N;
//
//            FS_COL_LOOP:
//            for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//
//                FS_GATHER_COL:
//                for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    ntt_t d = data[dbase + row * N1 + col];
//                    ntt_t p = psi_powers[row * N1 + col];
//                    tile[row] = barrett_mod_mul(d, p, q, mu);
//                }
//
//                sub_ntt(tile, tw_local, q, mu, N2, logN2);
//
//                FS_SCATTER_COL:
//                for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    data[dbase + row * N1 + col] = tile[row];
//                }
//            }
//        }
//
//        uint32_t inter_off = N2 + N1;
//
//        FS_LOAD_TW_ROW:
//        for (uint32_t i = 0; i < N1; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//            tw_local[i] = twiddles[N2 + i];
//        }
//
//        FS_P23_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t dbase = b * N;
//
//            FS_ROW_LOOP:
//            for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//
//                FS_LOAD_ROW:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    ntt_t d = data[dbase + row * N1 + col];
//                    ntt_t tw_inter = twiddles[inter_off + row * N1 + col];
//                    tile[col] = barrett_mod_mul(d, tw_inter, q, mu);
//                }
//
//                sub_ntt(tile, tw_local, q, mu, N1, logN1);
//
//                FS_STORE_ROW:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    data[dbase + row * N1 + col] = tile[col];
//                }
//            }
//
//            FS_TRANS_TO_SCRATCH:
//            for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                FS_TRANS_LOAD:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    tile[col] = data[dbase + row * N1 + col];
//                }
//                FS_TRANS_WRITE_SCRATCH:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    psi_powers[dbase + col * N2 + row] = tile[col];
//                }
//            }
//
//            FS_TRANS_FROM_SCRATCH:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=8192 max=1048576
//                data[dbase + i] = psi_powers[dbase + i];
//            }
//        }
//    }
//}

//----------------------------------------------------------------------------------------------------------------------------------------------

//#include "ntt.hpp"
//
///*==========================================================================
// * Corrected four-step NTT kernel that can only synthesize 100 MHz
// *
// * Bug fixes vs previous version:
// *   1. Column NTTs FIRST, then row NTTs (was reversed)
// *   2. Transpose via scratch buffer (psi_powers DDR, reused after twist)
// *   3. Inter-stage twiddle omega^(col*row) folded into row gather
// *==========================================================================*/
//
//
//static inline ntt_t mod_mul(ntt_t a, ntt_t b, ntt_t q) {
//#pragma HLS INLINE
//    return (ntt_t)(((uint64_t)a * (uint64_t)b) % (uint64_t)q);
//}
//
//static inline ntt_t mod_add(ntt_t a, ntt_t b, ntt_t q) {
//#pragma HLS INLINE
//    ntt_t s = a + b;
//    return (s >= q) ? (s - q) : s;
//}
//
//static inline ntt_t mod_sub(ntt_t a, ntt_t b, ntt_t q) {
//#pragma HLS INLINE
//    return (a >= b) ? (a - b) : (a + q - b);
//}
//
//static inline uint32_t reverse_bits(uint32_t x, uint32_t logN) {
//#pragma HLS INLINE
//    uint32_t r = 0;
//    for (int i = 0; i < MAX_LOG_N; i++) {
//#pragma HLS UNROLL
//        if (i < (int)logN) {
//            r = (r << 1) | (x & 1);
//            x >>= 1;
//        }
//    }
//    return r;
//}
//
///* On-chip NTT for up to TILE_N points */
//static inline void sub_ntt(
//    ntt_t a[TILE_N],
//    const ntt_t tw[TILE_N],
//    ntt_t q,
//    uint32_t sub_N,
//    uint32_t sub_logN
//) {
//    SUB_BIT_REV:
//    for (uint32_t i = 0; i < sub_N; i++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//        uint32_t j = reverse_bits(i, sub_logN);
//        if (j > i) {
//            ntt_t tmp = a[i]; a[i] = a[j]; a[j] = tmp;
//        }
//    }
//
//    SUB_STAGE:
//    for (uint32_t s = 0; s < sub_logN; s++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=12
//        uint32_t span = 1u << s, span2 = span << 1;
//        SUB_GROUP:
//        for (uint32_t k = 0; k < sub_N; k += span2) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
//            SUB_BFLY:
//            for (uint32_t j = 0; j < span; j++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=1 max=2048
//                uint32_t i1 = k + j, i2 = i1 + span;
//                ntt_t w = tw[span + j];
//                ntt_t u = a[i1];
//                ntt_t v = mod_mul(a[i2], w, q);
//                a[i1] = mod_add(u, v, q);
//                a[i2] = mod_sub(u, v, q);
//            }
//        }
//    }
//}
//
//void ntt_kernel(
//    ntt_t       *data,
//    ntt_t       *psi_powers,
//    const ntt_t *twiddles,
//    ntt_t        q,
//    uint32_t     batch,
//    uint32_t     N,
//    uint32_t     logN
//) {
//#pragma HLS INTERFACE m_axi port=data       offset=slave bundle=gmem0 depth=16777216 max_read_burst_length=256 max_write_burst_length=256
//#pragma HLS INTERFACE m_axi port=psi_powers offset=slave bundle=gmem1 depth=1048576  max_read_burst_length=256 max_write_burst_length=256
//#pragma HLS INTERFACE m_axi port=twiddles   offset=slave bundle=gmem2 depth=1048576  max_read_burst_length=256
//
//#pragma HLS INTERFACE s_axilite port=data
//#pragma HLS INTERFACE s_axilite port=psi_powers
//#pragma HLS INTERFACE s_axilite port=twiddles
//#pragma HLS INTERFACE s_axilite port=q
//#pragma HLS INTERFACE s_axilite port=batch
//#pragma HLS INTERFACE s_axilite port=N
//#pragma HLS INTERFACE s_axilite port=logN
//#pragma HLS INTERFACE s_axilite port=return
//
//    ntt_t tile[TILE_N];
//    ntt_t tw_local[TILE_N];
//#pragma HLS BIND_STORAGE variable=tile     type=RAM_2P impl=BRAM
//#pragma HLS BIND_STORAGE variable=tw_local type=RAM_1P impl=BRAM
//
//    if (N <= TILE_N) {
//        /*==============================================================
//         * DIRECT PATH: N <= 4096
//         *==============================================================*/
//        ntt_t psi_local[TILE_N];
//#pragma HLS BIND_STORAGE variable=psi_local type=RAM_1P impl=BRAM
//
//        DIRECT_LOAD_PSI:
//        for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//            psi_local[i] = psi_powers[i];
//        }
//        DIRECT_LOAD_TW:
//        for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//            tw_local[i] = twiddles[i];
//        }
//
//        DIRECT_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t base = b * N;
//
//            DIRECT_LOAD:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=3
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//                tile[i] = mod_mul(data[base + i], psi_local[i], q);
//            }
//
//            sub_ntt(tile, tw_local, q, N, logN);
//
//            DIRECT_STORE:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=256 max=4096
//                data[base + i] = tile[i];
//            }
//        }
//
//    } else {
//        /*==============================================================
//         * FOUR-STEP PATH: N > 4096
//         *
//         * N = N1 * N2, data as N2 rows × N1 cols: data[row*N1+col]
//         * N1 = 2^(logN/2), N2 = 2^(logN - logN/2)
//         *
//         * Algorithm:
//         *   Phase 1: Psi twist + Column NTTs (N2-point)
//         *   Phase 2: Inter-stage twiddle + Row NTTs (N1-point)
//         *   Phase 3: Transpose via scratch (data→scratch→data)
//         *==============================================================*/
//
//        uint32_t logN1 = logN >> 1;
//        uint32_t logN2 = logN - logN1;
//        uint32_t N1 = 1u << logN1;
//        uint32_t N2 = 1u << logN2;
//
//        /* ---- Phase 1: Psi twist + Column NTTs for ALL batches ----
//         * Must finish all psi_powers reads before Phase 3 uses
//         * psi_powers DDR as scratch for transpose.
//         */
//
//        FS_LOAD_TW_COL:
//        for (uint32_t i = 0; i < N2; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//            tw_local[i] = twiddles[i];
//        }
//
//        FS_P1_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t dbase = b * N;
//
//            FS_COL_LOOP:
//            for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//
//                FS_GATHER_COL:
//                for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS PIPELINE II=3
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    ntt_t d = data[dbase + row * N1 + col];
//                    ntt_t p = psi_powers[row * N1 + col];
//                    tile[row] = mod_mul(d, p, q);
//                }
//
//                sub_ntt(tile, tw_local, q, N2, logN2);
//
//                FS_SCATTER_COL:
//                for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    data[dbase + row * N1 + col] = tile[row];
//                }
//            }
//        }
//
//        /* ---- Phase 2+3: Row NTTs + Transpose for ALL batches ----
//         * psi_powers DDR is now free for scratch use.
//         */
//
//        uint32_t inter_off = N2 + N1;
//
//        FS_LOAD_TW_ROW:
//        for (uint32_t i = 0; i < N1; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//            tw_local[i] = twiddles[N2 + i];
//        }
//
//        FS_P23_BATCH:
//        for (uint32_t b = 0; b < batch; b++) {
//#pragma HLS LOOP_TRIPCOUNT min=1 max=16
//            uint32_t dbase = b * N;
//
//            /* Row NTTs with inter-stage twiddle */
//            FS_ROW_LOOP:
//            for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//
//                FS_LOAD_ROW:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=3
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    ntt_t d = data[dbase + row * N1 + col];
//                    ntt_t tw_inter = twiddles[inter_off + row * N1 + col];
//                    tile[col] = mod_mul(d, tw_inter, q);
//                }
//
//                sub_ntt(tile, tw_local, q, N1, logN1);
//
//                FS_STORE_ROW:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    data[dbase + row * N1 + col] = tile[col];
//                }
//            }
//
//            /* Transpose: data → scratch (transposed) → data */
//            FS_TRANS_TO_SCRATCH:
//            for (uint32_t row = 0; row < N2; row++) {
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                FS_TRANS_LOAD:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    tile[col] = data[dbase + row * N1 + col];
//                }
//                FS_TRANS_WRITE_SCRATCH:
//                for (uint32_t col = 0; col < N1; col++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=4 max=4096
//                    psi_powers[dbase + col * N2 + row] = tile[col];
//                }
//            }
//
//            FS_TRANS_FROM_SCRATCH:
//            for (uint32_t i = 0; i < N; i++) {
//#pragma HLS PIPELINE II=1
//#pragma HLS LOOP_TRIPCOUNT min=8192 max=1048576
//                data[dbase + i] = psi_powers[dbase + i];
//            }
//        }
//    }
//}
