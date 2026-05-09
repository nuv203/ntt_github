#ifndef NTT_KERNEL_HPP
#define NTT_KERNEL_HPP

#include <stdint.h>

/*==========================================================================
 * Negacyclic NTT — Zynq UltraScale+ (xck26, -2LV-c)
 *
 * N <= TILE_N : direct on-chip NTT
 * N >  TILE_N : four-step via DDR
 *
 * Four-step algorithm (N = N1 * N2):
 *   1. Psi twist (data[i] *= psi^i)
 *   2. Column NTTs (N2-point, omega_2 = omega^N1)
 *   3. Inter-stage twiddle + Row NTTs (N1-point, omega_1 = omega^N2)
 *   4. Transpose via scratch buffer (psi_powers reused after consumption)
 *
 * Twiddle packing for four-step:
 *   twiddles[0..N2-1]             N2-pt Stockham twiddles (col NTTs)
 *   twiddles[N2..N2+N1-1]         N1-pt Stockham twiddles (row NTTs)
 *   twiddles[N2+N1..N2+N1+N-1]    inter-stage omega^(col*row)
 *
 * Note: psi_powers is non-const because it is reused as DDR scratch
 * space for the transpose in the four-step path (after psi values
 * have been consumed in the twist phase).
 *
 * Target: 10ns (100MHz), xck26-sfvc784-2LV-c
 *==========================================================================*/

#define TILE_N     4096
#define LOG_TILE_N 12
#define MAX_LOG_N  20
#define MAX_BATCH  16

typedef uint32_t ntt_t;

void ntt_kernel(
    ntt_t       *data,
    ntt_t       *psi_powers,    /* non-const: reused as scratch for transpose */
    const ntt_t *twiddles,
    ntt_t        q,
    uint32_t     batch,
    uint32_t     N,
    uint32_t     logN
);

#endif