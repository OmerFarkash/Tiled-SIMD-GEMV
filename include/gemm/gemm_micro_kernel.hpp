#pragma once
#include <immintrin.h>

/*
 * AVX2 GEMM micro-kernel: accumulates an (MR x NR) output block of C.
 *
 * MR * (NR/8) YMM registers are used as accumulators (pre-zeroed by caller)
 * and stay live across all kk iterations — no C loads/stores inside the loop.
 *
 * Template parameters:
 *   MR  — rows handled simultaneously  (>= 1)
 *   NR  — cols handled simultaneously  (multiple of 8)
 *
 * Parameters:
 *   A_base      — pointer to A[row_i][k], row stride = K_stride
 *   B_row       — pointer to packed b_tile[0][jj], col stride = tile_stride
 *   valid_k     — number of k steps to accumulate
 *   acc[MR][NR/8] — accumulator array, written by caller after this returns
 *
 * Implementation is defined in GEMM_Kernels.cpp (only file that instantiates it).
 */
template<int MR, int NR>
void gemm_micro_kernel(
    const float* __restrict__ A_base,
    int K_stride,
    const float* __restrict__ B_row,
    int tile_stride,
    int valid_k,
    __m256 acc[MR][NR/8]);
