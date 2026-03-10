#include "../../include/gemm/GEMM_Common.hpp"
#include "../../include/gemm/gemm_micro_kernel.hpp"

// 1. Pure Naive (M x K) * (K x N)
void gemm_naive(const Matrix& A, const Matrix& B, Matrix& C) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * B.data[k * N + j];
            }
            C.data[i * N + j] = sum;
        }
    }
}

// 2. Naive Transpose
void gemm_transpose_naive(const Matrix& A, const Matrix& B, Matrix& C) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;

    // BT will be N x K
    Matrix BT(N, K);
    
    // Transpose B (K x N) into BT (N x K)
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            BT.data[j * K + k] = B.data[k * N + j];
        }
    }

    // Multiply A (M x K) with BT (N x K)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A.data[i * K + k] * BT.data[j * K + k];
            }
            C.data[i * N + j] = sum;
        }
    }
}
/* * PHASE 2: Tiled Packed (Scalar / Dot-Product)
 * Logic: Transpose B tile to SRAM for linear access.
 */
void gemm_tiled_packed_dynamic(const Matrix& A, const Matrix& B, Matrix& C, int tile_size) {
    int M = A.rows;
    int K = A.cols;
    int N = B.cols;
    
    // Allocate buffer once
    std::vector<float> b_tile(tile_size * tile_size, 0.0f);

    for (int i = 0; i < M; i += tile_size) {
        int valid_i = std::min(tile_size, M - i);
        
        for (int j = 0; j < N; j += tile_size) {
            int valid_j = std::min(tile_size, N - j);

            for (int k = 0; k < K; k += tile_size) {
                int valid_k = std::min(tile_size, K - k);

                // --- PACKING STEP ---
                // Load chunk of B (valid_k x valid_j) into transposed b_tile
                for (int rr = 0; rr < valid_k; ++rr) {
                    for (int cc = 0; cc < valid_j; ++cc) {
                        b_tile[cc * tile_size + rr] = B.data[(k + rr) * N + (j + cc)];
                    }
                }

                // --- COMPUTE STEP ---
                for (int ii = 0; ii < valid_i; ++ii) {
                    for (int jj = 0; jj < valid_j; ++jj) {
                        float partial_sum = 0;
                        for (int kk = 0; kk < valid_k; ++kk) {
                            partial_sum += A.data[(i + ii) * K + (k + kk)] * b_tile[jj * tile_size + kk];
                        }
                        C.data[(i + ii) * N + (j + jj)] += partial_sum;
                    }
                }
            }
        }
    }
}

// Keep the fixed TILE_SIZE version updated too, exactly like the dynamic one 
// but replacing tile_size with TILE_SIZE macro.
void gemm_tiled_packed(const Matrix& A, const Matrix& B, Matrix& C) {
    gemm_tiled_packed_dynamic(A, B, C, TILE_SIZE);
}

/* * PHASE 3: Tiled SIMD Core Kernel (Vectorized / Outer-Product)
 * The core mathematical implementation for a specific block.
 * Used by the Parallel Executor for multi-threading.
 */
void gemm_tiled_simd_block(const Matrix& A, const Matrix& B, Matrix& C, 
                           int start_row, int end_row, 
                           int start_col, int end_col, 
                           int tile_size) {
    int K = A.cols;
    int N = B.cols;
    
    // Buffer for the B tile (Notice: NO transpose needed for broadcasting!)
    std::vector<float> b_tile(tile_size * tile_size, 0.0f);

    // Iterate only over the assigned rows for this specific block
    for (int i = start_row; i < end_row; i += tile_size) {
        int valid_i = std::min(tile_size, end_row - i);
        
        // Iterate only over the assigned columns for this specific block
        for (int j = start_col; j < end_col; j += tile_size) {
            int valid_j = std::min(tile_size, end_col - j);

            // K dimension is shared and always fully traversed (0 to K)
            for (int k = 0; k < K; k += tile_size) {
                int valid_k = std::min(tile_size, K - k);

                // --- PACKING STEP (Row-Major, no transpose) ---
                for (int rr = 0; rr < valid_k; ++rr) {
                    for (int cc = 0; cc < valid_j; ++cc) {
                        b_tile[rr * tile_size + cc] = B.data[(k + rr) * N + (j + cc)];
                    }
                }

                // --- SIMD COMPUTE STEP ---
                for (int ii = 0; ii < valid_i; ++ii) {
                    // Process 8 columns of C simultaneously
                    for (int jj = 0; jj < valid_j; jj += 8) {
                        
                        // Check if we have a full 8-element block for SIMD
                        if (jj + 8 <= valid_j) {
                            // Load 8 elements from C
                            __m256 c_vec = _mm256_loadu_ps(&C.data[(i + ii) * N + (j + jj)]);

                            for (int kk = 0; kk < valid_k; ++kk) {
                                // Broadcast 1 element from A to all 8 slots
                                __m256 a_val = _mm256_set1_ps(A.data[(i + ii) * K + (k + kk)]);
                                
                                // Load 8 elements from the packed B tile
                                __m256 b_vec = _mm256_loadu_ps(&b_tile[kk * tile_size + jj]);
                                
                                // FMA: c_vec = a_val * b_vec + c_vec
                                c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);
                            }
                            
                            // Store the updated 8 elements back to C
                            _mm256_storeu_ps(&C.data[(i + ii) * N + (j + jj)], c_vec);
                        } 
                        else {
                            // --- SCALAR TAIL HANDLING (Fallback for remaining < 8 elements) ---
                            for (int rem_j = jj; rem_j < valid_j; ++rem_j) {
                                float partial_sum = 0;
                                for (int kk = 0; kk < valid_k; ++kk) {
                                    partial_sum += A.data[(i + ii) * K + (k + kk)] * b_tile[kk * tile_size + rem_j];
                                }
                                C.data[(i + ii) * N + (j + rem_j)] += partial_sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

/* Single-threaded wrapper — calls the block kernel over the full matrix dimensions. */
void gemm_tiled_simd(const Matrix& A, const Matrix& B, Matrix& C, int tile_size) {
    gemm_tiled_simd_block(A, B, C, 0, A.rows, 0, B.cols, tile_size);
}

/*
 * Register-tile block kernel.
 *
 * Adds a third tiling level inside the cache tile:
 *
 *   Thread macro-tile  (ParallelExecutor 2D grid)
 *     Cache tile       (tile_size x tile_size)
 *       Register tile  (MR x NR)  <-- this kernel
 *
 * MR rows x NR cols of C are computed simultaneously keeping MR*(NR/8)
 * __m256 accumulators live across the entire k-loop, maximising register-file
 * utilisation and eliminating redundant loads/stores on C.
 * A compile-time dispatch calls the templated micro-kernel so the compiler
 * can fully unroll and pin all accumulators to named YMM registers.
 */

// Template definition (only instantiated from this file via dispatch_micro_tile)
template<int MR, int NR>
void gemm_micro_kernel(
    const float* __restrict__ A_base,
    int K_stride,
    const float* __restrict__ B_row,
    int tile_stride,
    int valid_k,
    __m256 acc[MR][NR/8])
{
    static_assert(NR % 8 == 0, "NR must be a multiple of 8 (AVX2 float lane width)");

    for (int kk = 0; kk < valid_k; ++kk) {
        const float* b_ptr = B_row + kk * tile_stride;

        for (int mr = 0; mr < MR; ++mr) {
            __m256 a_broadcast = _mm256_set1_ps(A_base[mr * K_stride + kk]);

            for (int nr = 0; nr < NR/8; ++nr) {
                __m256 b_vec = _mm256_loadu_ps(b_ptr + nr * 8);
                acc[mr][nr] = _mm256_fmadd_ps(a_broadcast, b_vec, acc[mr][nr]);
            }
        }
    }
}

// Helper: one (MR, NR) body — accumulates into local acc[], then stores to C.
// Handles full MR×NR tiles. Tail rows/cols are handled by the caller.
// jj_local: column offset of this MR×NR subtile within the packed b_tile.
template<int MR, int NR>
static void process_mr_nr_tile(
    const Matrix& A, const Matrix& B, Matrix& C,
    const std::vector<float>& b_tile,
    int i, int j, int k, int jj_local,
    int valid_i, int valid_j, int valid_k,
    int tile_size)
{
    int K = A.cols;
    int N = B.cols;

    // Accumulator registers — pre-zeroed, live across all kk
    __m256 acc[MR][NR/8];
    for (int mr = 0; mr < MR; ++mr)
        for (int nr = 0; nr < NR/8; ++nr)
            acc[mr][nr] = _mm256_setzero_ps();

    // Base pointer into A — row i, k-panel starting at column k
    const float* A_base = &A.data[(i) * K + k];

    // B_row points to the start of this jj subtile within the packed b_tile
    const float* B_row = b_tile.data() + jj_local;

    // Run micro-kernel over all kk — accumulates without touching C
    gemm_micro_kernel<MR, NR>(
        A_base, K,
        B_row, tile_size,
        valid_k, acc);

    // Store accumulators back to C (add — C may already have partial sums
    // from previous cache-tile iterations over k)
    for (int mr = 0; mr < MR; ++mr) {
        if (mr >= valid_i) break;
        for (int nr = 0; nr < NR/8; ++nr) {
            int col_base = j + nr * 8;
            if (col_base + 8 <= j + valid_j) {
                // Full SIMD store
                __m256 existing = _mm256_loadu_ps(&C.data[(i + mr) * N + col_base]);
                _mm256_storeu_ps(&C.data[(i + mr) * N + col_base],
                                 _mm256_add_ps(existing, acc[mr][nr]));
            } else {
                // Scalar tail for partial NR group
                alignas(32) float tmp[8];
                _mm256_store_ps(tmp, acc[mr][nr]);
                for (int rem = 0; col_base + rem < j + valid_j; ++rem)
                    C.data[(i + mr) * N + col_base + rem] += tmp[rem];
            }
        }
    }
}

// All registered tile kernels share this signature (all args are runtime values).
using TileFn = void(*)(const Matrix&, const Matrix&, Matrix&,
                        const std::vector<float>&,
                        int, int, int, int,    // i, j, k, jj_local
                        int, int, int, int);   // valid_i, valid_j, valid_k, tile_size

// Returns the templated kernel for the given (mr, nr), or nullptr for unknown configs.
static TileFn find_tile_fn(int mr, int nr) {
    static const std::map<std::pair<int,int>, TileFn> table {
        {{1,  8},  process_mr_nr_tile<1,  8>},
        {{2,  8},  process_mr_nr_tile<2,  8>},
        {{4,  8},  process_mr_nr_tile<4,  8>},
        {{4, 16},  process_mr_nr_tile<4, 16>},
        {{6, 16},  process_mr_nr_tile<6, 16>},
        {{8,  8},  process_mr_nr_tile<8,  8>},
        {{4, 32},  process_mr_nr_tile<4, 32>},
        {{6, 32},  process_mr_nr_tile<6, 32>},
    };
    auto it = table.find({mr, nr});
    return it != table.end() ? it->second : nullptr;
}

// Dispatches to the registered templated kernel, or falls back to a generic SIMD loop.
static void dispatch_micro_tile(
    const Matrix& A, const Matrix& B, Matrix& C,
    const std::vector<float>& b_tile,
    int i, int j, int k, int jj_local,
    int valid_i, int valid_j, int valid_k,
    int tile_size, int mr, int nr)
{
    if (TileFn fn = find_tile_fn(mr, nr)) {
        fn(A, B, C, b_tile, i, j, k, jj_local, valid_i, valid_j, valid_k, tile_size);
        return;
    }

    // Generic fallback for any unregistered (mr, nr) — not fully unrolled by the compiler
    for (int ii = 0; ii < valid_i; ++ii) {
        for (int jj = 0; jj < valid_j; jj += 8) {
            int avail = std::min(8, valid_j - jj);
            if (avail == 8) {
                __m256 c_vec = _mm256_setzero_ps();
                for (int kk = 0; kk < valid_k; ++kk) {
                    __m256 a_val = _mm256_set1_ps(A.data[(i+ii)*A.cols + (k+kk)]);
                    __m256 b_vec = _mm256_loadu_ps(&b_tile[kk * tile_size + jj_local + jj]);
                    c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);
                }
                __m256 existing = _mm256_loadu_ps(&C.data[(i+ii)*B.cols + j+jj]);
                _mm256_storeu_ps(&C.data[(i+ii)*B.cols + j+jj], _mm256_add_ps(existing, c_vec));
            } else {
                for (int rem = 0; rem < avail; ++rem) {
                    float sum = 0;
                    for (int kk = 0; kk < valid_k; ++kk)
                        sum += A.data[(i+ii)*A.cols + (k+kk)] * b_tile[kk * tile_size + jj_local + jj + rem];
                    C.data[(i+ii)*B.cols + j+jj+rem] += sum;
                }
            }
        }
    }
}

void gemm_tiled_simd_block_mr_nr(const Matrix& A, const Matrix& B, Matrix& C,
                                  int start_row, int end_row,
                                  int start_col, int end_col,
                                  int tile_size, int mr, int nr)
{
    int K = A.cols;
    int N = B.cols;

    // Shared B tile buffer (same packing as Phase 3 — row-major, no transpose)
    std::vector<float> b_tile(tile_size * tile_size, 0.0f);

    // Outer loops: cache-tile strides over the assigned block
    for (int i = start_row; i < end_row; i += tile_size) {
        int valid_i_tile = std::min(tile_size, end_row - i);

        for (int j = start_col; j < end_col; j += tile_size) {
            int valid_j_tile = std::min(tile_size, end_col - j);

            for (int k = 0; k < K; k += tile_size) {
                int valid_k = std::min(tile_size, K - k);

                // --- PACK B (row-major, same as Phase 3) ---
                for (int rr = 0; rr < valid_k; ++rr)
                    for (int cc = 0; cc < valid_j_tile; ++cc)
                        b_tile[rr * tile_size + cc] = B.data[(k + rr) * N + (j + cc)];

                // --- REGISTER TILE LOOP (MR × NR strides) ---
                for (int ii = 0; ii < valid_i_tile; ii += mr) {
                    int valid_i = std::min(mr, valid_i_tile - ii);

                    for (int jj = 0; jj < valid_j_tile; jj += nr) {
                        int valid_j = std::min(nr, valid_j_tile - jj);

                        // jj is the LOCAL offset within the packed b_tile
                        // j+jj is the GLOBAL column of this subtile in C
                        dispatch_micro_tile(A, B, C, b_tile,
                                            i + ii, j + jj, k, jj,
                                            valid_i, valid_j, valid_k,
                                            tile_size, mr, nr);
                    }
                }
            }
        }
    }
}