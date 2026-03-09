#include "../../include/gemm/GEMM_Common.hpp"

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

// 3. Tiled Packed Dynamic (The Robust Version)
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