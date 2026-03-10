#ifndef GEMM_COMMON_HPP
#define GEMM_COMMON_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <immintrin.h> // Required for AVX2 intrinsics
#include <map>

// Define default tile size
// Note: We used 32 in our recent SIMD tests, but 64 is also fine to test!
const int TILE_SIZE = 64;

struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;

    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}
};

// ==========================================
// GEMM Kernel Declarations
// ==========================================

void gemm_tiled_simd_block(const Matrix& A, const Matrix& B, Matrix& C, 
                           int start_row, int end_row, 
                           int start_col, int end_col, 
                           int tile_size);

void gemm_tiled_simd_block_mr_nr(const Matrix& A, const Matrix& B, Matrix& C,
                                  int start_row, int end_row,
                                  int start_col, int end_col,
                                  int tile_size, int mr, int nr);

void gemm_tiled_simd(const Matrix& A, const Matrix& B, Matrix& C, int tile_size);
void gemm_naive(const Matrix& A, const Matrix& B, Matrix& C);
void gemm_tiled_packed_dynamic(const Matrix& A, const Matrix& B, Matrix& C, int tile_size);

#endif // GEMM_COMMON_HPP