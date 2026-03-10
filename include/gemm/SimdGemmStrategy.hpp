#pragma once
#include "GEMM_Strategy.hpp"

// SIMD tiled GEMM using AVX2 outer-product broadcasting
class SimdGemmStrategy : public GEMM_Strategy {
private:
    int tile_size;

public:
    explicit SimdGemmStrategy(int tile_size_ = 32);

    void execute(int start_row, int end_row,
                 int start_col, int end_col,
                 const Matrix& A, const Matrix& B, Matrix& C) override final;
};

// GEMM with an additional register-level micro-tile (MR rows x NR cols) inside the cache tile.
// MR * (NR/8) YMM registers are kept live as accumulators across the full k-loop,
// maximising register-file utilisation and reducing redundant loads/stores on C.
// NR must be a multiple of 8 (AVX2 float lane width).
class RegisterTileGemmStrategy : public GEMM_Strategy {
private:
    int tile_size;
    int mr;
    int nr;

public:
    // tile_size : cache-level tile size  (default 32)
    // mr        : register-tile rows     (default 4)
    // nr        : register-tile cols     (default 16, must be multiple of 8)
    explicit RegisterTileGemmStrategy(int tile_size = 32, int mr = 4, int nr = 16);

    void execute(int start_row, int end_row,
                 int start_col, int end_col,
                 const Matrix& A, const Matrix& B, Matrix& C) override final;
};