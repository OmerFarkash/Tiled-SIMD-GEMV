#include "../../include/gemm/SimdGemmStrategy.hpp"

// Forward-declare the new kernel from GEMM_Kernels.cpp
void gemm_tiled_simd_block_mr_nr(const Matrix& A, const Matrix& B, Matrix& C,
                                  int start_row, int end_row,
                                  int start_col, int end_col,
                                  int tile_size, int mr, int nr);

RegisterTileGemmStrategy::RegisterTileGemmStrategy(int tile_size_, int mr_, int nr_)
    : tile_size(tile_size_), mr(mr_), nr(nr_) {}

void RegisterTileGemmStrategy::execute(int start_row, int end_row,
                                       int start_col, int end_col,
                                       const Matrix& A, const Matrix& B, Matrix& C) {
    gemm_tiled_simd_block_mr_nr(A, B, C, start_row, end_row, start_col, end_col,
                                 tile_size, mr, nr);
}
