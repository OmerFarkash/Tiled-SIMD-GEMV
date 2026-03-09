#pragma once
#include "GEMM_Strategy.hpp"

// Concrete strategy implementing SIMD + Broadcasting (Outer Product)
class SimdGemmStrategy : public GEMM_Strategy {
private:
    int tile_size;

public:
    explicit SimdGemmStrategy(int tile_size_ = 32);

    void execute(int start_row, int end_row, 
                 int start_col, int end_col, 
                 const Matrix& A, const Matrix& B, Matrix& C) override final;
};