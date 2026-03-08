#pragma once
#include "ComputeStrategy.hpp"

class SIMDMatrixVectorStrategy : public ComputeStrategy {
public:
    void execute(int start_row, int end_row, float* matrix, float* inputs, 
                 float* output, int full_width, WeightLoader& loader) override;
};

class NaiveMatrixVectorStrategy : public ComputeStrategy {
public:
    void execute(int start_row, int end_row, float* matrix, float* inputs, 
                 float* output, int full_width, WeightLoader& loader) override;
};