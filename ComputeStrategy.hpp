#pragma once
#include "WeightLoader.hpp"

class ComputeStrategy {
public:
    virtual void execute(int start_row, int end_row, float* matrix, float* inputs, 
                        float* output, int full_width, WeightLoader& loader) = 0;
    virtual ~ComputeStrategy() = default;
};