#pragma once
#include <vector>
#include "ComputeStrategy.hpp"
#include "WeightLoader.hpp"

class ParallelExecutor {
public:
    // returning run time in ms
    static double run(ComputeStrategy& strategy, int num_threads, int total_rows, 
                    float* matrix, float* inputs, float* output, int full_width);
};