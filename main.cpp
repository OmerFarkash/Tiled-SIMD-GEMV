#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "WeightLoader.hpp"
#include "Strategies.hpp"
#include "ParallelExecutor.hpp"
#include <bits/std_thread.h>

// Utility to verify if two output vectors are identical within a small epsilon
bool verify_results(const std::vector<float>& ref, const std::vector<float>& target, float epsilon = 1e-4f) {
    if (ref.size() != target.size()) return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - target[i]) > epsilon) {
            std::cout << "Mismatch at index " << i << ": Ref=" << ref[i] << ", Target=" << target[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int ROWS = 8192;
    const int COLS = 8192;
    const int NUM_CORES = std::thread::hardware_concurrency();

    std::cout << "--- Matrix-Vector Multiplication Benchmark ---" << std::endl;
    std::cout << "Matrix Size: " << ROWS << "x" << COLS << std::endl;
    std::cout << "CPU Cores Detected: " << NUM_CORES << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    // Allocate aligned memory for DDR simulation
    alignas(32) std::vector<float> matrix(ROWS * COLS);
    alignas(32) std::vector<float> inputs(COLS);
    alignas(32) std::vector<float> output_naive(ROWS, 0.0f);
    alignas(32) std::vector<float> output_simd(ROWS, 0.0f);
    alignas(32) std::vector<float> output_parallel(ROWS, 0.0f);

    // Initialize with dummy data
    std::fill(matrix.begin(), matrix.end(), 0.2f);
    std::fill(inputs.begin(), inputs.end(), 1.5f);

    // Initialize strategies
    NaiveMatrixVectorStrategy naive_strategy;
    SIMDMatrixVectorStrategy simd_strategy;

    // 1. Run Naive (Baseline - 1 Thread)
    double t_naive = ParallelExecutor::run(naive_strategy, 1, ROWS, matrix.data(), 
                                           inputs.data(), output_naive.data(), COLS);
    std::cout << std::left << std::setw(30) << "Naive (1 Thread):" 
              << std::fixed << std::setprecision(2) << t_naive << " ms" << std::endl;

    // 2. Run SIMD (Single-core optimization - 1 Thread)
    double t_simd = ParallelExecutor::run(simd_strategy, 1, ROWS, matrix.data(), 
                                          inputs.data(), output_simd.data(), COLS);
    std::cout << std::left << std::setw(30) << "SIMD (1 Thread):" 
              << t_simd << " ms" 
              << " (Speedup: " << t_naive / t_simd << "x)" << std::endl;

    // 3. Run SIMD + Parallel (Multi-core optimization)
    double t_parallel = ParallelExecutor::run(simd_strategy, NUM_CORES, ROWS, matrix.data(), 
                                              inputs.data(), output_parallel.data(), COLS);
    std::cout << std::left << std::setw(30) << "SIMD (" + std::to_string(NUM_CORES) + " Threads):" 
              << t_parallel << " ms" 
              << " (Total Speedup: " << t_naive / t_parallel << "x)" << std::endl;

    // Verification
    std::cout << "----------------------------------------------" << std::endl;
    bool simd_ok = verify_results(output_naive, output_simd);
    bool parallel_ok = verify_results(output_naive, output_parallel);

    if (simd_ok && parallel_ok) {
        std::cout << "SUCCESS: All results verified!" << std::endl;
    } else {
        std::cout << "FAILURE: Results do not match baseline." << std::endl;
        return -1;
    }

    return 0;
}