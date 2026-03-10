#include "../../include/core/ParallelExecutor.hpp"

// ---------------------------------------------------------
// GEMV Implementation
// ---------------------------------------------------------
double ParallelExecutor::run(ComputeStrategy& strategy, int num_threads, int total_rows, 
                            float* matrix, float* inputs, float* output, int full_width) {
    
    auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    std::vector<WeightLoader> loaders(num_threads); 
    
    int rows_per_thread = total_rows / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int s_row = i * rows_per_thread;
        int e_row = (i == num_threads - 1) ? total_rows : s_row + rows_per_thread;
        
        threads.emplace_back(&ComputeStrategy::execute, &strategy, 
                             s_row, e_row, matrix, inputs, output, 
                             full_width, std::ref(loaders[i]));
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

// ---------------------------------------------------------
// GEMM Implementation (New 2D Grid Logic)
// ---------------------------------------------------------
namespace {
    // Helper function to find the most square-like 2D grid for a given number of threads
    void compute_optimal_grid(int num_threads, int& grid_rows, int& grid_cols) {
        grid_rows = 1;
        grid_cols = num_threads;
        
        for (int r = std::sqrt(num_threads); r > 0; --r) {
            if (num_threads % r == 0) {
                grid_rows = r;
                grid_cols = num_threads / r;
                break;
            }
        }
    }
}

// GEMM Implementation with Matrix references
double ParallelExecutor::run(GEMM_Strategy& strategy, int num_threads, 
                             const Matrix& A, const Matrix& B, Matrix& C) {
    
    auto start = std::chrono::steady_clock::now();
    
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Extract dimensions from the Matrix object
    int M = C.rows;
    int N = C.cols;

    int grid_rows, grid_cols;
    compute_optimal_grid(num_threads, grid_rows, grid_cols);

    int m_per_thread = M / grid_rows;
    int n_per_thread = N / grid_cols;

    for (int r = 0; r < grid_rows; ++r) {
        for (int c = 0; c < grid_cols; ++c) {
            
            int start_row = r * m_per_thread;
            int end_row = (r == grid_rows - 1) ? M : start_row + m_per_thread;
            
            int start_col = c * n_per_thread;
            int end_col = (c == grid_cols - 1) ? N : start_col + n_per_thread;

            // Pass Matrix objects via reference wrappers to avoid copying and respect thread safety
            threads.emplace_back(&GEMM_Strategy::execute, &strategy, 
                                 start_row, end_row, start_col, end_col, 
                                 std::cref(A), std::cref(B), std::ref(C));
        }
    }

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    return elapsed.count();
}