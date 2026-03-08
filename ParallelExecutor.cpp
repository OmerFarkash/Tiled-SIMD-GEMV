#include "ParallelExecutor.hpp"
#include <thread>
#include <chrono>
#include <vector>

double ParallelExecutor::run(ComputeStrategy& strategy, int num_threads, int total_rows, 
                            float* matrix, float* inputs, float* output, int full_width) {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    
    std::vector<WeightLoader> loaders(num_threads); 
    
    int rows_per_thread = total_rows / num_threads;

    for (int i = 0; i < num_threads; ++i) {
        int s_row = i * rows_per_thread;
        // handle left overs if need
        int e_row = (i == num_threads - 1) ? total_rows : s_row + rows_per_thread;
        // create thread
        threads.emplace_back(&ComputeStrategy::execute, &strategy, 
                            s_row, e_row, matrix, inputs, output, 
                            full_width, std::ref(loaders[i]));
    }

    // wait for all threads to finish
    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    return elapsed.count();
}