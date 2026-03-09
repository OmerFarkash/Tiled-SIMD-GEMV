#include "../include/gemm/GEMM_Common.hpp"
#include <iomanip>
#include <random>
#include <vector>
#include <string>

// Function declarations
void gemm_naive(const Matrix& A, const Matrix& B, Matrix& C);
void gemm_transpose_naive(const Matrix& A, const Matrix& B, Matrix& C);
void gemm_tiled_packed(const Matrix& A, const Matrix& B, Matrix& C);
void gemm_tiled_packed_dynamic(const Matrix& A, const Matrix& B, Matrix& C, int tile_size);

bool verify(const Matrix& C1, const Matrix& C2) {
    for (size_t i = 0; i < C1.data.size(); ++i) {
        if (std::abs(C1.data[i] - C2.data[i]) > 1e-3) return false;
    }
    return true;
}

int main() {
    // Deep Learning Asymmetric Profile with unaligned tails
    const int M = 250;   // e.g., Batch size
    const int K = 1000;  // e.g., Input features
    const int N = 4000;  // e.g., Output features

    std::cout << "\n======================================================\n";
    std::cout << "      Phase 2: GEMM Robustness & Cache Benchmark      \n";
    std::cout << "      Profile: Deep Learning FC Layer (Unaligned)     \n";
    std::cout << "      Dimensions: M=" << M << ", K=" << K << ", N=" << N << "   \n";
    std::cout << "======================================================\n\n";

    // Setup matrices with correct dimensions
    Matrix A(M, K), B(K, N), C_naive(M, N), C_trans(M, N), C_packed(M, N);

    const uint32_t seed = 42;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Initialize A (M x K)
    for (int i = 0; i < M * K; ++i) A.data[i] = dist(gen);
    
    // Initialize B (K x N)
    for (int i = 0; i < K * N; ++i) B.data[i] = dist(gen);

    std::cout << "--- Part 1: Algorithmic Comparison ---\n";
    std::cout << std::left << std::setw(25) << "Algorithm" 
              << std::setw(15) << "Time (ms)" 
              << "Speedup vs Naive" << std::endl;
    std::cout << "------------------------------------------------------\n";

    // 1. Benchmark Naive (Baseline)
    auto start = std::chrono::high_resolution_clock::now();
    gemm_naive(A, B, C_naive);
    auto end = std::chrono::high_resolution_clock::now();
    double t_naive = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << std::left << std::setw(25) << "Naive GEMM" 
              << std::fixed << std::setprecision(2) << std::setw(15) << t_naive 
              << "1.00x (Baseline)" << std::endl;

    // 2. Benchmark Naive Transpose
    start = std::chrono::high_resolution_clock::now();
    gemm_transpose_naive(A, B, C_trans);
    end = std::chrono::high_resolution_clock::now();
    double t_trans = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << std::left << std::setw(25) << "Naive Transpose" 
              << std::setw(15) << t_trans 
              << std::fixed << std::setprecision(2) << (t_naive / t_trans) << "x" << std::endl;

    // 3. Benchmark Tiled Packed (Fixed TILE_SIZE from HPP)
    start = std::chrono::high_resolution_clock::now();
    gemm_tiled_packed(A, B, C_packed);
    end = std::chrono::high_resolution_clock::now();
    double t_packed = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Create a dynamic name string that includes the TILE_SIZE macro
    std::string packed_name = "Tiled Packed (" + std::to_string(TILE_SIZE) + ")";
    std::cout << std::left << std::setw(25) << packed_name 
              << std::setw(15) << t_packed 
              << std::fixed << std::setprecision(2) << (t_naive / t_packed) << "x" << std::endl;

    std::cout << "\nValidation: ";
    if (verify(C_naive, C_trans) && verify(C_naive, C_packed)) {
        std::cout << "[SUCCESS] All kernels produce identical results.\n\n";
    } else {
        std::cout << "[ERROR] Discrepancy found in results!\n\n";
        return 1;
    }

    std::cout << "--- Part 2: Cache Sweet-Spot Analysis (Tiled Packed) ---\n";
    std::cout << std::left << std::setw(15) << "Tile Size" 
              << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Speedup"
              << "Status" << std::endl;
    std::cout << "------------------------------------------------------\n";

    std::vector<int> tile_sizes = {16, 32, 64, 128, 256, 512};
    
    for (int ts : tile_sizes) {
        Matrix C_test(N, N);
        
        start = std::chrono::high_resolution_clock::now();
        gemm_tiled_packed_dynamic(A, B, C_test, ts);
        end = std::chrono::high_resolution_clock::now();
        
        double t_dynamic = std::chrono::duration<double, std::milli>(end - start).count();
        double current_speedup = t_naive / t_dynamic;
        
        std::cout << std::left << std::setw(15) << ts 
                  << std::setw(15) << std::fixed << std::setprecision(2) << t_dynamic 
                  << std::setw(15) << std::fixed << std::setprecision(2) << current_speedup
                  << (verify(C_naive, C_test) ? "[Valid]" : "[Invalid]") << std::endl;
    }
    
    std::cout << "======================================================\n";

    return 0;
}