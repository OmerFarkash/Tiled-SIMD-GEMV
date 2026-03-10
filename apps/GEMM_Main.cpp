#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <thread>
#include <cmath>
#include <string>
#include <algorithm>

#include "../include/core/ParallelExecutor.hpp"
#include "../include/gemm/GEMM_Common.hpp"
#include "../include/gemm/SimdGemmStrategy.hpp"

// Kernel declarations from GEMM_Kernels.cpp
void gemm_naive            (const Matrix& A, const Matrix& B, Matrix& C);
void gemm_transpose_naive  (const Matrix& A, const Matrix& B, Matrix& C);
void gemm_tiled_packed     (const Matrix& A, const Matrix& B, Matrix& C);
void gemm_tiled_packed_dynamic(const Matrix& A, const Matrix& B, Matrix& C, int tile_size);
void gemm_tiled_simd       (const Matrix& A, const Matrix& B, Matrix& C, int tile_size);

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// Returns true if every element of C2 is within 1e-3 of C1 (the naive reference).
bool verify(const Matrix& C1, const Matrix& C2) {
    if (C1.data.size() != C2.data.size()) return false;
    for (size_t i = 0; i < C1.data.size(); ++i)
        if (std::abs(C1.data[i] - C2.data[i]) > 1e-3f) return false;
    return true;
}

// Column widths shared by all parts for a consistent look.
static constexpr int W_NAME    = 34;
static constexpr int W_TIME    = 14;
static constexpr int W_SPEED   = 18;
static constexpr int W_SPEED2  = 18;
static constexpr int W_STATUS  =  9;
static const std::string SEP   = std::string(W_NAME + W_TIME + W_SPEED + W_STATUS, '-');

static constexpr int W_SMALL   = 12;
static const std::string SEP2  = std::string(W_SMALL + W_TIME + W_SPEED + W_STATUS, '-');
static const std::string LSEP2  = std::string(W_SMALL + W_TIME + W_SPEED + W_SPEED2 + W_STATUS, '-');
static const std::string LSEP4  = std::string(8 + 8 + 8 + W_TIME + W_SPEED + W_SPEED + W_STATUS, '-');

const std::string ANSI_GREEN = "\033[32m";
const std::string ANSI_RED   = "\033[31m";
const std::string ANSI_RESET = "\033[0m";

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main() {
    const int M = 250;    // batch size / output rows
    const int K = 1000;   // shared inner dimension
    const int N = 4000;   // output cols

    std::cout << "\n================================================================\n"
              << "  GEMM Full System Benchmark — Deep Learning FC Layer Profile\n"
              << "  Dimensions: M=" << M << ", K=" << K << ", N=" << N << "\n"
              << "================================================================\n\n";

    Matrix A(M, K), B(K, N);
    Matrix C_naive(M, N), C_trans(M, N), C_packed(M, N), C_simd(M, N);

    // Deterministic, reproducible input
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : A.data) v = dist(gen);
    for (auto& v : B.data) v = dist(gen);

    auto bench = [](auto fn) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    };

    // -----------------------------------------------------------------------
    // Part 1: Single-thread algorithmic comparison
    // -----------------------------------------------------------------------
    std::cout << "--- Part 1: Algorithmic Comparison (Single Thread) ---\n"
              << std::left
              << std::setw(W_NAME)   << "Algorithm"
              << std::setw(W_TIME)   << "Time (ms)"
              << std::setw(W_SPEED)  << "Speedup vs Naive"
              << "Status\n"
              << SEP << "\n";

    struct Result1 { std::string name; double t; std::string s1; bool v; };
    std::vector<Result1> res1;

    double t_naive = bench([&]{ gemm_naive(A, B, C_naive); });
    res1.push_back({"Naive", t_naive, "1.00x (Baseline)", true});

    double t_trans = bench([&]{ gemm_transpose_naive(A, B, C_trans); });
    res1.push_back({"Naive Transpose", t_trans, std::to_string(t_naive / t_trans) + "x", verify(C_naive, C_trans)});

    double t_packed = bench([&]{ gemm_tiled_packed(A, B, C_packed); });
    res1.push_back({"Tiled Packed Transpose (" + std::to_string(TILE_SIZE) + ")", t_packed, std::to_string(t_naive / t_packed) + "x", verify(C_naive, C_packed)});

    double t_simd = bench([&]{ gemm_tiled_simd(A, B, C_simd, 32); });
    res1.push_back({"Tiled SIMD (tile=32)", t_simd, std::to_string(t_naive / t_simd) + "x", verify(C_naive, C_simd)});

    double min1 = 1e18, max1 = -1.0;
    for (const auto& r : res1) if (r.v) { min1 = std::min(min1, r.t); max1 = std::max(max1, r.t); }
    for (const auto& r : res1) {
        std::string col = (r.v && r.t == min1) ? ANSI_GREEN : (r.v && r.t == max1) ? ANSI_RED : "";
        std::cout << col << std::left << std::fixed << std::setprecision(2)
                  << std::setw(W_NAME) << r.name << std::setw(W_TIME) << r.t
                  << std::setw(W_SPEED) << r.s1 << (r.v ? "[Valid]" : "[Invalid]") << ANSI_RESET << "\n";
    }
    std::cout << "\n";

    // -----------------------------------------------------------------------
    // Part 2: Cache tile sweet-spot (tiled packed, single thread)
    // -----------------------------------------------------------------------
    std::cout << "--- Part 2: Cache Tile Sweet-Spot (Tiled Packed, Single Thread) ---\n"
              << std::left
              << std::setw(W_SMALL) << "Tile Size"
              << std::setw(W_TIME)  << "Time (ms)"
              << std::setw(W_SPEED) << "Speedup vs Naive"
              << "Status\n"
              << SEP2 << "\n";

    struct Result2 { int ts; double t; std::string s1; bool v; };
    std::vector<Result2> res2;
    for (int ts : {16, 32, 64, 128, 256, 512}) {
        Matrix C(M, N);
        double t = bench([&]{ gemm_tiled_packed_dynamic(A, B, C, ts); });
        res2.push_back({ts, t, std::to_string(t_naive / t) + "x", verify(C_naive, C)});
    }
    double min2 = 1e18, max2 = -1.0;
    for (const auto& r : res2) if (r.v) { min2 = std::min(min2, r.t); max2 = std::max(max2, r.t); }
    for (const auto& r : res2) {
        std::string col = (r.v && r.t == min2) ? ANSI_GREEN : (r.v && r.t == max2) ? ANSI_RED : "";
        std::cout << col << std::left << std::fixed << std::setprecision(2)
                  << std::setw(W_SMALL) << r.ts << std::setw(W_TIME) << r.t
                  << std::setw(W_SPEED) << r.s1 << (r.v ? "[Valid]" : "[Invalid]") << ANSI_RESET << "\n";
    }

    // -----------------------------------------------------------------------
    // Part 3: Multi-thread scaling — SIMD 2D grid
    // -----------------------------------------------------------------------
    std::cout << "\n--- Part 3: Multi-Threading Scaling (SIMD 2D Grid, tile=32) ---\n"
              << std::left
              << std::setw(W_SMALL)  << "Threads"
              << std::setw(W_TIME)   << "Time (ms)"
              << std::setw(W_SPEED)  << "Speedup vs Naive"
              << std::setw(W_SPEED2) << "Scaling vs 1-Thr"
              << "Status\n"
              << LSEP2 << "\n";

    int max_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<int> thread_counts = {1, 2, 4, 8};
    if (std::find(thread_counts.begin(), thread_counts.end(), max_threads) == thread_counts.end())
        thread_counts.push_back(max_threads);
    std::sort(thread_counts.begin(), thread_counts.end());

    SimdGemmStrategy simd_strat(32);
    double t_p3_1thr = 0;
    
    struct Result3 { int threads; double t; std::string s1, s2; bool v; };
    std::vector<Result3> res3;
    for (int threads : thread_counts) {
        Matrix C(M, N);
        double t = ParallelExecutor::run(simd_strat, threads, A, B, C);
        if (threads == 1) t_p3_1thr = t;
        double scale = (threads == 1) ? 1.0 : (t_p3_1thr / t);
        res3.push_back({threads, t, std::to_string(t_naive / t) + "x", std::to_string(scale) + "x", verify(C_naive, C)});
    }
    double min3 = 1e18, max3 = -1.0;
    for (const auto& r : res3) if (r.v) { min3 = std::min(min3, r.t); max3 = std::max(max3, r.t); }
    for (const auto& r : res3) {
        std::string col = (r.v && r.t == min3) ? ANSI_GREEN : (r.v && r.t == max3) ? ANSI_RED : "";
        std::cout << col << std::left << std::fixed << std::setprecision(2)
                  << std::setw(W_SMALL) << r.threads << std::setw(W_TIME) << r.t
                  << std::setw(W_SPEED) << r.s1 << std::setw(W_SPEED2) << r.s2 
                  << (r.v ? "[Valid]" : "[Invalid]") << ANSI_RESET << "\n";
    }

    // -----------------------------------------------------------------------
    // Part 4: Register-tile sweet-spot — MR × NR sweep (single thread)
    // -----------------------------------------------------------------------
    std::cout << "\n--- Part 4: Register-Tile Sweet-Spot (MR x NR Sweep, Single Thread) ---\n"
              << std::left
              << std::setw(8)       << "MR"
              << std::setw(8)       << "NR"
              << std::setw(8)       << "YMM"
              << std::setw(W_TIME)  << "Time (ms)"
              << std::setw(W_SPEED) << "vs Naive"
              << std::setw(W_SPEED) << "vs SIMD-P3"
              << "Status\n"
              << LSEP4 << "\n";

    struct Cfg { int mr, nr; };
    std::vector<Cfg> configs = {
        {1,  8}, {2,  8}, {4,  8}, {4, 16},
        {6, 16}, {8,  8}, {4, 32}, {6, 32},
    };

    struct Result4 { int mr, nr, ymm; double t; std::string s1, s2; bool v; };
    std::vector<Result4> res4;

    int    best_mr = 4, best_nr = 16;
    double best_t  = 1e18;

    for (auto& cfg : configs) {
        Matrix C(M, N);
        RegisterTileGemmStrategy strat(32, cfg.mr, cfg.nr);
        double t = ParallelExecutor::run(strat, 1, A, B, C);
        bool   ok = verify(C_naive, C);
        res4.push_back({cfg.mr, cfg.nr, cfg.mr * cfg.nr / 8, t, std::to_string(t_naive / t) + "x", std::to_string(t_simd / t) + "x", ok});
        if (ok && t < best_t) { best_t = t; best_mr = cfg.mr; best_nr = cfg.nr; }
    }
    
    double min4 = 1e18, max4 = -1.0;
    for (const auto& r : res4) if (r.v) { min4 = std::min(min4, r.t); max4 = std::max(max4, r.t); }
    for (const auto& r : res4) {
        std::string col = (r.v && r.t == min4) ? ANSI_GREEN : (r.v && r.t == max4) ? ANSI_RED : "";
        std::cout << col << std::left << std::fixed << std::setprecision(2)
                  << std::setw(8) << r.mr << std::setw(8) << r.nr << std::setw(8) << r.ymm
                  << std::setw(W_TIME) << r.t << std::setw(W_SPEED) << r.s1 << std::setw(W_SPEED) << r.s2
                  << (r.v ? "[Valid]" : "[Invalid]") << ANSI_RESET << "\n";
    }
    std::cout << "\n  >> Optimal: MR=" << best_mr << ", NR=" << best_nr
              << " (" << (best_mr * best_nr / 8) << " YMM accumulators)\n";

    // -----------------------------------------------------------------------
    // Part 5: Multi-thread scaling — optimal register tile
    // -----------------------------------------------------------------------
    std::cout << "\n--- Part 5: Multi-Threading Scaling (Register Tile MR="
              << best_mr << ", NR=" << best_nr << ") ---\n"
              << std::left
              << std::setw(W_SMALL)  << "Threads"
              << std::setw(W_TIME)   << "Time (ms)"
              << std::setw(W_SPEED)  << "Speedup vs Naive"
              << std::setw(W_SPEED2) << "Scaling vs 1-Thr"
              << "Status\n"
              << LSEP2 << "\n";

    RegisterTileGemmStrategy opt_strat(32, best_mr, best_nr);
    double t_p5_1thr = 0;
    
    // We can reuse Result3 struct here for Part 5 execution since it contains the same fields
    std::vector<Result3> res5;
    for (int threads : thread_counts) {
        Matrix C(M, N);
        double t = ParallelExecutor::run(opt_strat, threads, A, B, C);
        if (threads == 1) t_p5_1thr = t;
        double scale = (threads == 1) ? 1.0 : (t_p5_1thr / t);
        res5.push_back({threads, t, std::to_string(t_naive / t) + "x", std::to_string(scale) + "x", verify(C_naive, C)});
    }
    double min5 = 1e18, max5 = -1.0;
    for (const auto& r : res5) if (r.v) { min5 = std::min(min5, r.t); max5 = std::max(max5, r.t); }
    for (const auto& r : res5) {
        std::string col = (r.v && r.t == min5) ? ANSI_GREEN : (r.v && r.t == max5) ? ANSI_RED : "";
        std::cout << col << std::left << std::fixed << std::setprecision(2)
                  << std::setw(W_SMALL) << r.threads << std::setw(W_TIME) << r.t
                  << std::setw(W_SPEED) << r.s1 << std::setw(W_SPEED2) << r.s2
                  << (r.v ? "[Valid]" : "[Invalid]") << ANSI_RESET << "\n";
    }

    std::cout << "================================================================\n";
    return 0;
}