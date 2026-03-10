# Tiled SIMD Matrix Engine

A high-performance C++ Matrix Multiplication (GEMM) and Vector (GEMV) engine optimized for modern CPU architectures. This project demonstrates significant speedups through **AVX2 SIMD Intrinsics**, **Cache-aware Tiling**, **Register-level Micro-Kernels**, and **Multi-threaded 2D Grid Partitioning**.


## 🔹 Matrix-Vector Multiplication (GEMV)

Focuses on real-time streaming of weights using asynchronous double-buffering.

### GEMV Features:
* **Double Buffering:** Overlaps I/O (loading tiles) with computation.
* **SIMD & FMA:** Processes 8 floats per cycle using `AVX2`.
* **Robust Tail Handling:** Supports any matrix dimension without padding.

### GEMV Performance:
**Matrix Size:** $10013 \times 12452$ | **CPU Cores:** 4

| Implementation | Execution Time | Speedup |
| :--- | :--- | :--- |
| Naive (1 Thread) | 188.86 ms | 1.00x |
| SIMD (1 Thread) | 93.17 ms | ~2.03x |
| SIMD (4 Threads) | 54.40 ms | **~3.47x** |

---

## 🔸 Matrix-Matrix Multiplication (GEMM)

Unlike GEMV, GEMM is heavily compute-bound and requires aggressive cache reuse and vectorization. The project evolves through 4 distinct optimization phases.

### GEMM Performance:
*Benchmark conducted on a Deep Learning FC Layer Profile (M=250, K=1000, N=4000).*

| Phase | Optimization Level | Strategy | Threads | Time (ms) | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Baseline** | Naive Triple Loop | 1 | ~2434 | 1.00x |
| **1** | **Memory Opt** | Naive Transpose | 1 | ~1391 | 1.74x |
| **2** | **Cache Opt** | Tiled + Packed Transpose | 1 | ~894 | 2.72x |
| **3** | **Vectorized** | SIMD + Broadcasting | 1 | ~245 | 9.93x |
| **3** | **Parallel** | 2D Grid SIMD Engine | 4 | ~138 | 17.53x |
| **4** | **Register Tile** | MR×NR Micro-Kernel (4×16) | 1 | **~105** | **22.97x** |
| **4** | **Register Tile + Parallel (Optimal)** | **MR×NR + 2D Grid** | **4** | **~57** | **~42.47x** |

*Note: The single-thread Phase 4 register-tile kernel fully surpasses the 4-thread Phase 3 SIMD baseline.*

---

## 🧠 Key Technical Implementations

### 1. Vectorized Outer-Product (Broadcasting)
To leverage AVX2 registers, the engine transitions from a standard Dot Product to an **Outer-Product pattern**. 
* **The Operation:** A single scalar from Matrix $A$ is broadcasted across a 256-bit register.
* **SIMD Execution:** It is multiplied by 8 contiguous elements of Matrix $B$ using FMA (`_mm256_fmadd_ps`).
* **Efficiency:** This approach eliminates the need for transposing Matrix $B$ during the packing stage, significantly reducing memory overhead and maximizing ALU throughput.

### 2. 2D Grid Partitioning & Thread Scaling
The `ParallelExecutor` core utilizes a 2D Grid topology to distribute work, revealing key hardware boundaries:
* **Cache Reuse:** By splitting the output matrix $C$ into blocks (Macro-tiles) rather than simple row-strips, we maximize the reuse of shared **L3 Cache** for matrices $A$ and $B$ across different threads.
* **Load Balancing:** The engine dynamically calculates the most "square-like" grid for any given thread count, ensuring optimal work distribution and handling edge-case tails for unaligned dimensions.
* **The Hardware Ceiling:** Benchmarks show peak execution at 4 threads with a regression at 8 threads. This is a textbook example of **ALU saturation**. Dense AVX2/FMA operations fully occupy the physical CPU cores; therefore, logical cores cannot extract further instruction-level parallelism and only introduce memory subsystem bottlenecks.

### 3. Cache-Aware Tiling
Micro-tiling (fixed at $32 \times 32$) ensures that the active data set for any given computation fits entirely within the **L1 Data Cache**, preventing costly stalls and cache evictions during the inner-most computation loops.

### 4. Register-Level Micro-Tile (MR × NR) — Phase 4
The innermost tiling level targets the **CPU register file** itself. AVX2 provides **16 × 256-bit YMM registers**. The Phase 3 kernel used only 1 of these as an accumulator at a time; Phase 4 occupies up to 8 (`MR=4, NR=16` → `4 × 2 = 8 accumulators`).

> **Note:** "Cores saturated" (README Part 3) and "registers underutilised" are two different dimensions. At 4 threads the ALU throughput is core-saturated; but *within each core*, only 1 of 16 YMM registers held an accumulator. Phase 4 fills 8 accumulators per core, issuing more independent FMAs per cycle (higher IPC) without needing more threads.

**Three-level tiling hierarchy:**
```
Thread macro-tile  (2D grid via ParallelExecutor)
  Cache tile       (tile_size × tile_size, fits L1/L2)
    Register tile  (MR × NR — lives in YMM register file)  ← Phase 4
```

**Why it works:**
* **MR rows** of C are computed simultaneously — each A broadcast value is **reused NR/8 times** instead of once.
* **NR cols** of C per row: accumulators stay **in registers across the full k-loop**, eliminating redundant loads/stores to C.
* The micro-kernel is a **compile-time template** (`template<int MR, int NR>`), so the compiler fully unrolls and assigns named YMM registers.

**AVX2 Register Budget (optimal config):**

| Usage | Count |
| :--- | :--- |
| MR×(NR/8) accumulators | 8 |
| A broadcast operand | 1 |
| B load operand | 1 |
| **Total** | **10 / 16 YMM** |

### 5. Benchmark Variance & Environment Noise

You may notice that running the benchmark multiple times yields slightly different results, particularly regarding the optimal multi-thread configuration (e.g., 4 threads vs. 8 threads) and exact cache tile sweet-spots. This variance is entirely normal and stems from several hardware and OS-level factors:
* **OS Task Scheduling & Thread Migration:** Because we are not pinning threads to specific CPU cores (Thread Affinity), the OS scheduler may dynamically migrate threads across different physical/logical cores during execution. This can cause sudden L1/L2 cache invalidations.
* **Hyper-Threading (SMT) Collisions:** On a 4-core / 8-thread machine, scaling from 4 to 8 threads means logical cores begin sharing the physical ALU and L1 caches. Sometimes SMT successfully hides memory latency (resulting in a slight speedup at 8 threads); other times, they fight for the same saturated AVX2 units and cache lines, resulting in a regression. The optimal count often dances right on this hardware ceiling.
* **Thermal Throttling & Turbo Boost:** Modern processors dynamically adjust their clock frequencies based on thermal headroom. A cold run might boost significantly higher than a subsequent run where the CPU scales back clock speeds to maintain temperatures.
* **Background Noise:** Any other processes running on the machine (browser, background services) will steal CPU cycles and pollute the shared L3 cache, causing momentary micro-stutters in execution time.

---

### 🧠 Technical Deep Dive: Dot Product vs. Outer Product (Broadcasting)

This project showcases the transition between two fundamental execution patterns in GEMM optimization:

#### 1. The Dot Product Approach
In the scalar tiled version, we calculate each element $C[i][j]$ individually:
* **The Operation:** $C[i][j]=\sum(A[i][k]\times B[k][j])$.
* **The Memory Challenge:** Since $B$ is stored in Row-Major order, accessing $B[k][j]$ for a fixed $j$ while incrementing $k$ results in **vertical strides** through memory, causing severe Cache Misses.
* **The Solution:** We **transposed (packed)** the $B$ tiles into local SRAM-like buffers to ensure $B[k][j]$ elements are contiguous in memory during the dot product.

#### 2. The Outer Product & Broadcasting Approach
To leverage SIMD (AVX2), we flipped the logic to update multiple elements of $C$ at once:
* **The Operation:** One scalar $A[i][k]$ is **broadcasted** (duplicated) into a 256-bit SIMD register.
* **Vectorized Execution:** We load **8 contiguous elements** of $B$ (from the same row $k$) into another register.
* **FMA Acceleration:** We use `_mm256_fmadd_ps` to compute $A[i][k]\times B[k][j:j+7]$ and accumulate the result directly into 8 elements of the $C$ matrix.
* **The Efficiency Win:** Since we are now consuming $B$ along its rows (horizontal access), **we no longer need to transpose the B-tiles**. We simply load them as-is, saving significant packing overhead and maximizing the throughput of the Arithmetic Logic Unit (ALU).

---

## 📂 Project Structure
```text
.
├── apps/               # Benchmark drivers (GEMM_Main, GEMV_Main)
├── include/
│   ├── core/           # ParallelExecutor (The 2D Grid Engine)
│   ├── gemm/           # Strategy interfaces, SIMD Kernels & Micro-Kernel
│   └── gemv/           # WeightLoading & Streaming logic
├── src/                # Implementations
└── CMakeLists.txt      # Optimized build configuration (-O3, -mavx2)
```

---

## 🚀 Getting Started

### Prerequisites
* **CPU:** x86_64 with AVX2 and FMA support.
* **Compiler:** GCC 10+ or Clang 11+ (supporting C++20).
* **Build System:** CMake 3.10+.

### Compilation
```bash
mkdir build && cd build
cmake ..
make
```

## Running Benchmarks (Execution)
```bash
# Run GEMM Performance Benchmark (all 5 parts)
./build/gemm_bench

# Run GEMV Application
./build/gemv_app
```

### Cleaning
```bash
# Standard clean
cmake --build build --target clean
# Or simply
rm -rf build/
```