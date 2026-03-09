# Tiled-SIMD-GEMV 🚀

A high-performance **General Matrix-Vector Multiplication (GEMV)** engine implemented in C++20. This project demonstrates how to bypass the "memory wall" by utilizing SIMD vectorization, cache-friendly tiling, and asynchronous double-buffering.

---

## 🛠 Features

* **SIMD Vectorization:** Uses `AVX2` and `FMA` (Fused Multiply-Add) intrinsics to process 8 floating-point operations in a single CPU cycle.
* **Hardware-Inspired Tiling:** Processes the matrix in $16 \times 256$ tiles, simulating on-chip SRAM storage to maximize cache locality.
* **Double Buffering (DMA Simulation):** Implements a multi-buffered architecture where the next data tile is pre-fetched (simulated DMA) while the current tile is being computed.
* **Multi-Threaded Execution:** A thread-safe `ParallelExecutor` distributes matrix rows across all available CPU cores without shared-state contention (Lock-free row partitioning).
* **Robust Fringe (Tail) Handling:** Capable of processing matrices of **any arbitrary size**. The engine dynamically calculates safe tile dimensions (`valid_h`, `valid_w`), uses SIMD for the largest possible multiple of 8, and falls back to a highly optimized **Scalar Epilogue** for the remainder to prevent out-of-bounds memory access without padding overhead.
* **Strategy Pattern Architecture:** Easily switch between `Naive` and `SIMD` execution modes to benchmark the performance gains.

---

## 🏗 System Architecture

The project is built on the concept of **Software-Managed Memory**:

1.  **DDR (Main Memory):** Large, slow storage where the matrix resides.
2.  **SRAM (WeightLoader):** A simulated fast local buffer that stores current and next tiles.
3.  **Compute Unit:** An optimized kernel that performs dot products using `__m256` registers.
4.  **Double Buffering:** The `WeightLoader` uses atomic state transitions (`FREE`, `LOADING`, `READY`, `COMPUTE`) to overlap I/O with computation.

---

## 🚀 Getting Started

### Prerequisites
* A CPU supporting **AVX2** and **FMA** instructions.
* A C++20 compliant compiler (GCC 10+, Clang 11+, or MSVC 2019+).
* **CMake** 3.10 or higher.

### Compilation & Build
Using the provided `CMakeLists.txt`:

```bash
mkdir build && cd build
cmake ..
make
```

### Running the Benchmark
```Bash
./app
```

## 📊 Performance Benchmark Results
The engine remains highly efficient even when dealing with unaligned, non-power-of-two matrix dimensions, thanks to the dynamic tail handling.

**Matrix Size:** $10013 \times 12452$  
**CPU Cores:** 4

| Implementation | Execution Time | Speedup |
| :--- | :--- | :--- |
| **Naive (1 Thread)** | 188.86 ms | Baseline (1.0x) |
| **SIMD (1 Thread)** | 93.17 ms | ~2.03x |
| **SIMD (4 Threads)** | 54.40 ms | **~3.47x** |


**Matrix Size:** $8192 \times 8192$  
**CPU Cores:** 4

| Implementation | Execution Time | Speedup |
| :--- | :--- | :--- |
| **Naive (1 Thread)** | 162.60 ms | Baseline (1.0x) |
| **SIMD (1 Thread)** | 111.47 ms | ~1.46x |
| **SIMD (4 Threads)** | 51.25 ms | **~3.17x** |

> **Verification:** All results were verified against the baseline to ensure mathematical consistency across optimized kernels.

---

## 📂 Project Structure

* **`WeightLoader`**: Simulates the hardware buffer (SRAM) and DMA logic, managing atomic state transitions.
* **`ParallelExecutor`**: Manages thread lifecycle, workload partitioning, and high-resolution timing.
* **`ComputeStrategy`**: An abstract interface defining the contract for various computation kernels.
* **`Strategies`**: Concrete implementations of the `ComputeStrategy`, including the SIMD/FMA optimized kernel and the scalar baseline.