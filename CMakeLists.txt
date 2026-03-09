cmake_minimum_required(VERSION 3.10)
project(Tiled-SIMD-Matrix-Engine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Include path for all headers
include_directories(include)

# --- Shared Core Sources ---
# ParallelExecutor handles both GEMV and GEMM execution.
# It requires WeightLoader.cpp to resolve dependencies for the GEMV run() overload.
set(CORE_SOURCES
    src/core/ParallelExecutor.cpp
    src/gemv/WeightLoader.cpp
)

# --- Phase 1: GEMV App ---
set(GEMV_SOURCES
    apps/GEMV_Main.cpp
    src/gemv/Strategies.cpp
    ${CORE_SOURCES}
)

add_executable(gemv_app ${GEMV_SOURCES})
target_compile_options(gemv_app PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)
target_link_libraries(gemv_app PRIVATE Threads::Threads)

# --- Phase 2 & Phase 3: GEMM Benchmark ---
# Now includes the new SimdGemmStrategy and uses the shared core
set(GEMM_SOURCES
    apps/GEMM_Main.cpp
    src/gemm/GEMM_Kernels.cpp
    src/gemm/SimdGemmStrategy.cpp
    ${CORE_SOURCES}
)

add_executable(gemm_bench ${GEMM_SOURCES})
target_compile_options(gemm_bench PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)
target_link_libraries(gemm_bench PRIVATE Threads::Threads)

# --- Clean Functionality ---
set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES 
    "${CMAKE_BINARY_DIR}/gemv_app"
    "${CMAKE_BINARY_DIR}/gemm_bench"
)