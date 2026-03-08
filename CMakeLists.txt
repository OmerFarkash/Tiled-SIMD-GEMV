cmake_minimum_required(VERSION 3.10)
project(Tiled-SIMD-GEMV)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find threading library
find_package(Threads REQUIRED)

# Add executable and source files
add_executable(app 
    main.cpp 
    WeightLoader.cpp 
    ParallelExecutor.cpp 
    Strategies.cpp
)

# Target compile options:
# -O3: Maximum optimization
# -mavx2 -mfma: Enable SIMD instruction sets
# -Wall -Wextra: Helpful warnings for development
target_compile_options(app PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)

# Link the threading library
target_link_libraries(app PRIVATE Threads::Threads)