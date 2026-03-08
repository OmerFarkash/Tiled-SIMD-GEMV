#include "WeightLoader.hpp"
#include <cstring>
#include <immintrin.h> // SIMD AVX/FMA

WeightLoader::WeightLoader() {
    sram_buffer_state[0].store(FREE);
    sram_buffer_state[1].store(FREE);
    is_dma_hardware_busy[0] = false;
    is_dma_hardware_busy[1] = false;
}

// Function to start an asynchronous DMA load from DDR to SRAM
void WeightLoader::start_dma_load(float* ddr_weights_ptr, int target_buffer, int full_width, int row_offset, int col_offset) {
    uint8_t expected = FREE;
    while (!sram_buffer_state[target_buffer].compare_exchange_weak(expected, LOADING)) {
        expected = FREE;
    }
    
    // load - DMA simulation
    int sram_step = 0;
    int ddr_offset = row_offset * full_width + col_offset;
    const int copy_size = TILE_W * sizeof(float);
    for (int i = 0; i < TILE_H; i++)
    {
        std::memcpy(&sram_storage[target_buffer][sram_step], &ddr_weights_ptr[ddr_offset], copy_size);
        sram_step += TILE_W;
        ddr_offset += full_width;
    }
    
    is_dma_hardware_busy[target_buffer] = true;
}

// Function to wait (stall) until the DMA load to SRAM is complete
void WeightLoader::wait_for_dma(int target_buffer) {
    // it's simple in the simulation, just check register on real senario.
    if (is_dma_hardware_busy[target_buffer]) {
        is_dma_hardware_busy[target_buffer] = false;
    }
    sram_buffer_state[target_buffer].store(READY, std::memory_order_release);
}

// Function to trigger the compute unit for the specified block
void WeightLoader::compute_on_block(int buffer_index, float* inputs, float* output, int row_offset, int col_offset) {
    uint8_t expected = READY;
    while (!sram_buffer_state[buffer_index].compare_exchange_weak(expected, COMPUTE, std::memory_order_acquire)) {
        expected = READY;
    }
    
    // compute
    float* curr_output = output + row_offset;
    float* curr_input = inputs + col_offset;

    // --- SIMD Kernel ---
    for (int i = 0; i < TILE_H; i++)
    {
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < TILE_W; j += 8) {
            __m256 weights   = _mm256_loadu_ps(&sram_storage[buffer_index][i * TILE_W + j]);
            __m256 inputs_v  = _mm256_loadu_ps(&curr_input[j]);
            acc = _mm256_fmadd_ps(weights, inputs_v, acc);
        }
        // Horizontal reduction
        alignas(32) float res[8];
        _mm256_store_ps(res, acc);
        float row_sum = 0;
        for(int k = 0; k < 8; ++k) row_sum += res[k];
        
        curr_output[i] += row_sum;
    }

    sram_buffer_state[buffer_index].store(FREE, std::memory_order_release);
}

void WeightLoader::compute_on_block_naive(int buffer_index, float* inputs, float* output, int row_offset, int col_offset) {
    uint8_t expected = READY;
    while (!sram_buffer_state[buffer_index].compare_exchange_weak(expected, COMPUTE, std::memory_order_acquire)) {
        expected = READY;
    }

    float* curr_output = output + row_offset;
    float* curr_input = inputs + col_offset;

    for (int i = 0; i < 16; i++) {
        float row_sum = 0;
        for (int j = 0; j < 256; j++) {
            row_sum += sram_storage[buffer_index][i * 256 + j] * curr_input[j];
        }
        curr_output[i] += row_sum;
    }

    sram_buffer_state[buffer_index].store(FREE, std::memory_order_release);
}