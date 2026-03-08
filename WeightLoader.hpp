#pragma once
#include <atomic>
#include <cstdint>

#define FREE 0
#define LOADING 1
#define READY 2
#define COMPUTE 3

class WeightLoader {
    public:
        static constexpr int TILE_H = 16;
        static constexpr int TILE_W = 256;
        static constexpr int FLAT_TILE = TILE_H * TILE_W;
        static constexpr int BUFFER_LEN = 2;
        WeightLoader();
        
        // Function to start an asynchronous DMA load from DDR to SRAM
        void start_dma_load(float* ddr_weights_ptr, int target_buffer, int full_width, int row_offset, int col_offset);

        // Function to wait (stall) until the DMA load to SRAM is complete
        void wait_for_dma(int target_buffer);

        // Function to trigger the compute unit for the specified block
        void compute_on_block(int buffer_index, float* inputs, float* output, int row_offset, int col_offset);
        // naive version for comperison
        void compute_on_block_naive(int buffer_index, float* inputs, float* output, int row_offset, int col_offset);

    private:
        alignas(32) float sram_storage[BUFFER_LEN][FLAT_TILE]; 
        std::atomic_uint8_t sram_buffer_state[BUFFER_LEN];
        bool is_dma_hardware_busy[BUFFER_LEN];
};
