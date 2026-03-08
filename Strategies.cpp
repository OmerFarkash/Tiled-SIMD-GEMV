#include "Strategies.hpp"

void SIMDMatrixVectorStrategy::execute(int start_row, int end_row, float* matrix, float* inputs, 
                                      float* output, int full_width, WeightLoader& loader) {
    const int TILE_H = 16;
    const int TILE_W = 256;

    for (int r = start_row; r < end_row; r += TILE_H) {
        int current_buffer = 0;
        
        // Initial load for the first tile in the row
        loader.start_dma_load(matrix, current_buffer, full_width, r, 0);

        for (int c = 0; c < full_width; c += TILE_W) {
            int next_c = c + TILE_W;
            int next_buffer = 1 - current_buffer;

            if (next_c < full_width) {
                loader.start_dma_load(matrix, next_buffer, full_width, r, next_c);
            }

            loader.wait_for_dma(current_buffer);
            
            // SIMD Optimized computation
            loader.compute_on_block(current_buffer, inputs, output, r, c);

            current_buffer = next_buffer;
        }
    }
}

void NaiveMatrixVectorStrategy::execute(int start_row, int end_row, float* matrix, float* inputs, 
                                       float* output, int full_width, WeightLoader& loader) {
    const int TILE_H = 16;
    const int TILE_W = 256;

    for (int r = start_row; r < end_row; r += TILE_H) {
        int current_buffer = 0;
        
        loader.start_dma_load(matrix, current_buffer, full_width, r, 0);

        for (int c = 0; c < full_width; c += TILE_W) {
            int next_c = c + TILE_W;
            int next_buffer = 1 - current_buffer;

            if (next_c < full_width) {
                loader.start_dma_load(matrix, next_buffer, full_width, r, next_c);
            }

            loader.wait_for_dma(current_buffer);
            
            // Scalar (Naive) computation
            loader.compute_on_block_naive(current_buffer, inputs, output, r, c);

            current_buffer = next_buffer;
        }
    }
}