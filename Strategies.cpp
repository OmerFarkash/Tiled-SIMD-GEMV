#include "Strategies.hpp"
#include <algorithm>

void SIMDMatrixVectorStrategy::execute(int start_row, int end_row, float* matrix, float* inputs, 
                                      float* output, int full_width, WeightLoader& loader) {
    const int TILE_H = 16;
    const int TILE_W = 256;

    for (int r = start_row; r < end_row; r += TILE_H) {
        int valid_h = std::min(TILE_H, end_row - r);
        int current_buffer = 0;
        
        // Initial load for the first tile in the row
        int first_valid_w = std::min(TILE_W, full_width);
        loader.start_dma_load(matrix, current_buffer, full_width, r, 0, valid_h, first_valid_w);

        for (int c = 0; c < full_width; c += TILE_W) {
            int valid_w = std::min(TILE_W, full_width - c);
            int next_c = c + TILE_W;
            int next_buffer = 1 - current_buffer;

            if (next_c < full_width) {
                int next_valid_w = std::min(TILE_W, full_width - next_c);
                loader.start_dma_load(matrix, next_buffer, full_width, r, next_c, valid_h, next_valid_w);
            }

            loader.wait_for_dma(current_buffer);
            
            // SIMD Optimized computation
            loader.compute_on_block(current_buffer, inputs, output, r, c, valid_h, valid_w);

            current_buffer = next_buffer;
        }
    }
}

void NaiveMatrixVectorStrategy::execute(int start_row, int end_row, float* matrix, float* inputs, 
                                       float* output, int full_width, WeightLoader& loader) {
    const int TILE_H = 16;
    const int TILE_W = 256;

    for (int r = start_row; r < end_row; r += TILE_H) {
        int valid_h = std::min(TILE_H, end_row - r);
        int current_buffer = 0;
        
        int first_valid_w = std::min(TILE_W, full_width);
        loader.start_dma_load(matrix, current_buffer, full_width, r, 0, valid_h, first_valid_w);

        for (int c = 0; c < full_width; c += TILE_W) {
            int valid_w = std::min(TILE_W, full_width - c);
            int next_c = c + TILE_W;
            int next_buffer = 1 - current_buffer;

            if (next_c < full_width) {
                int next_valid_w = std::min(TILE_W, full_width - next_c);
                loader.start_dma_load(matrix, next_buffer, full_width, r, next_c, valid_h, next_valid_w);
            }

            loader.wait_for_dma(current_buffer);
            
            // Scalar (Naive) computation
            loader.compute_on_block_naive(current_buffer, inputs, output, r, c, valid_h, valid_w);

            current_buffer = next_buffer;
        }
    }
}