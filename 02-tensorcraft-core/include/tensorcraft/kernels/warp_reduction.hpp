#pragma once
/**
 * @file warp_reduction.hpp
 * @brief Warp-level reduction primitives using CUDA shuffle instructions
 *
 * Provides efficient warp-level reduction operations that can be reused
 * across different kernels (softmax, normalization, etc.).
 */

#include <cfloat>

#include "../core/features.hpp"

namespace tensorcraft {
namespace kernels {

// ============================================================================
// Warp Reduction Primitives
// ============================================================================

/**
 * @brief Warp-level max reduction using shuffle
 *
 * Reduces a value across all 32 threads in a warp to find the maximum.
 * Uses __shfl_down_sync for efficient intra-warp communication.
 *
 * @param val The value to reduce (each thread contributes one value)
 * @return The maximum value across the warp (same for all threads after call)
 *
 * @example
 * @code
 * float thread_max = ...; // each thread has its own max
 * float warp_max = warp_reduce_max(thread_max); // all threads get the same result
 * @endcode
 */
template <typename T>
TC_DEVICE_INLINE T warp_reduce_max(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

/**
 * @brief Warp-level sum reduction using shuffle
 *
 * Reduces a value across all 32 threads in a warp to compute the sum.
 * Uses __shfl_down_sync for efficient intra-warp communication.
 *
 * @param val The value to reduce (each thread contributes one value)
 * @return The sum across the warp (same for all threads after call)
 *
 * @example
 * @code
 * float thread_sum = ...; // each thread has its own partial sum
 * float warp_sum = warp_reduce_sum(thread_sum); // all threads get the same result
 * @endcode
 */
template <typename T>
TC_DEVICE_INLINE T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Online Reduction Utilities
// ============================================================================

/**
 * @brief Combine two (max, sum) pairs using online algorithm
 *
 * Used for online softmax and similar algorithms that maintain
 * running maximum and sum while processing data in chunks.
 *
 * @param max1 First maximum value
 * @param sum1 First sum (corresponding to max1)
 * @param max2 Second maximum value
 * @param sum2 Second sum (corresponding to max2)
 * @param out_max Output combined maximum
 * @param out_sum Output combined sum (properly rescaled)
 */
TC_DEVICE_INLINE void online_combine(float max1, float sum1, float max2, float sum2, float& out_max,
                                     float& out_sum) {
    if (max2 > max1) {
        // Rescale sum1 to new max
        out_max = max2;
        out_sum = sum1 * expf(max1 - max2) + sum2;
    } else {
        // Rescale sum2 to new max
        out_max = max1;
        out_sum = sum1 + sum2 * expf(max2 - max1);
    }
}

}  // namespace kernels
}  // namespace tensorcraft
