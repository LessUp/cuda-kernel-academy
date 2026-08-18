#pragma once
/**
 * @file cuda_foundations.hpp
 * @brief Main header for CUDA Kernel Academy common utilities
 *
 * Include this header to get access to all common utilities:
 * - Error checking macros (CA_CUDA_CHECK, etc.)
 * - Device memory RAII wrappers
 * - Timing utilities
 * - GPU information
 *
 * @example
 * @code
 * #include <cuda_foundations/cuda_foundations.hpp>
 *
 * int main() {
 *     // Print GPU info
 *     cuda_foundations::print_all_gpus();
 *
 *     // Allocate device memory
 *     cuda_foundations::DeviceMemory<float> d_data(1024);
 *
 *     // Time a kernel
 *     cuda_foundations::CudaTimer timer;
 *     timer.start();
 *     my_kernel<<<grid, block>>>(d_data.get(), 1024);
 *     timer.stop();
 *     std::cout << "Time: " << timer.elapsed_ms() << " ms" << std::endl;
 *
 *     return 0;
 * }
 * @endcode
 */

#include "core/cuda_check.hpp"
#include "core/device_memory.hpp"
#include "core/gpu_info.hpp"
#include "core/timer.hpp"

namespace cuda_foundations {

/**
 * @brief Library version
 */
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

/**
 * @brief Get version string
 */
inline const char* version() {
    return "1.0.0";
}

/**
 * @brief Common constants
 */
namespace constants {
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int DEFAULT_TILE_SIZE = 32;
constexpr int DEFAULT_BLOCK_DIM = 16;
}  // namespace constants

}  // namespace cuda_foundations
