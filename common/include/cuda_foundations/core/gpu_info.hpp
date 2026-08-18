#pragma once
/**
 * @file gpu_info.hpp
 * @brief GPU device information and query utilities
 *
 * Provides functions to query and display GPU device properties.
 *
 * @example
 * @code
 * #include <cuda_foundations/core/gpu_info.hpp>
 *
 * // Print info for current device
 * cuda_foundations::print_gpu_info();
 *
 * // Print info for all devices
 * cuda_foundations::print_all_gpus();
 *
 * // Get device properties
 * auto props = cuda_foundations::get_device_properties();
 * std::cout << "SM count: " << props.multiProcessorCount << std::endl;
 * @endcode
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cuda_foundations/core/cuda_check.hpp>

namespace cuda_foundations {

// ============================================================================
// Device Query Functions
// ============================================================================

/**
 * @brief Get the number of CUDA-capable devices
 * @return Number of devices, or 0 if no CUDA driver or devices
 */
inline int get_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return count;
}

/**
 * @brief Get the current device index
 * @return Current device index (0-based)
 * @throws CudaError if query fails
 */
inline int get_current_device() {
    int device = 0;
    CA_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

/**
 * @brief Get device properties for a specific device
 * @param device Device index (default: current device)
 * @return cudaDeviceProp structure with device properties
 * @throws CudaError if query fails
 */
inline cudaDeviceProp get_device_properties(int device = -1) {
    if (device < 0) {
        device = get_current_device();
    }
    cudaDeviceProp prop;
    CA_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return prop;
}

/**
 * @brief Check if a device supports a given compute capability
 * @param major Required major version
 * @param minor Required minor version
 * @param device Device index (default: current device)
 * @return true if device has >= required compute capability
 */
inline bool has_compute_capability(int major, int minor, int device = -1) {
    cudaDeviceProp prop = get_device_properties(device);
    if (prop.major > major) {
        return true;
    }
    if (prop.major == major) {
        return prop.minor >= minor;
    }
    return false;
}

// ============================================================================
// Display Functions
// ============================================================================

/**
 * @brief Print detailed information for a specific GPU
 * @param device Device index (default: current device)
 */
inline void print_gpu_info(int device = -1) {
    if (device < 0) {
        device = get_current_device();
    }

    cudaDeviceProp prop = get_device_properties(device);

    printf("GPU Device %d: %s\n", device, prop.name);
    printf("  Compute Capability:     %d.%d\n", prop.major, prop.minor);
    printf("  SM Count:               %d\n", prop.multiProcessorCount);
    printf("  Max Threads per Block:  %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads per SM:     %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Warp Size:              %d\n", prop.warpSize);
    printf("  Shared Memory per Block:%zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Shared Memory per SM:   %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("  Global Memory:          %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  L2 Cache Size:          %d KB\n", prop.l2CacheSize / 1024);
    printf("  Memory Bus Width:       %d bits\n", prop.memoryBusWidth);
    printf("  Max Grid Size:          (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1],
           prop.maxGridSize[2]);
    printf("  Max Block Size:         (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);
    printf("\n");
}

/**
 * @brief Print information for all CUDA-capable GPUs
 */
inline void print_all_gpus() {
    int count = get_device_count();

    if (count == 0) {
        printf("No CUDA-capable devices found.\n");
        return;
    }

    printf("Found %d CUDA-capable device(s):\n\n", count);

    int current = get_current_device();

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop = get_device_properties(i);
        printf("  [%d] %s (SM %d.%d, %d SMs)%s\n", i, prop.name, prop.major, prop.minor,
               prop.multiProcessorCount, (i == current ? " [current]" : ""));
    }
    printf("\n");

    // Print detailed info for current device
    printf("Current device details:\n");
    print_gpu_info(current);
}

// ============================================================================
// Memory Information
// ============================================================================

/**
 * @brief Get free and total memory on the current device
 * @param free Output: free memory in bytes
 * @param total Output: total memory in bytes
 * @throws CudaError if query fails
 */
inline void get_memory_info(size_t& free, size_t& total) {
    CA_CUDA_CHECK(cudaMemGetInfo(&free, &total));
}

/**
 * @brief Print memory usage information for the current device
 */
inline void print_memory_info() {
    size_t free_mem, total_mem;
    get_memory_info(free_mem, total_mem);

    size_t used_mem = total_mem - free_mem;

    printf("GPU Memory:\n");
    printf("  Total:  %.2f GB\n", total_mem / (1024.0 * 1024.0 * 1024.0));
    printf("  Used:   %.2f GB (%.1f%%)\n", used_mem / (1024.0 * 1024.0 * 1024.0),
           100.0 * used_mem / total_mem);
    printf("  Free:   %.2f GB (%.1f%%)\n", free_mem / (1024.0 * 1024.0 * 1024.0),
           100.0 * free_mem / total_mem);
    printf("\n");
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Calculate optimal block size for a kernel
 * @param kernel_func Pointer to kernel function
 * @param device Device index (default: current device)
 * @return Optimal block size, or 256 if unable to determine
 */
template <typename KernelFunc>
inline int optimal_block_size(KernelFunc kernel_func, int device = -1) {
    if (device < 0) {
        device = get_current_device();
    }

    int min_grid_size, block_size;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel_func);

    if (err != cudaSuccess) {
        return 256;  // Default fallback
    }

    return block_size;
}

/**
 * @brief Calculate the number of blocks needed for a given problem size
 * @param num_elements Total number of elements to process
 * @param block_size Number of threads per block
 * @return Minimum number of blocks needed
 */
inline int ceil_div(int num_elements, int block_size) {
    return (num_elements + block_size - 1) / block_size;
}

}  // namespace cuda_foundations
