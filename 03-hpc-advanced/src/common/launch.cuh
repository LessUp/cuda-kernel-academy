#pragma once

#include <cuda_runtime.h>
#include <concepts>

namespace hpc {

// Concept for kernel configuration
template <typename T>
concept KernelConfig = requires {
    { T::BLOCK_SIZE } -> std::convertible_to<int>;
};

// Launch configuration helper
template <int BlockSize = 256>
struct LaunchConfig {
    static constexpr int BLOCK_SIZE = BlockSize;

    [[nodiscard]] static constexpr dim3 grid_1d(size_t n) noexcept {
        return dim3(static_cast<unsigned int>((n + BlockSize - 1) / BlockSize));
    }

    [[nodiscard]] static constexpr dim3 block_1d() noexcept {
        return dim3(BlockSize);
    }

    [[nodiscard]] static constexpr dim3 grid_2d(int rows, int cols,
                                                 int tile_rows, int tile_cols) noexcept {
        return dim3(
            static_cast<unsigned int>((cols + tile_cols - 1) / tile_cols),
            static_cast<unsigned int>((rows + tile_rows - 1) / tile_rows)
        );
    }
};

// Compile-time shared memory size calculation
template <typename T, int TileSize>
constexpr size_t shared_mem_size() {
    return TileSize * TileSize * sizeof(T);
}

// Warp size constant
inline constexpr int WARP_SIZE = 32;

// Common block sizes
inline constexpr int BLOCK_64 = 64;
inline constexpr int BLOCK_128 = 128;
inline constexpr int BLOCK_256 = 256;
inline constexpr int BLOCK_512 = 512;
inline constexpr int BLOCK_1024 = 1024;

// Tile sizes for GEMM
inline constexpr int TILE_16 = 16;
inline constexpr int TILE_32 = 32;
inline constexpr int TILE_64 = 64;
inline constexpr int TILE_128 = 128;

} // namespace hpc
