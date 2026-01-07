/**
 * @file main.cu
 * @brief 基础 GEMM 示例 | Basic GEMM Example
 *
 * 本示例演示两种 GEMM 实现：
 * 1. Naive: 简单的三层循环
 * 2. Tiled: 使用共享内存优化
 *
 * This example demonstrates two GEMM implementations:
 * 1. Naive: Simple triple-nested loop
 * 2. Tiled: Optimized with shared memory
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

// ============================================================================
// CUDA 错误检查宏 | CUDA Error Checking Macro
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// 常量定义 | Constants
// ============================================================================
constexpr int TILE_SIZE = 16;  // 共享内存 tile 大小

// ============================================================================
// Naive GEMM Kernel
// ============================================================================
/**
 * @brief Naive GEMM: C = A * B
 *
 * 每个线程计算 C 的一个元素，直接从全局内存读取 A 和 B。
 * 这是最简单但效率最低的实现。
 *
 * Each thread computes one element of C, reading A and B directly from
 * global memory. This is the simplest but least efficient implementation.
 *
 * @param A 输入矩阵 A [M x K]
 * @param B 输入矩阵 B [K x N]
 * @param C 输出矩阵 C [M x N]
 * @param M A 的行数
 * @param N B 的列数
 * @param K A 的列数 / B 的行数
 */
__global__ void naive_gemm_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // 计算当前线程负责的 C 元素位置
    // Calculate the position of C element this thread is responsible for
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查 | Boundary check
    if (row < M && col < N) {
        float sum = 0.0f;

        // 累加 A 的一行与 B 的一列的点积
        // Accumulate dot product of row of A and column of B
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// ============================================================================
// Tiled GEMM Kernel (共享内存优化)
// ============================================================================
/**
 * @brief Tiled GEMM: C = A * B (使用共享内存)
 *
 * 使用共享内存减少全局内存访问次数。
 * 将 A 和 B 分块加载到共享内存，然后计算部分结果。
 *
 * Uses shared memory to reduce global memory accesses.
 * Loads tiles of A and B into shared memory, then computes partial results.
 *
 * 优化原理 | Optimization principle:
 * - 全局内存访问延迟高 (~400 cycles)
 * - 共享内存访问延迟低 (~5 cycles)
 * - 通过数据复用减少全局内存访问
 */
__global__ void tiled_gemm_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    // 共享内存声明 | Shared memory declaration
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 线程索引 | Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 遍历所有 tile | Iterate over all tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // 加载 A 的 tile 到共享内存
        // Load tile of A into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 加载 B 的 tile 到共享内存
        // Load tile of B into shared memory
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 同步，确保 tile 加载完成
        // Synchronize to ensure tile is loaded
        __syncthreads();

        // 计算部分点积 | Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 同步，确保计算完成后再加载下一个 tile
        // Synchronize before loading next tile
        __syncthreads();
    }

    // 写回结果 | Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// CPU 参考实现 | CPU Reference Implementation
// ============================================================================
void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ============================================================================
// 验证函数 | Verification Function
// ============================================================================
bool verify(const float* ref, const float* test, int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        float diff = std::fabs(ref[i] - test[i]);
        if (diff > tolerance) {
            printf("Mismatch at index %d: ref=%.6f, test=%.6f, diff=%.6f\n",
                   i, ref[i], test[i], diff);
            return false;
        }
    }
    return true;
}

// ============================================================================
// 性能测量 | Performance Measurement
// ============================================================================
float benchmark_kernel(void (*launch_func)(const float*, const float*, float*,
                                            int, int, int, dim3, dim3),
                       const float* d_A, const float* d_B, float* d_C,
                       int M, int N, int K, dim3 grid, dim3 block,
                       int warmup = 3, int iterations = 10) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        launch_func(d_A, d_B, d_C, M, N, K, grid, block);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        launch_func(d_A, d_B, d_C, M, N, K, grid, block);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(end - start).count();
    return ms / iterations;
}

// Kernel launch wrappers
void launch_naive(const float* A, const float* B, float* C,
                  int M, int N, int K, dim3 grid, dim3 block) {
    naive_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

void launch_tiled(const float* A, const float* B, float* C,
                  int M, int N, int K, dim3 grid, dim3 block) {
    tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

// ============================================================================
// 主函数 | Main Function
// ============================================================================
int main() {
    printf("=== CUDA Kernel Academy: Basic GEMM Example ===\n\n");

    // 矩阵维度 | Matrix dimensions
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("TILE_SIZE: %d\n\n", TILE_SIZE);

    // 分配主机内存 | Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C_naive = (float*)malloc(size_C);
    float* h_C_tiled = (float*)malloc(size_C);
    float* h_C_ref = (float*)malloc(size_C);

    // 初始化输入矩阵 | Initialize input matrices
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // 分配设备内存 | Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // 复制数据到设备 | Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // 配置 kernel 启动参数 | Configure kernel launch parameters
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Grid: (%d, %d), Block: (%d, %d)\n\n", grid.x, grid.y, block.x, block.y);

    // ========================================================================
    // 运行 Naive GEMM
    // ========================================================================
    printf("Running Naive GEMM...\n");
    float naive_time = benchmark_kernel(launch_naive, d_A, d_B, d_C,
                                        M, N, K, grid, block);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, size_C, cudaMemcpyDeviceToHost));

    // 计算 GFLOPS
    float gflops = 2.0f * M * N * K / (naive_time * 1e6f);
    printf("  Time: %.3f ms\n", naive_time);
    printf("  Performance: %.2f GFLOPS\n\n", gflops);

    // ========================================================================
    // 运行 Tiled GEMM
    // ========================================================================
    printf("Running Tiled GEMM...\n");
    float tiled_time = benchmark_kernel(launch_tiled, d_A, d_B, d_C,
                                        M, N, K, grid, block);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, size_C, cudaMemcpyDeviceToHost));

    gflops = 2.0f * M * N * K / (tiled_time * 1e6f);
    printf("  Time: %.3f ms\n", tiled_time);
    printf("  Performance: %.2f GFLOPS\n\n", gflops);

    // ========================================================================
    // 验证结果 | Verify Results
    // ========================================================================
    printf("Computing CPU reference (this may take a while)...\n");
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    printf("\nVerifying results...\n");
    bool naive_correct = verify(h_C_ref, h_C_naive, M * N);
    bool tiled_correct = verify(h_C_ref, h_C_tiled, M * N);

    printf("  Naive GEMM: %s\n", naive_correct ? "PASSED ✓" : "FAILED ✗");
    printf("  Tiled GEMM: %s\n", tiled_correct ? "PASSED ✓" : "FAILED ✗");

    // ========================================================================
    // 性能对比 | Performance Comparison
    // ========================================================================
    printf("\n=== Performance Summary ===\n");
    printf("Naive GEMM: %.3f ms\n", naive_time);
    printf("Tiled GEMM: %.3f ms\n", tiled_time);
    printf("Speedup: %.2fx\n", naive_time / tiled_time);

    // 清理 | Cleanup
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("\nDone!\n");
    return 0;
}
