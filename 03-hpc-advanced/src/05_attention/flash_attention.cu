#include "flash_attention.cuh"
#include "../common/cuda_check.cuh"
#include <cfloat>

namespace hpc::attention {

// Simplified FlashAttention forward pass
// Tiles Q, K, V into shared memory and computes attention in SRAM
template <typename T, int BLOCK_SIZE = 64, int HEAD_DIM = 64>
__global__ void flash_attention_kernel(const T* __restrict__ Q,
                                        const T* __restrict__ K,
                                        const T* __restrict__ V,
                                        T* __restrict__ O,
                                        int batch_size, int num_heads,
                                        int seq_len, int head_dim,
                                        float scale, bool causal) {
    // Shared memory for Q, K, V tiles
    extern __shared__ float smem[];
    float* q_tile = smem;
    float* k_tile = q_tile + BLOCK_SIZE * HEAD_DIM;
    float* v_tile = k_tile + BLOCK_SIZE * HEAD_DIM;
    float* scores = v_tile + BLOCK_SIZE * HEAD_DIM;

    int batch_head = blockIdx.x;
    int b = batch_head / num_heads;
    int h = batch_head % num_heads;
    int q_start = blockIdx.y * BLOCK_SIZE;

    // Offset to current batch and head
    int offset = (b * num_heads + h) * seq_len * head_dim;
    const T* Q_ptr = Q + offset;
    const T* K_ptr = K + offset;
    const T* V_ptr = V + offset;
    T* O_ptr = O + offset;

    // Load Q tile
    for (int i = threadIdx.x; i < BLOCK_SIZE * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        int q_idx = q_start + row;
        if (q_idx < seq_len) {
            q_tile[i] = static_cast<float>(Q_ptr[q_idx * head_dim + col]);
        } else {
            q_tile[i] = 0.0f;
        }
    }
    __syncthreads();

    // Initialize output accumulators
    float o_acc[HEAD_DIM] = {0.0f};
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    // Iterate over K, V tiles
    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_SIZE) {
        // Load K, V tiles
        for (int i = threadIdx.x; i < BLOCK_SIZE * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int kv_idx = kv_start + row;
            if (kv_idx < seq_len) {
                k_tile[i] = static_cast<float>(K_ptr[kv_idx * head_dim + col]);
                v_tile[i] = static_cast<float>(V_ptr[kv_idx * head_dim + col]);
            } else {
                k_tile[i] = 0.0f;
                v_tile[i] = 0.0f;
            }
        }
        __syncthreads();

        // Compute attention scores for this tile
        // Simplified: each thread handles one query position
        int q_idx = q_start + threadIdx.x;
        if (q_idx < seq_len && threadIdx.x < BLOCK_SIZE) {
            for (int j = 0; j < BLOCK_SIZE && (kv_start + j) < seq_len; ++j) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q_tile[threadIdx.x * head_dim + d] * k_tile[j * head_dim + d];
                }
                score *= scale;

                // Apply causal mask
                if (causal && (kv_start + j) > q_idx) {
                    score = -FLT_MAX;
                }

                // Online softmax update
                float m_new = fmaxf(m_prev, score);
                float exp_prev = expf(m_prev - m_new);
                float exp_curr = expf(score - m_new);
                float l_new = l_prev * exp_prev + exp_curr;

                // Update output accumulator
                for (int d = 0; d < head_dim; ++d) {
                    o_acc[d] = o_acc[d] * exp_prev + exp_curr * v_tile[j * head_dim + d];
                }

                m_prev = m_new;
                l_prev = l_new;
            }
        }
        __syncthreads();
    }

    // Write output
    int q_idx = q_start + threadIdx.x;
    if (q_idx < seq_len && threadIdx.x < BLOCK_SIZE) {
        float inv_l = 1.0f / l_prev;
        for (int d = 0; d < head_dim; ++d) {
            O_ptr[q_idx * head_dim + d] = static_cast<T>(o_acc[d] * inv_l);
        }
    }
}

template <>
void flash_attention_forward<float>(const float* Q, const float* K, const float* V,
                                    float* O, const FlashAttnConfig& config,
                                    cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 64;
    constexpr int HEAD_DIM = 64;
    
    dim3 grid(config.batch_size * config.num_heads,
              (config.seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    
    size_t smem_size = 3 * BLOCK_SIZE * HEAD_DIM * sizeof(float) + 
                       BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    
    flash_attention_kernel<float, BLOCK_SIZE, HEAD_DIM><<<grid, block, smem_size, stream>>>(
        Q, K, V, O,
        config.batch_size, config.num_heads,
        config.seq_len, config.head_dim,
        config.scale, config.causal);
    CUDA_CHECK_LAST();
}

} // namespace hpc::attention
