#pragma once
/**
 * @file attention.hpp
 * @brief LLM attention kernels including FlashAttention, RoPE, PagedAttention
 */

#include "../core/features.hpp"
#include "../core/cuda_check.hpp"
#include "../core/type_traits.hpp"
#include <cfloat>

namespace tensorcraft {
namespace kernels {

// ============================================================================
// FlashAttention-style Kernel (Simplified)
// ============================================================================

/**
 * @brief FlashAttention kernel with online softmax
 * 
 * Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
 * Uses tiling and online softmax for memory efficiency.
 */
template<typename T, int BLOCK_M = 64, int BLOCK_N = 64, int HEAD_DIM = 64>
__global__ void flash_attention_kernel(
    const T* TC_RESTRICT Q,   // [batch, heads, seq_len, head_dim]
    const T* TC_RESTRICT K,   // [batch, heads, seq_len, head_dim]
    const T* TC_RESTRICT V,   // [batch, heads, seq_len, head_dim]
    T* TC_RESTRICT O,         // [batch, heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    float scale) {
    
    // Shared memory for Q, K, V tiles
    __shared__ float Qs[BLOCK_M][HEAD_DIM];
    __shared__ float Ks[BLOCK_N][HEAD_DIM];
    __shared__ float Vs[BLOCK_N][HEAD_DIM];
    __shared__ float scores[BLOCK_M][BLOCK_N];
    
    const int batch_head = blockIdx.z;
    const int batch_idx = batch_head / num_heads;
    const int head_idx = batch_head % num_heads;
    const int m_block = blockIdx.x;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    
    // Offset pointers for this batch/head
    const int offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const T* q_ptr = Q + offset;
    const T* k_ptr = K + offset;
    const T* v_ptr = V + offset;
    T* o_ptr = O + offset;
    
    const int m_start = m_block * BLOCK_M;
    
    // Initialize output accumulators and running statistics
    float o_acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        o_acc[d] = 0.0f;
    }
    float m_prev = -FLT_MAX;  // Running max
    float l_prev = 0.0f;       // Running sum of exp
    
    // Load Q tile (persistent across K/V blocks)
    for (int d = tx; d < HEAD_DIM; d += blockDim.x) {
        const int m_idx = m_start + ty;
        Qs[ty][d] = (m_idx < seq_len) ? to_float(q_ptr[m_idx * HEAD_DIM + d]) : 0.0f;
    }
    __syncthreads();
    
    // Iterate over K/V blocks
    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;
    
    for (int n_block = 0; n_block < num_kv_blocks; ++n_block) {
        const int n_start = n_block * BLOCK_N;
        
        // Load K, V tiles
        for (int d = tx; d < HEAD_DIM; d += blockDim.x) {
            const int n_idx = n_start + ty;
            Ks[ty][d] = (n_idx < seq_len) ? to_float(k_ptr[n_idx * HEAD_DIM + d]) : 0.0f;
            Vs[ty][d] = (n_idx < seq_len) ? to_float(v_ptr[n_idx * HEAD_DIM + d]) : 0.0f;
        }
        __syncthreads();
        
        // Compute QK^T for this block
        if (ty < BLOCK_M && tx < BLOCK_N) {
            float qk = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                qk += Qs[ty][d] * Ks[tx][d];
            }
            scores[ty][tx] = qk * scale;
        }
        __syncthreads();
        
        // Online softmax update (per row)
        if (ty < BLOCK_M && m_start + ty < seq_len) {
            // Find max in this block
            float m_curr = m_prev;
            for (int n = 0; n < BLOCK_N && (n_start + n) < seq_len; ++n) {
                m_curr = fmaxf(m_curr, scores[ty][n]);
            }
            
            // Compute new sum with rescaling
            float l_curr = l_prev * expf(m_prev - m_curr);
            for (int n = 0; n < BLOCK_N && (n_start + n) < seq_len; ++n) {
                l_curr += expf(scores[ty][n] - m_curr);
            }
            
            // Rescale previous output accumulator
            float scale_prev = (l_prev > 0.0f) ? (l_prev * expf(m_prev - m_curr) / l_curr) : 0.0f;
            float scale_curr = 1.0f / l_curr;
            
            // Update output accumulator
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                o_acc[d] *= scale_prev;
            }
            
            for (int n = 0; n < BLOCK_N && (n_start + n) < seq_len; ++n) {
                float p = expf(scores[ty][n] - m_curr) * scale_curr;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    o_acc[d] += p * Vs[n][d];
                }
            }
            
            m_prev = m_curr;
            l_prev = l_curr;
        }
        __syncthreads();
    }
    
    // Write output
    const int m_idx = m_start + ty;
    if (m_idx < seq_len && ty < BLOCK_M) {
        for (int d = tx; d < HEAD_DIM; d += blockDim.x) {
            o_ptr[m_idx * HEAD_DIM + d] = from_float<T>(o_acc[d]);
        }
    }
}

// ============================================================================
// RoPE (Rotary Position Embedding) Kernel
// ============================================================================

/**
 * @brief RoPE kernel for applying rotary position embeddings
 * 
 * Applies rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
 */
template<typename T>
__global__ void rope_kernel(
    T* TC_RESTRICT x,                    // [batch, seq_len, num_heads, head_dim]
    const float* TC_RESTRICT cos_cache,  // [max_seq, head_dim/2]
    const float* TC_RESTRICT sin_cache,  // [max_seq, head_dim/2]
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int start_pos) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_head_dim = head_dim / 2;
    const int total = batch_size * seq_len * num_heads * half_head_dim;
    
    if (idx >= total) return;
    
    // Decode indices
    const int d = idx % half_head_dim;
    int remaining = idx / half_head_dim;
    const int h = remaining % num_heads;
    remaining /= num_heads;
    const int s = remaining % seq_len;
    const int b = remaining / seq_len;
    
    // Get position
    const int pos = start_pos + s;
    
    // Load cos/sin for this position and dimension
    const float cos_val = cos_cache[pos * half_head_dim + d];
    const float sin_val = sin_cache[pos * half_head_dim + d];
    
    // Compute offset in x
    const int base_offset = ((b * seq_len + s) * num_heads + h) * head_dim;
    
    // Load pair of values
    const float x0 = to_float(x[base_offset + d]);
    const float x1 = to_float(x[base_offset + d + half_head_dim]);
    
    // Apply rotation
    const float y0 = x0 * cos_val - x1 * sin_val;
    const float y1 = x0 * sin_val + x1 * cos_val;
    
    // Store
    x[base_offset + d] = from_float<T>(y0);
    x[base_offset + d + half_head_dim] = from_float<T>(y1);
}

/**
 * @brief Precompute RoPE cos/sin cache
 */
__global__ void rope_precompute_cache_kernel(
    float* TC_RESTRICT cos_cache,  // [max_seq, head_dim/2]
    float* TC_RESTRICT sin_cache,  // [max_seq, head_dim/2]
    int max_seq_len,
    int head_dim,
    float base = 10000.0f) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_head_dim = head_dim / 2;
    const int total = max_seq_len * half_head_dim;
    
    if (idx >= total) return;
    
    const int d = idx % half_head_dim;
    const int pos = idx / half_head_dim;
    
    // Compute frequency: 1 / (base^(2d/head_dim))
    const float freq = 1.0f / powf(base, (2.0f * d) / head_dim);
    const float angle = pos * freq;
    
    cos_cache[idx] = cosf(angle);
    sin_cache[idx] = sinf(angle);
}

// ============================================================================
// Simplified PagedAttention Kernel
// ============================================================================

/**
 * @brief PagedAttention kernel for non-contiguous KV cache
 * 
 * Supports paged memory layout for efficient KV cache management.
 */
template<typename T, int BLOCK_SIZE = 16, int HEAD_DIM = 64>
__global__ void paged_attention_kernel(
    const T* TC_RESTRICT Q,           // [batch, num_heads, head_dim]
    const T* TC_RESTRICT K_cache,     // [num_blocks, block_size, num_heads, head_dim]
    const T* TC_RESTRICT V_cache,     // [num_blocks, block_size, num_heads, head_dim]
    T* TC_RESTRICT O,                 // [batch, num_heads, head_dim]
    const int* TC_RESTRICT block_tables,  // [batch, max_blocks_per_seq]
    const int* TC_RESTRICT seq_lens,      // [batch]
    int batch_size,
    int num_heads,
    int max_blocks_per_seq,
    float scale) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const int seq_len = seq_lens[batch_idx];
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Load query for this batch/head
    __shared__ float q_shared[HEAD_DIM];
    const T* q_ptr = Q + (batch_idx * num_heads + head_idx) * HEAD_DIM;
    
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        q_shared[d] = to_float(q_ptr[d]);
    }
    __syncthreads();
    
    // Online softmax variables
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float o_acc[HEAD_DIM];
    
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        o_acc[d] = 0.0f;
    }
    
    // Iterate over KV cache blocks
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const int physical_block = block_tables[batch_idx * max_blocks_per_seq + block_idx];
        const int block_start = block_idx * BLOCK_SIZE;
        const int block_end = min(block_start + BLOCK_SIZE, seq_len);
        
        // Process each position in the block
        for (int pos = block_start; pos < block_end; ++pos) {
            const int pos_in_block = pos - block_start;
            
            // Load K for this position
            const T* k_ptr = K_cache + ((physical_block * BLOCK_SIZE + pos_in_block) * num_heads + head_idx) * HEAD_DIM;
            const T* v_ptr = V_cache + ((physical_block * BLOCK_SIZE + pos_in_block) * num_heads + head_idx) * HEAD_DIM;
            
            // Compute attention score
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += q_shared[d] * to_float(k_ptr[d]);
            }
            score *= scale;
            
            // Online softmax update
            float m_curr = fmaxf(m_prev, score);
            float l_curr = l_prev * expf(m_prev - m_curr) + expf(score - m_curr);
            
            float scale_prev = (l_prev > 0.0f) ? (l_prev * expf(m_prev - m_curr) / l_curr) : 0.0f;
            float scale_curr = expf(score - m_curr) / l_curr;
            
            // Update output
            for (int d = 0; d < HEAD_DIM; ++d) {
                o_acc[d] = o_acc[d] * scale_prev + scale_curr * to_float(v_ptr[d]);
            }
            
            m_prev = m_curr;
            l_prev = l_curr;
        }
    }
    
    // Write output
    T* o_ptr = O + (batch_idx * num_heads + head_idx) * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        o_ptr[d] = from_float<T>(o_acc[d]);
    }
}

// ============================================================================
// MoE Router Kernel
// ============================================================================

/**
 * @brief MoE (Mixture of Experts) top-k router kernel
 * 
 * Selects top-k experts for each token based on gating scores.
 */
template<typename T, int MAX_EXPERTS = 8>
__global__ void moe_router_kernel(
    const T* TC_RESTRICT gate_logits,  // [batch, num_experts]
    int* TC_RESTRICT expert_indices,    // [batch, top_k]
    float* TC_RESTRICT expert_weights,  // [batch, top_k]
    int batch_size,
    int num_experts,
    int top_k) {
    
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    const T* logits = gate_logits + batch_idx * num_experts;
    int* indices = expert_indices + batch_idx * top_k;
    float* weights = expert_weights + batch_idx * top_k;
    
    // Load logits and find top-k
    float scores[MAX_EXPERTS];
    int expert_ids[MAX_EXPERTS];
    
    for (int e = 0; e < num_experts && e < MAX_EXPERTS; ++e) {
        scores[e] = to_float(logits[e]);
        expert_ids[e] = e;
    }
    
    // Simple selection sort for top-k (efficient for small k)
    for (int i = 0; i < top_k; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < num_experts && j < MAX_EXPERTS; ++j) {
            if (scores[j] > scores[max_idx]) {
                max_idx = j;
            }
        }
        // Swap
        float tmp_score = scores[i];
        int tmp_id = expert_ids[i];
        scores[i] = scores[max_idx];
        expert_ids[i] = expert_ids[max_idx];
        scores[max_idx] = tmp_score;
        expert_ids[max_idx] = tmp_id;
    }
    
    // Compute softmax over top-k
    float max_score = scores[0];
    float sum_exp = 0.0f;
    
    for (int i = 0; i < top_k; ++i) {
        sum_exp += expf(scores[i] - max_score);
    }
    
    // Store results
    for (int i = 0; i < top_k; ++i) {
        indices[i] = expert_ids[i];
        weights[i] = expf(scores[i] - max_score) / sum_exp;
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

/**
 * @brief Launch FlashAttention kernel
 */
template<typename T>
void launch_flash_attention(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale,
    cudaStream_t stream = nullptr) {
    
    if (batch_size == 0 || seq_len == 0) return;
    
    // Use fixed head_dim=64 for now
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int HEAD_DIM = 64;
    
    dim3 block(HEAD_DIM, BLOCK_M);
    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, 1, batch_size * num_heads);
    
    flash_attention_kernel<T, BLOCK_M, BLOCK_N, HEAD_DIM><<<grid, block, 0, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, scale);
    
    TC_CUDA_CHECK_LAST();
}

/**
 * @brief Launch RoPE kernel
 */
template<typename T>
void launch_rope(
    T* x,
    const float* cos_cache,
    const float* sin_cache,
    int batch_size, int seq_len, int num_heads, int head_dim,
    int start_pos = 0,
    cudaStream_t stream = nullptr) {
    
    const int total = batch_size * seq_len * num_heads * (head_dim / 2);
    if (total == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    rope_kernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        x, cos_cache, sin_cache, batch_size, seq_len, num_heads, head_dim, start_pos);
    
    TC_CUDA_CHECK_LAST();
}

/**
 * @brief Precompute RoPE cache
 */
inline void precompute_rope_cache(
    float* cos_cache,
    float* sin_cache,
    int max_seq_len,
    int head_dim,
    float base = 10000.0f,
    cudaStream_t stream = nullptr) {
    
    const int total = max_seq_len * (head_dim / 2);
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    rope_precompute_cache_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        cos_cache, sin_cache, max_seq_len, head_dim, base);
    
    TC_CUDA_CHECK_LAST();
}

/**
 * @brief Launch MoE router kernel
 */
template<typename T>
void launch_moe_router(
    const T* gate_logits,
    int* expert_indices,
    float* expert_weights,
    int batch_size, int num_experts, int top_k,
    cudaStream_t stream = nullptr) {
    
    if (batch_size == 0) return;
    
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    moe_router_kernel<T, 8><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        gate_logits, expert_indices, expert_weights, batch_size, num_experts, top_k);
    
    TC_CUDA_CHECK_LAST();
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// FlashAttention
template<typename T>
void flash_attention(
    const T* Q, const T* K, const T* V, T* O,
    size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
    cudaStream_t stream = nullptr) {
    
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    launch_flash_attention(Q, K, V, O, 
                          static_cast<int>(batch_size),
                          static_cast<int>(num_heads),
                          static_cast<int>(seq_len),
                          static_cast<int>(head_dim),
                          scale, stream);
}

/// RoPE
template<typename T>
void rope(
    T* x,
    const float* cos_cache,
    const float* sin_cache,
    size_t batch_size, size_t seq_len, size_t num_heads, size_t head_dim,
    int start_pos = 0,
    cudaStream_t stream = nullptr) {
    
    launch_rope(x, cos_cache, sin_cache,
               static_cast<int>(batch_size),
               static_cast<int>(seq_len),
               static_cast<int>(num_heads),
               static_cast<int>(head_dim),
               start_pos, stream);
}

} // namespace kernels
} // namespace tensorcraft
