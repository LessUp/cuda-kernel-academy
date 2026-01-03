# FlashAttention 详解

本文档详细介绍 FlashAttention 的原理和实现。

## 1. 标准 Attention 的问题

### 标准 Attention 公式

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

其中:
- Q: (batch, heads, seq_len, head_dim)
- K: (batch, heads, seq_len, head_dim)
- V: (batch, heads, seq_len, head_dim)

### 内存问题

```
Q @ K^T 产生 (seq_len × seq_len) 的注意力矩阵

seq_len = 2048, heads = 32, batch = 8
注意力矩阵大小 = 8 × 32 × 2048 × 2048 × 4 bytes = 4.3 GB!
```

### IO 瓶颈

```
标准实现:
1. 计算 S = Q @ K^T          → 写入 HBM (N×N)
2. 计算 P = softmax(S)       → 读取 HBM, 写入 HBM (N×N)
3. 计算 O = P @ V            → 读取 HBM (N×N)

总 HBM 访问: O(N²) 次
```

---

## 2. FlashAttention 核心思想

### Tiling + Online Softmax

**核心思想**: 将 Q, K, V 分块加载到 SRAM (Shared Memory)，在 SRAM 中完成所有计算，避免写入 N×N 矩阵到 HBM。

```
FlashAttention:
1. 分块加载 Q, K, V 到 SRAM
2. 在 SRAM 中计算 attention score
3. 使用 Online Softmax 增量更新输出
4. 只写入最终输出 O 到 HBM

总 HBM 访问: O(N) 次
```

### 分块策略

```
Q: (seq_len, head_dim) → 分成 Tr 个块, 每块 Br 行
K: (seq_len, head_dim) → 分成 Tc 个块, 每块 Bc 行
V: (seq_len, head_dim) → 分成 Tc 个块, 每块 Bc 行

Br, Bc 选择使得块能放入 SRAM
```

---

## 3. 算法详解

### FlashAttention Forward 伪代码

```python
def flash_attention_forward(Q, K, V, Br, Bc):
    N, d = Q.shape
    Tr = ceil(N / Br)  # Q 的块数
    Tc = ceil(N / Bc)  # K, V 的块数
    
    # 初始化输出和统计量
    O = zeros(N, d)
    l = zeros(N)  # softmax 分母
    m = full(N, -inf)  # 当前最大值
    
    # 外层循环: 遍历 K, V 的块
    for j in range(Tc):
        Kj = K[j*Bc : (j+1)*Bc]  # 加载 K 块到 SRAM
        Vj = V[j*Bc : (j+1)*Bc]  # 加载 V 块到 SRAM
        
        # 内层循环: 遍历 Q 的块
        for i in range(Tr):
            Qi = Q[i*Br : (i+1)*Br]  # 加载 Q 块到 SRAM
            
            # 在 SRAM 中计算 attention score
            Sij = Qi @ Kj.T / sqrt(d)  # (Br, Bc)
            
            # Online Softmax 更新
            m_new = max(m[i*Br:(i+1)*Br], rowmax(Sij))
            P_tilde = exp(Sij - m_new[:, None])
            l_new = exp(m[i*Br:(i+1)*Br] - m_new) * l[i*Br:(i+1)*Br] + rowsum(P_tilde)
            
            # 更新输出
            O[i*Br:(i+1)*Br] = (
                exp(m[i*Br:(i+1)*Br] - m_new)[:, None] * O[i*Br:(i+1)*Br] +
                P_tilde @ Vj
            ) / l_new[:, None] * l[i*Br:(i+1)*Br][:, None]
            
            # 更新统计量
            m[i*Br:(i+1)*Br] = m_new
            l[i*Br:(i+1)*Br] = l_new
    
    return O
```

### Online Softmax 在 FlashAttention 中的应用

```
处理第 j 个 K, V 块时:

1. 计算当前块的 attention score: S_ij = Q_i @ K_j^T

2. 更新最大值:
   m_new = max(m_old, rowmax(S_ij))

3. 计算缩放后的 exp:
   P_tilde = exp(S_ij - m_new)

4. 更新 softmax 分母:
   l_new = l_old * exp(m_old - m_new) + rowsum(P_tilde)

5. 更新输出 (关键!):
   O_new = (O_old * l_old * exp(m_old - m_new) + P_tilde @ V_j) / l_new
```

---

## 4. CUDA 实现

### Kernel 结构

```cpp
__global__ void flash_attention_forward_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int batch, int heads, int seq_len, int head_dim,
    int Br, int Bc) {
    
    // 每个 Block 处理一个 (batch, head, Q_block)
    int batch_idx = blockIdx.x / heads;
    int head_idx = blockIdx.x % heads;
    int q_block_idx = blockIdx.y;
    
    // Shared Memory 分配
    extern __shared__ float smem[];
    float* Qi = smem;                           // (Br, head_dim)
    float* Kj = Qi + Br * head_dim;             // (Bc, head_dim)
    float* Vj = Kj + Bc * head_dim;             // (Bc, head_dim)
    float* Sij = Vj + Bc * head_dim;            // (Br, Bc)
    float* Oi = Sij + Br * Bc;                  // (Br, head_dim)
    float* li = Oi + Br * head_dim;             // (Br,)
    float* mi = li + Br;                        // (Br,)
    
    // 初始化
    // ...
    
    // 加载 Q 块
    load_tile(Q, Qi, batch_idx, head_idx, q_block_idx * Br, Br, head_dim);
    
    // 遍历 K, V 块
    int num_kv_blocks = (seq_len + Bc - 1) / Bc;
    for (int j = 0; j < num_kv_blocks; ++j) {
        // 加载 K, V 块
        load_tile(K, Kj, batch_idx, head_idx, j * Bc, Bc, head_dim);
        load_tile(V, Vj, batch_idx, head_idx, j * Bc, Bc, head_dim);
        __syncthreads();
        
        // 计算 S = Q @ K^T
        compute_qk(Qi, Kj, Sij, Br, Bc, head_dim);
        __syncthreads();
        
        // Online Softmax 更新
        online_softmax_update(Sij, Vj, Oi, li, mi, Br, Bc, head_dim);
        __syncthreads();
    }
    
    // 写回输出
    store_tile(Oi, O, batch_idx, head_idx, q_block_idx * Br, Br, head_dim);
}
```

### Warp 分配策略 (FlashAttention-2)

```
FlashAttention-1: 每个 Block 处理一个 Q 块
FlashAttention-2: 更细粒度的 Warp 分配

Block 内 Warp 分配:
- Warp 0-3: 计算 S = Q @ K^T 的不同部分
- Warp 4-7: 计算 P @ V 的不同部分

优势:
- 更好的并行度
- 减少同步开销
```

---

## 5. 内存分析

### SRAM 使用

```
Br = 64, Bc = 64, head_dim = 128

Qi: 64 × 128 × 4 = 32 KB
Kj: 64 × 128 × 4 = 32 KB
Vj: 64 × 128 × 4 = 32 KB
Sij: 64 × 64 × 4 = 16 KB
Oi: 64 × 128 × 4 = 32 KB
li, mi: 64 × 4 × 2 = 0.5 KB

总计: ~145 KB (需要 Hopper 的大 SRAM)
```

### HBM 访问对比

```
标准 Attention:
- 读 Q, K, V: 3 × N × d
- 写 S: N × N
- 读 S, 写 P: 2 × N × N
- 读 P, V, 写 O: N × N + N × d + N × d
总计: O(N²)

FlashAttention:
- 读 Q: N × d (一次)
- 读 K, V: Tr × N × d (每个 Q 块读一次)
- 写 O: N × d (一次)
总计: O(N × d × Tr) = O(N²/Br) ≈ O(N) (当 Br 足够大)
```

---

## 6. 性能优化技巧

### 1. 使用 Tensor Core

```cpp
// 使用 WMMA 加速 Q @ K^T
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;

wmma::load_matrix_sync(q_frag, Qi + ...);
wmma::load_matrix_sync(k_frag, Kj + ...);
wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
```

### 2. 异步数据加载 (CUDA 11+)

```cpp
// 使用 cp.async 异步加载
__pipeline_memcpy_async(Kj + offset, K + global_offset, sizeof(float4));
__pipeline_commit();
// ... 计算 ...
__pipeline_wait_prior(0);
```

### 3. 寄存器优化

```cpp
// 将频繁访问的数据放入寄存器
float reg_o[8];  // 每个线程的输出累加器
float reg_l, reg_m;  // 每个线程的统计量
```

---

## 7. Causal Mask 支持

### Causal Attention

```
Causal Mask: 只关注当前位置及之前的 token

Mask[i][j] = 1 if j <= i else 0
```

### 实现优化

```cpp
// 跳过不需要计算的块
if (j * Bc > (q_block_idx + 1) * Br) {
    continue;  // K 块完全在 Q 块之后，跳过
}

// 块内 Mask
for (int i = 0; i < Br; ++i) {
    for (int j = 0; j < Bc; ++j) {
        int q_pos = q_block_idx * Br + i;
        int k_pos = kv_block_idx * Bc + j;
        if (k_pos > q_pos) {
            Sij[i * Bc + j] = -INFINITY;
        }
    }
}
```

---

## 8. 性能对比

### 内存使用

| 实现 | seq_len=2048 | seq_len=4096 | seq_len=8192 |
|------|--------------|--------------|--------------|
| 标准 | 4.3 GB | 17.2 GB | 68.7 GB |
| FlashAttention | 0.5 GB | 0.5 GB | 0.5 GB |

### 速度对比

| 实现 | seq_len=2048 | seq_len=4096 |
|------|--------------|--------------|
| PyTorch | 45 ms | 180 ms |
| FlashAttention | 12 ms | 48 ms |
| 加速比 | 3.75× | 3.75× |

---

## 9. 参考资料

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [Online Softmax Trick](https://arxiv.org/abs/2112.05682)
