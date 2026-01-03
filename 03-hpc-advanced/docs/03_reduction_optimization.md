# 归约优化详解

本文档详细介绍 CUDA 归约操作的优化技术，包括 Warp Shuffle、Block Reduce 和 Online Softmax。

## 1. 归约基础

### 什么是归约？

归约是将一组数据通过某种操作（如求和、求最大值）合并为单个值的过程。

```
输入: [1, 2, 3, 4, 5, 6, 7, 8]
求和归约: 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
```

### Naive 实现 (原子操作)

```cpp
__global__ void reduce_naive(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(output, input[idx]);  // 极慢！所有线程竞争同一地址
    }
}
```

**问题**: 原子操作串行化，性能极差。

---

## 2. Warp Shuffle 归约

### Warp Shuffle 原语

Warp Shuffle 允许 Warp 内线程直接交换寄存器数据，无需 Shared Memory。

```cpp
// __shfl_down_sync: 从高 lane 获取数据
// mask: 参与的线程掩码 (0xffffffff = 所有 32 个线程)
// var: 要交换的变量
// delta: 偏移量
float val = __shfl_down_sync(0xffffffff, var, delta);
```

### Warp 级归约

```cpp
__device__ float warp_reduce_sum(float val) {
    // 二分归约
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // 只有 lane 0 有正确结果
}
```

### 归约过程示意

```
初始:  [0] [1] [2] [3] [4] [5] [6] [7] ... [31]

offset=16:
       [0+16] [1+17] [2+18] ... [15+31] [16] [17] ... [31]

offset=8:
       [0+8+16+24] [1+9+17+25] ... [7+15+23+31] ...

offset=4:
       [0+4+8+12+16+20+24+28] ...

offset=2:
       [0+2+4+6+8+10+12+14+16+18+20+22+24+26+28+30] ...

offset=1:
       [sum of all 32 elements] ...
```

---

## 3. Block 级归约

### 结合 Warp Shuffle 和 Shared Memory

```cpp
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // 每个 Warp 一个槽位
    
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    
    // Step 1: Warp 内归约
    val = warp_reduce_sum(val);
    
    // Step 2: Warp 0 的 lane 0 写入 Shared Memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    // Step 3: 第一个 Warp 归约所有 Warp 的结果
    if (warp_id == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;  // 只有线程 0 有正确结果
}
```

### 完整的 Block Reduce Kernel

```cpp
__global__ void reduce_block(const float* input, float* output, int n) {
    float sum = 0.0f;
    
    // Grid Stride Loop 累加
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += input[i];
    }
    
    // Block 内归约
    sum = block_reduce_sum(sum);
    
    // Block 0 的线程 0 写入结果
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);  // 只有少量 Block 竞争
    }
}
```

---

## 4. Softmax 优化

### Softmax 公式

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

### Naive 实现 (三次遍历)

```cpp
// Pass 1: 找最大值
float max_val = -INFINITY;
for (int i = 0; i < n; ++i) {
    max_val = fmaxf(max_val, input[i]);
}

// Pass 2: 计算 exp 和求和
float sum = 0.0f;
for (int i = 0; i < n; ++i) {
    sum += expf(input[i] - max_val);
}

// Pass 3: 归一化
for (int i = 0; i < n; ++i) {
    output[i] = expf(input[i] - max_val) / sum;
}
```

### Online Softmax (单次遍历)

**核心思想**: 在遍历过程中同时维护 max 和 sum。

```cpp
__device__ void online_softmax(const float* input, float* output, int n) {
    float max_val = -INFINITY;
    float sum = 0.0f;
    
    // 单次遍历: 同时计算 max 和 sum
    for (int i = 0; i < n; ++i) {
        float x = input[i];
        float old_max = max_val;
        max_val = fmaxf(max_val, x);
        
        // 关键: 调整之前的 sum
        // sum = sum * exp(old_max - new_max) + exp(x - new_max)
        sum = sum * expf(old_max - max_val) + expf(x - max_val);
    }
    
    // 归一化
    for (int i = 0; i < n; ++i) {
        output[i] = expf(input[i] - max_val) / sum;
    }
}
```

### Online Softmax 数学推导

```
设 S_k = sum_{i=1}^{k} exp(x_i - m_k), 其中 m_k = max_{i=1}^{k} x_i

当处理 x_{k+1} 时:
- 如果 x_{k+1} > m_k:
  m_{k+1} = x_{k+1}
  S_{k+1} = S_k * exp(m_k - m_{k+1}) + exp(x_{k+1} - m_{k+1})
          = S_k * exp(m_k - x_{k+1}) + 1

- 如果 x_{k+1} <= m_k:
  m_{k+1} = m_k
  S_{k+1} = S_k + exp(x_{k+1} - m_k)
```

### 并行 Online Softmax

```cpp
__global__ void softmax_online(const float* input, float* output, int batch, int seq_len) {
    int batch_idx = blockIdx.x;
    const float* row = input + batch_idx * seq_len;
    float* out_row = output + batch_idx * seq_len;
    
    // 每个线程处理部分元素
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float x = row[i];
        float old_max = local_max;
        local_max = fmaxf(local_max, x);
        local_sum = local_sum * expf(old_max - local_max) + expf(x - local_max);
    }
    
    // Warp 内合并 (max, sum) 对
    // 需要特殊处理: 合并两个 (max, sum) 对
    // ...
    
    // Block 内合并
    // ...
    
    // 最终归一化
    // ...
}
```

---

## 5. LayerNorm 优化

### LayerNorm 公式

```
y = (x - mean) / sqrt(var + eps) * gamma + beta

mean = sum(x) / n
var = sum((x - mean)^2) / n
```

### Welford 算法 (数值稳定)

```cpp
__device__ void welford_update(float& mean, float& m2, float& count, float x) {
    count += 1.0f;
    float delta = x - mean;
    mean += delta / count;
    float delta2 = x - mean;
    m2 += delta * delta2;
}

__device__ void welford_combine(float& mean1, float& m2_1, float& count1,
                                 float mean2, float m2_2, float count2) {
    if (count2 == 0) return;
    
    float count = count1 + count2;
    float delta = mean2 - mean1;
    mean1 = (count1 * mean1 + count2 * mean2) / count;
    m2_1 = m2_1 + m2_2 + delta * delta * count1 * count2 / count;
    count1 = count;
}
```

### 优化的 LayerNorm Kernel

```cpp
__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta,
                                   float* output, int batch, int hidden_size, float eps) {
    int batch_idx = blockIdx.x;
    const float* x = input + batch_idx * hidden_size;
    float* y = output + batch_idx * hidden_size;
    
    // Welford 算法计算 mean 和 variance
    float mean = 0.0f, m2 = 0.0f, count = 0.0f;
    
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        welford_update(mean, m2, count, x[i]);
    }
    
    // Warp 内合并
    // Block 内合并
    // ...
    
    float var = m2 / count;
    float inv_std = rsqrtf(var + eps);
    
    // 归一化
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
```

---

## 6. RMSNorm 优化

### RMSNorm 公式

```
y = x / sqrt(mean(x^2) + eps) * gamma

RMS = sqrt(sum(x^2) / n)
```

### RMSNorm 比 LayerNorm 更简单

- 不需要计算 mean
- 只需要计算平方和

```cpp
__global__ void rms_norm_kernel(const float* input, const float* gamma,
                                 float* output, int batch, int hidden_size, float eps) {
    int batch_idx = blockIdx.x;
    const float* x = input + batch_idx * hidden_size;
    float* y = output + batch_idx * hidden_size;
    
    // 计算平方和
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        sum_sq += x[i] * x[i];
    }
    
    // Block 归约
    sum_sq = block_reduce_sum(sum_sq);
    
    // 广播 RMS
    __shared__ float rms;
    if (threadIdx.x == 0) {
        rms = rsqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();
    
    // 归一化
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y[i] = x[i] * rms * gamma[i];
    }
}
```

---

## 7. 性能对比

### Softmax 性能

| 实现 | 遍历次数 | 相对性能 |
|------|----------|----------|
| Naive (原子操作) | 3 | 1× |
| Warp Shuffle | 3 | 10× |
| Online Softmax | 2 | 15× |
| Fused Online | 1 | 20× |

### LayerNorm vs RMSNorm

| 操作 | 计算量 | 相对性能 |
|------|--------|----------|
| LayerNorm | mean + var + norm | 1× |
| RMSNorm | rms + norm | 1.3× |

---

## 8. 最佳实践

1. **优先使用 Warp Shuffle**: 比 Shared Memory 更快
2. **使用 Online 算法**: 减少遍历次数
3. **Welford 算法**: 数值稳定的方差计算
4. **融合操作**: 减少 Kernel 启动和内存访问

## 参考资料

- [Online Softmax Paper](https://arxiv.org/abs/2205.14135)
- [Welford's Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance)
- [NVIDIA CUB Library](https://nvlabs.github.io/cub/)
