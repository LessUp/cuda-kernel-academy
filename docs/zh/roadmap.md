---
outline: [2, 3]
---

# 学习路线图

CUDA Kernel Academy 提供从入门到进阶的系统性 CUDA 算子工程学习路径。

## 学习阶段

```mermaid
journey
    title CUDA Kernel Academy 学习路径
    section 第一阶段：基础
      SGEMM 入门: 5: 掌握 Naive GEMM
      Shared Memory: 5: 理解 Tiling 和 Bank Conflict
      Double Buffer: 4: 隐藏内存延迟
    section 第二阶段：核心库
      TensorCraft Core: 5: 工业级 API 设计
      现代 C++ CUDA: 4: Concepts / constexpr if
      算子融合: 4: Kernel Fusion 和 Epilogue
    section 第三阶段：进阶 HPC
      GEMM 优化: 5: Register Tiling / WMMA
      FlashAttention: 4: Tiling + Online Softmax
      CUDA 13 特性: 3: TMA / Cluster / FP8
    section 第四阶段：推理系统
      Inference Engine: 5: 内存池 / Stream / Tensor
      性能调优: 4: AutoTuner / Profiler
      量化部署: 3: INT8 / FP8 推理
```

## 阶段详情

### 第一阶段：Fundamentals（约 2-3 周）

**预计学习时长**: 40-60 小时

**关键技能点**:

- CUDA 编程模型（Grid / Block / Thread / Warp）
- 全局内存 vs Shared Memory
- 合并访问与 Bank Conflict
- 双缓冲技术
- 基本性能分析（nsys / ncu）

**目标**: 能够独立编写并优化一个 Tiled GEMM，达到 cuBLAS 10-20% 性能。

### 第二阶段：Core Library（约 2-3 周）

**预计学习时长**: 40-60 小时

**关键技能点**:

- Header-Only 库设计
- C++17/20 现代特性在 CUDA 中的应用
- 模板元编程与编译时优化
- 算子融合（GEMM + Bias + Activation）
- 内存池与 RAII 资源管理

**目标**: 能够设计和实现一个可复用的高性能 CUDA 算子库。

### 第三阶段：Advanced HPC（约 3-4 周）

**预计学习时长**: 60-80 小时

**关键技能点**:

- Register Tiling 与软件流水线
- Tensor Core (WMMA / MMA PTX)
- FlashAttention 算法与实现
- 归约优化（Warp Shuffle / Online Softmax）
- Hopper 新特性（TMA / Cluster / FP8）

**目标**: 能够针对特定 GPU 架构编写接近 cuBLAS 80%+ 性能的生产级 kernel。

### 第四阶段：Inference System（约 2-3 周）

**预计学习时长**: 40-60 小时

**关键技能点**:

- 推理引擎架构设计
- 多 Stream 并发与异步执行
- 内存池与缓存策略
- 量化推理（INT8 / FP8）
- 端到端性能 profiling 与调优

**目标**: 能够将优化后的 kernel 集成到完整的推理系统中，实现端到端加速。

## 整体架构

```mermaid
flowchart TB
    subgraph M1["01-SGEMM Tutorial"]
        M1_out["Naive → Tiled → Double Buffer → Tensor Core"]
    end

    subgraph M2["02-TensorCraft Core"]
        M2_out["Reusable Kernel Library"]
    end

    subgraph M3["03-HPC Advanced"]
        M3_out["FlashAttention / TMA / FP8"]
    end

    subgraph M4["04-Inference Engine"]
        M4_out["End-to-End Inference System"]
    end

    M1 -->|"Kernel implementations"| M2
    M2 -->|"Reusable operators"| M3
    M2 -->|"Core library"| M4
    M3 -->|"Advanced kernels"| M4

    style M1 fill:#161b22,stroke:#76B900,color:#e6edf3
    style M2 fill:#161b22,stroke:#76B900,color:#e6edf3
    style M3 fill:#161b22,stroke:#76B900,color:#e6edf3
    style M4 fill:#161b22,stroke:#76B900,color:#e6edf3
    style M1_out fill:#161b22,stroke:#76B900,color:#e6edf3
    style M2_out fill:#161b22,stroke:#76B900,color:#e6edf3
    style M3_out fill:#161b22,stroke:#76B900,color:#e6edf3
    style M4_out fill:#161b22,stroke:#76B900,color:#e6edf3
```

## References

[^1]: NVIDIA. "CUDA C++ Programming Guide," v12.6. 2024. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
[^2]: Simon Boehm. "How to Optimize a CUDA Matmul Kernel." 2022. https://siboehm.com/articles/22/CUDA-MMM
[^3]: Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS* 2022. https://arxiv.org/abs/2205.14135
[^4]: NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass
