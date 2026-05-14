# 系统架构设计

CUDA Kernel Academy 的系统架构设计遵循"渐进式学习"与"生产级实践"双重原则。本文档深入解析整体架构、模块依赖关系、构建系统设计以及核心设计模式。

## 设计哲学

### 核心定位

本项目定位于 **"介于入门示例与生产框架之间"** 的学习型项目：

- **非玩具项目**：代码质量、错误处理、API 设计遵循生产标准
- **非重量级框架**：保持可读性，避免过度抽象
- **渐进式路径**：从 naive 实现逐步演进到优化版本

### 架构原则

```mermaid
mindmap
  root((设计原则))
    可读性优先
      清晰的命名
      详尽的注释
      渐进式复杂度
    生产级质量
      RAII 资源管理
      统一错误处理
      完善的测试
    模块化设计
      松耦合接口
      Header-only 库
      可独立构建
    性能导向
      多架构支持
      自动调优
      基准测试
```

## 模块架构

### 整体依赖关系

```mermaid
flowchart TB
    subgraph 学习路径
        M1[01-sgemm-tutorial]
        M2[02-tensorcraft-core]
        M3[03-hpc-advanced]
        M4[04-inference-engine]
    end
    
    M1 --> M2
    M2 --> M3
    M2 --> M4
    M3 --> M4
    
    subgraph 外部依赖
        CUDA[CUDA Runtime]
        cuBLAS[cuBLAS]
        CUTLASS[CUTLASS]
        GTest[GoogleTest]
        pybind[pybind11]
    end
    
    M1 --> CUDA
    M1 --> cuBLAS
    M1 --> GTest
    M2 --> CUDA
    M2 --> pybind
    M2 --> GTest
    M3 --> CUDA
    M3 --> CUTLASS
    M3 --> GTest
    M4 --> M2
    M4 --> CUDA
    M4 --> cuBLAS
    M4 --> GTest
```

### 模块职责划分

| 模块 | 职责 | 构建系统 | 关键特性 |
|------|------|----------|----------|
| **01-sgemm-tutorial** | SGEMM 优化入门 | Makefile | 独立、自包含、教学导向 |
| **02-tensorcraft-core** | 可复用算子库 | CMake | Header-only、多架构、Python 绑定 |
| **03-hpc-advanced** | 高级优化模式 | CMake | CUTLASS 集成、CUDA 12/13 特性 |
| **04-inference-engine** | 端到端推理 | CMake | 内存池、流管理、AutoTuner |

## 构建系统设计

### 双构建系统策略

项目采用 **混合构建系统** 设计：

```
项目根目录
├── CMakeLists.txt          # 主构建系统（模块 02-04）
├── cmake/                  # 共享 CMake 模块
│   ├── CudaArch.cmake      # CUDA 架构检测
│   ├── Dependencies.cmake  # 依赖管理
│   └── Testing.cmake       # 测试配置
└── 01-sgemm-tutorial/
    └── Makefile            # 独立构建系统（教学导向）
```

#### 为什么保留 Makefile？

```mermaid
flowchart LR
    subgraph Makefile 优势
        A1[零依赖学习]
        A2[直接命令]
        A3[快速迭代]
    end
    
    subgraph CMake 优势
        B1[跨平台]
        B2[依赖管理]
        B3[IDE 集成]
    end
    
    A1 --> 教学模块
    A2 --> 教学模块
    A3 --> 教学模块
    
    B1 --> 生产模块
    B2 --> 生产模块
    B3 --> 生产模块
```

### CMake 预设配置

```json
{
  "configurePresets": [
    {
      "name": "default",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CUDAARCHS": "70;75;80;86",
        "BUILD_TENSORCRAFT": "ON",
        "BUILD_HPC_ADVANCED": "ON",
        "BUILD_INFERENCE_ENGINE": "ON"
      }
    },
    {
      "name": "minimal",
      "cacheVariables": {
        "BUILD_TENSORCRAFT": "ON",
        "BUILD_HPC_ADVANCED": "OFF",
        "BUILD_INFERENCE_ENGINE": "OFF"
      }
    }
  ]
}
```

## 核心设计模式

### 1. RAII 资源管理

所有 GPU 资源采用 RAII 模式管理：

```cpp
// Tensor 类：自动管理 GPU 内存
class Tensor {
public:
    Tensor(size_t size) : data_(cuda_malloc(size)), size_(size) {}
    ~Tensor() { cudaFree(data_); }
    
    // 移动语义
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // 禁止拷贝
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
private:
    void* data_;
    size_t size_;
};
```

### 2. 统一错误处理

```cpp
// 错误检查宏层级
#define TC_CUDA_CHECK(call)          // 检查返回值
#define TC_CUDA_CHECK_LAST()         // 检查最后一个错误
#define TC_CUDA_SYNC_CHECK()         // 同步并检查（调试用）

// 使用示例
TC_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
TC_CUDA_SYNC_CHECK();  // 调试时捕获异步错误
```

### 3. 版本枚举模式

```cpp
enum class GemmVersion {
    Naive,           // 基础实现
    Tiled,           // 分块优化
    DoubleBuffer,    // 双缓冲
    TensorCore,      // Tensor Core 加速
    Auto             // 自动选择最优
};

// 运行时选择
template<GemmVersion V>
void gemm(const Tensor& A, const Tensor& B, Tensor& C);
```

### 4. 内存池模式

```mermaid
sequenceDiagram
    participant App as 应用层
    participant Pool as MemoryPool
    participant Cache as 缓存块
    participant GPU as GPU 内存
    
    App->>Pool: allocate(size)
    Pool->>Cache: 查找合适块
    
    alt 命中缓存
        Cache-->>Pool: 返回缓存块
    else 未命中
        Pool->>GPU: cudaMalloc(size)
        GPU-->>Pool: 新块
    end
    
    Pool-->>App: 内存指针
    
    Note over App,GPU: 使用完毕
    
    App->>Pool: deallocate(ptr)
    Pool->>Cache: 归还缓存（不释放）
```

## 内存层级优化策略

### GPU 内存层级

```mermaid
flowchart TB
    subgraph 内存层级
        HBM[HBM<br/>高带宽内存<br/>~1TB/s]
        L2[L2 Cache<br/>~40MB]
        SMEM[Shared Memory<br/>每 SM 48-164KB]
        REG[寄存器<br/>每线程 255 个]
    end
    
    HBM -->|全局访问| L2
    L2 -->|缓存| SMEM
    SMEM -->|bank 访问| REG
    
    subgraph 优化策略
        S1[数据重用 → Shared Memory]
        S2[延迟隐藏 → 双缓冲]
        S3[带宽优化 → 合并访问]
        S4[bank 冲突 → 填充]
    end
```

### 优化约束计算

GEMM 分块参数必须满足以下约束：

```
threads_per_block = (BM/TM) * (BN/TN) ≤ 1024
shared_memory = (BM*BK + BK*BN) * sizeof(float) ≤ 48KB
registers_per_thread = TM*TN + TM + TN + overhead ≤ 255
```

典型配置示例：

| 配置 | BM×BN×BK | TM×TN | 线程数 | Shared Mem |
|------|----------|-------|--------|------------|
| 小块 | 64×64×8 | 4×4 | 256 | 8KB |
| 中块 | 128×128×8 | 8×8 | 256 | 32KB |
| 大块 | 128×128×16 | 8×8 | 256 | 64KB (动态) |

## 测试策略

### 测试层级

```mermaid
flowchart TB
    subgraph 测试金字塔
        E2E[端到端测试<br/>推理引擎集成]
        INT[集成测试<br/>模块间交互]
        UNIT[单元测试<br/>单个 kernel]
        PROP[属性测试<br/>参数化验证]
    end
    
    E2E --> INT
    INT --> UNIT
    UNIT --> PROP
    
    subgraph 工具
        GTest[GoogleTest]
        GBench[Google Benchmark]
        RapidCheck[RapidCheck]
    end
    
    UNIT --> GTest
    E2E --> GTest
    PROP --> RapidCheck
```

### 正确性验证

所有 GEMM 实现必须通过 cuBLAS 对比验证：

```cpp
TEST(SgemmTest, Correctness) {
    // 生成随机矩阵
    auto A = random_matrix(M, K);
    auto B = random_matrix(K, N);
    
    // 计算
    auto C_custom = gemm_custom(A, B);
    auto C_cublas = gemm_cublas(A, B);
    
    // 验证相对误差
    EXPECT_NEAR(C_custom, C_cublas, 1e-5);
}
```

## 扩展性设计

### 添加新 Kernel 的流程

```mermaid
flowchart LR
    A[实现 kernel] --> B[添加版本枚举]
    B --> C[编写单元测试]
    C --> D[添加基准测试]
    D --> E[更新文档]
    E --> F[CI 验证]
```

### 插件式架构

TensorCraft Core 支持运行时加载自定义 kernel：

```cpp
// kernel 注册机制
struct KernelRegistry {
    template<typename Func>
    void register_kernel(const std::string& name, Func func);
    
    auto get_kernel(const std::string& name);
};

// 使用示例
registry.register_kernel("custom_gemm", my_custom_gemm);
```

---

## 总结

CUDA Kernel Academy 的架构设计平衡了 **教学清晰度** 与 **工程质量**：

1. **模块化**：清晰的职责划分，渐进式依赖
2. **双构建系统**：教学用 Makefile + 生产用 CMake
3. **RAII 优先**：所有资源自动管理，无内存泄漏
4. **可测试性**：单元测试、集成测试、属性测试三层覆盖
5. **可扩展性**：版本枚举、注册机制、插件架构

这套架构既适合学习 CUDA 优化原理，也可作为生产级代码的参考模板。
