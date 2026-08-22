# 修复全部已核实 Bug 的实现计划

> **For agentic workers:** 使用 superpowers:executing-plans（本会话内联执行）逐任务实施。
> 步骤使用 checkbox（`- [ ]`）跟踪。

**Goal:** 修复此前全面复核确认的 7 严重 + 10 中等 + 13 轻微 bug，并对可优化点做最小正确性优化。

**Architecture:** 按模块分组（03-hpc-advanced / 04-inference-engine / 02-tensorcraft-core / 01-sgemm-tutorial / common+examples），每项为独立小改动；无 GPU 运行时，验证采用 CPU 模拟 + g++/nvcc 编译检查。

**Tech Stack:** C++17 / CUDA 12.0 (nvcc, sm_86)，CPU 侧验证用 g++。

**Spec:** 本计划依据 `docs/superpowers/plans/` 目录外提交的复核报告（S1-S7 严重、M1-M10 中等、L1-L13 轻微）。每任务在「Fix」中直接给出可应用代码。

## Global Constraints

- 不改动公共 API 签名（除非必要）；保持头文件库风格（inline/template）。
- 所有对维度/对齐的防御性校验：非法输入抛 `std::invalid_argument` 或 `TC_CUDA_CHECK` 前置断言，绝不静默返回错误结果。
- 不改动已由测试验证通过的正确内核逻辑（如 tiled/double-buffer GEMM 主逻辑）。
- 死代码（无调用者）路径：优先加校验使其明确失败，或补 launcher 使其可用；不静默删除导出符号。
- 不引入新依赖。

---

## Group A: 03-hpc-advanced

### Task A1: 修复 gemm_software_pipeline_kernel 线程映射（S2）

**Files:** Modify `03-hpc-advanced/src/03_gemm/gemm.cu:542-543`
**验证:** CPU 模拟索引覆盖 64×64 全部元素 + 无越界读

- [ ] **Step 1: 修正线程映射**

原：`thread_row = threadIdx.x / PIPE_TILE_N(64)`；`thread_col = threadIdx.x % 64`
错：16×16 线程数组被映射成 4×64，导致 n_idx 到 255、m_idx 只到 15。

```cuda
    // 16 x 16 thread grid, each thread computes a 4 x 4 tile -> 64 x 64 block tile
    int thread_col = threadIdx.x % (PIPE_TILE_N / 4);
    int thread_row = threadIdx.x / (PIPE_TILE_N / 4);
```

- [ ] **Step 2: 运行 CPU 模拟验证**（`/tmp/opencode/verify_s2_fixed.cpp`）
- [ ] **Step 3: 编译检查** `nvcc -arch=sm_86 -c 03-hpc-advanced/src/03_gemm/gemm.cu`（含 -I include 路径）

### Task A2: topk n>1024 静默错误 → 明确校验（S4）

**Files:** Modify `03-hpc-advanced/src/05_attention/topk.cu:63-71`
**验证:** 无 GPU，改后 launcher 对 n>1024 抛异常（阅读确认）

- [ ] **Step 1: launcher 增加上限校验**

```cpp
template <>
void topk<float>(const float* input, float* output, int* indices,
                 int batch, int n, int k, cudaStream_t stream) {
    // Kernel loads the whole row into shared memory; rows larger than 1024
    // would silently drop elements. Refuse instead of returning wrong results.
    if (n > 1024) {
        throw std::invalid_argument("topk: n > 1024 not supported (shared-memory bound)");
    }
    int block_size = n > 0 ? n : 1;
    size_t smem_size = block_size * (sizeof(float) + sizeof(int));
    topk_kernel<float><<<batch, block_size, smem_size, stream>>>(
        input, output, indices, n, k);
    CUDA_CHECK_LAST();
}
```

（同时把 `min(n, 1024)` 改为 n，因为已由上限校验保证 n<=1024；注意 k<=n 也应加校验。）

- [ ] **Step 2: 编译检查**

### Task A3: flash_attention head_dim 校验（S5）

**Files:** Modify `03-hpc-advanced/src/05_attention/flash_attention.cu:116-135`（launcher）
**验证:** 阅读确认

- [ ] **Step 1: launcher 校验 head_dim == HEAD_DIM**

在 launcher 内、启动前加：

```cpp
    if (config.head_dim != HEAD_DIM) {
        throw std::invalid_argument("flash_attention: head_dim must equal HEAD_DIM (" +
                                    std::to_string(HEAD_DIM) + ")");
    }
```

- [ ] **Step 2: 编译检查**

### Task A4: gemm WMMA 16 对齐校验（M3）

**Files:** Modify `03-hpc-advanced/src/03_gemm/gemm.cu`（`gemm<__half, GemmOpt::TensorCoreWMMA>` 与对应 float WMMA 特化，约 :436-449）
**验证:** 阅读确认

- [ ] **Step 1: 加 16 对齐校验**（与 02 gemm.hpp:374 一致）

```cpp
    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        throw std::invalid_argument("TensorCoreWMMA: M/N/K must be multiples of 16");
    }
```

- [ ] **Step 2: 编译检查**

### Task A5: vector_add/relu Vectorized n<4 空 grid（M1）

**Files:** Modify `03-hpc-advanced/src/01_elementwise/vector_add.cu:59`、`relu.cu:63`
**验证:** 阅读确认

- [ ] **Step 1: 修正 grid 计算**（n<4 时也至少 1 个 block）

```cpp
    int grid_size = std::max(1, static_cast<int>((n / 4 + block_size - 1) / block_size));
```

- [ ] **Step 2: 编译检查**

### Task A6: gemm.cu `#include <mma.h>` 移至文件头（L11）

**Files:** Modify `03-hpc-advanced/src/03_gemm/gemm.cu`（:326 移除，头部添加）
**验证:** 编译检查

- [ ] **Step 1: 移动 include**

---

## Group B: 04-inference-engine

### Task B1: vectorized_gemm float4 对齐修复（S1）

**Files:** Modify `04-inference-engine/include/vectorized_gemm.cuh:138-158`
**验证:** CPU 模拟确认 N=130 时无 misaligned store

- [ ] **Step 1: 写回增加对齐判断**

```cuda
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int out_row = row_start + m;
        if (out_row < M) {
            // Row starting element index mod 4 controls float4 alignment.
            bool row_aligned = (static_cast<size_t>(out_row) * N) % 4 == 0;
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                int out_col = col_start + n;
                if (out_col + 3 < N && row_aligned) {
                    float4 result = make_float4(
                        regC[m][n], regC[m][n+1], regC[m][n+2], regC[m][n+3]);
                    store_float4(&C[out_row * N + out_col], result);
                } else {
                    for (int nn = 0; nn < 4 && out_col + nn < N; nn++) {
                        C[out_row * N + out_col + nn] = regC[m][n + nn];
                    }
                }
            }
        }
    }
```

- [ ] **Step 2: 运行 CPU 模拟验证**（`/tmp/opencode/verify_s1_fixed.cpp`）
- [ ] **Step 3: 编译检查** `nvcc -arch=sm_86 -c src/vectorized_gemm.cu`

### Task B2: memory_pool 双重释放 + cached_size 统计（S7）

**Files:** Modify `04-inference-engine/include/memory_pool.h:40-47, 69-88`
**验证:** CPU 状态机模拟（含 fixed 版）

- [ ] **Step 1: deallocate 增加缓存块防护**（仿 02 memory_pool.hpp:99-106）

```cpp
    void deallocate(void* ptr) override {
        if (!ptr)
            return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            // A block already returned to the free list must not be freed
            // again (would cudaFree pooled memory -> use-after-free).
            if (free_blocks_.count(ptr) || std::any_of(free_blocks_.begin(), free_blocks_.end(),
                    [ptr](const auto& kv){ return kv.second == ptr; })) {
                return;
            }
            cudaFree(ptr);
            return;
        }
        size_t size = it->second;
        allocated_blocks_.erase(it);
        free_blocks_.insert({size, ptr});
        stats_.cached_size += size;
    }
```

（注：`free_blocks_` 为 multimap<size_t, void*>，需按值搜索；更优做法：新增 `std::unordered_set<void*> cached_blocks_` 精确 O(1) 判断，仿 02。）

- [ ] **Step 2: allocate 命中时扣减 cached_size**

```cpp
        if (it != free_blocks_.end()) {
            void* ptr = it->second;
            free_blocks_.erase(it);
            stats_.cached_size -= it->first;  // 命中块离开缓存
            allocated_blocks_[ptr] = size;
            stats_.cache_hits++;
            return ptr;
        }
```

- [ ] **Step 3: CPU 状态机模拟验证**

### Task B3: compute_quant_params n==0 越界（M6）

**Files:** Modify `04-inference-engine/include/quantization.h:24-48`
**验证:** g++ 单测

- [ ] **Step 1: n==0 提前返回**

```cpp
inline QuantizationParams compute_quant_params(const float* data, size_t n) {
    QuantizationParams params;
    if (n == 0) {
        return params;  // scale=1.0f, zero_point=0, min/max default
    }
    ...
```

### Task B4: load_weights 单层尺寸上限（M7）

**Files:** Modify `04-inference-engine/src/inference_engine.cpp:71-110`
**验证:** 阅读确认

- [ ] **Step 1: 单层校验**

```cpp
        if (meta.in_features == 0 || meta.out_features == 0 ||
            static_cast<size_t>(meta.in_features) * meta.out_features > kMaxLayerBytes) {
            layers_.clear();
            return false;
        }
```
（`kMaxLayerBytes` 例如 1<<30。）

### Task B5: tensor.h clone 保留 allocator（M8）

**Files:** Modify `04-inference-engine/include/tensor.h:52-59`
**验证:** 阅读确认

- [ ] **Step 1: clone 传 allocator**

```cpp
    Tensor clone() const {
        Tensor result(shape_, allocator_);
        result.data_.copy_from_host(data_.get(), size_ * sizeof(float));
        return result;
    }
```
（需确认 `Tensor` 构造签名与 `copy_from_host` 使用场景。）

### Task B6: save_weights 检查写入（L9）

**Files:** Modify `04-inference-engine/src/inference_engine.cpp:115-154`
**验证:** 阅读确认

- [ ] **Step 1: 各次 write 后检查**

```cpp
    if (!file.good()) return false;
```
（在 header 与每层 write 后追加；循环内失败返回 false。）

### Task B7: Config::get_bool 大小写不敏感（L12）

**Files:** Modify `04-inference-engine/include/config.h`
**验证:** g++ 单测

- [ ] **Step 1: 转小写比较**

```cpp
    bool get_bool(const std::string& key, bool default_value = false) const {
        std::string value = get(key);
        if (value.empty()) return default_value;
        for (auto& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return value == "true" || value == "1" || value == "yes" || value == "on";
    }
```
（补 `#include <cctype>`。）

### Task B8: random_init 线程安全（L5）

**Files:** Modify `04-inference-engine/include/common.h`
**验证:** 阅读确认

- [ ] **Step 1: 加互斥或 thread_local**

```cpp
inline void random_init(float* data, size_t n, float min_val = -1.0f, float max_val = 1.0f) {
    static std::mt19937 gen(42);
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}
```
（补 `#include <mutex>`。）

### Task B9: StreamManager/Logger 锁（L6/L7）

**Files:** Modify `04-inference-engine/include/stream_manager.h:70-79`、`include/logger.h`
**验证:** 阅读确认

- [ ] **Step 1: cleanup() 加锁**（`std::lock_guard<std::mutex> lock(mutex_);`）
- [ ] **Step 2: Logger set_* 加锁**（`set_level` 等以 `std::lock_guard` 保护）

### Task B10: add_bias grid.y 上限（L8 部分）

**Files:** Modify `04-inference-engine/src/tensor.cu:191-193`
**验证:** 阅读确认

- [ ] **Step 1: grid.y 上限防护**（行数 > 65535*16 时抛异常）

---

## Group C: 02-tensorcraft-core

### Task C1: compute_quant_params_kernel atomicMin/Max 负浮点（S6）

**Files:** Modify `02-tensorcraft-core/include/tensorcraft/kernels/fusion.hpp:192-195`
**验证:** CPU 模拟（符号翻转变换在含负值时正确）

- [ ] **Step 1: 用符号翻转位变换**

```cuda
    if (tid == 0) {
        // Signed-int compare of raw float bits is wrong for negatives
        // (larger-magnitude negatives sort as smaller ints). Transform so
        // the resulting unsigned keys sort in float order.
        // key = sign ? ~bits : (bits | 0x80000000u)
        unsigned int min_key = (__float_as_uint(s_min[0]) >> 31)
            ? ~__float_as_uint(s_min[0])
            : (__float_as_uint(s_min[0]) | 0x80000000u);
        unsigned int max_key = (__float_as_uint(s_max[0]) >> 31)
            ? ~__float_as_uint(s_max[0])
            : (__float_as_uint(s_max[0]) | 0x80000000u);
        atomicMin(reinterpret_cast<unsigned int*>(min_val), min_key);
        atomicMax(reinterpret_cast<unsigned int*>(max_val), max_key);
    }
```
（注意：min_val/max_val 入参须由调用方以 `FLT_MAX`/`-FLT_MAX` 对应的变换 key 初始化；调用方需先清零并写入 `0xFF800000u` 与 `0x7F800000u`。因无调用者，补一段注释说明正确的初始化契约。）

- [ ] **Step 2: CPU 模拟验证变换后的比较序**

### Task C2: launch_elementwise_binary 对齐检查（M2）

**Files:** Modify `02-tensorcraft-core/include/tensorcraft/kernels/elementwise.hpp:348-364`
**验证:** 阅读确认

- [ ] **Step 1: 仿 unary 加 is_aligned 检查**

```cpp
    if (is_aligned<T, vec_size>(input1) && is_aligned<T, vec_size>(input2) &&
        is_aligned<T, vec_size>(output)) {
        // vectorized kernel (原逻辑)
    } else {
        // 标量 fallback elementwise_binary_kernel_naive
    }
```
（需存在/新增 naive binary kernel；若无则降级为 vec_size=1 路径。）

### Task C3: transpose_shared 用 T 类型 tile（M10）

**Files:** Modify `02-tensorcraft-core/include/tensorcraft/kernels/gemm.hpp:287-312`
**验证:** g++ 静态检查

- [ ] **Step 1: tile 类型改为 T**

```cuda
    __shared__ T tile[TILE_SIZE][TILE_SIZE + 1];
    ...
    tile[ty][tx] = input[in_row * cols + in_col];
    ...
    output[out_row * rows + out_col] = tile[tx][ty];
```
（删除 to_float/from_float 中间转换，double 保精度；half 亦保真。）

### Task C4: spmm_csr_tiled 补 launcher（M4）

**Files:** Modify `02-tensorcraft-core/include/tensorcraft/kernels/sparse.hpp`（:202-258 之后）
**验证:** 阅读确认

- [ ] **Step 1: 新增 launcher**

```cpp
template <typename T, int TILE_N = 32>
void launch_spmm_csr_tiled(const T* A_values, const int* A_col_indices, const int* A_row_ptrs,
                           const T* B, T* C, int M, int K, int N, cudaStream_t stream = nullptr) {
    if (M == 0 || N == 0)
        return;
    dim3 block(32);  // kernel assumes exactly one warp
    dim3 grid((N + TILE_N - 1) / TILE_N, M);
    spmm_csr_tiled_kernel<T, TILE_N><<<grid, block, 0, stream>>>(A_values, A_col_indices, A_row_ptrs, B, C, M, K, N);
    TC_CUDA_CHECK_LAST();
}
```

### Task C5: bindings numpy 连续性检查（M9）

**Files:** Modify `02-tensorcraft-core/src/python_ops/bindings.cpp:18-30`
**验证:** 阅读确认

- [ ] **Step 1: 强制 C 连续拷贝**

```cpp
template <typename T>
T* numpy_to_device(py::array_t<T> arr, size_t& size) {
    auto c_arr = py::array_t<T>::ensure(arr, py::array::c_style | py::array::forcecast);
    if (!c_arr) {
        throw std::runtime_error("numpy_to_device: cannot obtain C-contiguous array");
    }
    py::buffer_info buf = c_arr.request();
    size = buf.size;
    T* d_ptr;
    TC_CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
    TC_CUDA_CHECK(cudaMemcpy(d_ptr, buf.ptr, size * sizeof(T), cudaMemcpyHostToDevice));
    return d_ptr;
}
```
（确保 `ensure` 复制非连续输入，杜绝错位拷贝。）

### Task C6: conv2d/attention grid.z 校验（L8）

**Files:** Modify `02-tensorcraft-core/include/tensorcraft/kernels/conv2d.hpp:142`、`attention.hpp:432`
**验证:** 阅读确认

- [ ] **Step 1: launcher 校验 `N*K <= 65535` / `batch*heads <= 65535`，超限抛异常**

### Task C7: MemoryPool 流语义文档化（L4）

**Files:** Modify `02-tensorcraft-core/include/tensorcraft/memory/memory_pool.hpp:29-41`
**验证:** 阅读确认

- [ ] **Step 1: 在类注释补充跨流使用约束**

---

## Group D: 01-sgemm-tutorial

### Task D1: launch_tensor_core_sgemm 16 对齐校验（S3）

**Files:** Modify `01-sgemm-tutorial/src/kernels/tensor_core_sgemm.cuh`（wrapper）
**验证:** 阅读确认

- [ ] **Step 1: wrapper 加校验**

```cpp
    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        throw std::invalid_argument(
            "tensor_core_sgemm: M, N, K must be multiples of 16 (WMMA fragment size)");
    }
```

### Task D2: double_buffer register-tiled 共享内存上限（M5）

**Files:** Modify `01-sgemm-tutorial/src/kernels/double_buffer_sgemm.cuh:296`
**验证:** 阅读确认

- [ ] **Step 1: kernel 内加 static_assert**

```cuda
    // 2 buffers x 2 matrices x TILE_SIZE x (TILE_SIZE+1) x sizeof(float)
    static_assert(2 * 2 * TILE_SIZE * (TILE_SIZE + 1) * (int)sizeof(float) <= 48 * 1024,
                  "TILE_SIZE too large: shared memory would exceed the 48KB static limit");
```
（默认 TILE_SIZE=64 编译即失败，调用方改用 32。）

### Task D3: verify.cuh 容差语义一致 + NaN 对称（L1/L2）

**Files:** Modify `01-sgemm-tutorial/src/utils/verify.cuh:150,185,201`
**验证:** g++ 单测

- [ ] **Step 1: compareMatrices 增加 ref NaN/Inf 检查**
- [ ] **Step 2: matricesApproxEqual/findFirstMismatch 改用 allclose 语义**（`abs_error > atol + rtol*|ref|`）

---

## Group E: common + examples

### Task E1: device_memory 拷贝边界校验（L3）

**Files:** Modify `common/include/cuda_foundations/core/device_memory.hpp:132,160`
**验证:** g++ 单测

- [ ] **Step 1: count 超过 size_ 抛异常**

### Task E2: examples rand seed（L13）

**Files:** Modify `examples/02_tensor_operations/main.cu:139`（fill_random）
**验证:** 阅读确认

- [ ] **Step 1: 改用固定种子分布或 `srand` 初始化说明**

---

## 自检清单（Self-Review）

- S1-S7、M1-M10、L1-L13 全覆盖：见上任务映射。
- 无占位符：每任务给出可应用代码。
- 类型一致性：无跨任务共享新类型；launcher 签名沿用现有模式。
- 交叉验证：Group A/B/D 的索引/状态机问题均附 CPU 模拟命令；所有 CUDA 改动做 `nvcc -arch=sm_86 -c` 语法编译；纯 host 改动做 `g++ -fsyntax-only`。
