# Changelog

The repository keeps a lightweight changelog focused on release-level milestones, not day-by-day narrative notes.

## [Unreleased]

### Changed
- README 标题与仓名对齐为 `cuda-foundations`（教学品牌仍为 CUDA Kernel Academy）
- 中文 README 补齐 IN/OUT；`02-tensorcraft-core` 标明历史模块名而非独立产品
- `03-hpc-advanced` FlashAttention 教学核与文档标明非权威，指向 `cuflash-attn`
- 面向用户的 GitHub 链接统一为 `github.com/aicl-lab/...`

## [1.0.1] - 2026-08-17

### Added
- 01-sgemm-tutorial now integrated into root CMake (`BUILD_SGEMM_TUTORIAL`) and `ctest`
- `scripts/run_benchmarks.sh`: one-shot benchmark collection (GPU/CUDA info, build, 01 + 02 benchmarks)
- `docs/en/guides/profiling.md`: copy-pasteable nsys/ncu runbook with metric interpretation
- `02-tensorcraft-core` correctness tests: `tests/test_conv.cpp`, `tests/test_sparse.cpp`, `tests/test_fusion.cpp`
- `.github/workflows/gpu-tests.yml`: optional full GPU `ctest` for self-hosted/cloud GPU runners

### Fixed
- MemoryWrapper default-allocator bug: Tensor({shape}) no longer leaves device ptr null
- 01-sgemm-tutorial tests compile errors, tolerance, and Makefile gtest dependency
- FlashAttention: block size 2048 -> 256 threads, linear K/V load, `head_dim != 64` throws
- launch_gemm_wmma: rejects non-multiples of 16 with `std::invalid_argument`
- InferenceEngine: ping-pong intermediate buffers, explicit alias/protection checks
- Python bindings: removed references to missing source files, now builds
- `launch_moe_router`: validates `num_experts`/`top_k` bounds, null pointers; empty batch no-op
- Tensor: `reshape` rejects negative dims; `matmul` validates B batch shape for 3D+ inputs
- `load_weights`: rejects truncated files (mid-layer read failures) and unreasonable layer counts
- Removed dead `initRandomMatrixGPU` (and orphaned cuRAND macro/include)
- Root CMake: CUDA architecture default now actually applies on fresh configure (was silently building the toolkit's legacy `sm_52` default, which broke WMMA kernels)

### Changed
- Removed `GemmVersion::TensorCore` and `GemmVersion::Auto` from public enum
- All README/docs: replaced "production-grade"/"工业级" with "teaching"/"教学"
- Performance tables: removed unverified numbers; replaced with measured data
- CI: added CUDA build smoke job (nvidia/cuda container, compile only)
- ROADMAP: checked off items completed during this archive pass
- Root CMake now owns the single GoogleTest FetchContent; 02/03/04 fall back to self-fetch for standalone builds
- `03-hpc-advanced` honors the root `BUILD_TESTS` option
- Removed unused parameters/dead constants (`benchmark.cpp` print_results M/N/K, `mnist_demo.cpp` argc/argv, `vectorized_gemm.cuh` A_STRIDE/B_STRIDE)
- README/README.zh-CN: state the repository is in maintenance mode

### Verified
- 199/199 `ctest` tests pass on sm_86 (RTX 3060 Laptop, CUDA 12.0) — was 192, +7 new failure-path tests
- 52/52 module01 tests pass after tolerance fix
- 36/36 `tensorcraft_tests` pass, including new conv/sparse/fusion suites
- Module01 benchmark ladder measured and published
- Fresh `cmake --preset default` configure + build + test passes end to end; `scripts/run_benchmarks.sh` runs repeatably
- `nsys` profiling verified working on WSL2; `ncu` confirmed blocked by `ERR_NVGPUCTRPERM` (runbook documents bare-metal path)


### Changed

- merged the useful RMSNorm verification from the retired HPC lab and fixed the kernel's cross-warp reduction so every thread in a row uses the same full-row statistic
- established the four Academy modules as the sole maintained successors for the overlapping SGEMM, TensorCraft, HPC optimization, and mini-inference repositories
- retired duplicate historical variants instead of importing them back into the maintained teaching surface
- removed repository-tracked AI agent control files and dropped the docs `vitepress-plugin-llms` dependency
- simplified `04-inference-engine` to a single root-build path that always consumes the parent `TensorCraft::tensorcraft` target
- collapsed redundant 04-module GEMM wrapper files into one TensorCraft-backed dispatch unit and deleted dead duplicate kernel bodies
- pinned root CMake presets to system GCC/G++ and restored 04 inference/common compatibility so CUDA 12 builds stop drifting onto incompatible Conda host compilers
- removed legacy spec archives and AI control scaffolding from the tracked repository surface
- consolidated change history into the root `CHANGELOG.md` and removed module-level changelog drift
- simplified contributor and agent guidance around a small, verification-first workflow
- rebuilt the landing page and Pages navigation around a showcase-first information architecture
- tightened Pages-facing docs around the repository's actual build seams and learning path

## [1.0.0] - 2026-01-07

### Added

- four-module CUDA learning path covering SGEMM, reusable kernels, advanced optimization, and lightweight inference
- root CMake build for shared modules plus a standalone Makefile flow for `01-sgemm-tutorial`
- documentation set for installation, troubleshooting, coding style, and module-level deep dives

[Unreleased]: https://github.com/aicl-lab/cuda-foundations/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/aicl-lab/cuda-foundations/releases/tag/v1.0.0
