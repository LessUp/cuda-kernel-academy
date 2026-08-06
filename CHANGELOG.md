# Changelog

The repository keeps a lightweight changelog focused on release-level milestones, not day-by-day narrative notes.

## [Unreleased]

### Changed

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

[Unreleased]: https://github.com/LessUp/cuda-kernel-academy/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LessUp/cuda-kernel-academy/releases/tag/v1.0.0
