# CUDA Kernel Academy - Agent Guide

High-signal guidance for AI agents and maintainers working in this repository.

## 1. Project shape

CUDA Kernel Academy is a multi-module CUDA learning repository with **two build systems**:

| Module | Build system | Standard | Notes |
| --- | --- | --- | --- |
| `01-sgemm-tutorial/` | standalone `Makefile` | C++17 | not part of root CMake |
| `02-tensorcraft-core/` | root/module CMake | C++17+ | header-only, reused by 04 |
| `03-hpc-advanced/` | root/module CMake | C++20 | requires CUDA 12+ and modern host compiler |
| `04-inference-engine/` | root/module CMake | C++17 | depends on 02 |
| `common/` | root CMake | C++17 | shared CUDA helpers |

## 2. Project posture

This repository is in **closeout / stabilization mode**:

- prefer correctness, clarity, and documentation quality over feature growth
- prefer deleting stale or generic process artifacts over keeping parallel systems
- do not introduce broad new frameworks, workflows, or automation unless they solve a repository-specific problem
- keep GitHub Actions lean; CI is for CPU-safe checks, not pretend-GPU validation

## 3. Canonical workflow

This repository uses **OpenSpec as the only authoritative change-management workflow**.

### Required lifecycle

1. Create or update a change under `openspec/changes/<change-name>/`
2. Keep `proposal.md`, `specs/spec.md`, `design.md`, and `tasks.md` aligned
3. Implement from the tasks list
4. Run targeted verification
5. Use `/review` before merge for non-trivial changes
6. Archive completed work to `openspec/changes/archive/YYYY-MM-DD-<change-name>/`

### Do not use

- `.kiro/` specs, prompts, or skills
- duplicate process docs that disagree with OpenSpec
- long-lived local/cloud branch drift

See `docs/DEVELOPMENT_WORKFLOW.md` for the human-readable workflow and `openspec/README.md` for the repo layout.

## 4. Build and verification

### Root CMake workflow

```bash
cmake --list-presets
cmake --preset default
cmake --build --preset default
ctest --preset default
```

Recommended presets:

- `default` for general release builds
- `debug` for debugging
- `native` for local GPU tuning
- `minimal` for lightweight library-only work

### Module 01 workflow

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
make test
```

### Quality gates

```bash
pre-commit run --all-files
npm ci
npm run docs:build
cmake --list-presets
```

Notes:

- GitHub-hosted CI only runs CPU-safe checks
- CUDA builds and runtime tests must be validated on a real GPU machine
- `03-hpc-advanced` is skipped when CUDA/compiler requirements are not met
- `04-inference-engine` requires `BUILD_TENSORCRAFT=ON`

## 5. Code conventions

- formatting is defined by `.clang-format` and `.editorconfig`
- use `#pragma once`, not include guards
- namespaces: `cuda_academy::`, `tensorcraft::`, `hpc::`, `mini_inference::`
- class names use PascalCase; functions use lower_case; members use trailing `_`
- prefer existing helpers and narrow fixes over new abstraction layers

### CUDA error handling

Prefer the shared macros from `common/include/cuda_academy/`:

```cpp
#include "cuda_academy/cuda_academy.hpp"

CUDA_CHECK(cudaMalloc(&ptr, size));
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

## 6. Docs and Pages

- `README.md` is the public landing page for both GitHub and Pages
- `SUMMARY.md` controls the Pages navigation
- `docs/README.md` indexes durable reference docs
- documentation should be concise, specific to this repository, and maintained only when it materially helps users finish work

Do not add generic “best practice” docs unless they change contributor behavior for this codebase.

## 7. Tooling policy

- prefer `clangd` backed by `compile_commands.json` for C/C++/CUDA navigation
- prefer `scripts/dev/sync-compile-commands.sh <preset>` after CMake configure
- prefer built-in tool features and lightweight skills over heavy MCP stacks
- use `/research` only for external facts that are actually needed
- use `/remote` for contained background tasks, not as the default
- avoid `/fleet` unless a task genuinely needs a large parallel context; prefer fewer, longer, focused sessions

Tool-specific guidance lives in:

- `CLAUDE.md`
- `.github/copilot-instructions.md`
- `docs/AI_TOOLING.md`

## 8. Review and merge discipline

- keep branches short-lived and scoped
- merge once reviewed; do not accumulate unrelated local/cloud divergence
- use `/review` before merge on non-trivial work
- update specs and docs in the same change when behavior or workflow changes

## 9. Key paths

- `CMakeLists.txt` - root build entry
- `CMakePresets.json` - build presets and compile commands export
- `.pre-commit-config.yaml` - repo hooks
- `docs/DEVELOPMENT_WORKFLOW.md` - canonical workflow
- `docs/AI_TOOLING.md` - Copilot / Claude / Codex / LSP / MCP guidance
- `openspec/` - specs and archives
