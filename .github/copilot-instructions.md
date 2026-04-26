# Copilot Instructions for CUDA Kernel Academy

## Repository intent

This repository is a multi-module CUDA learning project in **stabilization / closeout mode**. Favor targeted fixes, cleanup, documentation clarity, and workflow simplification over new feature growth.

## Repository shape

- `01-sgemm-tutorial/` uses a standalone `Makefile`
- `02-tensorcraft-core/`, `03-hpc-advanced/`, `04-inference-engine/`, `common/`, and `examples/` use root/module CMake
- `04-inference-engine/` depends on `02-tensorcraft-core`
- `03-hpc-advanced/` requires CUDA 12+ and modern compiler support

## Build and verification

Use the existing commands:

```bash
cmake --list-presets
cmake --preset default
cmake --build --preset default
ctest --preset default

pre-commit run --all-files
npm ci
npm run docs:build
```

For module 01:

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
make test
```

GitHub Actions only runs CPU-safe checks. CUDA runtime validation belongs on a real GPU machine.

## Workflow rules

- use **OpenSpec** for non-trivial changes
- while a repo-wide closeout program is active, extend its umbrella change instead of opening parallel umbrella specs
- keep specs under `openspec/changes/<change-name>/`
- archive completed changes under `openspec/changes/archive/`
- do not introduce or revive `.kiro/`-based workflows
- use `/review` before merging non-trivial changes

## Style and design rules

- follow `.clang-format`, `.editorconfig`, and `docs/CODING_STYLE.md`
- prefer editing existing files over adding new abstractions
- do not add generic docs or generic engineering files
- prefer deleting stale content over maintaining parallel explanations
- keep GitHub Actions and engineering config minimal and repository-specific

## Tooling rules

- prefer `clangd` backed by repo compile commands for navigation
- use `scripts/dev/sync-compile-commands.sh <preset>` after configuring CMake
- keep MCP usage minimal unless it clearly pays for itself
- keep durable agent guidance in visible repo docs and OpenSpec artifacts; do not add hidden shadow instruction trees
- use `gh` for repository metadata updates instead of hand-editing descriptions or homepage settings in docs only
- treat `.github/workflows/copilot-setup-steps.yml` as a manual support workflow rather than routine CI
- avoid `/fleet` unless a task truly needs large parallel context
