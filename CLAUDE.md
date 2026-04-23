# CLAUDE.md

Repository-level instructions for Claude Code and Claude-compatible agents.

## Read first

1. `AGENTS.md`
2. `docs/DEVELOPMENT_WORKFLOW.md`
3. `docs/AI_TOOLING.md`

## Operating mode

- this repository is in **stabilize and finish** mode
- prefer bug fixes, cleanup, doc clarity, and workflow simplification
- avoid speculative features and generic engineering ceremony
- remove stale parallel process artifacts instead of preserving them

## OpenSpec is mandatory

Use only `openspec/changes/<change-name>/` and `openspec/changes/archive/`.

Every non-trivial change should keep these artifacts aligned:

- `proposal.md`
- `specs/spec.md`
- `design.md`
- `tasks.md`

Use `/review` before merge for non-trivial changes.

## Build commands

### Root

```bash
cmake --list-presets
cmake --preset default
cmake --build --preset default
ctest --preset default
```

### Module 01

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
make test
```

### Docs and hygiene

```bash
pre-commit run --all-files
npm ci
npm run docs:build
```

## Repository-specific cautions

- `01-sgemm-tutorial` is not part of the root CMake graph
- `03-hpc-advanced` needs CUDA 12+ and modern compiler support
- `04-inference-engine` requires `02-tensorcraft-core`
- GitHub Actions only covers CPU-safe checks; GPU validation stays local

## Tooling expectations

- prefer `clangd` with repo compile commands
- prefer repo guidance files over custom hidden prompts
- keep MCP usage lean; use built-in tools and skills first
- avoid `/fleet` unless a task clearly benefits from heavy parallelization

## Local overrides

`CLAUDE.local.md` is reserved for untracked personal notes only and should not be committed.
