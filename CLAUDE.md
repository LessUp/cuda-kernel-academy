# CLAUDE.md

Minimal repository guidance for Claude-compatible agents.

## Operating mode

- this repository is in stabilize-and-finish mode
- prefer cleanup, correctness, and doc clarity over feature growth
- remove stale process artifacts instead of preserving them

## Build commands

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

## Repository cautions

- `01-sgemm-tutorial` stays outside the root CMake graph
- `03-hpc-advanced` needs CUDA 12+ and a modern compiler
- `04-inference-engine` depends on `02-tensorcraft-core`
- GPU validation must happen on a real CUDA machine

## Documentation rules

- keep Pages focused on learning and reference material
- keep `CHANGELOG.md` at the repository root as the only changelog
- avoid committing local tool state or generated agent artifacts

## Local overrides

`CLAUDE.local.md` is reserved for untracked personal notes only.
