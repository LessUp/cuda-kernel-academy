# Contributing to CUDA Kernel Academy

Thanks for your interest in contributing.

## Development setup

### Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| CUDA Toolkit | 12.0 | 12.x |
| CMake | 3.20 | 3.24+ |
| GCC / Clang | GCC 9 / Clang 10 | GCC 11+ |
| GPU | sm_70 | sm_80+ |

### Clone and build

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy

# If you forked the repository, also add the upstream remote:
git remote add upstream https://github.com/LessUp/cuda-kernel-academy.git

cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

List presets with:

```bash
cmake --list-presets
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Workflow

1. Sync with upstream:

   ```bash
   git fetch upstream
   git checkout master
   git merge upstream/master
   ```

2. Create a branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make focused changes.
4. Run the relevant local checks.
5. Push your branch and open a pull request.

## What to validate locally

At minimum:

```bash
pre-commit run --all-files
npm ci
npx honkit build . _site
```

If you changed CUDA/CMake code, also run a local GPU build/test flow, for example:

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

## Code style

- Follow [docs/CODING_STYLE.md](docs/CODING_STYLE.md)
- Use the repository `.clang-format`
- Prefer updating existing files over introducing new abstractions
- Keep changes focused on the requested fix or feature

## Commit messages

Conventional Commits are preferred:

```text
feat(scope): short description
fix(scope): short description
docs(scope): short description
ci(scope): short description
```

## Pull requests

Please ensure:

- the change is scoped and explained clearly
- docs are updated when behavior or commands changed
- local checks pass
- GPU-dependent changes were validated on an actual CUDA machine when applicable
