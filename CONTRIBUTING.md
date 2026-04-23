# Contributing to CUDA Kernel Academy

Thanks for helping improve the repository.

## What belongs here

This project is no longer expanding aggressively. The highest-value contributions are:

- bug fixes and correctness repairs
- targeted CUDA or build-system cleanup
- documentation that removes ambiguity for real users
- workflow simplification and repository hardening

Please avoid drive-by framework additions, generic boilerplate docs, or speculative new subsystems.

## Development environment

| Requirement | Minimum | Recommended |
| --- | --- | --- |
| CUDA Toolkit | 12.0 | 12.x |
| CMake | 3.20 | 3.24+ |
| GCC / Clang | GCC 9 / Clang 10 | GCC 11+ |
| Node.js | 18 | 22 |
| GPU | sm_70 | sm_80+ |

## Quick start

```bash
git clone https://github.com/LessUp/cuda-kernel-academy.git
cd cuda-kernel-academy

pip install pre-commit
pre-commit install

cmake --list-presets
cmake --preset default
cmake --build --preset default
```

For the standalone tutorial module:

```bash
cd 01-sgemm-tutorial
make GPU_ARCH=sm_86
make test
```

## Canonical workflow

### 1. Keep the change small

Use a short-lived branch and keep the scope focused. Do not let local branches and cloud-side branches diverge for long periods.

### 2. Start with OpenSpec for non-trivial work

This repository uses [OpenSpec](https://github.com/Fission-AI/OpenSpec) as the only spec workflow.

```text
/opsx:propose "describe the change"
```

Work from:

- `openspec/changes/<change-name>/proposal.md`
- `openspec/changes/<change-name>/specs/spec.md`
- `openspec/changes/<change-name>/design.md`
- `openspec/changes/<change-name>/tasks.md`

When the work is done, archive it to:

`openspec/changes/archive/YYYY-MM-DD-<change-name>/`

See `docs/DEVELOPMENT_WORKFLOW.md` for the full process.

### 3. Implement and verify

Run the checks that match your change:

```bash
pre-commit run --all-files
npm ci
npm run docs:build
cmake --list-presets
```

If you changed CUDA/CMake code, also validate on a real GPU machine:

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

### 4. Review before merge

For non-trivial work, use `/review` before merging. The repo favors fast review and timely merge over long-lived branch stacks.

## Code and documentation rules

- follow `docs/CODING_STYLE.md` and `.clang-format`
- use existing helpers before adding new abstractions
- update docs in the same change when commands, behavior, or workflow change
- prefer deleting stale material over adding another layer of explanation beside it

## AI-assisted development

AI-assisted contributions are welcome when they stay specific to this repository and are actually verified.

Use these repo docs as the source of truth:

- `AGENTS.md`
- `CLAUDE.md`
- `.github/copilot-instructions.md`
- `docs/AI_TOOLING.md`

## Pull requests

A good pull request for this repository:

- is scoped to one change or one tightly related fix set
- links the relevant OpenSpec change for non-trivial work
- explains user-facing or workflow-facing impact clearly
- includes the exact local checks that were run when they matter

## Commit messages

Conventional Commits are preferred:

```text
fix(scope): short description
docs(scope): short description
ci(scope): short description
refactor(scope): short description
```
