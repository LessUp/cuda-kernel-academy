---
outline: [2, 3]
---

# Development Workflow

This repository follows a single, lightweight workflow designed for shipping focused fixes without process drift.

## Principles

- **OpenSpec first** for non-trivial work
- **Short-lived branches**
- **Review before merge**
- **Minimal ceremony**
- **No parallel spec systems**

## The Canonical Loop

1. Decide whether the work needs a spec
2. Create or refresh the OpenSpec change
3. Implement from the tasks list
4. Verify only what matters
5. Review before merge
6. Merge promptly
7. Archive completed work

## Verification

```bash
pre-commit run --all-files
npm ci
npm run docs:build
cmake --list-presets
```

GPU-dependent changes must be validated on a real CUDA machine.

## References

[^1]: CUDA Kernel Academy OpenSpec. See `openspec/README.md`.
