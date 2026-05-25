---
outline: [2, 3]
---

# Development Workflow

This repository follows a single, lightweight workflow designed for shipping focused fixes without process drift.

## Principles

- **Small, focused changes**
- **Targeted verification**
- **Docs stay honest**
- **Short-lived branches**
- **Minimal ceremony**

## The Canonical Loop

1. Pick one tightly scoped change
2. Implement the smallest complete fix
3. Run only the checks that match the change
4. Update docs and the root changelog when the public surface changes
5. Open a focused pull request and merge promptly

## Verification

```bash
pre-commit run --all-files
npm ci
npm run docs:build
cmake --list-presets
```

GPU-dependent changes must be validated on a real CUDA machine.

## Documentation and history

- `CHANGELOG.md` in the repository root is the only changelog
- GitHub Pages should stay focused on guides, references, and module learning paths
- Prefer deleting stale instructions over keeping parallel explanations

## References
