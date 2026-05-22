# Proposal: Pages and Docs Seams Hardening

## Problem

The repository's public documentation surface has drifted in several real, user-facing ways:

1. Canonical GitHub and Pages URLs disagree across README files, VitePress config, issue templates, and docs pages.
2. `docs/index.md` is a hardcoded redirect shim tied to `/cuda-kernel-academy/...` instead of a stable landing page.
3. GitHub Pages content is rich inside locale homepages, but the root entrypoint does not explain the site structure or let users choose a reading path intentionally.
4. `04-inference-engine/README.md` documents build artifacts that do not match the CMake targets.

These seams are shallow: users must reconcile repository identity, site entrypoints, and build outputs by reading multiple files. That reduces locality for maintainers and leverage for readers.

## Solution

Create one closeout change that:

- normalizes canonical GitHub and Pages URLs across public-facing docs and metadata
- replaces the redirect-only docs root with a real bilingual landing page
- adds bilingual reading-path pages to deepen the GitHub Pages learning surface
- fixes the `04-inference-engine` README/CMake artifact seam
- adds CPU-safe static docs tests so these seams stay tight

## Scope

| In Scope | Out of Scope |
| --- | --- |
| `README.md`, `README.zh-CN.md`, `CONTRIBUTING.md`, issue template links | CUDA kernel implementation changes |
| `docs/index.md`, `docs/.vitepress/config.ts`, selected docs pages | GPU runtime validation |
| new docs tests under `docs/tests/` | new large documentation systems |
| `04-inference-engine/README.md` docs/build alignment | refactoring inference runtime code |

## Success Criteria

1. Public URLs consistently point to `AICL-Lab/cuda-kernel-academy` and `https://aicl-lab.github.io/cuda-kernel-academy/`.
2. `docs/index.md` becomes a real bilingual entrypoint with explicit locale and reading-path links.
3. Pages gains bilingual reading-path guides linked from nav/sidebar and the root entrypoint.
4. `04-inference-engine/README.md` names binaries that actually exist in `04-inference-engine/CMakeLists.txt`.
5. Added static docs tests fail before the change and pass after the change.
6. `npm run docs:build` stays green.
