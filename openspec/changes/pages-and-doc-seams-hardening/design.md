# Design: Pages and Docs Seams Hardening

## Overview

This change deepens three weak seams:

1. **Repository identity seam** - public files should agree on the canonical GitHub owner/repo and Pages hostname.
2. **Docs entrypoint seam** - the docs root should be a stable landing page, not a client-side redirect shim with hardcoded base paths.
3. **Docs/build seam** - docs describing build artifacts should stay aligned with real CMake target names.

The implementation stays repository-specific and CPU-verifiable.

## Decisions

### 1. Canonical metadata by concentrated edits, not a new abstraction

We will normalize URLs in the existing public-facing files instead of introducing a new generated metadata system. The repo is in closeout mode; concentrated edits keep locality high without adding another workflow or generator.

### 2. Stable docs root instead of automatic redirect

`docs/index.md` will become a real bilingual landing page with:

- repository overview
- explicit links to `/en/` and `/zh/`
- learning-path summaries
- links to the new reading-path guides

This removes hardcoded runtime redirects and improves leverage for both readers and maintainers.

### 3. Reading-path guides as the content expansion

Add:

- `docs/en/guides/reading-paths.md`
- `docs/zh/guides/workflow-reading-paths.md`

Each page will explain how to read the repository by goal:

- first-contact / newcomer
- kernel optimization
- reusable library design
- end-to-end inference integration

We will wire these into the relevant nav/sidebar areas.

### 4. Static tests as the contract

Add static Node tests under `docs/tests/` to verify:

- public files no longer reference the stale `LessUp/...` slug or `lessup.github.io/...`
- `docs/index.md` no longer contains auto-redirect logic or hardcoded `/cuda-kernel-academy/...` redirects
- `04-inference-engine/README.md` stays aligned with its documented CMake artifacts

## File Map

### Modify

- `README.md`
- `README.zh-CN.md`
- `CONTRIBUTING.md`
- `.github/ISSUE_TEMPLATE/config.yml`
- `docs/index.md`
- `docs/.vitepress/config.ts`
- `docs/en/index.md`
- `docs/zh/index.md`
- `docs/en/guides/getting-started.md`
- `docs/zh/guides/getting-started.md`
- `docs/zh/reference/integration-examples.md`
- `docs/zh/reference/troubleshooting.md`
- `04-inference-engine/README.md`

### Add

- `docs/en/guides/reading-paths.md`
- `docs/zh/guides/reading-paths.md`
- `docs/tests/site-canonical-links.test.mjs`
- `docs/tests/site-entrypoint.test.mjs`
- `docs/tests/inference-readme-artifacts.test.mjs`

## Verification

- `node --test docs/tests/*.test.mjs`
- `npm run docs:build`
- `cmake --list-presets`

GPU-dependent verification remains out of scope for this session because the environment has no CUDA toolkit.
