# Development Workflow

This repository follows a single, lightweight workflow designed for **shipping focused fixes without process drift**.

## Principles

- **OpenSpec first** for non-trivial work
- **short-lived branches**
- **review before merge**
- **minimal ceremony**
- **no parallel spec systems**
- **while a repo-wide closeout program is active, keep it in one umbrella change unless a subdomain becomes truly independent**

## The canonical loop

### 1. Decide whether the work needs a spec

Use OpenSpec when the change is:

- architectural
- cross-cutting
- ambiguous at the start
- large enough that implementation should not depend on chat history alone

For tiny typo fixes or obvious one-file corrections, a full proposal is optional.

### 2. Create or refresh the OpenSpec change

```text
/opsx:propose "describe the change"
```

Expected structure:

```text
openspec/
├── changes/
│   └── <change-name>/
│       ├── proposal.md
│       ├── specs/spec.md
│       ├── design.md
│       └── tasks.md
└── changes/archive/
    └── YYYY-MM-DD-<change-name>/
```

Keep the artifacts honest:

- `proposal.md` = why this change exists
- `specs/spec.md` = requirements and scenarios
- `design.md` = decisions and trade-offs
- `tasks.md` = actionable implementation checklist

### 3. Implement from the tasks list

```text
/opsx:apply
```

Keep the implementation branch narrow. If the work expands materially, update the spec instead of letting code drift away from the original intent.

### 4. Verify only what matters

Typical checks:

```bash
pre-commit run --all-files
npm ci
npm run docs:build
cmake --list-presets
```

GPU-dependent changes must be validated on a real CUDA machine.

### 5. Review before merge

Use `/review` for non-trivial changes. The review step is part of the standard loop, not an optional extra.

Recommended review points:

- requirement coverage
- correctness and risk
- workflow/docs consistency
- unnecessary complexity

### 6. Merge promptly

Avoid long-lived divergence between:

- local branches
- cloud-agent branches
- generated code branches
- delayed review branches

The repository favors fast convergence over stacked process overhead.

### 7. Archive completed work

```text
/opsx:archive
```

Archive to:

`openspec/changes/archive/YYYY-MM-DD-<change-name>/`

Do not leave “finished” work in `openspec/changes/`.

## Tool guidance

### `/review`

Use before merge for:

- multi-file changes
- workflow/config changes
- docs that redefine contributor behavior
- anything architectural

### `/research`

Use only for external facts or up-to-date tool documentation that the repository genuinely depends on.

### `/remote`

Use for isolated background execution or contained research tasks. Do not use it as the default mode for routine development.

### `/fleet`

Use sparingly. This repository prefers fewer long-running, focused sessions to reduce quota burn and avoid parallel drift.

## Closeout bias

When in doubt, choose the option that:

- reduces maintenance burden
- removes stale or low-signal content
- keeps the public surface polished
- makes future final bug-fix work easier to land
