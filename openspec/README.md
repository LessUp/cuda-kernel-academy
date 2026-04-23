# OpenSpec in this repository

CUDA Kernel Academy uses OpenSpec as the single source of truth for non-trivial changes.

## Layout

```text
openspec/
├── changes/
│   ├── <active-change>/
│   │   ├── proposal.md
│   │   ├── specs/spec.md
│   │   ├── design.md
│   │   └── tasks.md
│   └── archive/
│       └── YYYY-MM-DD-<change-name>/
```

## Standard flow

1. `/opsx:propose "describe the change"`
2. refine proposal, spec, design, and tasks
3. `/opsx:apply`
4. verify the change
5. `/review`
6. `/opsx:archive`

## Repository rule

Do not keep parallel spec systems such as `.kiro/` in active use. If the OpenSpec artifacts and some other process artifact disagree, OpenSpec wins.

For the human workflow, see `docs/DEVELOPMENT_WORKFLOW.md`.
