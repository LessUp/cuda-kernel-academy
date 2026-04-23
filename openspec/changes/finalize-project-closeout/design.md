# Design: Finalize Project Closeout

## Context

The repository already has the teaching modules. The main risk now is maintenance drift: stale process artifacts, weak public presentation, and tooling/docs that describe more systems than the project actually needs.

## Decisions

### Decision 1: Keep OpenSpec as the only change-management layer

Legacy `.kiro/` artifacts are removed from the repository surface. The cost of parallel workflow systems is higher than the value they provide during closeout.

### Decision 2: Make the landing page showcase-first

GitHub Pages should not be a plain README mirror. The landing page should explain why the project matters, what each module teaches, and how to start.

### Decision 3: Keep AI/tooling configuration documentation-led

Official instruction surfaces are:

- `AGENTS.md`
- `CLAUDE.md`
- `.github/copilot-instructions.md`

We avoid checking in low-confidence tool-specific config files unless the schema is stable and worth maintaining.

### Decision 4: Standardize on clangd plus compile commands

`clangd` is the most portable LSP substrate for the repository. CMake presets already export compile commands; the repo adds a small sync script so multiple tools can reuse the same database.

### Decision 5: Keep MCP lean

Heavy MCP integration consumes context and tends to decay. For this repository, focused built-in tools and clear repo docs are the default, with extra integrations added only for recurring value.

## Risks and trade-offs

### Risk: Over-pruning useful documentation

Mitigation: keep documents that directly help learners, contributors, or future maintainers complete work; remove narrative clutter and duplicate process docs.

### Risk: Pages remains constrained by HonKit

Mitigation: improve the content model, SUMMARY structure, CSS, and post-build logic rather than adding a new static-site stack late in the project.
