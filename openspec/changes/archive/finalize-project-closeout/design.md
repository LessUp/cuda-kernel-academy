# Design: Finalize Project Closeout

## Context

The repository already has the teaching modules. The main risk now is maintenance drift: stale process artifacts, weak public presentation, and tooling/docs that describe more systems than the project actually needs.

## Decisions

### Decision 1: Keep OpenSpec as the only change-management layer

Legacy `.kiro/` artifacts are removed from the repository surface. The cost of parallel workflow systems is higher than the value they provide during closeout.

### Decision 2: Keep the entire closeout program in one umbrella change

The repository already has an active `finalize-project-closeout` change. We extend that change instead of creating a separate tree of loosely related specs.

This matches the repository posture:

- short-lived, convergent work
- low branch drift
- no parallel process systems

### Decision 3: Execute closeout in four ordered phases

The closeout work spans multiple surfaces, but it should still move through a single ordered program:

1. audit and normalize the repo surface
2. align public presentation and GitHub integration
3. finalize AI/tooling configuration
4. produce the final backlog and handoff semantics

This order matters. The public surface should not be polished before the rules and source-of-truth docs are stable, and the final backlog should be derived from real findings rather than guesswork.

### Decision 4: Make the landing page showcase-first without changing the site stack

GitHub Pages should not be a plain README mirror. The landing page should explain why the project matters, what each module teaches, and how to start.

HonKit already exists in the repository and is good enough for final closeout. Re-platforming the site would create churn late in the project and does not help the repository reach archive-ready stability faster.

### Decision 5: Keep AI/tooling configuration documentation-led

Official instruction surfaces are:

- `AGENTS.md`
- `CLAUDE.md`
- `.github/copilot-instructions.md`

We avoid checking in low-confidence tool-specific config files unless the schema is stable and worth maintaining.

### Decision 6: Standardize on clangd plus compile commands

`clangd` is the most portable LSP substrate for the repository. CMake presets already export compile commands; the repo adds a small sync script so multiple tools can reuse the same database.

### Decision 7: Keep MCP lean and prefer lightweight Skills when repetition exists

Heavy MCP integration consumes context and tends to decay. For this repository, focused built-in tools and clear repo docs are the default, with extra integrations added only for recurring value.

CLI skills are favored when they:

- encode a repeatable repo-specific workflow
- cost less context than always-on MCP integrations
- do not require external state

### Decision 8: Bound bug fixing to closeout-relevant surfaces

"Fix all bugs" is too open-ended to be a useful execution rule. During closeout, bug work is bounded to:

- failures in the documented verification path
- correctness or configuration issues discovered while auditing high-risk surfaces
- tightly coupled defects uncovered in files already being normalized

This keeps the work finite and aligned with the repository's final-maintenance goal.

### Decision 9: Unify version anchors across docs, workflows, and tooling guidance

Version claims should not drift between docs, CI, and setup guidance. The closeout pass will normalize the visible anchors for CUDA, CMake, Node, GitHub Actions runtimes, and related developer tooling where the repository makes explicit promises.

## Risks and trade-offs

### Risk: Over-pruning useful documentation

Mitigation: keep documents that directly help learners, contributors, or future maintainers complete work; remove narrative clutter and duplicate process docs.

### Risk: Pages remains constrained by HonKit

Mitigation: improve the content model, SUMMARY structure, CSS, and post-build logic rather than adding a new static-site stack late in the project.

### Risk: The umbrella change becomes too coarse again

Mitigation: keep the single change, but rewrite `tasks.md` into phase-level execution units with explicit acceptance boundaries and ordering.

### Risk: Workflow cleanup removes useful checks

Mitigation: preserve checks that directly protect docs, repo instructions, or CPU-safe correctness; remove only those that are duplicative, misleading, or detached from actual contributor value.
