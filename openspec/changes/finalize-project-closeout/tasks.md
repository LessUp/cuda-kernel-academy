# Tasks: Finalize Project Closeout

## Phase 1: Aggressive audit and normalization

- [x] 1.1 Audit the repository surface across OpenSpec, docs, Pages inputs, workflows, AI guidance files, and version anchors
- [x] 1.2 Rewrite `proposal.md`, `design.md`, `specs/spec.md`, and this `tasks.md` so the closeout change becomes execution-ready
- [x] 1.3 Prune stale or low-value docs, changelog clutter, and redundant contributor guidance while preserving durable project knowledge
- [x] 1.4 Normalize `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `docs/DEVELOPMENT_WORKFLOW.md`, and `docs/AI_TOOLING.md` around the same closeout posture
- [x] 1.5 Scan high-risk build, workflow, docs, and configuration surfaces for correctness issues and fix tightly coupled defects
- [x] 1.6 Unify visible version anchors across documentation, workflows, and setup guidance

## Phase 2: Engineering and GitHub integration

- [x] 2.1 Rework `README.md`, `README.zh-CN.md`, `SUMMARY.md`, `book.json`, and related Pages assets into one coherent showcase-first narrative
- [x] 2.2 Simplify `.github/workflows/ci.yml`, `.github/workflows/pages.yml`, and `.github/workflows/copilot-setup-steps.yml` so only high-value CPU-safe automation remains
- [x] 2.3 Update GitHub repository About metadata with `gh` so description, homepage URL, and topics match the landing-page narrative
- [x] 2.4 Encode a lightweight OpenSpec-driven development loop that avoids long-lived branch drift and treats `/review` as mandatory for non-trivial work

## Phase 3: AI tooling and Vibe Coding configuration

- [x] 3.1 Rebuild or calibrate `.github/copilot-instructions.md` from current repository reality rather than generic initialization output
- [x] 3.2 Refine `AGENTS.md`, `CLAUDE.md`, and `docs/AI_TOOLING.md` so they remain repository-specific, closeout-biased, and friendly to future GLM handoff
- [x] 3.3 Standardize the LSP path around `clangd`, shared compile commands, and minimal editor recommendations
- [x] 3.4 Decide which recurring workflows deserve lightweight CLI Skills versus additional MCP surface area and keep only the minimal durable toolchain

## Phase 4: Final backlog and handoff

- [x] 4.1 Aggregate audit findings and rewrite the final execution backlog in OpenSpec terms
- [x] 4.2 Classify follow-up work by prerequisites such as visual review, GitHub metadata write access, or real GPU validation
- [x] 4.3 Define the archive-ready stop line: what must be complete, what can remain out of scope, and when the repository is ready for final maintenance mode
- [x] 4.4 Queue the next execution batches with explicit `/review` checkpoints so future work lands without process drift

## Final execution backlog produced by this closeout pass

### Batch A: Completed in this change

- [x] remove tracked hidden `.claude/skills` bundles and block their reintroduction
- [x] upgrade the active closeout OpenSpec change from a rough outline to an execution-ready umbrella change
- [x] align visible AI guidance files around one closeout-biased instruction surface
- [x] tighten Pages navigation and site metadata so README, Pages, and GitHub About share one narrative
- [x] simplify workflow triggers so support automation is manual where appropriate and routine CI stays CPU-safe

### Batch B: Future work that requires GitHub write access

- use `gh` to keep repository description and homepage URL aligned with the landing-page narrative
- if topics materially drift from the repo narrative, update them with `gh repo edit --add-topic/--remove-topic` in the same change that updates README or Pages copy
- `/review` checkpoint: any metadata change that lands with README, Pages, or workflow edits should be reviewed before merge because it changes the public repo surface

### Batch C: Future work that requires visual review

- only revisit the HonKit landing-page layout when the public narrative changes enough to justify new mockups; route that work through visual review first rather than ad hoc CSS churn
- `/review` checkpoint: any non-trivial layout or navigation change should be reviewed together with the updated Pages build output

### Batch D: Future work that requires a real GPU machine

- any CUDA runtime, kernel, or performance-sensitive follow-up must validate on a real GPU machine using the documented root or module-specific build/test paths
- `/review` checkpoint: GPU-sensitive follow-up should not merge until the local GPU validation result is recorded in the change notes or PR discussion

## Archive-ready stop line

The repository is considered ready for final maintenance mode when all of the following remain true:

- OpenSpec is the only authoritative workflow surface for non-trivial changes
- visible repo docs, Pages inputs, and AI guidance files agree on closeout posture and tool boundaries
- hidden or duplicate instruction systems are absent from the tracked repository surface
- GitHub Actions remains limited to high-value CPU-safe checks plus manual support workflows
- GitHub About metadata, README, and Pages continue to describe the same project narrative
- GPU-dependent validation is kept explicit and out of pretend-CI workflows

Anything outside that boundary is intentionally out of scope for closeout unless it directly breaks the documented verification path or public repository narrative.
