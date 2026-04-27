# Proposal: Finalize Project Closeout

## Why

CUDA Kernel Academy already contains the core learning material, but the repository still needs a final closeout-oriented pass so future maintenance stays small, predictable, and high-signal.

The remaining work is not about expanding scope. It is about finishing the repository well:

- stabilize the public narrative
- fix lingering correctness or workflow issues
- keep documentation and automation aligned with the real project shape
- reduce the amount of AI/process drift that accumulates over time
- leave a trustworthy execution backlog that another model or maintainer can finish without re-discovering the repository

## What Changes

### New Capabilities

1. A phase-driven closeout program that turns repository hardening into an executable OpenSpec backlog
2. A single operating model for OpenSpec-driven maintenance work during the final stabilization pass
3. A final handoff-oriented task structure that can be executed incrementally by future AI agents or maintainers

### Modified Capabilities

1. Existing workflow, documentation, and AI guidance will be trimmed toward closeout-friendly maintenance
2. Pages, repo metadata, and engineering docs will emphasize clarity over coverage
3. Verification guidance, version anchors, and workflow boundaries will be made explicit and internally consistent

### Removed Capabilities

1. Legacy parallel process surfaces that do not contribute to final maintenance quality
2. Low-value docs, changelog clutter, or workflow/config complexity that creates maintenance drag without helping real contributors

## Impact

- lower maintenance drag
- cleaner handoff to future AI-assisted final fixes
- more trustworthy public presentation
- a smaller and more explicit set of engineering and AI instruction surfaces

## Implementation Strategy

This closeout program will be driven as a **single umbrella OpenSpec change** rather than split into long-lived parallel changes.

The execution order is intentionally sequential:

1. **Phase 1: aggressive audit and normalization**
   - tighten the OpenSpec artifact set
   - prune stale docs and inconsistent workflow guidance
   - bound bug fixing and version cleanup to high-value surfaces
2. **Phase 2: engineering and GitHub integration**
   - simplify workflows
   - align Pages, README, and GitHub About metadata
3. **Phase 3: AI tooling and Vibe Coding configuration**
   - rebuild the AI instruction surfaces around the real repository
   - standardize LSP guidance and keep MCP usage selective
4. **Phase 4: final backlog and handoff**
   - turn findings into a durable task list for the final execution pass

The repository already has partial closeout infrastructure. This proposal upgrades it into a rigorous closeout program rather than introducing a new process layer.
