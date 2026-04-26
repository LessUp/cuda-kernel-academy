# Specs: Finalize Project Closeout

## ADDED Requirements

### Requirement: Closeout-oriented repository governance

User story: As the maintainer, I want the repository to optimize for finishing well rather than expanding indefinitely, so that future work stays focused and low-noise.

#### Scenario: Workflow stays single-source

- **WHEN** contributors perform non-trivial work
- **THEN** the repository SHALL use OpenSpec as the only authoritative spec workflow

#### Scenario: Closeout bias is explicit

- **WHEN** contributors read the repo guidance
- **THEN** the guidance SHALL emphasize stabilization, cleanup, and correctness over feature growth

#### Scenario: Active closeout change is execution-ready

- **WHEN** maintainers open `openspec/changes/finalize-project-closeout/`
- **THEN** the change SHALL contain phase-structured proposal, design, spec, and tasks artifacts detailed enough to drive implementation without relying on chat history alone

### Requirement: Documentation stays curated and low-noise

User story: As a maintainer in the final stabilization phase, I want a small, trustworthy documentation set, so that I do not have to sort through stale or duplicate guidance.

#### Scenario: Low-value docs are pruned

- **WHEN** the repository is reviewed for closeout
- **THEN** stale, duplicate, or low-signal docs and changelog clutter SHALL be removed or consolidated into durable source-of-truth documents

#### Scenario: Durable docs stay aligned

- **WHEN** a contributor reads workflow, setup, AI tooling, or navigation docs
- **THEN** those docs SHALL agree on the real repository shape, supported workflows, and verification path

### Requirement: Public presentation stays coherent

User story: As a prospective user, I want the repository and Pages site to present a clear learning narrative, so that I can quickly decide whether the project is useful to me.

#### Scenario: Landing page explains the project clearly

- **WHEN** a visitor opens the repository or Pages home page
- **THEN** the page SHALL present the project value proposition, module map, and entry points without requiring deep scrolling through generic docs

#### Scenario: Repo metadata matches the site

- **WHEN** a visitor reads the GitHub About section
- **THEN** the description, homepage URL, and topics SHALL match the project narrative exposed by the landing page

#### Scenario: Navigation is curated

- **WHEN** a visitor browses the Pages sidebar or docs index
- **THEN** the navigation SHALL reflect the curated docs set rather than exposing stale or redundant paths

### Requirement: Final maintenance work stays verifiable

User story: As a future maintainer using AI tooling, I want a compact but trustworthy verification path, so that final bug-fix work can ship without workflow confusion.

#### Scenario: Verification path is explicit

- **WHEN** a contributor lands a change
- **THEN** the repository SHALL document the exact repo-native checks to run for docs, workflow, and CUDA-related changes

#### Scenario: Tooling guidance is repo-specific

- **WHEN** contributors use Copilot, Claude Code, Codex-style agents, or editor LSPs
- **THEN** the repository SHALL provide repo-specific guidance instead of generic AI boilerplate

#### Scenario: Workflow automation stays lean

- **WHEN** GitHub Actions workflows are reviewed during closeout
- **THEN** the repository SHALL keep only CPU-safe, high-value checks and SHALL avoid redundant triggers or overdesigned CI behavior

#### Scenario: Version anchors stay consistent

- **WHEN** the repository documents minimum or recommended tool versions
- **THEN** the visible version anchors across docs, workflows, and setup guidance SHALL not contradict one another without an explicit reason

### Requirement: AI tooling stays bounded and handoff-friendly

User story: As a future AI-assisted maintainer, I want the repository's instruction surface and tool choices to be explicit and minimal, so that I can continue closeout work without re-learning hidden conventions.

#### Scenario: Official instruction surfaces agree

- **WHEN** contributors read `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, or `docs/AI_TOOLING.md`
- **THEN** those files SHALL agree on closeout posture, OpenSpec-first workflow, review expectations, and tool boundaries

#### Scenario: LSP default path is explicit

- **WHEN** contributors configure code navigation
- **THEN** the repository SHALL document `clangd` plus shared compile commands as the primary LSP path

#### Scenario: MCP and Skills trade-off is explicit

- **WHEN** contributors consider heavier AI integrations
- **THEN** the repository SHALL state when built-in tooling or lightweight CLI Skills are preferred over additional MCP surface area

### Requirement: Final execution backlog is durable

User story: As the maintainer handing the project to a later execution phase, I want a structured final backlog, so that follow-up work can be executed in order without redefining the scope again.

#### Scenario: Backlog captures execution constraints

- **WHEN** the closeout audit finishes
- **THEN** the resulting task list SHALL classify work by prerequisites such as visual review, GitHub metadata write access, or real GPU validation

#### Scenario: Archive-ready boundary is explicit

- **WHEN** contributors decide whether the repository is ready for final archival posture
- **THEN** the closeout artifacts SHALL define what must be true before the work is considered complete and what is intentionally left out of scope
