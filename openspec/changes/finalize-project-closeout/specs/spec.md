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

### Requirement: Public presentation stays coherent

User story: As a prospective user, I want the repository and Pages site to present a clear learning narrative, so that I can quickly decide whether the project is useful to me.

#### Scenario: Landing page explains the project clearly

- **WHEN** a visitor opens the repository or Pages home page
- **THEN** the page SHALL present the project value proposition, module map, and entry points without requiring deep scrolling through generic docs

#### Scenario: Repo metadata matches the site

- **WHEN** a visitor reads the GitHub About section
- **THEN** the description, homepage URL, and topics SHALL match the project narrative exposed by the landing page

### Requirement: Final maintenance work stays verifiable

User story: As a future maintainer using AI tooling, I want a compact but trustworthy verification path, so that final bug-fix work can ship without workflow confusion.

#### Scenario: Verification path is explicit

- **WHEN** a contributor lands a change
- **THEN** the repository SHALL document the exact repo-native checks to run for docs, workflow, and CUDA-related changes

#### Scenario: Tooling guidance is repo-specific

- **WHEN** contributors use Copilot, Claude Code, Codex-style agents, or editor LSPs
- **THEN** the repository SHALL provide repo-specific guidance instead of generic AI boilerplate
