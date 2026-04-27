# Spec: Repository Cleanup Finalization

## Requirements

### REQ-1: Directory Cleanup
The repository shall not contain directories that violate AGENTS.md guidelines:
- `.claude/skills/` - prohibited hidden instruction bundles
- `_bmad/` - redundant framework in closeout mode
- `_bmad-output/` - historical artifacts
- `changelog/` - empty directory

### REQ-2: Documentation Conciseness
Documentation files shall be project-specific and concise:
- `docs/CODING_STYLE.md` < 300 lines
- `docs/TROUBLESHOOTING.md` < 300 lines
- Generic content removed, linked to external resources instead

### REQ-3: File Naming Consistency
Documentation files shall use kebab-case naming:
- `docs/integration-examples.md` (not `integration_examples.md`)

### REQ-4: Git Pages Quality
Git Pages navigation shall be:
- Visually appealing with icons/emojis
- Hierarchically organized
- Include external resource links

### REQ-5: Dependency Version Anchoring
GitHub Actions shall use pinned versions, not `latest`

## Scenarios

### Scenario 1: Clean Repository
**Given** the repository is in closeout mode
**When** a user inspects the directory structure
**Then** no violating directories shall be present

### Scenario 2: Concise Documentation
**Given** a user reads project documentation
**When** they open CODING_STYLE.md or TROUBLESHOOTING.md
**Then** they find project-specific content under 300 lines

### Scenario 3: Git Pages Navigation
**Given** a user visits the GitHub Pages site
**When** they view the sidebar navigation
**Then** they see organized sections with visual elements
