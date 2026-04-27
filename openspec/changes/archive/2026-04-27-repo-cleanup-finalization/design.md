# Design: Repository Cleanup Finalization

## Design Decisions

### D1: Aggressive Documentation Trimming

**Decision**: Remove all generic C++/CUDA best practices from documentation.

**Rationale**:
- Project is in closeout mode
- Generic content is available in NVIDIA documentation
- Reduces maintenance burden
- Users seeking generic info should use official sources

**Trade-offs**:
- (+) Less documentation to maintain
- (+) More focused content
- (-) Users may need to reference external docs for basics

### D2: FAQ Merge into TROUBLESHOOTING

**Decision**: Merge FAQ.md into TROUBLESHOOTING.md and README.md.

**Rationale**:
- Significant overlap between FAQ and TROUBLESHOOTING
- Reduces duplicate content
- Single source for troubleshooting

**Trade-offs**:
- (+) No duplicate content
- (+) Single troubleshooting reference
- (-) Slightly longer TROUBLESHOOTING (still under 300 lines)

### D3: Visual Navigation Enhancement

**Decision**: Add emojis and visual elements to SUMMARY.md.

**Rationale**:
- Improves user experience
- Makes navigation more intuitive
- Highlights key sections

**Trade-offs**:
- (+) Better UX
- (+) More engaging presentation
- (-) May not render perfectly in all environments

### D4: Pinned Dependency Versions

**Decision**: Use pinned versions for all GitHub Actions.

**Rationale**:
- Reproducible builds
- No surprise breakages from upstream changes
- Follows best practices

**Trade-offs**:
- (+) Stable CI
- (-) Requires manual updates for new versions

## File Changes Summary

| File | Action | Lines Before | Lines After |
|------|--------|--------------|-------------|
| `.claude/skills/` | Delete | - | - |
| `_bmad/` | Delete | - | - |
| `_bmad-output/` | Delete | - | - |
| `changelog/` | Delete | - | - |
| `docs/CODING_STYLE.md` | Rewrite | 569 | 139 |
| `docs/TROUBLESHOOTING.md` | Rewrite | 532 | 123 |
| `docs/FAQ.md` | Delete | 333 | - |
| `docs/integration_examples.md` | Rename | 374 | 374 |
| `SUMMARY.md` | Rewrite | 37 | 62 |
| `book.json` | Update | 95 | 83 |
| `.github/workflows/ci.yml` | Update | 83 | 83 |
