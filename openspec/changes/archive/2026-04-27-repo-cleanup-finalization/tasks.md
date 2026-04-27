# Tasks: Repository Cleanup Finalization

## Phase 1: Cleanup

- [x] 1.1 Delete `.claude/skills/` directory
- [x] 1.2 Delete `_bmad/` directory
- [x] 1.3 Delete `_bmad-output/` directory
- [x] 1.4 Delete empty `changelog/` directory
- [x] 1.5 Verify `.omc/` not tracked by git

## Phase 2: Documentation Governance

- [x] 2.1 Aggressively trim `docs/CODING_STYLE.md` (569 → 139 lines)
- [x] 2.2 Aggressively trim `docs/TROUBLESHOOTING.md` (532 → 123 lines)
- [x] 2.3 Merge `docs/FAQ.md` into TROUBLESHOOTING and README
- [x] 2.4 Delete `docs/FAQ.md`
- [x] 2.5 Rename `docs/integration_examples.md` to `docs/integration-examples.md`
- [x] 2.6 Update references in SUMMARY.md and docs/README.md

## Phase 3: Git Pages Enhancement

- [x] 3.1 Restructure SUMMARY.md with visual elements
- [x] 3.2 Optimize book.json configuration
- [x] 3.3 Remove non-existent plugin (flexible-link)
- [x] 3.4 Verify docs build successfully

## Phase 4: GitHub Integration

- [x] 4.1 Pin `lukka/get-cmake` to v4 in ci.yml
- [x] 4.2 Update repository description via gh CLI
- [x] 4.3 Update repository homepage URL via gh CLI
- [x] 4.4 (Attempted) Update repository topics via gh CLI

## Phase 5: Verification

- [x] 5.1 Run `pre-commit run --all-files` - all passed
- [x] 5.2 Check for TODO/FIXME comments - none found
- [x] 5.3 Verify docs build: `npm run docs:build-prod`
- [x] 5.4 Verify OpenSpec changes archived

## Archive-ready Stop Line

Repository is ready for final maintenance mode when:
- [x] All violating directories removed
- [x] Documentation concise and project-specific
- [x] Git Pages builds successfully
- [x] CI workflow uses pinned versions
- [x] OpenSpec change created and ready for archive
