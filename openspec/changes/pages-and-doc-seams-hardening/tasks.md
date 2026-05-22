# Tasks: Pages and Docs Seams Hardening

## Task 1 - Lock the failing tests first

- [ ] Add a static test for canonical GitHub/Pages URLs across public docs files
- [ ] Add a static test that rejects redirect-only logic and hardcoded base-path redirects in `docs/index.md`
- [ ] Add a static test for `04-inference-engine/README.md` artifact names against `04-inference-engine/CMakeLists.txt`
- [ ] Run the new tests and confirm they fail for the intended reasons

## Task 2 - Tighten public documentation seams

- [ ] Normalize public GitHub and Pages URLs across README files, contributing docs, issue template config, VitePress config, and selected docs pages
- [ ] Replace `docs/index.md` redirect shell with a real bilingual landing page
- [ ] Update locale homepages and nav/sidebar wiring to expose the deeper reading-path content

## Task 3 - Expand GitHub Pages content

- [ ] Add bilingual reading-path guides under `docs/en/guides/` and `docs/zh/guides/`
- [ ] Link the new guides from the docs root and locale navigation

## Task 4 - Fix the inference docs/build seam

- [ ] Align `04-inference-engine/README.md` target names and usage text with the real CMake targets
- [ ] Re-run the static artifact test

## Task 5 - Verify and close out

- [ ] Run `node --test docs/tests/*.test.mjs`
- [ ] Run `npm run docs:build`
- [ ] Run `cmake --list-presets`
