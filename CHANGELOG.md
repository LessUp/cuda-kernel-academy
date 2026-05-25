# Changelog

The repository keeps a lightweight changelog focused on release-level milestones, not day-by-day narrative notes.

## [Unreleased]

### Changed

- removed legacy spec archives and AI control scaffolding from the tracked repository surface
- consolidated change history into the root `CHANGELOG.md` and removed module-level changelog drift
- simplified contributor and agent guidance around a small, verification-first workflow
- rebuilt the landing page and Pages navigation around a showcase-first information architecture
- tightened Pages-facing docs around the repository's actual build seams and learning path

## [1.0.0] - 2026-01-07

### Added

- four-module CUDA learning path covering SGEMM, reusable kernels, advanced optimization, and lightweight inference
- root CMake build for shared modules plus a standalone Makefile flow for `01-sgemm-tutorial`
- documentation set for installation, troubleshooting, coding style, and module-level deep dives

[Unreleased]: https://github.com/LessUp/cuda-kernel-academy/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LessUp/cuda-kernel-academy/releases/tag/v1.0.0
