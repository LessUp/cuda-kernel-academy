# AI Tooling Guide

This repository supports AI-assisted development, but the toolchain is intentionally conservative.

## Source-of-truth instruction surfaces

Use these in order:

1. `AGENTS.md`
2. `CLAUDE.md`
3. `.github/copilot-instructions.md`
4. the active OpenSpec change in `openspec/changes/<change-name>/`

Do not create competing instruction systems.

## Recommended workflow by tool

### GitHub Copilot

- rely on `.github/copilot-instructions.md` for repo context
- use `/review` before merge on non-trivial work
- use `/research` only when you need current external information
- use `/remote` for bounded background tasks
- avoid `/fleet` unless the task clearly needs large parallel context

### Claude Code

- rely on `CLAUDE.md` plus the active OpenSpec change
- prefer the repo `verify` skill for repeated validation
- keep local-only preferences in untracked `CLAUDE.local.md`

### Codex-style agents

- follow `AGENTS.md` and the active OpenSpec change
- keep work scoped and merge-oriented
- do not invent parallel specs or shadow docs

### OpenCode / other agents

This repository does **not** check in an OpenCode-specific config file by default.

Reason:

- a repo-specific config is only worth committing when the schema is stable and official
- low-confidence or tool-specific config files quickly become stale
- the durable source of truth here is the repo documentation plus OpenSpec artifacts

## LSP strategy

For this repository, the primary language-server stack is:

1. **clangd** for C / C++ / CUDA navigation
2. **CMake Tools or editor-native CMake support** for CMake editing
3. editor-native YAML / JSON / Markdown support

### Why clangd

- it works across many editors and agent environments
- it benefits directly from CMake-generated `compile_commands.json`
- it is the most portable common denominator across Copilot, Claude-oriented editors, and Codex-oriented editors

### Recommended local setup

```bash
sudo apt-get install clangd

cmake --preset default
./scripts/dev/sync-compile-commands.sh default
```

The repository does **not** require `cmake-language-server` by default. It can be useful in some editors, but its packaging and Python-version compatibility drifts more often than `clangd`, so it is treated as optional rather than part of the canonical repo setup.

### Important note on “tool universality”

The **language server itself is tool-agnostic**. What changes between Copilot, Claude Code, Codex, or editor plugins is **how each client consumes LSP data**.

For this repository, the project-side action is:

- generate compile commands with CMake presets
- sync them to the repo root
- let each editor or agent consume the same compilation database

Use:

```bash
cmake --preset default
./scripts/dev/sync-compile-commands.sh default
```

## MCP policy

Keep MCP usage lean.

Use extra MCP integrations only when they provide recurring value that outweighs context cost. For this repository, lightweight repo instructions plus built-in tooling are usually better than large always-on MCP stacks.

## Recommended editor extensions

If your editor supports workspace recommendations, prefer:

- `clangd`
- CMake support
- GitHub Copilot / Copilot Chat
- YAML support
- Markdown linting

See `.vscode/extensions.json` for the repo recommendation set.
