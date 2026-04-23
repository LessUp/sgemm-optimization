# Copilot Instructions

This repository is in a **closeout-oriented** phase. Optimize for consolidation, clarity, and archive-ready stability.

## Repository Rules

- Treat `openspec/specs/*` as the authoritative stable spec source.
- Use active change artifacts under `openspec/changes/<change>/` as the implementation contract.
- Prefer deleting or merging redundant docs, workflows, and configs over preserving duplicate files.
- Keep `README.md` as the repo entry point and `index.md` + `docs/` as the public landing/documentation surface.

## Working Preferences

- Prefer CMake-based validation and `openspec validate --all` for repository-wide changes.
- Use `gh` for repository metadata and GitHub-side operations.
- Assume `clangd` + `compile_commands.json` is the shared LSP baseline.
- Avoid adding project-level instructions unless they are specific to this repository and do not duplicate existing guidance.
