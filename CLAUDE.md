# CLAUDE.md

Claude-specific operating notes for this repository.

## Preferred Command Flow

1. Use `/opsx:explore` if the cleanup scope or trade-offs are not clear.
2. Use `/opsx:propose` and `/opsx:apply` for any non-trivial repo-wide cleanup.
3. Use `/review` before:
   - deleting multiple files
   - changing workflow topology
   - archiving an OpenSpec change
4. Use `/opsx:archive` only after specs, docs, workflows, and validation are aligned.

## How to Work Here

- The repository is in a **finish-and-stabilize** phase, not a feature-expansion phase.
- Prefer one longer implementation pass over many small speculative branches.
- Use subagents for bounded research or log inspection, not for broad uncontrolled fan-out.
- Avoid `/fleet` by default; only use it when tasks are truly independent and parallelizable.

## Tooling Baseline

- Regenerate `compile_commands.json` through the normal CMake configure step.
- Treat `clangd` as the shared C/CUDA navigation baseline across tools.
- Prefer `gh` for About/topics/homepage, Actions, issues, and PR metadata.
- Prefer skills/native CLI over heavyweight MCP integration unless a concrete gain is obvious.
- Use `/review` as the primary quality gate before major cleanup milestones.
- Use `/research` or remote-only assistance sparingly, mainly when repository-local evidence is insufficient.

## Repository-Specific Reminders

- `openspec/specs/*` is the stable source of truth.
- `README.md` is the repository entry point; `index.md` + `docs/` are the public landing/documentation surface.
- Remove or merge redundant docs instead of preserving decorative placeholders.
- Keep instructions concise and repository-specific; do not re-import generic AI handbook content.
