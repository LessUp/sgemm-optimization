# CLAUDE.md

Claude-specific deltas for this repository. Shared repository rules live in `AGENTS.md`; OpenSpec rules live in `openspec/AGENTS.md`.

## Command Flow

1. Use `/opsx:explore`, `/opsx:propose`, `/opsx:apply`, `/review`, and `/opsx:archive` for non-trivial repository-wide changes.
2. Use `/review` before deleting multiple files, changing workflow topology, or archiving an OpenSpec change.
3. Avoid `/fleet` unless the work is genuinely independent and will not create branch/worktree cleanup debt.

## Claude Operating Notes

- Keep one long-running closeout pass on `master` unless isolation is necessary; temporary worktrees must be merged and removed before completion.
- Prefer repository-local evidence, OpenSpec artifacts, `gh`, and clangd/compile commands over remote-only research.
- Do not duplicate `AGENTS.md` content here; update the shared guide if a rule applies to all AI tools.
