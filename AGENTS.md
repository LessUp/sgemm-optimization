# AGENTS.md

Repository-wide guidance for AI agents and automated contributors.

## Project Posture

This repository is in a **closeout-oriented** stage.

- Prefer **consolidation over expansion**
- Prefer **one authoritative source** over duplicate explanations
- Prefer **deleting or merging** low-value files over keeping placeholders
- Prefer **one longer apply/autopilot session** over frequent `/fleet` fan-out

## Authoritative Sources

| Need | Source |
|------|--------|
| Stable requirements and governance | `openspec/specs/*` |
| Active implementation plan | `openspec/changes/<change>/` |
| OpenSpec workflow details | `openspec/README.md`, `openspec/AGENTS.md` |
| Repository entry point | `README.md` |
| Public landing/documentation entry | `index.md` + `docs/` |

If two files say the same thing, keep one and remove the other.

## Default Workflow

1. `/opsx:explore` when scope or trade-offs are unclear
2. `/opsx:propose "description"` for repo-wide or behavior-affecting changes
3. `/opsx:apply` to execute the task list
4. `/review` before large deletions, workflow changes, or archive
5. `/opsx:archive` only after specs, docs, workflows, and validation agree

## Branch and Worktree Policy

- `master` is the only long-lived branch.
- Short-lived local branches or worktrees are allowed only as temporary isolation for a specific task.
- Before closeout, merge completed work back to `master`, delete temporary branches/worktrees, and ensure local and remote branch lists contain no stale task branches.
- Do not add release/version branch automation; this repository favors a single-mainline closeout flow.

## Repository Boundaries

### Code

- CUDA kernels live in `src/kernels/`
- Shared utilities live in `src/utils/`
- Entry point lives in `src/main.cu`
- Tests live in `tests/test_sgemm.cu`

### Documentation

- `README.md`: concise repo entry and quick-start
- `index.md` + `docs/`: public project presentation and deeper learning content
- `CHANGELOG.md`: meaningful release history only
- `CONTRIBUTING.md`: compact collaboration instructions
- `openspec/*`: normative process and requirements

Do not duplicate the same explanation across these surfaces.

## Validation Baseline

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
openspec validate --all
```

- Hosted CI is compile-time and structure validation only
- GPU runtime verification and benchmarking are local-only

## Tooling Preferences

- Use `gh` for repository metadata, workflow inspection, and GitHub-side operations
- Treat CMake-generated `compile_commands.json` plus `clangd` as the shared LSP baseline
- Keep hooks minimal and project-specific
- Prefer native CLI tools, OpenSpec skills, and focused reviews over heavy MCP usage

## Tool Routing

- Use `/review` before major deletions, workflow topology changes, or OpenSpec archive.
- Use subagents for bounded research, log inspection, or clearly separable analysis.
- Use native CLI plus `gh` for repository metadata, workflows, and GitHub maintenance.
- Treat Claude, Copilot, OpenCode, and similar tools as sharing the same OpenSpec and clangd baseline; avoid tool-specific drift.
- Add MCP or plugins only when built-in tools or skills leave a concrete gap.

## Quality Rules

- Keep kernel interfaces consistent with the existing template launcher pattern
- Preserve RAII-based CUDA resource handling and exception-based error reporting
- Avoid generic governance boilerplate; write instructions that are specific to this repository
- When cleaning up docs or workflows, ensure retained files have a single obvious responsibility
