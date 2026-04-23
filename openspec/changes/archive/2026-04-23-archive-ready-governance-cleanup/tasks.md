## 1. OpenSpec foundation

- [x] 1.1 Normalize `openspec/config.yaml`, `specs.md`, and OpenSpec guidance docs so `openspec/specs/*` is the only authoritative stable spec path.
- [x] 1.2 Rewrite `AGENTS.md`, `openspec/AGENTS.md`, and create `CLAUDE.md` with clear ownership and a concise closeout-oriented OpenSpec workflow.
- [x] 1.3 Add only the thinnest necessary project-level Copilot/AI instruction files and remove or avoid redundant governance surfaces.

## 2. Documentation consolidation

- [x] 2.1 Rewrite `README.md` and `README.zh-CN.md` into concise repository entry points aligned with the cleaned project positioning.
- [x] 2.2 Redesign `index.md`, `_config.yml`, and key `docs/*.md` pages so GitHub Pages becomes a real project landing/documentation experience instead of a README mirror.
- [x] 2.3 Consolidate or remove redundant documentation artifacts such as low-value landing pages, duplicated architecture/process writeups, and `RELEASE_NOTES.md` if its content is fully absorbed elsewhere.
- [x] 2.4 Clean `CHANGELOG.md` and `CONTRIBUTING.md` so they keep only high-signal release and collaboration information.

## 3. Automation and tooling

- [x] 3.1 Prune `.github/workflows/*` to a minimal high-signal set and align triggers with the repository default branch and closeout-state maintenance policy.
- [x] 3.2 Add a minimal hook strategy and shared clangd/LSP baseline (`compile_commands.json` workflow, optional `.clangd`, hook install path) that works across Claude/Codex/Copilot style environments.
- [x] 3.3 Document the preferred usage boundaries for `/review`, subagents, OpenSpec commands, native CLI tools, and minimal MCP/plugin adoption.

## 4. Repository consistency and metadata

- [x] 4.1 Sweep repository bugs and inconsistencies across code, docs, specs, workflows, and configuration; fix the confirmed closeout-blocking issues.
- [x] 4.2 Run the relevant validation commands for formatting, buildability, tests, and OpenSpec/workflow integrity; adjust files until the closeout baseline is coherent.
- [x] 4.3 Use `gh` to update repository description, homepage/about link, and curated topics so the public metadata matches the cleaned Pages and README positioning.

## 5. Closeout readiness

- [x] 5.1 Do a final pass to ensure retained files have a single clear responsibility and removed files no longer leave broken references.
- [x] 5.2 Update the change artifacts as needed so the closeout cleanup is ready to archive once implementation is complete.
