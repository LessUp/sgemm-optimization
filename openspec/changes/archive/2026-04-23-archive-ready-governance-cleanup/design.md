## Context

This repository is functionally rich enough to close out, but it is not yet governable in a clean way. The core SGEMM code layout is reasonable, yet the surrounding repository has drifted:

- `openspec/specs/*` is acting as the practical source of truth, while `openspec/config.yaml`, `specs.md`, and `openspec/README.md` still describe older paths and conventions.
- Guidance is split between root `AGENTS.md`, `openspec/AGENTS.md`, generated `.claude/` assets, and missing project-level `CLAUDE.md` / Copilot-specific instructions.
- README, Pages, docs, changelog, and release notes overlap heavily, which weakens both maintainability and project presentation.
- Workflow triggers are not fully aligned with the current default branch or with the desired archive-ready maintenance posture.
- The repository already has GitHub metadata and Pages enabled, so the work is not greenfield; it is a consolidation and correction pass.

The chosen operating mode for this closeout is **strict pruning**: when two artifacts serve the same purpose, the repository should keep one authoritative version and remove the rest.

## Goals / Non-Goals

**Goals:**

- Make OpenSpec the single authoritative governance and requirements system for repository-level decisions.
- Collapse redundant instructions, release-history files, and documentation surfaces into a small, high-signal set of maintained artifacts.
- Redesign the GitHub-facing surfaces (README, Pages, metadata) so the project is legible and attractive to new users.
- Keep automation and tooling intentionally small: only workflows, hooks, and configs that materially improve correctness or contributor throughput should remain.
- Establish a shared project baseline for clangd/LSP, hooks, and AI-assistant usage that works across Claude/Codex/Copilot style tools with minimal duplication.
- Produce an implementation sequence that is safe to execute in a long-running autopilot session without relying on `/fleet` as the default mode.

**Non-Goals:**

- Adding new SGEMM kernels, new benchmark features, or new end-user product scope.
- Replacing the CUDA/CMake/Google Test stack.
- Building a large multi-runner CI platform or adding heavyweight external services.
- Preserving redundant historical files solely as placeholders once their content is safely represented elsewhere.

## Decisions

### 1. `openspec/specs/` becomes the only authoritative stable spec location

**Decision:** Repository documentation and configuration will be aligned so that `openspec/specs/<capability>/spec.md` is the sole stable-spec source, while change-specific deltas continue to live under `openspec/changes/<change>/specs/`.

**Why this choice:** The repository already stores the living capability specs in `openspec/specs/*`. Keeping that structure avoids another migration and lets `specs.md` become a documentation entry point instead of a second spec system.

**Alternatives considered:**

- **Reintroduce root `specs/` as stable specs**: rejected because it would re-expand the same duplication problem that already caused drift.
- **Keep both paths and explain the difference better**: rejected because two normative sources would continue to confuse both humans and AI agents.

### 2. Governance documents will be split by responsibility, not by tool vanity

**Decision:** Root `AGENTS.md` will carry concise cross-tool project rules, `CLAUDE.md` will carry only Claude-specific operational guidance, and any Copilot instruction file will be added only if it contains project-specific guidance that does not merely restate the other files.

**Why this choice:** The problem is not the absence of documents; it is duplicated, generic documents with fuzzy ownership. Clear boundaries make the docs maintainable and easier for automation to trust.

**Alternatives considered:**

- **One giant governance file for every tool**: rejected because it becomes verbose and hard to keep accurate.
- **Separate fully independent instructions for every tool**: rejected because it maximizes drift.

### 3. Documentation will follow a three-surface model

**Decision:** The repository will treat documentation as three surfaces:

1. **README**: compact repository entry point and quick-start
2. **GitHub Pages + `docs/`**: presentation and guided technical learning
3. **OpenSpec**: normative requirements, process, and engineering governance

`CHANGELOG.md` stays only for meaningful version changes. `RELEASE_NOTES.md` will be removed unless it retains unique user-facing information after consolidation.

**Why this choice:** Each surface serves a different audience. Mixing them causes duplication and weakens both onboarding and maintenance.

**Alternatives considered:**

- **Keep long-form README and mirror it into Pages**: rejected because it duplicates content while satisfying neither repository entry nor landing-page goals well.
- **Move all docs into OpenSpec**: rejected because OpenSpec should govern the system, not replace tutorial or presentation content.

### 4. Automation must reflect archive-ready maintenance, not expansion

**Decision:** Workflows will be trimmed to a minimal set of high-signal checks: formatting/style, CI-safe compile/build validation, OpenSpec/repository-structure validation, and Pages deployment. Triggers will be aligned with the default branch and pull requests, with no redundant workflow fan-out.

**Why this choice:** The project is entering a finishing phase. Noise in CI is a maintenance liability, not a feature.

**Alternatives considered:**

- **Add more workflows for more coverage**: rejected because runtime GPU coverage is not available in GitHub-hosted CI and extra workflows would mostly increase noise.
- **Collapse everything into one mega-workflow**: rejected because Pages deployment and repository validation have different trigger patterns and failure modes.

### 5. Shared tooling baseline will prefer portable foundations over tool-specific magic

**Decision:** The project-wide developer tooling baseline will center on CMake-generated `compile_commands.json`, `clangd`, minimal `.githooks`, and native CLI/skill usage. MCP integrations will be treated as optional and added only where they clearly outperform built-in tools or skills.

**Why this choice:** LSP fundamentals are largely tool-agnostic. Building the shared foundation once yields value across Claude/Codex/Copilot workflows without committing the repository to heavy context-consuming integrations.

**Alternatives considered:**

- **Tool-specific IDE/plugin configurations first**: rejected because they solve the thinnest layer while leaving the common foundation inconsistent.
- **Broad MCP adoption**: rejected because the repository size and user preference do not justify the extra context and operational cost.

## Risks / Trade-offs

- **Aggressive pruning may remove historical breadcrumbs users still read** → Mitigation: keep genuinely useful historical context in Git history, archived OpenSpec changes, and a smaller but higher-quality changelog.
- **Dirty worktree increases merge risk during cleanup** → Mitigation: inspect affected files carefully, avoid destructive resets, and converge existing edits instead of overwriting them blindly.
- **GitHub metadata and Pages changes require authenticated `gh` access and repo permissions** → Mitigation: validate access early and stage metadata updates after the content model is settled.
- **No GPU runtime in CI means closeout validation cannot rely on hosted runners for full correctness** → Mitigation: make the boundary explicit in specs and keep CI focused on compile-time and repository-structure guarantees.
- **Cross-tool instruction files can drift again later** → Mitigation: centralize generic rules in one authoritative document and keep tool-specific files intentionally thin.

## Migration Plan

1. Normalize OpenSpec config, spec paths, and governance document ownership.
2. Consolidate README, docs, changelog/release-notes, and Pages information architecture under the new authoritative model.
3. Simplify workflows and engineering configs to match the closeout-state repository.
4. Add or refine shared tooling/hook/LSP guidance and only the thinnest necessary tool-specific instructions.
5. Sync GitHub metadata after the presentation layer is finalized.
6. Run a bug and consistency sweep across code, docs, specs, and automation before archiving the change.

Rollback is straightforward at the repository level because the cleanup is file- and config-oriented; each phase can be reverted via Git if a consolidation decision proves too aggressive.

## Open Questions

- None for the repository structure itself.
- Full CUDA configure/build/test reruns still require a host with the CUDA toolkit and `nvcc` available; the governance and OpenSpec validation baseline can run without that dependency.
