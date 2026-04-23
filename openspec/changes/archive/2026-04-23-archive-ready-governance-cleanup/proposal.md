## Why

The repository has reached a closeout stage where the main risk is no longer missing features, but drift: OpenSpec structure and config disagree with the actual repo layout, project guidance is split across overlapping documents, GitHub Pages is not acting as a strong project entry point, and workflows/tooling are noisier than the maintenance goal requires. This change is needed now to aggressively consolidate the repository into one authoritative, archive-ready shape that is easier to trust, maintain, and eventually freeze.

## What Changes

- Normalize OpenSpec structure, configuration, and project instructions so the repo follows one authoritative spec-driven workflow.
- Rewrite governance documents (`AGENTS.md`, `CLAUDE.md`, and Copilot guidance) around a concise closeout-oriented development loop.
- Aggressively consolidate or remove low-value documentation, including duplicate landing content and redundant release-history material.
- Redesign GitHub Pages as a project presentation and documentation entry point instead of a README mirror.
- Prune GitHub workflows and engineering configuration to keep only high-signal automation aligned with the default branch and archive-ready maintenance goals.
- Standardize project-specific developer tooling guidance for hooks, clangd/LSP, Claude/Copilot/OpenCode usage, and minimal-MCP tradeoffs.
- Sync repository metadata (description, homepage, topics) with the cleaned project positioning using `gh`.
- Sweep and fix repository bugs and inconsistencies discovered during the closeout pass.

## Capabilities

### New Capabilities
- `repository-governance`: Defines the authoritative OpenSpec layout, project governance documents, automation policy, developer tooling conventions, and archive-ready maintenance workflow.
- `project-presentation`: Defines how README, GitHub Pages, and repository metadata present the project to new users and route them to the right technical content.

### Modified Capabilities
- `architecture`: Refine architectural and engineering decisions so repository structure, build/CI expectations, and documentation boundaries match the real closeout-state system.
- `testing`: Refine validation expectations to distinguish local GPU verification, CI-safe checks, and repository-structure/spec validation during the cleanup.

## Impact

- Affected code and configuration: `openspec/**`, root governance docs, documentation files, GitHub Pages files, `.github/workflows/*`, developer tooling config, and GitHub repository metadata.
- No intended public API expansion; the main impact is repository structure, quality gates, documentation clarity, and maintenance workflow.
- May remove or merge existing files whose content is redundant, outdated, or no longer authoritative.
