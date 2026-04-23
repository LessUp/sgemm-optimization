## ADDED Requirements

### Requirement: OpenSpec is the authoritative governance system
The repository MUST use OpenSpec as the single authoritative system for repository requirements, workflow rules, and closeout-state governance decisions.

#### Scenario: Contributor needs the normative source of truth
- **WHEN** a contributor or AI agent needs project requirements, workflow rules, or repository governance guidance
- **THEN** the authoritative source MUST be the stable capability specs under `openspec/specs/` together with active change artifacts under `openspec/changes/`

### Requirement: Governance instructions have clear ownership
The repository MUST keep governance instructions concise, non-redundant, and separated by responsibility rather than by repetitive tool-specific restatement.

#### Scenario: Contributor reads repository instruction files
- **WHEN** a contributor opens root governance files such as `AGENTS.md`, `CLAUDE.md`, or any project-level Copilot instruction file
- **THEN** each file MUST have a distinct responsibility, avoid generic duplicated boilerplate, and defer to the authoritative OpenSpec workflow where normative process rules apply

### Requirement: Automation is archive-ready and high-signal
The repository MUST keep only automation that materially improves closeout-state quality, consistency, or publication.

#### Scenario: Repository change triggers automation
- **WHEN** a pull request or push matches the repository automation rules
- **THEN** workflows MUST be limited to high-signal validation or publishing tasks, align with the default branch strategy, and avoid redundant or low-value executions

### Requirement: Shared developer tooling is lightweight and portable
The repository MUST provide a project-level tooling baseline that works across multiple AI and editor environments without depending on heavy, context-expensive integrations.

#### Scenario: Contributor sets up local tooling
- **WHEN** a contributor enables hooks, language tooling, or AI-assisted workflow features for the repository
- **THEN** the primary baseline MUST rely on shared project foundations such as CMake-generated compile commands, clangd-compatible configuration, minimal hooks, and native CLI or skill-driven workflows before optional MCP-style integrations are considered
