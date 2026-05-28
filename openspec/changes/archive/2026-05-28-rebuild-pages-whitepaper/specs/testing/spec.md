## MODIFIED Requirements

### Requirement: Repository validation includes governance integrity
The repository test and validation model MUST cover not only code correctness but also the integrity of governance, documentation structure, mirrored public routes, and theme-aware figure conventions.

#### Scenario: Governance-related files change
- **WHEN** OpenSpec files, workflow files, governance documents, or documentation structure are modified
- **THEN** the repository MUST provide CI-safe validation that checks the relevant specification and structural invariants before those changes are treated as complete

#### Scenario: Whitepaper presentation surfaces are reorganized
- **WHEN** Pages navigation, mirrored route inventory, or curated figure embedding changes
- **THEN** CI-safe docs tests MUST verify that the mirrored public routes still exist
- **AND** those tests MUST verify that curated whitepaper figures use the shared theme-aware convention instead of theme-fragile ad hoc markup
