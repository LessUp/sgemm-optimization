## ADDED Requirements

### Requirement: Repository validation includes governance integrity
The repository test and validation model MUST cover not only code correctness but also the integrity of governance, specification, and documentation structure.

#### Scenario: Governance-related files change
- **WHEN** OpenSpec files, workflow files, governance documents, or documentation structure are modified
- **THEN** the repository MUST provide CI-safe validation that checks the relevant specification and structural invariants before those changes are treated as complete

### Requirement: Validation expectations are split by execution environment
The repository MUST document which checks are expected to run in hosted CI and which checks require a local GPU-capable environment.

#### Scenario: Contributor reads validation guidance
- **WHEN** a contributor prepares to validate repository changes
- **THEN** the documented workflow MUST separate CI-safe checks such as formatting, compilation, OpenSpec validation, or Pages buildability from GPU-required runtime verification and benchmarking
