## ADDED Requirements

### Requirement: Published architecture matches the real repository structure
The repository architecture guidance MUST describe only the directory structure, documentation boundaries, and engineering surfaces that actually exist and are maintained.

#### Scenario: Contributor consults architecture guidance
- **WHEN** a contributor reads architecture-facing documentation or specifications
- **THEN** all referenced repository paths, layers, and responsibilities MUST correspond to the real maintained layout and MUST NOT reference stale or superseded structures as authoritative

### Requirement: Engineering boundaries are explicit
The repository architecture MUST make local-only and CI-safe responsibilities explicit so maintainers can reason correctly about build, test, and validation coverage.

#### Scenario: Contributor decides how to validate a change
- **WHEN** a contributor evaluates required validation steps for code, docs, specs, or workflow changes
- **THEN** the architecture guidance MUST clearly distinguish local GPU-dependent verification from CI-safe compile, structure, and publication checks
