# AGENTS.md — AI Agent Workflow Instructions

## Project Philosophy: Spec-Driven Development (SDD)

This project strictly follows the **Spec-Driven Development (SDD)** paradigm. All code implementations must use the `/specs` directory as the Single Source of Truth.

## Directory Context

| Directory | Purpose |
|-----------|---------|
| `/specs/product/` | Product feature definitions and acceptance criteria |
| `/specs/rfc/` | Technical design documents and architecture decisions |
| `/specs/api/` | API interface definitions (if applicable) |
| `/specs/testing/` | Test specifications and BDD scenarios |
| `/docs/` | User-facing documentation, tutorials, and guides |
| `/src/` | Source code implementations |
| `/tests/` | Test implementations |

## AI Agent Workflow Instructions

When you (the AI) are asked to develop a new feature, modify existing functionality, or fix a bug, **you must strictly follow this workflow without skipping any steps**:

### Step 1: Review Specs

- **First**, read the relevant documents in `/specs` (product requirements, RFCs, API definitions, test specs).
- **If** the user's request conflicts with existing specs, **stop immediately** and point out the conflict. Ask the user whether to update the specs first.
- **Never** start coding without understanding the spec context.

### Step 2: Spec-First Update

- **If** this is a new feature, or if existing interfaces/database structures need to change, **you must first propose modifications to the appropriate spec documents** (e.g., `specs/product/*.md`, `specs/rfc/*.md`, or `specs/testing/*.md`).
- **Wait** for user confirmation on the spec changes before entering the code implementation phase.
- **Never** implement code changes that would invalidate existing specs without updating the specs first.

### Step 3: Code Implementation

- When writing code, **100% adhere to the spec definitions** (including variable naming, API paths, data types, status codes, etc.).
- **No gold-plating**: Do not add features in code that are not defined in the specs.
- Follow the architectural patterns and design decisions documented in `/specs/rfc/`.
- Use the project's established coding conventions (clang-format, RAII, exception-based error handling).

### Step 4: Test Against Specs

- Write unit tests and integration tests based on the acceptance criteria in `/specs`.
- Ensure test cases cover all boundary conditions described in the specs.
- Verify that implementations match the verification tolerances and performance expectations defined in the specs.

## Code Generation Rules

1. **Spec Compliance**: Any code that exposes API changes or behavioral changes must be justified by corresponding spec documents.
2. **No Spec Violations**: Do not implement patterns that contradict spec definitions (e.g., using `exit()` when specs require exception-based error handling).
3. **Reference Existing RFCs**: When uncertain about technical details, consult `/specs/rfc/` for architectural conventions. Do not invent design patterns independently.
4. **Test Coverage**: All new code must have corresponding tests that verify spec compliance.
5. **Documentation Sync**: When adding new features, update relevant spec documents and user-facing documentation (`/docs/`, `README.md`) accordingly.

## Project-Specific Guidelines

### SGEMM Kernel Development

When working on kernel implementations:

1. **Review** `specs/product/sgemm-kernel-requirements.md` for functional requirements.
2. **Review** `specs/rfc/0001-core-architecture.md` for interface design and error handling strategy.
3. **Review** `specs/testing/kernel-verification.md` for test scenarios and tolerance specifications.
4. **Implement** code that matches the unified kernel interface template.
5. **Verify** against cuBLAS with spec-defined tolerances (`rtol=1e-3, atol=1e-4` for standard, `rtol=5e-2, atol=1e-2` for Tensor Core).

### Build & Test Commands

```bash
# Build (CMake - recommended)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run benchmark
./build/bin/sgemm_benchmark
./build/bin/sgemm_benchmark -a  # all dimensions

# Run tests
cmake --build build --target test_sgemm
ctest --test-dir build

# Build & test (Make - quick local)
make GPU_ARCH=sm_86
make benchmark
make test
```

### Code Style

- **CUDA C++17** with clang-format enforcement
- **RAII** for all resource management (no raw `cudaFree`, use wrappers)
- **Exceptions** for error handling (no `exit()` in library code)
- **Template-based** kernel interfaces with default `TILE_SIZE=32`

## Why This Matters

1. **Prevents AI Hallucinations**: Forcing the AI to read `/specs` first anchors its reasoning to documented requirements and designs.
2. **Enforces Modification Path**: "Specs before code" ensures documentation and code stay synchronized (Document-Code Synchronization).
3. **Improves PR Quality**: When the AI generates Pull Requests, implementations align closely with business logic because they are derived from spec-defined acceptance criteria.
