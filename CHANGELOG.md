# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Documentation Reorganization**: Migrated to Spec-Driven Development (SDD) structure
  - Removed `.kiro/specs/` directory
  - Created `/specs/` directory with standardized structure:
    - `specs/product/` - Product requirements
    - `specs/rfc/` - Technical design documents (RFCs)
    - `specs/testing/` - Test specifications
  - Migrated and translated all spec documents to English
  - Added `AGENTS.md` for AI agent workflow instructions
  - Updated `README.md` and `CONTRIBUTING.md` to reference new spec structure

---

## [2.1.0] - 2026-04-16

### Changed
- **Documentation Refactor**: Complete documentation restructure
  - Consolidated changelog entries into single `CHANGELOG.md`
  - Rewrote `.kiro/specs/` documentation (design.md, requirements.md, tasks.md)
  - Enhanced `index.md` GitHub Pages landing page
  - Updated `_config.yml` for better Jekyll configuration
- **GitHub Workflows**: Simplified and optimized
  - `ci.yml`: Cleaner structure, better step naming
  - `pages.yml`: Fixed paths filter, improved concurrency group

### Removed
- **Dead Code Cleanup**: Removed 514 lines of unused code across 7 source files
  - `src/utils/cuda_utils.cuh`: Removed unused utility functions
  - `src/utils/verify.cuh`: Removed unused verification functions
  - `src/kernels/naive_sgemm.cuh`: Removed scaled kernel variants
  - `src/kernels/tiled_sgemm.cuh`: Removed scaled kernel variants
  - `src/kernels/bank_conflict_free_sgemm.cuh`: Removed transposed variant
  - `src/kernels/double_buffer_sgemm.cuh`: Removed register tiled variant
  - `src/kernels/tensor_core_sgemm.cuh`: Removed unused optimized kernel
- Deleted `changelog/` directory (consolidated into `CHANGELOG.md`)

## [2.0.0] - 2026-03-13

### Fixed
- **Critical**: CI workflow adjusted for CPU-safe execution
  - Removed GPU-dependent CUDA container build from GitHub Hosted Runner
  - Restored `push`, `pull_request`, and `workflow_dispatch` triggers
  - Unified format check using `jidicula/clang-format-action`

### Changed
- CI now only performs static format checking (CPU-safe)
- GPU runtime tests remain local/dedicated-runner only

## [2.0.0-rc.2] - 2026-03-10

### Added
- GitHub Pages configuration with SEO metadata in `_config.yml`
- Professional Chinese landing page in `index.md`

### Changed
- README.md: Added CI/Pages badges, timing column, ASCII roadmap
- pages.yml: Narrowed path triggers and sparse-checkout
- .gitignore: Added Jekyll-related entries

### Fixed
- Workflow deep standardization: unified `permissions`, `concurrency`, path filtering

## [2.0.0-rc.1] - 2026-03-09

### Fixed
- **Critical Memory Leak**: Error checking macros now throw exceptions instead of `exit()`
- **Memory Leak**: `launch_tensor_core_sgemm` now uses RAII wrappers
- **Memory Leak**: `SGEMMBenchmark::run()` now uses RAII wrappers

### Changed
- **Breaking**: Removed external dependency — project is now self-contained

### Added
- `CMakeLists.txt` for modern CMake build
- Standardized CI workflow with `clang-format` check
- CUDA container-based build validation

## [1.0.0] - 2025-02-13

### Added
- MIT LICENSE file
- `.gitignore` with CUDA/Profiling/IDE rules
- `.editorconfig` for unified code formatting
- Standardized badges in README

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 2.1.0 | 2026-04-16 | Documentation refactor, dead code cleanup |
| 2.0.0 | 2026-03-13 | CPU-safe CI, stable release |
| 2.0.0-rc.2 | 2026-03-10 | GitHub Pages, documentation |
| 2.0.0-rc.1 | 2026-03-09 | Memory leak fixes, CMake, self-contained |
| 1.0.0 | 2025-02-13 | Initial project infrastructure |
