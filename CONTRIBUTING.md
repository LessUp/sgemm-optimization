# Contributing

Thank you for your interest in this project! Contributions via Issues and Pull Requests are welcome.

## Spec-Driven Development

This project follows **Spec-Driven Development (SDD)**. All technical specifications are maintained in `/specs`:

- 📋 [Product Requirements](specs/product/sgemm-kernel-requirements.md)
- 🏗️ [Core Architecture RFC](specs/rfc/0001-core-architecture.md)
- 🗺️ [Implementation Roadmap RFC](specs/rfc/0002-implementation-roadmap.md)
- 🧪 [Test Specifications](specs/testing/kernel-verification.md)

**When contributing new features or changes:**
1. **Review** the relevant spec documents in `/specs` first.
2. **Update** specs if your changes affect interfaces, requirements, or behavior.
3. **Implement** code that 100% adheres to the spec definitions.
4. **Verify** against spec-defined test criteria.

For detailed AI and human contributor workflow, see [`AGENTS.md`](AGENTS.md).

## Development Workflow

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "feat: add your feature"`
4. Push branch: `git push origin feature/your-feature`
5. Create a Pull Request

## Build & Test

Recommended: CMake (primary build system):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark
cmake --build build --target test_sgemm
ctest --test-dir build
```

Quick local builds with Make:

```bash
make GPU_ARCH=sm_86
make benchmark
make test
```

Note: GitHub Actions currently runs format checks and containerized CUDA compile-only builds. CUDA runtime tests must be executed locally or on a GPU-enabled runner.

## Code Style

- CUDA code follows project conventions (clang-format enforced)
- Use indentation and formatting rules defined in `.editorconfig`
- New kernel variants must include correctness verification against cuBLAS
- Ensure all existing tests pass

## Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature or kernel variant
- `fix:` Bug fix
- `perf:` Performance optimization
- `docs:` Documentation update
- `test:` Test-related changes
