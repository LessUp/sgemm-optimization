---
layout: default
title: Contributing
nav_order: 11
permalink: /CONTRIBUTING/
---

# Contributing

Keep changes tight, evidence-based, and easy to audit. This repository favors simplification over framework layering.

## Working principles

- Preserve the SGEMM kernel ladder as the core teaching surface.
- Prefer one authoritative explanation over duplicated docs.
- Remove stale or duplicate files instead of keeping placeholders.
- Keep public docs, README, and workflows aligned when behavior changes.

## Branch model

`master` is the only long-lived branch. Temporary local branches are fine for isolation, but merge them back quickly and delete them once the work lands.

## Validation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
npm --prefix docs test
npm --prefix docs run build
```

- Hosted CI covers formatting and docs-site checks.
- Local CUDA-capable machines remain the source of truth for building, runtime correctness, and benchmarking.

## Code and docs

- Keep kernel launcher interfaces consistent unless the behavior itself changes.
- Preserve RAII-based CUDA resource management and explicit error reporting.
- Use CMake as the supported build path.
- Keep docs compact and repository-specific.
