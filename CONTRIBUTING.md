---
layout: default
title: Contributing
nav_order: 11
permalink: /CONTRIBUTING/
---

# Contributing

Focused improvements are welcome. For this repository, the goal is clarity, correctness, and compactness rather than feature sprawl.

## When to use OpenSpec

Use the OpenSpec workflow for any non-trivial change that affects:

- repository structure
- documentation roles or public positioning
- validation rules or workflows
- kernel behavior or engineering requirements

Stable specs live under `openspec/specs/`. Active changes live under `openspec/changes/<change>/`.

## Recommended flow

1. `/opsx:explore` for scope and trade-offs
2. `/opsx:propose "description"` for the actual change
3. `/opsx:apply` to execute the task list
4. `/review` before major deletions, workflow changes, or archive
5. `/opsx:archive` once tasks, docs, specs, and validation all agree

## Validation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
openspec validate --all
```

- Hosted CI covers formatting, CUDA compile validation, OpenSpec/repository checks, and Pages.
- Runtime verification and benchmarking still require a local GPU-capable machine.

## Code and doc expectations

- Keep the existing kernel launcher interface shape intact unless the specs require a change.
- Preserve RAII-based CUDA resource management and exception-style error handling.
- Avoid adding generic governance boilerplate or duplicate docs.
- If two files serve the same purpose, prefer one authoritative file over two partially-overlapping ones.

## Tooling notes

- CMake is the primary build path.
- `clangd` plus `compile_commands.json` is the shared LSP baseline.
- Use `gh` for repository metadata, Actions, issues, and PR operations.
- Install the lightweight local hooks with `scripts/install-hooks.sh` if you want repository-specific guardrails before commit.
