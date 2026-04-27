---
layout: default
title: Architecture
nav_order: 8
permalink: /docs/architecture/
lang: en
page_key: architecture
lang_ref: zh-architecture
---

# Architecture
{: .fs-8 }

What the repository contains, and where validation responsibility lives
{: .fs-6 .fw-300 }

---

## System shape

```text
main.cu
  ├── benchmark orchestration
  ├── verification flow
  └── CLI argument handling

src/kernels/
  ├── naive
  ├── tiled
  ├── bank-conflict-free
  ├── double-buffer
  └── tensor-core

src/utils/
  ├── CUDA RAII and error handling
  ├── benchmark helpers
  └── verification helpers

tests/
  └── Google Test coverage against cuBLAS
```

---

## Repository surfaces

| Surface | Role |
|---------|------|
| `README.md` | Repository entry point and quick-start |
| `index.md` + `docs/` | Public landing page and learning-oriented documentation |
| `openspec/specs/` | Stable authoritative requirements and governance |
| `openspec/changes/` | Active implementation plans and delta specs |
| `.github/workflows/` | CI-safe validation and Pages deployment |

The repository intentionally separates **public explanation** from **normative process**. OpenSpec governs; docs teach; README introduces.

---

## Kernel contract

All kernel launchers follow the same shape:

```cpp
template<int TILE_SIZE = 32>
void launch_xxx_sgemm(
    const float* A, const float* B, float* C,
    int M, int K, int N,
    cudaStream_t stream = 0
);
```

That shared launcher contract makes it easy to benchmark, swap, and verify kernels without changing the surrounding harness.

---

## Validation boundaries

| Area | Local GPU machine | Hosted CI |
|------|-------------------|-----------|
| CUDA compilation | Yes | Yes |
| Runtime correctness | Yes | No |
| Benchmarking | Yes | No |
| OpenSpec/repository checks | Yes | Yes |
| GitHub Pages buildability | Optional | Yes |

This split is deliberate. The repository does not pretend CI can replace a real CUDA runtime environment.

---

## Repository-level design choices

1. **Progressive kernels** keep optimization steps readable.
2. **RAII wrappers and exception-style error propagation** keep CUDA resource handling predictable.
3. **OpenSpec governs repo-wide changes** so docs, workflows, and validation stay aligned.
4. **Docs stay role-based**: README introduces, Pages teach, OpenSpec defines rules.

---

## Related references

- [Learning Path](learning-path/)
- [Getting Started](getting-started/)
- [Specifications Index](../specs/)
- [Stable architecture spec](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/architecture/spec.md)
