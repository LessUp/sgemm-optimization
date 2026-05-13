---
title: Architecture
---

# Architecture

What the repository contains, and where validation responsibility lives



## Repository surfaces

| Surface | Role |
|---------|------|
| `README.md` | Repository entry point and quick-start |
| `index.md` + `docs/` | Public landing page and learning-oriented documentation |
| `openspec/specs/` | Stable authoritative requirements and governance |
| `openspec/changes/` | Active implementation plans and delta specs |
| `.github/workflows/` | CI-safe validation and Pages deployment |

The repository intentionally separates **public explanation** from **normative process**. OpenSpec governs; docs teach; README introduces.



## Validation boundaries

| Area | Local GPU machine | Hosted CI |
|------|-------------------|-----------|
| CUDA compilation | Yes | Yes |
| Runtime correctness | Yes | No |
| Benchmarking | Yes | No |
| OpenSpec/repository checks | Yes | Yes |
| GitHub Pages buildability | Optional | Yes |

This split is deliberate. The repository does not pretend CI can replace a real CUDA runtime environment.



## Related references

- [Learning Path](/en/learning-path)
- [Getting Started](/en/getting-started)
- [Specifications Index](https://github.com/LessUp/sgemm-optimization/tree/master/openspec/specs/)
- [Stable architecture spec](https://github.com/LessUp/sgemm-optimization/blob/master/openspec/specs/architecture/spec.md)
