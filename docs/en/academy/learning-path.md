---
title: Learning Path
---

# Learning Path

Follow the optimization ladder in the order the repository was designed to teach it



## What each stage teaches

### Naive -> Tiled

- Thread/block mapping
- Memory coalescing
- Shared-memory reuse

### Tiled -> Bank-Free

- 32-bank shared-memory behavior
- Why `[32][33]` matters

### Bank-Free -> Double Buffer

- Pipeline thinking
- Tile staging and latency hiding

### Double Buffer -> Tensor Core

- WMMA fragments
- Mixed precision
- Safe fallback behavior for unsupported shapes



## Before you start

- Make sure your environment follows [Getting Started](/en/overview/getting-started)
- Use the [Architecture](/en/architecture/) page if you want the repository-level map first
- Keep [Getting Started](/en/overview/getting-started) and [Validation](/en/validation/) nearby if you want the repository-level constraints and evidence boundary
