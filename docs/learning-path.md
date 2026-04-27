---
layout: default
title: Learning Path
nav_order: 3
permalink: /docs/learning-path/
lang: en
page_key: learning-path
lang_ref: zh-learning-path
---

# Learning Path
{: .fs-8 }

Follow the optimization ladder in the order the repository was designed to teach it
{: .fs-6 .fw-300 }

---

## Recommended order

| Step | Kernel | Why it comes here |
|------|--------|-------------------|
| 1 | [Naive](kernel-naive/) | Establish the baseline cost model |
| 2 | [Tiled](kernel-tiled/) | Introduce shared-memory reuse |
| 3 | [Bank-Free](kernel-bank-free/) | Show why shared-memory layout still matters |
| 4 | [Double Buffer](kernel-double-buffer/) | Add staging and overlap concepts |
| 5 | [Tensor Core](kernel-tensor-core/) | Move to WMMA and mixed-precision hardware |

---

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

---

## Suggested reading rhythm

1. Build and run the project first
2. Read the kernel page for one stage
3. Run the benchmark again
4. Compare the code with the previous stage
5. Move to the next optimization only after the current one is clear

---

## Before you start

- Make sure your environment follows [Getting Started](getting-started/)
- Use the [Architecture](architecture/) page if you want the repository-level map first
- Keep the [Specifications Index](../specs/) nearby if you want the normative requirements
