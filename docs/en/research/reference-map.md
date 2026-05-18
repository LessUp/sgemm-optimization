---
title: Reference Map
---

# Reference Map

This page is a structured index of the external sources that back the claims in this whitepaper. Each entry is classified by type and linked to the section it supports most directly.

## Primary technical references

### CUDA and GPU architecture

| Source | What it establishes | Relevant section |
|---|---|---|
| [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Memory hierarchy, warp execution model, shared memory layout | Architecture, Academy |
| [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | Memory coalescing, occupancy, bank conflict avoidance | Academy (kernel pages) |
| [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/) | WMMA instruction semantics, matrix fragment layout | Tensor Core path |

### cuBLAS

| Source | What it establishes | Relevant section |
|---|---|---|
| [cuBLAS Developer Guide](https://docs.nvidia.com/cuda/cublas/) | GEMM API, precision modes, leading-dimension conventions | Validation (oracle definition) |

### Tensor Core / WMMA

| Source | What it establishes | Relevant section |
|---|---|---|
| [WMMA API documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma) | Fragment types, load/store/compute API | Academy (kernel-tensor-core), Architecture (tensor-core-path) |
| [Volta architecture whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) | First-generation Tensor Core throughput model | Research (evolution), Performance model |

## Foundational papers

| Paper | Contribution | Primary support for |
|---|---|---|
| Goto & van de Geijn (2008) — [Anatomy of High-Performance Matrix Multiplication](https://dl.acm.org/doi/10.1145/1356052.1356053) | Hierarchical blocking theory for GEMM on CPUs | Tiled kernel design, shared-memory staging rationale |
| Lai & Seznec (2013) — [Performance Upper Bound Analysis and Optimization of SGEMM on Fermi and Kepler GPUs](https://dl.acm.org/doi/10.1145/2464996.2465013) | GPU SGEMM tiling and occupancy analysis | Tiled kernel, double-buffer motivation |
| Whaley & Dongarra (1998) — ATLAS | Automated tuning of block sizes | Historical context for tile-size sensitivity |
| Markidis et al. (2018) — [NVIDIA Tensor Core Programmability, Performance & Precision](https://ieeexplore.ieee.org/document/8425500) | WMMA programming model and mixed-precision behavior | Tensor Core path design |

## Related open-source implementations

| Repository | Relationship | Notes |
|---|---|---|
| [CUTLASS](https://github.com/NVIDIA/cutlass) | Authoritative production GEMM kernel library | The ceiling above which this project does not claim to compete |
| [tinygrad / BEAM SGEMM](https://github.com/tinygrad/tinygrad) | Community SGEMM exploration | Different educational framing; useful for contrast |
| [siboehm/CUDA-GEMM-Optimization](https://github.com/siboehm/CUDA-GEMM-Optimization) | Step-by-step SGEMM tutorial | Most directly comparable educational structure |
| [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) | Chinese-language SGEMM practice repository | Bilingual contrast; different kernel progression |

## How to use this map

This reference map is not a bibliography to be cited at the end of a paper. It is a **live index** that connects each claim in the whitepaper to its supporting source.

If you want to challenge a claim:
1. Find the section in the whitepaper that makes the claim.
2. Find the supporting source in the table above.
3. Open the source and check whether the claim is appropriately scoped.

If the claim is not in the table, it is either derived from the implementation itself (verifiable by reading the code) or it is an open question explicitly labeled as such in the text.

## Related pages

- [Curated References](./references) — full annotated reference list with reading notes
- [Papers](./papers) — focused academic reading list
- [Related Projects](./related-projects) — comparative context for the project scope
- [Evolution Notes](./evolution) — how the external sources shaped the current design
- [Performance Casebook](./performance-casebook) — how to interpret measured results against external benchmarks
