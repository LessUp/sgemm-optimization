---
title: Resources Hub
---

# Resources Hub

This is the curated handoff for readers who want to keep learning after the architecture and methodology sections.

Use this section when you need more than a flat bibliography: each route below explains what to open next, why it matters, and which project question it helps answer.

## Start with the question you are trying to answer

| If your question sounds like this | Start here | Why this route is useful |
|---|---|---|
| "Why does this kernel spend so much effort on memory layout?" | [CUDA Memory Cheat Sheet](/en/cuda-memory-cheatsheet) | Fast refresher on coalescing, shared-memory layout, Tensor Core alignment, and the checks worth doing before you trust a timing result. |
| "Which official docs justify the constraints used in this whitepaper?" | [Curated References](/en/references#official-cuda-and-nvidia-docs) | Jumps straight to the CUDA, WMMA, and runtime references behind the execution-model and memory-model claims. |
| "What do strong SGEMM implementations look like in the wild?" | [Curated References](/en/references#exemplary-codebases-and-production-grade-samples) | Points to mature repositories and sample implementations so you can compare this project's teaching ladder with production-style code. |
| "What should I study after finishing this site?" | [Further Reading Routes](/en/resources/further-reading) | Organizes adjacent topics into deliberate routes instead of expecting you to guess which external paper or tool matters next. |
| "How do I turn a benchmark symptom into evidence?" | [Diagnosis Loop](/en/methodology/diagnosis-loop) + [Profiler and tooling references](/en/references#profiler-tooling-and-diagnosis-references) | Connects the site's internal workflow with the external tools that help confirm occupancy, memory, and scheduling hypotheses. |

## Curated shelves

### Official docs for constraints and terminology

Start here when you need the authoritative wording behind a claim in the whitepaper.

- [CUDA C++ Programming Guide](/en/references#official-cuda-and-nvidia-docs) for execution model, synchronization, and memory hierarchy rules.
- [WMMA and Tensor Core references](/en/references#official-cuda-and-nvidia-docs) for fragment constraints, alignment assumptions, and mixed-precision behavior.
- [CUDA Runtime API and Compute Sanitizer references](/en/references#profiler-tooling-and-diagnosis-references) for practical debugging and validation work.

### Papers and mental models for reasoning, not just citation

Use these when you want the design logic behind the kernel ladder, not just the API surface.

- [Foundational papers and performance models](/en/references#papers-and-performance-mental-models) explain why tiling, reuse, and arithmetic intensity dominate SGEMM work.
- [Further Reading Routes](/en/resources/further-reading#roofline-thinking-for-sgemm) turns those ideas into concrete next-study steps for roofline thinking and occupancy trade-offs.

### Exemplary codebases when you want to compare styles

These links help you see where this repository is intentionally simplified for teaching and where production code adds more abstraction.

- [CUTLASS, BLIS, and CUDA samples](/en/references#exemplary-codebases-and-production-grade-samples) show different answers to the same GEMM problem.
- [Kernel Ladder](/en/architecture/kernel-ladder) stays useful as a contrast: it explains the staged learning path before you dive into industrial-strength template stacks.

### Tooling when performance numbers stop being self-explanatory

Use these when "the benchmark changed" is not enough and you need evidence.

- [Profiler, tooling, and diagnosis references](/en/references#profiler-tooling-and-diagnosis-references) covers Nsight Compute, Nsight Systems, the occupancy calculator, and correctness-oriented tools.
- [Benchmark Scope](/en/validation/benchmark-scope) and [Reproducibility](/en/validation/reproducibility) explain how to interpret and report what those tools show.

## Suggested study routes

### Route 1: Understand memory before touching another optimization

1. Read [Memory Flow](/en/architecture/memory-flow).
2. Use the [CUDA Memory Cheat Sheet](/en/cuda-memory-cheatsheet) to sanity-check what one warp is loading, where reuse appears, and how shared memory changes the access pattern.
3. Continue to [Further Reading: GEMM tiling](/en/resources/further-reading#gemm-tiling-and-hierarchy-thinking) when you want stronger mental models.

### Route 2: Validate Tensor Core claims without hand-waving

1. Read [Tensor Core Path](/en/architecture/tensor-core-path) and [Tensor Core WMMA](/en/kernel-tensor-core).
2. Open the [WMMA and mixed-precision references](/en/references#official-cuda-and-nvidia-docs) to confirm shape, alignment, and fallback constraints.
3. Continue to [Further Reading: Tensor Core constraints](/en/resources/further-reading#tensor-core-constraints-and-fallback-design) for the adjacent topics that usually get skipped.

### Route 3: Move from benchmark curiosity to profiler-led diagnosis

1. Start with [Diagnosis Loop](/en/methodology/diagnosis-loop).
2. Pair it with [Nsight and occupancy references](/en/references#profiler-tooling-and-diagnosis-references).
3. Continue to [Further Reading: Profiling workflow](/en/resources/further-reading#profiling-from-symptoms-to-evidence) when you want a next-step checklist.

## Related pages inside the site

- [Curated References](/en/references)
- [CUDA Memory Cheat Sheet](/en/cuda-memory-cheatsheet)
- [Further Reading Routes](/en/resources/further-reading)
- [Diagnosis Loop](/en/methodology/diagnosis-loop)
- [Validation Overview](/en/validation/)
