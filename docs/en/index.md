---
layout: home
title: SGEMM Architecture Whitepaper
---

<div class="home-shell">
  <div class="home-hero-grid">
    <div>
      <p class="home-eyebrow">CUDA SGEMM ARCHITECTURE GUIDE</p>
      <h1 class="home-main-title">SGEMM Architecture Whitepaper</h1>
      <p class="home-main-subtitle">
        A bilingual CUDA SGEMM guide organized around architecture, optimization methodology, validation evidence,
        and reusable engineering references. Every optimization step is tied to correctness constraints,
        benchmark evidence, and explicit validation boundaries.
      </p>
      <div class="home-action-row">
        <a class="btn" href="/en/getting-started">Start in 5 minutes</a>
        <a class="btn btn-outline" href="/en/architecture">Explore architecture</a>
        <a class="btn btn-outline" href="/en/learning-path">Follow learning path</a>
        <a class="btn btn-outline" href="https://github.com/LessUp/sgemm-optimization">GitHub</a>
      </div>
      <div class="home-kicker-row">
        <span class="home-chip">cuBLAS-verified</span>
        <span class="home-chip">OpenSpec-governed</span>
        <span class="home-chip">EN / ZH mirrored</span>
      </div>
    </div>
    <div class="signal-grid">
      <div class="signal-card">
        <div class="signal-title">Kernel Ladder</div>
        <div class="signal-value">5</div>
        <div class="signal-note">naive -> tiled -> bank-free -> double-buffer -> WMMA</div>
      </div>
      <div class="signal-card">
        <div class="signal-title">Correctness Oracle</div>
        <div class="signal-value">cuBLAS</div>
        <div class="signal-note">separate tolerances for FP32 and Tensor Core paths</div>
      </div>
      <div class="signal-card">
        <div class="signal-title">Validation Boundary</div>
        <div class="signal-value">CI + GPU</div>
        <div class="signal-note">hosted CI for build health, local GPU for runtime and performance</div>
      </div>
      <div class="signal-card">
        <div class="signal-title">Public Surfaces</div>
        <div class="signal-value">EN / 中文</div>
        <div class="signal-note">mirrored pages for architecture, methodology, and references</div>
      </div>
    </div>
  </div>

  <div class="home-proof-strip">
    <div class="proof-grid">
      <div class="proof-item">
        <div class="proof-label">Benchmark Scope</div>
        <div class="proof-value">End-to-end and compute-only WMMA are reported separately.</div>
      </div>
      <div class="proof-item">
        <div class="proof-label">Numerical Policy</div>
        <div class="proof-value">FP32 and Tensor Core paths use different tolerance budgets by design.</div>
      </div>
      <div class="proof-item">
        <div class="proof-label">Engineering Contract</div>
        <div class="proof-value">Unified launcher signature keeps kernels swappable and testable.</div>
      </div>
      <div class="proof-item">
        <div class="proof-label">Governance</div>
        <div class="proof-value">OpenSpec keeps docs, process, and implementation intent aligned.</div>
      </div>
    </div>
  </div>
</div>

## Why this repository is worth attention

<div class="perf-grid">
  <div class="perf-card">
    <div class="perf-label">Learning Depth</div>
    <div class="perf-value">Progressive</div>
    <div class="perf-note">Each kernel stage teaches one specific performance concept.</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">Evidence Model</div>
    <div class="perf-value">Traceable</div>
    <div class="perf-note">Speedup claims are attached to correctness checks and scope labels.</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">Architecture Utility</div>
    <div class="perf-value">Practical</div>
    <div class="perf-note">The project can be explained as a clear engineering decision chain.</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">Community Value</div>
    <div class="perf-value">Reusable</div>
    <div class="perf-note">Includes playbooks, references, and architecture-aware tuning guidance.</div>
  </div>
</div>

## Project map in one diagram

```mermaid
flowchart LR
    A[Naive FP32\none thread -> one output] --> B[Tiled FP32\nshared-memory reuse]
    B --> C[Bank-Free FP32\npadding for bank conflicts]
    C --> D[Double Buffer FP32\noverlap load and compute]
    D --> E[Tensor Core WMMA\nmixed precision]

    E --> F{M/K/N all aligned to 16?}
    F -- yes --> G[WMMA compute path]
    F -- no --> H[Guarded FP32 fallback]

    T1[Google Test + cuBLAS comparison] -. correctness rail .-> A
    T1 -. correctness rail .-> E
    T2[Benchmark labels:\nend-to-end vs compute-only] -. evidence rail .-> E
    T3[OpenSpec governance] -. process rail .-> A
    T3 -. process rail .-> H
```

## Choose your route

<div class="route-grid">
  <div class="route-card">
    <h3>Build and run quickly</h3>
    <p>Get from clone to benchmark execution with clear local-vs-CI expectations.</p>
    <div class="route-links">
      <a href="/en/getting-started">Getting Started</a>
      <a href="/en/benchmark-results">Benchmark Results</a>
    </div>
  </div>
  <div class="route-card">
    <h3>Learn the optimization ladder</h3>
    <p>Understand what each stage changes in memory behavior and performance profile.</p>
      <div class="route-links">
        <a href="/en/architecture">Architecture Overview</a>
        <a href="/en/learning-path">Learning Path</a>
      </div>
  </div>
  <div class="route-card">
    <h3>Trace architecture decisions</h3>
    <p>See how design choices, kernel stages, and benchmark evidence fit together.</p>
    <div class="route-links">
      <a href="/en/architecture">Architecture</a>
      <a href="/en/benchmark-results">Benchmark Results</a>
    </div>
  </div>
  <div class="route-card">
    <h3>Validate technical lineage</h3>
    <p>Trace implementation choices to official docs, papers, and high-quality repos.</p>
    <div class="route-links">
      <a href="/en/references">References</a>
      <a href="/en/optimization-playbook">Optimization Playbook</a>
    </div>
  </div>
</div>

## Knowledge hub

<div class="knowledge-grid">
  <a class="knowledge-card" href="/en/architecture">
    <h3>Architecture</h3>
    <p>A guided map of kernel stages, validation boundaries, and the decisions that hold the project together.</p>
  </a>
  <a class="knowledge-card" href="/en/learning-path">
    <h3>Learning Path</h3>
    <p>A structured route through the kernel ladder so each optimization concept builds on the last.</p>
  </a>
  <a class="knowledge-card" href="/en/references">
    <h3>References</h3>
    <p>Curated papers, official docs, and repositories mapped to concrete design decisions.</p>
  </a>
  <a class="knowledge-card" href="/en/benchmark-results">
    <h3>Benchmark Results</h3>
    <p>Use the validation entry point to interpret benchmark scope, correctness budgets, and trust boundaries.</p>
  </a>
  <a class="knowledge-card" href="/en/getting-started">
    <h3>Getting Started</h3>
    <p>Use the support entry point for local setup, first build, and the repository's validation boundary.</p>
  </a>
  <a class="knowledge-card" href="/en/benchmark-results">
    <h3>Validation Boundary</h3>
    <p>Use the validation entry point to separate what hosted CI proves from what only local GPU runs can prove.</p>
  </a>
</div>

> Legacy narrative pages are being consolidated into the canonical architecture, methodology, and validation sections. They remain available during migration but are no longer primary entry points.

## Command cockpit

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Validate
ctest --test-dir build
openspec validate --all

# Benchmark
./build/bin/sgemm_benchmark -a
./build/bin/sgemm_benchmark --dims 256 384 640
```

## Language and entry points

- Chinese mirrored home: [中文首页](/zh/)
- Repository entry: [README](https://github.com/LessUp/sgemm-optimization/blob/master/README.md)
- OpenSpec source of truth: [openspec/specs](https://github.com/LessUp/sgemm-optimization/tree/master/openspec/specs)
