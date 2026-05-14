---
layout: home
title: SGEMM Architecture Whitepaper
---

<div class="home-shell">
  <div class="home-hero-grid">
    <div>
      <p class="home-eyebrow">CUDA SGEMM WHITEPAPER LANDING PAGE</p>
      <h1 class="home-main-title">SGEMM as a technical argument</h1>
      <p class="home-main-subtitle">
        This site explains the SGEMM project as a coherent engineering argument: begin with a simple CUDA matrix
        multiplication baseline, climb through staged architectural changes, and only accept performance claims when
        correctness checks, benchmark scope labels, and validation boundaries still hold. Architecture, methodology,
        resources, validation, and support are organized as one knowledge model so a new technical reader can see both
        the implementation ladder and the evidence model from the homepage alone.
      </p>
      <div class="home-action-row">
        <a class="btn" href="/en/architecture/">Read the architecture map</a>
        <a class="btn btn-outline" href="/en/learning-path">Follow the methodology</a>
        <a class="btn btn-outline" href="/en/benchmark-results">Check the validation boundary</a>
        <a class="btn btn-outline" href="https://github.com/LessUp/sgemm-optimization">GitHub</a>
      </div>
      <div class="home-kicker-row">
        <span class="home-chip">5-stage kernel ladder</span>
        <span class="home-chip">cuBLAS-anchored evidence</span>
        <span class="home-chip">EN / ZH mirrored</span>
      </div>
    </div>
    <div class="signal-grid">
      <div class="signal-card">
        <div class="signal-title">Core Question</div>
        <div class="signal-value">Why is each kernel faster?</div>
        <div class="signal-note">Every stage should earn its complexity by changing memory behavior or execution shape.</div>
      </div>
      <div class="signal-card">
        <div class="signal-title">Evidence Model</div>
        <div class="signal-value">cuBLAS + scope labels</div>
        <div class="signal-note">Correctness comparisons and benchmark labeling keep claims interpretable instead of promotional.</div>
      </div>
      <div class="signal-card">
        <div class="signal-title">Validation Boundary</div>
        <div class="signal-value">CI is not a GPU</div>
        <div class="signal-note">Hosted automation proves build and structure health; local hardware proves runtime and performance.</div>
      </div>
      <div class="signal-card">
        <div class="signal-title">Reader Model</div>
        <div class="signal-value">5 canonical domains</div>
        <div class="signal-note">Architecture, methodology, resources, validation, and support form the site's top-level map.</div>
      </div>
    </div>
  </div>

  <div class="home-proof-strip">
    <div class="proof-grid">
      <div class="proof-item">
        <div class="proof-label">Thesis</div>
        <div class="proof-value">This repository is about how optimization decisions are justified, not just how many kernels exist.</div>
      </div>
      <div class="proof-item">
        <div class="proof-label">Method</div>
        <div class="proof-value">One optimization stage, one bottleneck shift, one clearer explanation of what changed.</div>
      </div>
      <div class="proof-item">
        <div class="proof-label">Engineering Contract</div>
        <div class="proof-value">A unified launcher shape and guarded WMMA fallback keep kernels swappable and claims comparable.</div>
      </div>
      <div class="proof-item">
        <div class="proof-label">Outcome</div>
        <div class="proof-value">A first-time reader can enter from architecture, methodology, resources, validation, or support without losing the technical thread.</div>
      </div>
    </div>
  </div>
</div>

## Thesis and positioning

This homepage is the entry point to a whitepaper-style documentation set. The project is not presented as a feature list or benchmark trophy board; it is presented as a chain of technical claims about CUDA SGEMM, each claim tied to implementation structure, optimization intent, and validation evidence.

The site's knowledge model is intentionally explicit:

- **Architecture** explains what the kernel ladder contains, how stages relate, and where interface constraints live.
- **Methodology** explains how to read, learn, and tune the ladder without skipping the logic behind each step.
- **Resources** traces decisions back to papers, official docs, and high-value repositories.
- **Validation** explains what the evidence means, what the benchmark labels mean, and where trust stops.
- **Support** gets a reader from clone to local verification with the right expectations about CI versus GPU proof.

## Why this matters

<div class="perf-grid">
  <div class="perf-card">
    <div class="perf-label">Interpretability</div>
    <div class="perf-value">Clear</div>
    <div class="perf-note">A new reader can tell what changed between kernels and why the change should matter.</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">Transferability</div>
    <div class="perf-value">Reusable</div>
    <div class="perf-note">The staged explanation turns one SGEMM implementation into a reusable CUDA optimization case study.</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">Benchmark honesty</div>
    <div class="perf-value">Scoped</div>
    <div class="perf-note">End-to-end WMMA and compute-only WMMA are separated so readers know what each number actually proves.</div>
  </div>
  <div class="perf-card">
    <div class="perf-label">Trust model</div>
    <div class="perf-value">Explicit</div>
    <div class="perf-note">The homepage tells you upfront which claims are validated in hosted CI and which require a real GPU machine.</div>
  </div>
</div>

## Architecture at a glance

The project revolves around a progressive kernel ladder, but the ladder is only meaningful because correctness rails, benchmark labels, and process governance stay attached to it.

```mermaid
flowchart LR
    A[Naive FP32\none thread -> one output] --> B[Tiled FP32\nshared-memory reuse]
    B --> C[Bank-Free FP32\npadding for bank conflicts]
    C --> D[Double Buffer FP32\noverlap load and compute]
    D --> E[Tensor Core WMMA\nmixed precision]

    E --> F{M/K/N all aligned to 16?}
    F -- yes --> G[WMMA compute path]
    F -- no --> H[Guarded FP32 fallback]

    M1[Architecture\nkernel ladder + interfaces] --> M2[Methodology\nlearning path + tuning logic]
    M2 --> M3[Resources\npapers + docs + repos]
    M2 --> M4[Validation\ncuBLAS checks + scope labels]
    M4 --> M5[Support\nbuild + run locally]

    A -. implementation spine .-> M1
    E -. implementation spine .-> M1
    M4 -. correctness rail .-> A
    M4 -. correctness rail .-> E
    M3 -. design lineage .-> H
```

## Methodology entry points

<div class="route-grid">
  <div class="route-card">
    <h3>I need the big picture first</h3>
    <p>Start with the system view of the repository, then use validation context to interpret what the architecture is allowed to claim.</p>
    <div class="route-links">
      <a href="/en/architecture/">Architecture</a>
      <a href="/en/benchmark-results">Validation and benchmark scope</a>
    </div>
  </div>
  <div class="route-card">
    <h3>I want to learn the optimization ladder in order</h3>
    <p>Use the staged reading path when you want each performance concept to build on the previous kernel rather than jump straight to WMMA.</p>
    <div class="route-links">
      <a href="/en/learning-path">Learning Path</a>
      <a href="/en/architecture/">Architecture overview</a>
    </div>
  </div>
  <div class="route-card">
    <h3>I want tuning heuristics and follow-up material</h3>
    <p>Use the methodology and resource surfaces together when you need actionable optimization guidance plus technical lineage.</p>
    <div class="route-links">
      <a href="/en/optimization-playbook">Optimization Playbook</a>
      <a href="/en/references">References</a>
    </div>
  </div>
  <div class="route-card">
    <h3>I want to reproduce or audit the claims</h3>
    <p>Use the support and validation surfaces together to understand what to run locally, what CI already proves, and how results should be interpreted.</p>
    <div class="route-links">
      <a href="/en/getting-started">Getting Started</a>
      <a href="/en/benchmark-results">Benchmark Results</a>
    </div>
  </div>
</div>

## Resource hub entry points

<div class="knowledge-grid">
  <a class="knowledge-card" href="/en/architecture/">
    <h3>Architecture</h3>
    <p>The structural map of the kernel ladder, interface boundaries, and the decisions that hold the implementation together.</p>
  </a>
  <a class="knowledge-card" href="/en/learning-path">
    <h3>Methodology</h3>
    <p>The guided path for learning the stages in order, with optimization logic that stays connected to the architecture.</p>
  </a>
  <a class="knowledge-card" href="/en/references">
    <h3>Resources</h3>
    <p>The source trail behind the project: official docs, papers, and mature repositories mapped to concrete decisions.</p>
  </a>
  <a class="knowledge-card" href="/en/benchmark-results">
    <h3>Validation</h3>
    <p>The evidence surface for correctness budgets, benchmark scope, fallback behavior, and what the published numbers actually mean.</p>
  </a>
  <a class="knowledge-card" href="/en/getting-started">
    <h3>Support</h3>
    <p>The practical entry point for cloning, building, testing, and running the project with the right local-versus-CI expectations.</p>
  </a>
</div>

## Validation boundary

The validation model is deliberately split so readers do not over-trust hosted automation or under-value local GPU evidence.

| Evidence surface | What it can prove | Where it runs |
|------------------|-------------------|---------------|
| OpenSpec and repository checks | Specs, documentation structure, workflow alignment, Pages fitness | Hosted CI and local CLI |
| CUDA compilation | The codebase still builds in a configured CUDA toolchain | Hosted CI and local machines |
| Google Test + cuBLAS comparisons | Runtime correctness against the project oracle | Local GPU machines |
| Benchmark execution | Performance behavior, WMMA scope differences, fallback consequences | Local GPU machines |

This boundary is a first-class concept, not a footnote: CI keeps the repository healthy, but only local GPU execution can validate runtime behavior and performance claims.

## Canonical entry points

- Chinese mirrored home: [中文首页](/zh/)
- Repository entry point: [README](https://github.com/LessUp/sgemm-optimization/blob/master/README.md)
- Stable process and requirements: [openspec/specs](https://github.com/LessUp/sgemm-optimization/tree/master/openspec/specs)
