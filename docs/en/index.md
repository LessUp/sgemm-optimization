---
layout: home
title: SGEMM Whitepaper
---

<div class="home-shell">
  <div class="home-hero-grid">
    <div>
      <p class="home-eyebrow">CUDA SGEMM WHITEPAPER · ARCHITECTURE SITE · KERNEL ACADEMY</p>
      <h1 class="home-main-title">A CUDA SGEMM project that reads like a defended technical case</h1>
      <p class="home-main-subtitle">
        This site is built for interviewers and advanced GitHub readers who care about more than “one fast kernel”.
        It frames the repository as a chain of architectural claims, optimization decisions, validation boundaries,
        and research lineage. Read it as a whitepaper first, then as an academy.
      </p>
      <div class="home-action-row">
        <a class="btn" href="./overview/">Open the project guide</a>
        <a class="btn btn-outline" href="./architecture/">Inspect the architecture</a>
        <a class="btn btn-outline" href="./academy/">Enter the academy</a>
        <a class="btn btn-outline" href="./research/">Open the research desk</a>
      </div>
      <div class="home-kicker-row">
        <span class="home-chip">5-stage kernel ladder</span>
        <span class="home-chip">cuBLAS-grounded validation</span>
        <span class="home-chip">EN / ZH mirrored routes</span>
      </div>
    </div>
    <ThemedFigure
      light="/figures/whitepaper-system-light.svg"
      dark="/figures/whitepaper-system-dark.svg"
      alt="Whitepaper map connecting overview, architecture, academy, validation, and research around the SGEMM kernel ladder."
      caption="The public narrative is organized like a technical argument: thesis, architecture, academy, proof, then lineage."
    />
  </div>

  <div class="thesis-grid">
    <div class="signal-card">
      <div class="signal-title">Project thesis</div>
      <div class="signal-value">Optimization must stay explainable</div>
      <div class="signal-note">Each kernel exists because it changes one bottleneck class, not because another benchmark screenshot was possible.</div>
    </div>
    <div class="signal-card">
      <div class="signal-title">Audience contract</div>
      <div class="signal-value">Readable under interview pressure</div>
      <div class="signal-note">The site is written so an interviewer can audit the design, a candidate can defend it, and a CUDA reader can keep digging.</div>
    </div>
    <div class="signal-card">
      <div class="signal-title">Trust model</div>
      <div class="signal-value">CI is structural, GPU is empirical</div>
      <div class="signal-note">Repository health, docs checks, and Pages fitness live in automation. Runtime correctness and performance still belong to real hardware.</div>
    </div>
  </div>
</div>

## Read this site by intent

<div class="route-grid">
  <div class="route-card">
    <h3>I need the 90-second project brief</h3>
    <p>Open the guide first, then jump to architecture if you need the system story behind the summary.</p>
    <div class="route-links">
      <a href="./overview/">Project guide</a>
      <a href="./architecture/">Architecture overview</a>
    </div>
  </div>
  <div class="route-card">
    <h3>I need to understand why each kernel exists</h3>
    <p>Start with the ladder and memory model before opening the academy pages that inspect each stage in detail.</p>
    <div class="route-links">
      <a href="./architecture/kernel-ladder">Kernel ladder</a>
      <a href="./academy/">Academy overview</a>
    </div>
  </div>
  <div class="route-card">
    <h3>I care about proof, not posture</h3>
    <p>Use validation when you want the correctness policy, benchmark scope, and reproducibility boundary before trusting any number.</p>
    <div class="route-links">
      <a href="./validation/">Validation overview</a>
      <a href="./validation/benchmark-results">Benchmark results</a>
    </div>
  </div>
  <div class="route-card">
    <h3>I want lineage and comparative context</h3>
    <p>Use the research desk for papers, related repositories, and notes on how this project’s current shape emerged.</p>
    <div class="route-links">
      <a href="./research/">Research desk</a>
      <a href="./research/related-projects">Related projects</a>
    </div>
  </div>
</div>

## The whitepaper spine

| Surface | What it answers | Why it exists |
|---|---|---|
| [Overview](./overview/) | What is this project, why does it matter, how should I read it? | Gives reviewers and new readers one decisive orientation surface. |
| [Architecture](./architecture/) | How is the SGEMM system structured, and what are its core invariants? | Turns implementation detail into a defendable system map. |
| [Academy](./academy/) | How do I study the optimization ladder in a rigorous order? | Packages the repository as a curriculum, not a pile of notes. |
| [Validation](./validation/) | What can the evidence prove, and what can it not prove? | Keeps the project technically honest. |
| [Research](./research/) | Where do these ideas come from, and what should I compare against? | Adds academic and comparative depth. |

## Architecture, rendered as a controlled figure

<ThemedFigure
  :wide="true"
  light="/figures/kernel-ladder-light.svg"
  dark="/figures/kernel-ladder-dark.svg"
  alt="Kernel ladder moving from naive FP32 to tiled, bank-free, double-buffer, and Tensor Core WMMA, with architecture, validation, and research rails."
  caption="The ladder is not a trophy rack. It is a map of bottleneck shifts, interface constraints, and evidence requirements."
/>

## What makes this presentation different

1. It treats SGEMM as a technical argument, not a showcase.
2. It separates architecture, academy, validation, and research so each page has a single job.
3. It uses mirrored English and Chinese routes because public depth is part of the project, not an afterthought.

## Start from the repository if needed

- Repository executive summary: [README](https://github.com/LessUp/sgemm-optimization/blob/master/README.md)
- Chinese repository entry: [README.zh-CN](https://github.com/LessUp/sgemm-optimization/blob/master/README.zh-CN.md)
- Build and validation guide: [CONTRIBUTING](https://github.com/LessUp/sgemm-optimization/blob/master/CONTRIBUTING.md)
