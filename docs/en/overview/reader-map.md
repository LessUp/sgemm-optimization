---
title: Reader Map
---

# Reader Map

This page is a navigational index for the SGEMM whitepaper site. Choose an entry point based on your intent, background, and available time.

## Entry paths by intent

| Who you are | Best entry point | Expected time |
|---|---|---|
| Interviewer auditing system clarity | [Architecture Overview](../architecture/) | 8 min |
| Candidate preparing a walkthrough | [Academy Overview](../academy/) + [Learning Path](../academy/learning-path) | 25 min |
| CUDA learner starting from scratch | [Getting Started](./getting-started) → [Architecture](../architecture/) → [Academy](../academy/) | 45 min |
| Performance skeptic | [Validation Overview](../validation/) → [Benchmark Results](../validation/benchmark-results) | 12 min |
| Researcher tracing lineage | [Research Desk](../research/) → [Papers](../research/papers) → [Related Projects](../research/related-projects) | 20 min |
| Reader who wants the fastest overview | This page + [Architecture Overview](../architecture/) | 5 min |

## The whitepaper argument structure

The site is organized as a sequence of claims, each backed by a dedicated section:

```
Claim: SGEMM optimization should read as a reasoning chain
  └─ Architecture  → system map, invariants, bottleneck ladder
  └─ Academy       → ordered kernel study with causal explanations
  └─ Validation    → correctness policy, benchmark scope, trust boundary
  └─ Research      → paper lineage, related repos, evolution notes
```

Each section has one primary job. The Reader Map helps you skip directly to the section that matches your question.

## Depth tiers

### Tier 1: 5-minute reviewer sweep

1. [Architecture Overview](../architecture/) — system claim and kernel ladder
2. [Validation Overview](../validation/) — what the evidence can and cannot prove

### Tier 2: 20-minute technical audit

1. [Architecture Overview](../architecture/)
2. [Kernel Ladder](../architecture/kernel-ladder)
3. [Memory Flow](../architecture/memory-flow)
4. [Validation Overview](../validation/)
5. [Benchmark Scope](../validation/benchmark-scope)

### Tier 3: Full whitepaper reading

Follow the [Learning Path](../academy/learning-path) in Academy, which maps the complete order from orientation through each kernel deep dive.

## Cross-section dependencies

Some pages build on each other. Here are the important pairs:

- Read **Tensor Core Path** after **Memory Flow**, not before.
- Read **Benchmark Results** after **Benchmark Scope** and **Reproducibility**, not before.
- Read **Performance Casebook** after **Validation Overview**, not before.
- Read any **kernel deep dive** after **Learning Path** establishes the ladder order.

## Related pages

- [Getting Started](./getting-started) — environment setup and first run
- [System Blueprint](../architecture/system-blueprint) — full component dependency map
- [Performance Model](../validation/performance-model) — quantitative cost model behind the ladder
