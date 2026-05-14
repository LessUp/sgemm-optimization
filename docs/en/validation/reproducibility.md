---
title: Reproducibility
---

# Reproducibility

Reproducibility in this repository means another reader can tell **what was run, where it was run, and what kind of claim the result supports**.

## Minimum local workflow

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build
./build/bin/sgemm_benchmark -a --warmup 10 --benchmark 50
```

This sequence is the minimum bar for a locally reproduced performance statement.

## Record the environment

Every reported run should capture:

- GPU model
- CUDA toolkit / driver context
- benchmark command
- dimensions or benchmark set used
- warmup count and benchmark count
- whether the number is end-to-end or compute-only

Without that metadata, readers cannot tell whether the number is directly comparable to the published snapshot.

## Hosted CI versus local reruns

Hosted CI is still valuable because it proves the documentation, Pages, and governance surfaces stay coherent. But CI runners are not the evidence source for runtime behavior.

Only local GPU reruns can confirm:

- correctness against cuBLAS on the actual machine
- whether Tensor Core fast-path conditions were met
- whether a measured gain survives the chosen workload mix

## Reporting checklist

Before publishing or repeating a result, make sure you can answer all of these:

- Which GPU produced the number?
- Which command produced the number?
- Which benchmark label applies?
- Which correctness run guarded the benchmark?
- Which irregular shape prevents the claim from being aligned-only cherry-picking?

If you cannot answer those questions, rerun the experiment before you cite it.
