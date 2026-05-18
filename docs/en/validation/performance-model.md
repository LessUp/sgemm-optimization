---
title: Performance Model
---

# Performance Model

This page describes the analytical performance model behind the kernel ladder. The model predicts the dominant cost at each optimization stage and explains why each architectural change shifts the bottleneck.

## Roofline framing

SGEMM performance on a given GPU is bounded by two resources:

1. **Memory bandwidth** — the rate at which data can be moved between global memory and SMs
2. **Compute throughput** — the rate at which multiply-accumulate operations can be retired

The **arithmetic intensity** (FLOPs per byte of DRAM traffic) determines which bound is active:

```
I = FLOPs / bytes_transferred
```

For an N×N matrix multiplication:
- FLOPs: 2N³
- Naïve DRAM traffic: reads A and B once per output element = 2N³ data elements = O(N³) bytes
- Tiled DRAM traffic: reads each element O(N/tile) times = O(N²) bytes → arithmetic intensity rises to O(tile)

When `I < ridge_point`, the kernel is memory-bound. When `I > ridge_point`, it is compute-bound.

## Per-stage cost model

### Naïve FP32

```
Arithmetic intensity ≈ 1 FLOP/byte (at FP32 density)
Status: strongly memory-bound
Bottleneck: every multiply-accumulate requires a fresh global memory read
```

The naïve kernel does not reuse any loaded value. Every thread independently fetches the same A and B elements that neighboring threads also fetch, resulting in massive redundant DRAM traffic.

### Tiled FP32

```
Arithmetic intensity ≈ tile_size / 2 FLOPs/byte
Status: partially memory-bound (tile-size dependent)
Bottleneck: shared-memory bandwidth + synchronization overhead
```

Cooperative loading into shared tiles eliminates most redundant DRAM traffic. The dominant cost shifts from global memory bandwidth to shared-memory throughput and `__syncthreads` serialization.

### Bank-Free FP32

```
Arithmetic intensity: same as tiled (no new reuse)
Status: same roofline point, lower effective shared-memory latency
Bottleneck: residual shared-memory bank conflicts → eliminated by padding
```

Padding eliminates multi-way bank conflicts in the tiled layout. This does not change arithmetic intensity but removes a source of shared-memory serialization that degraded effective bandwidth.

### Double Buffer

```
Arithmetic intensity: same as tiled
Status: compute-bound on sufficient hardware
Bottleneck: memory latency hidden by prefetch overlap
```

Double buffering overlaps the fetch of tile `k+1` with the computation of tile `k`. The dominant cost shifts from memory latency to compute throughput, approaching the compute roof on capable hardware.

### Tensor Core WMMA

```
Arithmetic intensity: high (hardware fragment accumulation)
Status: compute-bound, higher throughput ceiling
Bottleneck: FP32→FP16 conversion overhead + shape constraints
```

WMMA instructions retire 16×16×16 mixed-precision multiply-accumulates per instruction, yielding 8× higher throughput on Tensor Core hardware compared to FP32 CUDA cores. The real costs are the FP32→FP16 conversion and the requirement that matrix dimensions be divisible by the WMMA fragment size.

## Performance predictions vs. observed behavior

| Kernel | Predicted dominant cost | Observed behavior |
|---|---|---|
| Naïve | DRAM bandwidth | ✓ Very low GFLOPS, GPU memory-bound |
| Tiled | Shared-memory bandwidth | ✓ Significant improvement, tile-size sensitive |
| Bank-Free | Reduced shared-memory serialization | ✓ Moderate improvement over tiled on conflict-prone shapes |
| Double Buffer | Memory latency (hidden) | ✓ Improvement on high-occupancy shapes |
| Tensor Core | FP32→FP16 conversion + compute | ✓ Large improvement for large shapes under capability guard |

> These comparisons require local GPU execution. See [Benchmark Results](./benchmark-results) and [Reproducibility](./reproducibility) for specific numbers and hardware context.

## Model limitations

- The roofline model assumes perfect caching behavior; actual occupancy, warp scheduling, and SM resource limits create deviations.
- The FP32→FP16 conversion cost in the Tensor Core path is not captured by arithmetic intensity alone.
- The model does not account for L2 cache effects, which can substantially affect medium-size matrix results.
- Tile size selection interacts with occupancy and register file pressure in ways the basic roofline model does not predict.

## Related pages

- [Benchmark Scope](./benchmark-scope) — which shape classes and hardware contexts are covered
- [Benchmark Results](./benchmark-results) — measured numbers with hardware and shape labels
- [Reproducibility](./reproducibility) — how to run and interpret a result responsibly
- [Kernel Ladder](../architecture/kernel-ladder) — the architectural counterpart of this cost model
- [Memory Flow](../architecture/memory-flow) — data-movement story behind the arithmetic intensity shifts
