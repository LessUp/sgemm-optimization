---
layout: default
title: 4. Double Buffer
parent: Home
nav_order: 6
permalink: /docs/kernel-double-buffer/
lang: en
page_key: kernel-double-buffer
lang_ref: zh-kernel-double-buffer
---

# Kernel 4: Double Buffer
{: .fs-8 }

Overlapping memory loads with computation
{: .fs-6 .fw-300 }

---

## Overview

The double buffer (or "ping-pong buffering") technique overlaps **global memory loads** with **shared memory computation**. While computing on one tile, we load the next tile — hiding memory latency.

<div class="highlight-box info">
  <strong>Key Insight</strong><br>
  Modern GPUs can execute memory operations concurrently with computation. Double buffering exploits this to keep the ALUs busy while waiting for memory.
</div>

---

## The Problem: Sequential Execution

In the tiled kernel:

```
Timeline:
  Load Tile 0 ──────────────────▶
                                Compute Tile 0 ───────────────▶
                                                              Load Tile 1 ───▶
                                                                              Compute Tile 1 ──▶

Problem: GPU is IDLE during loads, MEMORY is IDLE during compute
```

### Execution Timeline

```
Without Double Buffering:
┌────────────────────────────────────────────────────────────┐
│ Cycle:  0    100   200   300   400   500   600   700       │
│                                                             │
│ Load0  [████████████]                                      │
│        ↓ Idle                                            │
│ Comp0         [████████████]                               │
│                      ↓                                     │
│ Load1                 [████████████]                       │
│                           ↓                               │
│ Comp1                          [████████████]              │
│                                                             │
│ ALU Utilization:  ████░░░░████░░░░████░░░░  (~50%)        │
└────────────────────────────────────────────────────────────┘
```

---

## The Solution: Double Buffering

Use **two shared memory buffers** that alternate roles:
- **Buffer 0**: Being computed on
- **Buffer 1**: Being loaded from global memory

```
Timeline with Double Buffering:
  Load Tile 0 ──────────────────▶
  Load Tile 1 ───────────────────────▶
          Compute Tile 0 ─────────────▶
          Load Tile 2 ───────────────────────▶
                      Compute Tile 1 ─────────▶
                      Load Tile 3 ───────────────────▶
                                  Compute Tile 2 ─────▶

Result: Computation overlaps with loading!
```

### Execution Timeline

```
With Double Buffering:
┌────────────────────────────────────────────────────────────┐
│ Cycle:  0    100   200   300   400   500   600   700       │
│                                                             │
│ Load0  [████████████]                                      │
│ Load1       [████████████]  ← OVERLAP!                     │
│ Comp0         [████████████]                               │
│ Load2               [████████████]  ← OVERLAP!             │
│ Comp1              [████████████]                          │
│                                                             │
│ ALU Utilization:  ████████████████  (~100%)                │
│ Memory Util:     ████████████████████  (also high)         │
│                                                             │
│ Total Time: ~550 cycles (vs ~700 without DB)              │
└────────────────────────────────────────────────────────────┘
```

---

## Shared Memory Layout

```cpp
// Single buffer (before)
__shared__ float As[TILE_SIZE][TILE_SIZE + 1];
__shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

// Double buffer (after)
__shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];  // [2] for ping-pong
__shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];
```

**Trade-off**: 2× shared memory usage for latency hiding.

---

## Implementation

```cpp
// File: src/kernels/double_buffer_sgemm.cuh

template<int TILE_SIZE = 32>
__global__ void sgemm_double_buffer_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // Double buffers for ping-pong
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Helper lambda to load a tile
    auto load_tile = [&](int buf, int tile_idx) {
        int a_col = tile_idx * TILE_SIZE + tx;
        int b_row = tile_idx * TILE_SIZE + ty;

        if (row < M && a_col < K)
            As[buf][ty][tx] = A[row * K + a_col];
        else
            As[buf][ty][tx] = 0.0f;

        if (b_row < K && col < N)
            Bs[buf][ty][tx] = B[b_row * N + col];
        else
            Bs[buf][ty][tx] = 0.0f;
    };

    // Pre-load first tile
    load_tile(0, 0);
    __syncthreads();

    // Main loop with double buffering
    for (int t = 0; t < num_tiles; ++t) {
        int curr = t % 2;        // Current buffer for computing
        int next = (t + 1) % 2;  // Next buffer for loading

        // Asynchronously load next tile (if exists)
        if (t + 1 < num_tiles) {
            load_tile(next, t + 1);
        }

        // Compute on current tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[curr][ty][k] * Bs[curr][k][tx];
        }

        __syncthreads();  // Wait for compute AND next load to finish
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

## Key Concepts

### 1. Buffer Indexing

```cpp
int curr = t % 2;        // 0, 1, 0, 1, 0, 1, ...
int next = (t + 1) % 2;  // 1, 0, 1, 0, 1, 0, ...
```

The modulo operator creates the ping-pong pattern.

### 2. Overlap Requirements

For effective overlap, the compute time should be ≈ load time:

```
Compute Time = TILE_SIZE³ / (threads × FLOPs/cycle)
Load Time = TILE_SIZE² × 2 × sizeof(float) / bandwidth

Ideal: TILE_SIZE chosen so these are balanced
```

### 3. Synchronization

Only **one** `__syncthreads()` is needed per iteration:
- Ensures all threads finish computing current tile
- Ensures all threads finish loading next tile

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Double Buffer Layout                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Buffer 0                    Buffer 1                      │
│   ┌─────────────┐             ┌─────────────┐              │
│   │ As[0][][]   │             │ As[1][][]   │              │
│   │ Bs[0][][]   │             │ Bs[1][][]   │              │
│   └──────┬──────┘             └──────┬──────┘              │
│          │                          │                      │
│          │    COMPUTE                │    LOAD              │
│          │    (current)              │    (next)            │
│          │       ↓                   │       ↓              │
│          └───────┬───────────────────┘       │              │
│                  │                           │              │
│         ┌────────▼───────────────────────────┤              │
│         │         REGISTERS                  │              │
│         │   (partial sums accumulation)      │              │
│         └────────────────────────────────────┘              │
│                                                             │
│   Swaps each iteration:                                     │
│   T0: Load(0), Comp(0)                                     │
│   T1: Load(1), Comp(0)  ← overlap!                         │
│   T2: Load(0), Comp(1)  ← overlap!                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

| Metric | Bank-Free | Double Buffer | Improvement |
|--------|-----------|---------------|-------------|
| **GFLOPS (1024³)** | 673 | 701 | **+4%** |
| **Shared Memory** | 8.4 KB | 16.8 KB | 2× |
| **Register Pressure** | Low | Moderate | — |
| **Occupancy** | Higher | Lower | Trade-off |

<div class="highlight-box warning">
  <strong>Note</strong><br>
  The performance improvement (~4%) is modest because modern GPUs have effective memory latency hiding through warp scheduling. Double buffering becomes more impactful on memory-bound kernels with minimal compute.
</div>

---

## Advanced: Software Pipelining

Double buffering is a form of **software pipelining**:

```
Traditional Loop:        Pipelined Loop:
┌──────────┐            ┌──┬──┬────────┐
│ Prologue │            │L0│C0│        │
├──────────┤            ├──┼──┼──┬──┐  │
│   Load   │            │L1│C0│C1│  │  │
│  Compute │      →     ├──┼──┼──┼──┤  │
│   Store  │            │L2│C1│C2│S0│  │
├──────────┤            ├──┼──┼──┼──┤  │
│ Epilogue │            │  │C2│S0│S1│  │
└──────────┘            └──┴──┴──┴──┘  │
                          └─────────┘
                          Pipeline stages overlap
```

---

## When Double Buffering Helps Most

| Scenario | Benefit |
|----------|---------|
| Very memory-bound kernels | High |
| Large TILE_SIZE | High (more compute to overlap) |
| Smaller GPUs (fewer warps) | Higher (less natural latency hiding) |
| Compute-heavy kernels | Low (already compute-bound) |

---

## Next Steps

We've optimized:
- ✓ Global memory coalescing
- ✓ Shared memory bank conflicts
- ✓ Memory latency hiding

The final frontier: **dedicated matrix hardware**. Modern GPUs have Tensor Cores that can perform 4×4×4 matrix multiply-accumulate in one cycle.

→ Continue to [Tensor Core Kernel](kernel-tensor-core/){: .btn .btn-primary }

---

## Key Takeaways

1. **Double Buffer**: Two buffers alternate between load and compute roles
2. **Overlap**: Hide memory latency by computing while loading
3. **Ping-Pong**: Use `t % 2` to alternate buffer indices
4. **Trade-off**: 2× shared memory for better latency hiding
5. **Synchronization**: Single barrier handles both operations
