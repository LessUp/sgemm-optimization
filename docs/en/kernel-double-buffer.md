---
title: 4. Double Buffer
---

# Kernel 4: Double Buffer

Overlapping memory loads with computation



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



## When Double Buffering Helps Most

| Scenario | Benefit |
|----------|---------|
| Very memory-bound kernels | High |
| Large TILE_SIZE | High (more compute to overlap) |
| Smaller GPUs (fewer warps) | Higher (less natural latency hiding) |
| Compute-heavy kernels | Low (already compute-bound) |



## Key Takeaways

1. **Double Buffer**: Two buffers alternate between load and compute roles
2. **Overlap**: Hide memory latency by computing while loading
3. **Ping-Pong**: Use `t % 2` to alternate buffer indices
4. **Trade-off**: 2× shared memory for better latency hiding
5. **Synchronization**: Single barrier handles both operations
