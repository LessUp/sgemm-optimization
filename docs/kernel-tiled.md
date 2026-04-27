---
layout: default
title: 2. Tiled Kernel
parent: Home
nav_order: 4
permalink: /docs/kernel-tiled/
lang: en
page_key: kernel-tiled
lang_ref: zh-kernel-tiled
---

# Kernel 2: Tiled Implementation
{: .fs-8 }

Shared memory blocking for better data reuse
{: .fs-6 .fw-300 }

---

## Overview

The tiled kernel introduces **shared memory** to dramatically reduce global memory traffic. Instead of each thread loading data from global memory K times, we load tiles once into fast shared memory and reuse them.

<div class="highlight-box info">
  <strong>Key Insight</strong><br>
  Each element of A and B is used N and M times respectively. Shared memory reduces global memory reads by <strong>TILE_SIZE×</strong>.
</div>

---

## The Problem with Naïve

In the naïve kernel:
- To compute one row of C, we read that row of A **N times**
- To compute one column of C, we read that column of B **M times**

```
┌─────────────────────────────────────────┐
│           C[row, :] (1 row)             │
├─────────────────────────────────────────┤
│  = A[row, :] × B[0, :]  ← read A row    │
│  = A[row, :] × B[1, :]  ← read SAME row!│
│  = A[row, :] × B[2, :]  ← and again!    │
└─────────────────────────────────────────┘
```

---

## The Solution: Tiling

Divide matrices into **TILE_SIZE × TILE_SIZE** tiles:

```
A (M×K)          B (K×N)          C (M×N)
┌───┬───┐       ┌───┬───┐       ┌───┬───┐
│A00│A01│       │B00│B01│       │C00│C01│
├───┼───┤   ×   ├───┼───┤   =   ├───┼───┤
│A10│A11│       │B10│B11│       │C10│C11│
└───┴───┘       └───┴───┘       └───┴───┘

C00 = A00×B00 + A01×B10
```

Each tile of C is computed by loading corresponding tiles from A and B into shared memory.

---

## Implementation

```cpp
// File: src/kernels/tiled_sgemm.cuh

template<int TILE_SIZE = 32>
__global__ void sgemm_tiled_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Global position
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles
    for (int t = 0; t < num_tiles; ++t) {
        // Calculate tile positions
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        // Load tile from A (coalesced)
        if (row < M && a_col < K)
            As[ty][tx] = A[row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        // Load tile from B (coalesced)
        if (b_row < K && col < N)
            Bs[ty][tx] = B[b_row * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();  // Wait for all loads to complete

        // Compute tile multiplication
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();  // Wait for all threads to finish computing
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

## Memory Access Pattern

### Before (Naïve)
```
Global Memory Reads = M × N × K elements × 2 matrices
                    = 2 × M × N × K reads
```

### After (Tiled)
```
Global Memory Reads = (M × K) for A tiles + (K × N) for B tiles
                    = K × (M + N) reads per tile iteration
                    = O((M × N × K) / TILE_SIZE) total
Reduction factor: TILE_SIZE×
```

### Coalesced Access

Consecutive threads now read consecutive memory addresses:

```
Thread 0: A[row, t*TILE+0], B[t*TILE+0, col]
Thread 1: A[row, t*TILE+1], B[t*TILE+1, col]
Thread 2: A[row, t*TILE+2], B[t*TILE+2, col]
          ↑ consecutive! ✓
```

---

## Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     GPU Architecture                    │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌───────────────────────────────┐  │
│  │   Global    │    │         Shared Memory         │  │
│  │   Memory    │───▶│      (per thread block)       │  │
│  │   (slow)    │    │  ┌─────────┐    ┌─────────┐   │  │
│  └─────────────┘    │  │  As[][] │    │  Bs[][] │   │  │
│                     │  │ TILE×TIL│    │ TILE×TIL│   │  │
│                     │  └────┬────┘    └────┬────┘   │  │
│                     │       │              │        │  │
│                     └───────┼──────────────┼────────┘  │
│                             ▼              ▼           │
│                         └──────────────────┐           │
│                           Compute (registers)           │
│                                         │              │
│                                         ▼              │
│                                     Write to Global    │
└─────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

| Metric | Naïve | Tiled | Improvement |
|--------|-------|-------|-------------|
| **GFLOPS (1024³)** | 604 | 753 | **+25%** |
| **Global Mem Traffic** | 2MNK | 2MNK/TILE | **-97%** |
| **Shared Memory** | 0 KB | ~8 KB | new |
| **Memory Bound?** | Yes | Still yes | — |

---

## Synchronization Points

Two `__syncthreads()` barriers are critical:

1. **After loading tiles**: Ensures all data is in shared memory before any thread starts computing
2. **After computing**: Prevents threads from overwriting shared memory while others are still reading

```cpp
__syncthreads();  // Load complete

// Compute phase...

__syncthreads();  // Compute complete, safe to load next tile
```

<div class="highlight-box warning">
  <strong>Common Bug</strong><br>
  Missing either <code>__syncthreads()</code> causes race conditions — some threads read garbage data or write before others finish.
</div>

---

## Boundary Handling

For matrices not divisible by TILE_SIZE:

```cpp
if (row < M && a_col < K)
    As[ty][tx] = A[row * K + a_col];
else
    As[ty][tx] = 0.0f;  // Zero padding
```

The zero padding ensures correct computation without special-case logic.

---

## Tile Size Selection

| TILE_SIZE | Shared Memory | Occupancy | Performance |
|-----------|---------------|-----------|-------------|
| 16 | 2 KB | High | Lower (less reuse) |
| **32** | **8 KB** | **Medium** | **Good balance** |
| 64 | 32 KB | Low | Limited by SM capacity |

Default `TILE_SIZE = 32` fits well in typical 48-64 KB shared memory per SM.

---

## Next Steps

While we've improved global memory access, we've introduced a new problem: **shared memory bank conflicts**. When threads access the same memory bank, their requests are serialized.

→ Continue to [Bank Conflict Free Kernel](kernel-bank-free/){: .btn .btn-primary }

---

## Key Takeaways

1. **Shared Memory** is ~100× faster than global memory
2. **Tiling** reduces global memory bandwidth by reusing data
3. **Coalesced Access** is achieved when consecutive threads read consecutive addresses
4. **Synchronization** is required when threads share data
5. **Template Parameters** allow compile-time tile size selection
