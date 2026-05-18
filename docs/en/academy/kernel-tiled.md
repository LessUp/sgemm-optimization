---
title: 2. Tiled Kernel
---

# Kernel 2: Tiled Implementation

Shared memory blocking for better data reuse



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



## Tile Size Selection

| TILE_SIZE | Shared Memory | Occupancy | Performance |
|-----------|---------------|-----------|-------------|
| 16 | 2 KB | High | Lower (less reuse) |
| **32** | **8 KB** | **Medium** | **Good balance** |
| 64 | 32 KB | Low | Limited by SM capacity |

Default `TILE_SIZE = 32` fits well in typical 48-64 KB shared memory per SM.



## Key Takeaways

1. **Shared Memory** is ~100× faster than global memory
2. **Tiling** reduces global memory bandwidth by reusing data
3. **Coalesced Access** is achieved when consecutive threads read consecutive addresses
4. **Synchronization** is required when threads share data
5. **Template Parameters** allow compile-time tile size selection
