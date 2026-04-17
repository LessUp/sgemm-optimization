---
layout: default
title: Kernel Implementations
nav_order: 3
---

# Kernel Implementations

Deep dive into each SGEMM kernel variant, from the naive triple-loop to Tensor Core WMMA.

## 1. Naive Kernel

### Implementation
`src/kernels/naive_sgemm.cuh`

### Algorithm

```cpp
// Each thread computes one output element: C[row, col] = A[row, :] × B[:, col]
__global__ void sgemm_naive_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### Characteristics

| Aspect | Detail |
|--------|--------|
| **Memory Access** | Uncoalesced for B (stride-N access) |
| **Data Reuse** | None (each element read once per use) |
| **Shared Memory** | None |
| **Complexity** | O(M×N×K) global memory reads |

### Performance Bottlenecks

1. **No memory coalescing**: B matrix accessed with stride N
2. **No data reuse**: Same elements read repeatedly from global memory
3. **Low arithmetic intensity**: 2 FLOPs per 8 bytes (for K large)

### When to Use

- Baseline for comparison
- Understanding SGEMM fundamentals
- Small matrices where optimization overhead dominates

---

## 2. Tiled Kernel

### Implementation
`src/kernels/tiled_sgemm.cuh`

### Algorithm

```cpp
template<int TILE_SIZE = 32>
__global__ void sgemm_tiled_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from A
        int a_row = row;
        int a_col = t * TILE_SIZE + tx;
        if (a_row < M && a_col < K)
            As[ty][tx] = A[a_row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        // Load tile from B
        int b_row = t * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < K && b_col < N)
            Bs[ty][tx] = B[b_row * N + b_col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute tile multiplication
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Key Improvement: Shared Memory Tiling

```
Global Memory          Shared Memory         Registers
    │                      │                     │
    ├── A[row*K:k] ───────▶│ As[ty][tx]          │
    ├── B[K*N+col] ───────▶│ Bs[ty][tx]          │
    │                      │                     │
    │                      ├── sync ────────────▶│ sum += As[ty][k] * Bs[k][tx]
    │                      │                     │
```

### Characteristics

| Aspect | Detail |
|--------|--------|
| **Memory Access** | Coalesced (consecutive threads read consecutive addresses) |
| **Data Reuse** | TILE_SIZE× (each shared element used TILE_SIZE times) |
| **Shared Memory** | 2 × TILE_SIZE² × 4 bytes = 8KB for TILE_SIZE=32 |
| **Complexity** | O(M×N×K/TILE_SIZE) global memory reads |

### Performance Gain

- **Global memory traffic reduced**: TILE_SIZE× less reads
- **Coalesced access**: Bandwidth utilization ~100% vs ~12.5% in naive
- **Typical improvement**: 2-5× over naive depending on matrix size

---

## 3. Bank Conflict Free Kernel

### Implementation
`src/kernels/bank_conflict_free_sgemm.cuh`

### The Problem: Shared Memory Bank Conflicts

Shared memory is divided into 32 banks. When multiple threads in a warp access the same bank, accesses are serialized.

```
// ❌ 32-way bank conflict
__shared__ float As[32][32];

// Thread i accesses: As[k][i]
// Bank index: (k * 32 + i) % 32 = i % 32 = i
// ALL 32 threads hit DIFFERENT banks in SAME cycle → NO conflict... 
// Wait, actually when accessing COLUMN-WISE:
// Thread i accesses As[i][k]
// Bank index: (i * 32 + k) % 32 = k % 32 = k
// ALL 32 threads hit the SAME bank → 32-way conflict!
```

### The Solution: Padding

```cpp
// ✅ Eliminate bank conflicts with +1 padding
__shared__ float As[32][33];  // Note: 33, not 32!

// Thread i accesses: As[k][i]
// Bank index: (k * 33 + i) % 32
// = (k + i) % 32  (since 33 ≡ 1 mod 32)
// Each thread accesses a DIFFERENT bank → No conflict!
```

### Algorithm

```cpp
template<int TILE_SIZE = 32>
__global__ void sgemm_bank_free_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // Padding eliminates bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // ... same tiling logic, but with padded shared memory
}
```

### Characteristics

| Aspect | Detail |
|--------|--------|
| **Shared Memory** | 2 × 32 × 33 × 4 = 8,448 bytes (+5.5% overhead) |
| **Bank Conflicts** | Eliminated (32-way → 0-way) |
| **Bandwidth Gain** | ~1.1-1.3× improvement over tiled |

### Bank Conflict Visualization

```
Without Padding (32×32):
  Thread 0 → Bank 5
  Thread 1 → Bank 5  ← CONFLICT!
  Thread 2 → Bank 5  ← CONFLICT!
  ...
  Thread 31 → Bank 5 ← CONFLICT!
  Result: 32 accesses serialized → 32× slower

With Padding (32×33):
  Thread 0 → Bank 5
  Thread 1 → Bank 6
  Thread 2 → Bank 7
  ...
  Thread 31 → Bank 4
  Result: All 32 banks accessed in parallel → Full bandwidth
```

---

## 4. Double Buffer Kernel

### Implementation
`src/kernels/double_buffer_sgemm.cuh`

### Concept

Overlap global memory loads with shared memory compute using two buffers:

```
Time →
Without DB:  [Load 0] [Compute 0] [Load 1] [Compute 1] ...
With DB:     [Load 0] [Load 1 + Compute 0] [Load 2 + Compute 1] ...
```

### Algorithm

```cpp
template<int TILE_SIZE = 32>
__global__ void sgemm_double_buffer_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // Double buffers: [2][TILE_SIZE][TILE_SIZE]
    __shared__ float As[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Pre-load first tile
    load_tile(As[0], Bs[0], A, B, row, col, 0, M, N, K);
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        int curr = t % 2;
        int next = (t + 1) % 2;

        // Async load next tile (if exists)
        if (t + 1 < num_tiles) {
            load_tile(As[next], Bs[next], A, B, row, col, t + 1, M, N, K);
        }

        // Compute current tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[curr][ty][k] * Bs[curr][k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Characteristics

| Aspect | Detail |
|--------|--------|
| **Shared Memory** | 2× tiled (16,896 bytes for TILE_SIZE=32) |
| **Latency Hiding** | Global memory load overlaps with compute |
| **Complexity** | More complex control flow |
| **Typical Gain** | 1.04-1.12× over bank-free |

### Pipeline Timing

```
SM Execution Timeline:

Cycle:     0    100   200   300   400   500
          [Load0]─────────▶│
                          [Compute0]──────▶│
                                [Load1]─────────▶│
                                               [Compute1]───▶

With Double Buffer:
Cycle:     0    100   200   300   400   500
          [Load0]─────────▶│
                          │[Load1]─────────▶│
                          [Compute0]──────▶│
                                        [Compute1]───▶
                                        ↑ Overlap saves ~50 cycles
```

---

## 5. Tensor Core Kernel (WMMA)

### Implementation
`src/kernels/tensor_core_sgemm.cuh`

### Architecture Guard

```cpp
#if __CUDA_ARCH__ >= 700
    // WMMA available (Volta and newer)
    #include <mma.h>
    using namespace nvcuda::wmma;
#else
    // Fallback to FP32 kernel
#endif
```

### WMMA API Basics

Each warp (32 threads) collaboratively computes a 16×16×16 matrix multiply:

```cpp
// Declare fragments (data pieces)
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// Zero accumulator
fill_fragment(c_frag, 0.0f);

// Load data (FP16)
load_matrix_sync(a_frag, A_fp16, lda);
load_matrix_sync(b_frag, B_fp16, ldb);

// Execute matrix multiply-accumulate
mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store result (FP32)
store_matrix_sync(C, c_frag, ldc, mem_row_major);
```

### Complete Kernel Structure

```cpp
#if __CUDA_ARCH__ >= 700
void launch_tensor_core_sgemm(const float* A, const float* B, float* C,
                               int M, int N, int K, cudaStream_t stream)
{
    // Check alignment
    bool aligned = (M % 16 == 0) && (K % 16 == 0) && (N % 16 == 0);

    if (!aligned) {
        // Fallback to FP32 tiled kernel
        sgemm_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
        return;
    }

    // Allocate FP16 buffers
    half *A_fp16, *B_fp16;
    cudaMalloc(&A_fp16, M * K * sizeof(half));
    cudaMalloc(&B_fp16, K * N * sizeof(half));

    // Convert FP32 → FP16
    // ... conversion kernels ...

    // Launch WMMA kernel
    // ... WMMA configuration and launch ...

    // Cleanup
    cudaFree(A_fp16);
    cudaFree(B_fp16);
}
#endif
```

### Characteristics

| Aspect | Detail |
|--------|--------|
| **Precision** | FP16 input, FP32 accumulation (mixed precision) |
| **Alignment** | M, K, N must be multiples of 16 |
| **Fallback** | Automatic FP32 tiled kernel for non-aligned sizes |
| **Peak Performance** | ~8× FP32 CUDA cores (Ampere) |
| **Verification** | Relaxed tolerances: `rtol=5e-2, atol=1e-2` |

### WMMA Fragment Layout

```
Warp (32 threads) collaboratively computes:

    16 columns
  ┌────────────┐
  │            │ 16 rows
  │  D = A×B+C │
  │            │
  └────────────┘

Each thread holds a fragment of the 16×16 matrix.
Thread lane ID determines which elements it owns.
```

### Mixed Precision Impact

| Operation | Precision | Notes |
|-----------|-----------|-------|
| Input A | FP16 | Converted from FP32 |
| Input B | FP16 | Converted from FP32 |
| Accumulation | FP32 | Maintains precision for sum |
| Output C | FP32 | Final result |
| **Max Error** | ~2.1e-2 | Acceptable for many ML/HPC workloads |

---

## Comparison Summary

| Feature | Naive | Tiled | Bank-Free | Double Buffer | Tensor Core |
|---------|-------|-------|-----------|---------------|-------------|
| Shared Memory | ✗ | ✓ | ✓ (padded) | ✓ (2×) | ✓ (WMMA) |
| Coalesced Access | ✗ | ✓ | ✓ | ✓ | ✓ |
| Bank Conflict | ✗ | ✗ | ✓ | ✓ | ✓ |
| Latency Hiding | ✗ | ✗ | ✗ | ✓ | ✓ |
| Mixed Precision | ✗ | ✗ | ✗ | ✗ | ✓ |
| GFLOPS (1024³) | 604 | 753 | 673 | 701 | 2300 |
| Code Complexity | Low | Medium | Medium | High | High |

---

## Further Reading

- [Architecture Guide](architecture.md) - System design
- [Benchmark Results](benchmark-results.md) - Performance data
- [RFC 0001](../specs/rfc/0001-core-architecture.md) - Technical decisions
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
