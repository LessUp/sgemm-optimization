---
title: 1. Naïve Kernel
---

# Kernel 1: Naïve Implementation

The simplest approach — each thread computes one output element



## Algorithm

For matrix multiplication C = A × B where:
- A is M × K
- B is K × N  
- C is M × N

Each thread computes:
```
C[row, col] = Σ A[row, k] × B[k, col]  for k = 0 to K-1
```

### Thread Mapping

```
Grid:  (N + 15) / 16 × (M + 15) / 16 blocks
Block: 16 × 16 threads

Thread (tx, ty) in block (bx, by) computes:
  row = by × 16 + ty
  col = bx × 16 + tx
  C[row, col] if row < M and col < N
```



## Memory Access Pattern

### The Problem: Uncoalesced Access

```
Thread (0,0) reads: A[0,0], A[0,1], A[0,2] ...  →  CONSECUTIVE ✓
                   B[0,0], B[N,0], B[2N,0] ...   →  STRIDE-N ✗

Thread (0,1) reads: A[0,0], A[0,1], A[0,2] ...  →  Same as thread 0!
                   B[0,1], B[N,1], B[2N,1] ...   →  STRIDE-N ✗
```

When reading matrix **B**, consecutive threads access elements separated by **N** floats (stride-N access). This causes:

1. **Memory request serialization** — GPU must issue separate loads
2. **Cache inefficiency** — loaded data isn't shared between threads
3. **~12.5% bandwidth utilization** — vs 100% with coalesced access

### Visual Representation

```
Memory Layout:
A (row-major):    [0,0] [0,1] [0,2] ... [1,0] [1,1] ...
B (row-major):    [0,0] [0,1] [0,2] ... [1,0] [1,1] ...
                          ↑
                    Threads need [k, col]
                    With col = 0,1,2,3...
                    This is NOT consecutive in memory!
```



## Code Walkthrough

### 1. Kernel Signature

```cpp
__global__ void sgemm_naive_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K)
```

- `__global__` — CUDA kernel launchable from CPU
- Pointers marked `const` for read-only matrices A and B
- Dimensions passed as `int` (matrices assumed row-major)

### 2. Thread Indexing

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

- Uses 2D grid/block structure for natural matrix mapping
- `blockIdx` identifies which tile of output
- `threadIdx` identifies position within tile

### 3. Bounds Checking

```cpp
if (row < M && col < N) {
```

Essential for matrices not perfectly divisible by block size. Prevents out-of-bounds writes.

### 4. Inner Product Computation

```cpp
float sum = 0.0f;
for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
}
```

- Accumulator in register (`float sum`)
- Single loop over K dimension
- Row of A × Column of B



## Learning Checkpoints

After understanding this kernel, you should be able to:

- [x] Explain CUDA's thread hierarchy (grid/block/thread)
- [x] Identify uncoalesced memory access patterns
- [x] Calculate arithmetic intensity for matrix multiply
- [x] Understand why this kernel is memory-bound



## Further Reading

- [CUDA Best Practices Guide — Memory Coalescing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
- [NVIDIA Blog: CUDA Pro Tip](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
