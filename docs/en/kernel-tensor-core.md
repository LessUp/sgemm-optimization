---
title: 5. Tensor Core
---

# Kernel 5: Tensor Core (WMMA)

Leveraging dedicated matrix multiply-accumulate hardware



## Tensor Core Architecture

### Hardware Capabilities

| Generation | Architecture | Operations/Cycle | Precision |
|------------|-------------|------------------|-----------|
| Volta (V100) | sm_70 | 64 FMA | FP16/FP32 |
| Turing (RTX 20) | sm_75 | 64 FMA | FP16/INT8/INT32 |
| Ampere (A100/RTX 30) | sm_80/sm_86 | 256 FMA | FP16/BF16/TF32 |
| Ada (RTX 40) | sm_89 | 512 FMA | FP16/BF16/TF32 |
| Hopper (H100) | sm_90 | 1024 FMA | FP8/FP16/BF16 |

### WMMA Fragment Size

```
Warp Matrix Multiply Accumulate (WMMA):

Fragment A: 16×16 FP16 matrix (row-major)
Fragment B: 16×16 FP16 matrix (row-major)
Fragment C: 16×16 FP32 matrix (row-major)
             ↓
          D = A × B + C
             ↓
Fragment D: 16×16 FP32 matrix

One warp (32 threads) collaborates on one 16×16×16 operation.
```



## Implementation

```cpp
// File: src/kernels/tensor_core_sgemm.cuh

#include <mma.h>
using namespace nvcuda::wmma;

// WMMA uses 16×16 tiles
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void sgemm_tensor_core_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Each warp computes one 16×16 output tile
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    
    // Check if this warp has work
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N)
        return;

    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    
    // Initialize accumulator to zero
    fill_fragment(acc_frag, 0.0f);
    
    // Iterate over K dimension in 16-element chunks
    for (int k = 0; k < K; k += WMMA_K) {
        // Calculate pointers for this tile
        const half* a_ptr = A + warpM * WMMA_M * K + k;
        const half* b_ptr = B + k * N + warpN * WMMA_N;
        
        // Load fragments
        load_matrix_sync(a_frag, a_ptr, K);
        load_matrix_sync(b_frag, b_ptr, N);
        
        // Perform MMA
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    // Store result
    float* c_ptr = C + warpM * WMMA_M * N + warpN * WMMA_N;
    store_matrix_sync(c_ptr, acc_frag, N, mem_row_major);
}
```



## Mixed Precision Considerations

### Accuracy Trade-off

```
FP32 precision: ~7 decimal digits
FP16 precision: ~3 decimal digits

Conversion FP32 → FP16 introduces quantization error.
But FP32 accumulation maintains precision for the sum.
```

### Verification Tolerances

```cpp
// Standard FP32 kernels
const float rtol_fp32 = 1e-3f;
const float atol_fp32 = 1e-4f;

// Tensor Core mixed precision
const float rtol_tc = 5e-2f;   // 50× looser
const float atol_tc = 1e-2f;   // 100× looser
```

### When FP16 is Acceptable

| Use Case | FP16 OK? |
|----------|----------|
| Deep Learning Training | ✓ Yes |
| Deep Learning Inference | ✓ Yes |
| Scientific Computing | ⚠️ Check |
| Financial Calculations | ✗ No |



## Fragment Layout Details

Each warp's 32 threads hold the 16×16 matrix collaboratively:

```
16×16 matrix distributed across 32 threads:

Thread Layout (8 rows × 4 columns of threads):
┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │  Row 0 holds elements at column 0-3
├───┼───┼───┼───┤
│ 4 │ 5 │ 6 │ 7 │  Row 1 holds elements at column 0-3
├───┼───┼───┼───┤
│...│   │   │   │
└───┴───┴───┴───┘

Each thread holds 8 FP16 values (4 rows × 2 columns).
```

The exact mapping is hardware-defined and managed by WMMA APIs.



## Optimization Opportunities

Our Tensor Core kernel achieves only **40% of cuBLAS** performance. The gap comes from:

1. **Multi-level tiling**: Warp-level + thread-level tiles
2. **Instruction pipelining**: Issue multiple MMAs concurrently
3. **Shared memory staging**: Better FP16 data layout
4. **Epilogue fusion**: Combine output processing with MMA

For production use, **always use cuBLAS** or **CUTLASS**.

---
