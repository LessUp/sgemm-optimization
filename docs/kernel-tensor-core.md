---
layout: default
title: 5. Tensor Core
parent: Home
nav_order: 7
permalink: /docs/kernel-tensor-core
---

# Kernel 5: Tensor Core (WMMA)
{: .fs-8 }

Leveraging dedicated matrix multiply-accumulate hardware
{: .fs-6 .fw-300 }

---

## Overview

NVIDIA Tensor Cores are specialized units that perform **mixed-precision matrix multiply-accumulate** operations. A single instruction can compute a 4×4×4 or larger matrix operation — achieving **~8× theoretical peak throughput** of CUDA cores (~3-4× in practice).

<div class="highlight-box info">
  <strong>Key Insight</strong><br>
  Tensor Cores use FP16 inputs with FP32 accumulation. This mixed precision provides significant speedup while maintaining accuracy for most deep learning and HPC workloads.
</div>

---

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

---

## The WMMA API

### Fragment Declaration

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Matrix fragments
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;
fragment<accumulator, 16, 16, 16, float> d_frag;
```

### WMMA Operations

```cpp
// Initialize accumulator to zero
fill_fragment(c_frag, 0.0f);

// Load data from global/shared memory
load_matrix_sync(a_frag, A_ptr, lda);
load_matrix_sync(b_frag, B_ptr, ldb);

// Perform matrix multiply-accumulate
mma_sync(d_frag, a_frag, b_frag, c_frag);

// Store result
store_matrix_sync(D_ptr, d_frag, ldd, mem_row_major);
```

---

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

---

## FP32 → FP16 Conversion

Since Tensor Cores require FP16 input, we must convert:

```cpp
// FP32 to FP16 conversion kernel
__global__ void convert_fp32_to_fp16(
    const float* __restrict__ in,
    half* __restrict__ out,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// Host function to launch Tensor Core kernel
void launch_tensor_core_sgemm(
    const float* A_fp32,
    const float* B_fp32,
    float* C_fp32,
    int M, int N, int K,
    cudaStream_t stream)
{
    // Check alignment for WMMA
    bool aligned = (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0);
    
    if (!aligned) {
        // Fall back to FP32 tiled kernel for non-aligned sizes
        sgemm_tiled<<<grid, block, 0, stream>>>(A_fp32, B_fp32, C_fp32, M, N, K);
        return;
    }

    // Allocate temporary FP16 buffers via RAII wrappers
    DeviceMemory<half> A_fp16(M * K);
    DeviceMemory<half> B_fp16(K * N);

    // Convert inputs
    int threads = 256;
    int blocks_A = (M * K + threads - 1) / threads;
    int blocks_B = (K * N + threads - 1) / threads;
    
    convert_fp32_to_fp16<<<blocks_A, threads, 0, stream>>>(A_fp32, A_fp16.get(), M * K);
    convert_fp32_to_fp16<<<blocks_B, threads, 0, stream>>>(B_fp32, B_fp16.get(), K * N);

    // Launch WMMA kernel
    dim3 block(16, 4);   // 64 threads (2 warps)
    dim3 grid((N + WMMA_N - 1) / WMMA_N / 2, 
              (M + WMMA_M - 1) / WMMA_M);
    
    sgemm_tensor_core_kernel<<<grid, block, 0, stream>>>(
        A_fp16.get(), B_fp16.get(), C_fp32, M, N, K);

    // No manual cleanup: the RAII wrappers release device memory automatically
}
```

---

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

---

## Performance Characteristics

| Metric | Double Buffer | Tensor Core | Improvement |
|--------|---------------|-------------|-------------|
| **GFLOPS (1024³)** | 701 | 2300 | **3.3×** |
| **vs cuBLAS** | 12.2% | 40.2% | — |
| **Precision** | FP32 | FP16→FP32 | Mixed |
| **Alignment Required** | No | 16× | Yes |
| **Compute Units** | CUDA Cores | Tensor Cores | Dedicated |

---

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

---

## Architecture Guards

Use conditional compilation for different GPU generations:

```cpp
#if __CUDA_ARCH__ >= 700
    // WMMA available (Volta+)
    #include <mma.h>
    // Tensor Core implementation
#else
    // Fallback to FP32 kernel
    // CUDA < 7.0 or no Tensor Cores
#endif
```

---

## Optimization Opportunities

Our Tensor Core kernel achieves only **40% of cuBLAS** performance. The gap comes from:

1. **Multi-level tiling**: Warp-level + thread-level tiles
2. **Instruction pipelining**: Issue multiple MMAs concurrently
3. **Shared memory staging**: Better FP16 data layout
4. **Epilogue fusion**: Combine output processing with MMA

For production use, **always use cuBLAS** or **CUTLASS**.

---

## Key Takeaways

1. **Tensor Cores**: Dedicated matrix units, ~8× theoretical peak (~3-4× achieved)
2. **WMMA API**: Warp-level abstraction for Tensor Core programming
3. **Mixed Precision**: FP16 inputs, FP32 accumulation
4. **Alignment**: M, K, N must be multiples of 16 for WMMA
5. **Fallback**: Always provide FP32 kernel for edge cases
