## Context

The project serves as an educational and demonstrative implementation of core GPU optimization techniques in the HPC domain. The architecture must support five distinct kernel implementations while maintaining a clean, testable, and benchmarkable codebase.

## Goals / Non-Goals

### Goals
- Support progressive optimization from naive to Tensor Core
- Maintain consistent kernel interface across all implementations
- Enable accurate performance benchmarking
- Ensure numerical correctness verification

### Non-Goals
- Production-grade library (educational focus)
- Multi-GPU support
- Other GEMM variants (DGEMM, CGEMM)

## Decisions

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.cu                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Benchmark  │  │   Verify    │  │  CLI Parser │              │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘              │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
          ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Kernel Implementations                        │
│  ┌────────┐ ┌────────┐ ┌────────────┐ ┌─────────────┐ ┌───────┐ │
│  │ Naive  │ │ Tiled  │ │ Bank-Free  │ │ Dbl-Buffer  │ │ TC    │ │
│  └────────┘ └────────┘ └────────────┘ └─────────────┘ └───────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Unified Kernel Interface

All kernels conform to a unified template interface:

```cpp
template<int TILE_SIZE = 32>
void launch_xxx_sgemm(
    const float* A,      // M×K input matrix
    const float* B,      // K×N input matrix
    float* C,            // M×N output matrix
    int M, int K, int N,
    cudaStream_t stream = 0
);
```

### Exception-Based Error Handling

```cpp
struct CudaError : std::runtime_error {
    explicit CudaError(const std::string& msg) : std::runtime_error(msg) {}
};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw CudaError(cudaGetErrorString(err)); \
        } \
    } while(0)
```

### Verification Tolerances

| Kernel Type | rtol | atol |
|-------------|------|------|
| Standard FP32 | 1e-3 | 1e-4 |
| Tensor Core | 5e-2 | 1e-2 |

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Kernel interface changes would require updating all implementations | Interface is stable; use delta specs for any changes |
| Error handling exceptions may impact performance in hot paths | Exceptions only on setup/teardown, not kernel execution |
