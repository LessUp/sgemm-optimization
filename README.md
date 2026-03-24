# SGEMM Optimization: From Naive to Tensor Core

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

Hand-written, progressively optimized CUDA matrix multiplication — the "Hello World" of HPC. Five kernel variants demonstrate core GPU optimization techniques, from a naive triple loop to a guarded Tensor Core WMMA path with explicit mixed-precision benchmarking.

## Performance

The exact GFLOPS you see will depend on GPU model, CUDA version, and problem size.
The benchmark now reports two Tensor Core views:

- **Tensor Core (WMMA end-to-end)**: includes FP32→FP16 conversion and safe fallback for non-WMMA-compatible dimensions.
- **Tensor Core (WMMA compute-only)**: times only the WMMA compute path and is shown only when `M`, `K`, and `N` are multiples of 16.

Verification tolerances are centralized in code:

- Standard FP32 kernels: `rtol=1e-3`, `atol=1e-4`
- Tensor Core mixed-precision path: `rtol=5e-2`, `atol=1e-2`

The default benchmark set includes:

- aligned square cases: `512x512x512`, `1024x1024x1024`
- one aligned non-square case: `256x384x640`
- one unaligned edge case: `511x513x1025` to exercise safe Tensor Core fallback

> Note: the printed theoretical peak and roofline numbers are approximate analytical references, not exact hardware limits.

## Optimization Roadmap

```
  Naive  ->  Tiled  ->  Bank-Free  ->  Double Buffer  ->  Tensor Core (WMMA)
```

| Stage | What Changes | Why It Helps |
|-------|-------------|--------------|
| **Naive → Tiled** | Load tiles into shared memory | Data reuse reduces global memory traffic by TILE_SIZE× |
| **Tiled → Bank-Free** | Pad shared memory `[32][33]` | Eliminates 32-way bank conflicts on column access |
| **Bank-Free → Double Buffer** | Two shared-memory buffers | Restructures tile staging and buffering to reduce memory stalls |
| **→ Tensor Core** | WMMA API `mma_sync` | Dedicated matrix units, ~8× peak over CUDA cores |

## Build & Run

Recommended path: CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark
./build/bin/sgemm_benchmark --dims 256 384 640
./build/bin/sgemm_benchmark -a
```

Quick local path: Makefile

```bash
make GPU_ARCH=sm_86
make benchmark
make test
```

## Project Structure

```
sgemm-optimization/
├── src/
│   ├── kernels/
│   │   ├── naive_sgemm.cuh              # Naive: basic triple loop
│   │   ├── tiled_sgemm.cuh              # Tiled: shared memory blocking
│   │   ├── bank_conflict_free_sgemm.cuh # Bank conflict elimination
│   │   ├── double_buffer_sgemm.cuh      # Double buffer pipeline
│   │   └── tensor_core_sgemm.cuh        # Tensor Core (WMMA API)
│   ├── utils/
│   │   ├── cuda_utils.cuh               # CUDA error checking & utilities
│   │   ├── benchmark.cuh                # Benchmark framework (CUDA Events)
│   │   └── verify.cuh                   # Correctness verification (vs cuBLAS)
│   └── main.cu                          # Entry point
├── tests/
│   └── test_sgemm.cu                    # Google Test property tests
├── roofline_data_*.csv                  # Roofline analysis data
├── CMakeLists.txt                       # CMake build (recommended)
└── Makefile                             # Make build (quick start)
```

## Testing

Google Test coverage includes:

| Property | What It Verifies |
|----------|-----------------|
| **Numerical correctness** | Standard kernels match cuBLAS across square and non-square cases |
| **Tensor Core fast path** | WMMA path is validated on `16`-aligned dimensions |
| **Tensor Core fallback** | Non-aligned dimensions safely fall back to an FP32 kernel |
| **Small/edge inputs** | Includes `1x1x1` and unaligned edge cases |
| **Error detection** | Verification helpers stay consistent with benchmark tolerances |

```bash
cmake --build build --target test_sgemm
ctest --test-dir build

# or
make test
```

## GPU Architecture Reference

| GPU Family | Architecture | Compute Capability | Build Flag |
|------------|-------------|-------------------|-----------|
| Tesla V100 | Volta | sm_70 | `GPU_ARCH=sm_70` |
| RTX 2080 | Turing | sm_75 | `GPU_ARCH=sm_75` |
| RTX 3090 / A100 | Ampere | sm_80 / sm_86 | `GPU_ARCH=sm_86` |
| RTX 4090 / L40 | Ada Lovelace | sm_89 | `GPU_ARCH=sm_89` |
| H100 | Hopper | sm_90 | `GPU_ARCH=sm_90` |

## Engineering Quality

- **Build**: CMake 3.18+ is the primary build system; Makefile remains available for quick local use
- **Code style**: clang-format enforced via CI
- **CI**: GitHub Actions runs format checks and a containerized CUDA compile-only build; GPU runtime tests are still local / dedicated-runner only
- **Testing**: Google Test verification against cuBLAS, including Tensor Core fallback and edge-size coverage

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's high-performance GEMM library
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Roofline Model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)

## License

MIT License
