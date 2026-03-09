# SGEMM Optimization: From Naive to Tensor Core

[![CI](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/sgemm-optimization/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/sgemm-optimization/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/sgemm-optimization/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

Hand-written, progressively optimized CUDA matrix multiplication — the "Hello World" of HPC. Five kernel variants demonstrate core GPU optimization techniques, from a naive triple loop to **Tensor Core WMMA reaching 40% of cuBLAS throughput**.

## Performance (RTX 3060 Laptop, 1024×1024×1024)

| Kernel | GFLOPS | vs cuBLAS | Time | Key Technique |
|--------|-------:|----------:|-----:|---------------|
| **cuBLAS** (ref) | 5727 | 100% | 0.375 ms | NVIDIA optimized library |
| **Tensor Core** (WMMA) | 2300 | 40.2% | 0.934 ms | FP16→FP32 mixed precision |
| **Tiled** (32×32) | 753 | 13.1% | 2.853 ms | Shared memory blocking |
| **Double Buffer** | 701 | 12.2% | 3.064 ms | Compute-memory overlap |
| **Bank Conflict Free** | 673 | 11.8% | 3.190 ms | Shared memory padding (+1) |
| **Naive** | 604 | 10.6% | 3.553 ms | One thread per output element |

*All kernels verified against cuBLAS (allclose: rtol=1e-3, atol=1e-4; Tensor Core: rtol=5e-2)*

## Optimization Roadmap

```
  ┌─────────┐     ┌──────────┐     ┌──────────────┐     ┌───────────────┐
  │  Naive  │────▶│  Tiled   │────▶│  Bank-Free   │────▶│ Double Buffer │
  │ 604 GF  │     │ 753 GF   │     │   673 GF     │     │   701 GF      │
  └─────────┘     └──────────┘     └──────────────┘     └───────┬───────┘
                                                                │
                                                                ▼
                                                    ┌───────────────────┐
                                                    │   Tensor Core     │
                                                    │   2300 GF (WMMA)  │
                                                    └───────────────────┘
```

| Stage | What Changes | Why It Helps |
|-------|-------------|--------------|
| **Naive → Tiled** | Load tiles into shared memory | Data reuse reduces global memory traffic by TILE_SIZE× |
| **Tiled → Bank-Free** | Pad shared memory `[32][33]` | Eliminates 32-way bank conflicts on column access |
| **Bank-Free → Double Buffer** | Two shared-memory buffers | Overlaps next-tile load with current-tile compute |
| **→ Tensor Core** | WMMA API `mma_sync` | Dedicated matrix units, ~8× peak over CUDA cores |

## Build & Run

```bash
# Makefile (adjust GPU arch for your hardware)
make GPU_ARCH=sm_86
make benchmark

# Or CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/bin/sgemm_benchmark
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

Property-based tests with Google Test:

| Property | What It Verifies |
|----------|-----------------|
| **Numerical correctness** | All kernels match cuBLAS output (allclose) |
| **Tensor Core tolerance** | Correct under relaxed FP16 tolerance |
| **Error detection** | Verification system catches injected errors |
| **Dimension invariance** | All kernels handle arbitrary aligned sizes |

```bash
make test
# Or: cmake --build build --target test_sgemm && ctest --test-dir build
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

- **Build**: CMake 3.18+ with `target_include_directories`, `target_compile_options` (generator expressions), FetchContent for GTest v1.14.0
- **Code style**: clang-format enforced via CI
- **CI**: GitHub Actions — CUDA container build + format check
- **Testing**: Google Test property-based verification against cuBLAS

## References

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Simon Boehm
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's high-performance GEMM library
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [Roofline Model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)

## License

MIT License
