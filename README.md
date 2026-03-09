# SGEMM Optimization: From Naive to Tensor Core

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)

English | [简体中文](README.zh-CN.md)

Hand-written, progressively optimized matrix multiplication — the "Hello World" of HPC.

## Performance (RTX 3060 Laptop, 1024×1024×1024)

| Kernel | GFLOPS | vs cuBLAS |
|--------|--------|-----------|
| cuBLAS (ref) | 5727 | 100% |
| Tensor Core (WMMA) | 2300 | 40.2% |
| Tiled (32×32) | 753 | 13.1% |
| Double Buffer | 701 | 12.2% |
| Bank Conflict Free | 673 | 11.8% |
| Naive | 604 | 10.6% |

## Optimization Levels

| Level | Description | Key Technique |
|-------|-------------|---------------|
| Naive | Basic triple loop | One thread per output element |
| Tiled | Shared memory tiling | Data reuse, reduced global memory access |
| Bank Conflict Free | Eliminate bank conflicts | Shared memory padding (+1) |
| Double Buffer | Pipeline overlap | Compute/memory overlap |
| Tensor Core | WMMA API | Hardware-accelerated matrix ops (FP16→FP32) |

## Build & Run

```bash
make GPU_ARCH=sm_86   # Adjust for your GPU
./build/sgemm_benchmark
```

## Key Optimization Techniques

1. **Memory Coalescing** — Warp-aligned memory access for full bandwidth
2. **Shared Memory Tiling** — O(N³/TILE_SIZE) global memory reduction
3. **Bank Conflict Elimination** — +1 padding for 32x bandwidth recovery
4. **Double Buffering** — Overlap next-tile load with current-tile compute
5. **Tensor Core (WMMA)** — 16×16×16 hardware MMA, ~8x over CUDA Cores

## Project Structure

```
├── src/kernels/           # 5 kernel implementations
├── src/utils/             # CUDA utils, benchmark, verification
├── src/main.cu            # Entry point
├── tests/test_sgemm.cu    # Google Test property tests
└── Makefile
```

## License

MIT License
