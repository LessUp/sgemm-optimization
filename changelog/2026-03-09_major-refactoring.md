# Major Refactoring - v2.0.0

Date: 2026-03-09

## Critical Bug Fixes

### Error checking macros: exit() → exceptions
- `CUDA_CHECK`, `CUBLAS_CHECK`, `CURAND_CHECK` all called `exit(EXIT_FAILURE)` on error
- This prevented RAII destructors (`DeviceMemory`, `CublasHandle`, `SGEMMBenchmark`) from running — **leaked all GPU memory on any error**
- Added `CudaError` exception type; all macros now throw instead of exiting
- All existing RAII wrappers now properly clean up on error paths

### Tensor Core launch: per-call cudaMalloc/cudaFree → RAII
- `launch_tensor_core_sgemm` allocated and freed FP16 conversion buffers on every call
- Replaced with `DeviceMemory<half>` RAII wrappers — no leak on error
- Also cleaned up confusing dead code: multiple overwritten `blockDim`/`gridDim` calculations replaced with a single clean configuration

### Benchmark: per-call cudaMalloc/cudaFree → DeviceMemory RAII
- `SGEMMBenchmark::run()` used raw `cudaMalloc`/`cudaFree` for all test matrices
- Replaced with `DeviceMemory<float>` RAII wrappers and `.copyFromHost()`/`.copyToHost()`/`.zero()` methods
- GPU memory is now properly freed even if benchmarking throws an exception

### External dependency removed
- `main.cu` included `"tensorcraft/kernels/gemm.hpp"` from `../modern-ai-kernels/include` — an external sibling project that may not exist
- Kernel wrappers (`naive_kernel`, `tiled_kernel`, `double_buffer_kernel`) now call local `launch_*_sgemm` functions directly
- Removed `-I../modern-ai-kernels/include` from Makefile
- Project is now fully self-contained

## Build System

### Added CMakeLists.txt
- Proper CMake build alongside existing Makefile
- CUDA architecture auto-detect (`native` on CMake 3.24+)
- `CMAKE_EXPORT_COMPILE_COMMANDS` for IDE support
- FetchContent for GoogleTest
- Imported `CUDA::cublas` and `CUDA::curand` targets

### Files Modified
- `src/utils/cuda_utils.cuh` — CudaError exception, throw instead of exit
- `src/kernels/tensor_core_sgemm.cuh` — RAII FP16 buffers, clean launch config
- `src/utils/benchmark.cuh` — DeviceMemory RAII in run()
- `src/main.cu` — removed external dependency, use local kernels
- `Makefile` — removed external include path

### Files Created
- `CMakeLists.txt` — modern CMake build
- `changelog/2026-03-09_major-refactoring.md`
