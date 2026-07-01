cmake_minimum_required(VERSION 3.18)

if(UNIX)
    if(EXISTS "/usr/bin/gcc-12")
        set(SGEMM_DEFAULT_C_COMPILER "/usr/bin/gcc-12")
    elseif(EXISTS "/usr/bin/cc")
        set(SGEMM_DEFAULT_C_COMPILER "/usr/bin/cc")
    endif()

    if(EXISTS "/usr/bin/g++-12")
        set(SGEMM_DEFAULT_CXX_COMPILER "/usr/bin/g++-12")
    elseif(EXISTS "/usr/bin/c++")
        set(SGEMM_DEFAULT_CXX_COMPILER "/usr/bin/c++")
    endif()

    if(NOT DEFINED CMAKE_C_COMPILER AND DEFINED SGEMM_DEFAULT_C_COMPILER)
        set(CMAKE_C_COMPILER "${SGEMM_DEFAULT_C_COMPILER}")
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER AND DEFINED SGEMM_DEFAULT_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "${SGEMM_DEFAULT_CXX_COMPILER}")
    endif()
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND DEFINED SGEMM_DEFAULT_CXX_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER "${SGEMM_DEFAULT_CXX_COMPILER}")
    endif()
endif()

project(sgemm_optimization
    VERSION 2.1.0
    DESCRIPTION "SGEMM optimization from naive to Tensor Core"
    LANGUAGES CXX CUDA
)

# ── C++ / CUDA 标准 ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── CUDA 配置 ────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
get_target_property(SGEMM_CUDART_LIBRARY CUDA::cudart IMPORTED_LOCATION)
get_filename_component(SGEMM_CUDA_LIBRARY_DIR "${SGEMM_CUDART_LIBRARY}" DIRECTORY)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # `native` has proven unreliable with some toolkit/driver combinations and
    # can silently drop WMMA-capable codegen. Default to a portable set that
    # keeps pre-Volta fallback builds working while still emitting Tensor Core
    # code for modern GPUs.
    set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86 89 90)
endif()

set(SGEMM_HAS_WMMA_TARGET 0)
foreach(cuda_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REGEX MATCH "^[0-9]+" SGEMM_CUDA_ARCH_NUMBER "${cuda_arch}")
    if(SGEMM_CUDA_ARCH_NUMBER AND SGEMM_CUDA_ARCH_NUMBER GREATER_EQUAL 70)
        set(SGEMM_HAS_WMMA_TARGET 1)
        break()
    endif()
endforeach()

# ── 输出目录 ─────────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── 主程序 ───────────────────────────────────────────────────────
add_executable(sgemm_benchmark src/main.cu)

target_include_directories(sgemm_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(sgemm_benchmark PRIVATE SGEMM_HAS_WMMA_TARGET=${SGEMM_HAS_WMMA_TARGET})

target_link_libraries(sgemm_benchmark PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

target_compile_options(sgemm_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
)

# ── 测试 ─────────────────────────────────────────────────────────
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Include test helper functions
    include(cmake/SgemmTestHelpers.cmake)

    # ═══════════════════════════════════════════════════════════════
    # CPU-only Tests (no CUDA required)
    # ═══════════════════════════════════════════════════════════════

    # Benchmark settings module test - pure C++, no CUDA dependencies
    sgemm_add_cpu_test(
        NAME test_benchmark_settings
        SOURCES tests/test_benchmark_settings.cpp
    )

    # Device info provider CPU tests - uses fake device properties
    # Compiled as .cu because it includes kernel headers with CUDA launch syntax
    sgemm_add_cpu_test(
        NAME test_device_info_cpu
        SOURCES tests/test_device_info_cpu.cu
    )

    # ═══════════════════════════════════════════════════════════════
    # CUDA Tests (requires CUDA device, skipped if unavailable)
    # ═══════════════════════════════════════════════════════════════

    # Kernel correctness tests with property-based testing
    sgemm_add_cuda_test(
        NAME test_sgemm
        SOURCES tests/test_sgemm.cu
        CUDA_LIBRARIES CUDA::curand
        REQUIRES_WMMA
    )

    # Utility layer tests (DeviceMemory, verifier RAII, etc.)
    sgemm_add_cuda_test(
        NAME test_utils
        SOURCES tests/test_utils.cu
        CUDA_LIBRARIES CUDA::curand
        REQUIRES_WMMA
    )

    # Kernel catalog module test - requires device memory and kernel launch
    sgemm_add_cuda_test(
        NAME test_kernel_catalog
        SOURCES tests/test_kernel_catalog.cu
        REQUIRES_WMMA
    )

    # Device info provider CUDA tests - requires real GPU for production adapter
    sgemm_add_cuda_test(
        NAME test_device_info_cuda
        SOURCES tests/test_device_info_cuda.cu
    )

    # ═══════════════════════════════════════════════════════════════
    # Performance Tests (CUDA + performance label)
    # ═══════════════════════════════════════════════════════════════

    # Performance regression tests
    sgemm_add_cuda_perf_test(
        NAME test_performance
        SOURCES tests/test_performance.cu
        CUDA_LIBRARIES CUDA::curand
        REQUIRES_WMMA
    )

endif()
