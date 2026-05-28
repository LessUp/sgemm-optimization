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

    add_executable(test_sgemm tests/test_sgemm.cu)
    target_include_directories(test_sgemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(test_sgemm PRIVATE SGEMM_HAS_WMMA_TARGET=${SGEMM_HAS_WMMA_TARGET})
    target_link_options(test_sgemm PRIVATE -L${SGEMM_CUDA_LIBRARY_DIR})
    target_link_libraries(test_sgemm PRIVATE
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
    target_compile_options(test_sgemm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # 工具层测试
    add_executable(test_utils tests/test_utils.cu)
    target_include_directories(test_utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(test_utils PRIVATE SGEMM_HAS_WMMA_TARGET=${SGEMM_HAS_WMMA_TARGET})
    target_link_options(test_utils PRIVATE -L${SGEMM_CUDA_LIBRARY_DIR})
    target_link_libraries(test_utils PRIVATE
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
    target_compile_options(test_utils PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # 性能回归测试
    add_executable(test_performance tests/test_performance.cu)
    target_include_directories(test_performance PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(test_performance PRIVATE SGEMM_HAS_WMMA_TARGET=${SGEMM_HAS_WMMA_TARGET})
    target_link_options(test_performance PRIVATE -L${SGEMM_CUDA_LIBRARY_DIR})
    target_link_libraries(test_performance PRIVATE
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
    target_compile_options(test_performance PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # Benchmark 设置模块测试
    add_executable(test_benchmark_settings tests/test_benchmark_settings.cu)
    target_include_directories(test_benchmark_settings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_benchmark_settings PRIVATE
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
    )
    target_compile_options(test_benchmark_settings PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # Kernel catalog module test
    add_executable(test_kernel_catalog tests/test_kernel_catalog.cu)
    target_include_directories(test_kernel_catalog PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_kernel_catalog PRIVATE
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
    )
    target_compile_options(test_kernel_catalog PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # Device info seam test
    add_executable(test_device_info_seam tests/test_device_info_seam.cu)
    target_include_directories(test_device_info_seam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(test_device_info_seam PRIVATE
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
    )
    target_compile_options(test_device_info_seam PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    include(GoogleTest)
    gtest_discover_tests(test_sgemm)
    gtest_discover_tests(test_utils)
    gtest_discover_tests(test_performance)
    gtest_discover_tests(test_benchmark_settings)
    gtest_discover_tests(test_kernel_catalog)
    gtest_discover_tests(test_device_info_seam)
endif()
