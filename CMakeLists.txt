cmake_minimum_required(VERSION 3.18)
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

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
    endif()
endif()

# ── 输出目录 ─────────────────────────────────────────────────────
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ── 主程序 ───────────────────────────────────────────────────────
add_executable(sgemm_benchmark src/main.cu)

target_include_directories(sgemm_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

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
    target_link_libraries(test_sgemm PRIVATE
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
    )
    target_compile_options(test_sgemm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    include(GoogleTest)
    gtest_discover_tests(test_sgemm)
endif()
