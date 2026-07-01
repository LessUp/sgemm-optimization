# SGEMM CMake Test Helper Functions
#
# This module provides helper functions to standardize test target creation
# and ensure consistent handling of CPU-only vs CUDA-required tests.
#
# Test Categories:
# - CPU-only: Can compile and run without CUDA runtime/device. May still need
#   CUDA headers for struct definitions (e.g., cudaDeviceProp).
# - CUDA: Requires CUDA device at runtime. Tests are skipped if no GPU.
# - Performance: CUDA tests with performance regression labels.

# sgemm_add_cpu_test(
#   NAME <test_name>
#   SOURCES <source1> [<source2> ...]
#   [LIBRARIES <lib1> <lib2> ...]
# )
#
# Creates a CPU-only test target that does not require CUDA runtime.
# These tests:
# - Can run on any system with CUDA toolkit (headers only)
# - Are labeled with "cpu" for CTest filtering
# - Do NOT use runCudaAwareTests() - just gtest_main
# - Will NOT be skipped due to missing GPU
function(sgemm_add_cpu_test)
    set(options "")
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_NAME)
        message(FATAL_ERROR "sgemm_add_cpu_test: NAME is required")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "sgemm_add_cpu_test: SOURCES is required")
    endif()

    add_executable(${ARG_NAME} ${ARG_SOURCES})

    target_include_directories(${ARG_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # CUDA compile options for .cu sources (if any)
    target_compile_options(${ARG_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # CPU-only tests still need CUDA headers but not runtime
    # Use CUDA::cudart header-only interface if available, otherwise link minimal
    target_link_libraries(${ARG_NAME} PRIVATE
        GTest::gtest_main
        CUDA::cudart
        ${ARG_LIBRARIES}
    )

    # Set C++ standard
    target_compile_features(${ARG_NAME} PRIVATE cxx_std_17)

    # Register with CTest
    gtest_discover_tests(${ARG_NAME}
        PROPERTIES
            LABELS "cpu"
        DISCOVERY_MODE PRE_TEST
    )
endfunction()

# sgemm_add_cuda_test(
#   NAME <test_name>
#   SOURCES <source1> [<source2> ...]
#   [LIBRARIES <lib1> <lib2> ...]
#   [CUDA_LIBRARIES <cuda_lib1> <cuda_lib2> ...]
#   [REQUIRES_WMMA]
# )
#
# Creates a CUDA test target that requires a CUDA device.
# These tests:
# - Require CUDA toolkit and a CUDA-capable GPU
# - Are automatically skipped when no GPU is available
# - Are labeled with "cuda" for CTest filtering
# - Use runCudaAwareTests() for proper environment setup
#
# REQUIRES_WMMA: If set, adds SGEMM_HAS_WMMA_TARGET compile definition
function(sgemm_add_cuda_test)
    set(options "REQUIRES_WMMA")
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LIBRARIES CUDA_LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_NAME)
        message(FATAL_ERROR "sgemm_add_cuda_test: NAME is required")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "sgemm_add_cuda_test: SOURCES is required")
    endif()

    add_executable(${ARG_NAME} ${ARG_SOURCES})

    target_include_directories(${ARG_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Standard CUDA compile options
    target_compile_options(${ARG_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    # Link against CUDA runtime and libraries
    target_link_options(${ARG_NAME} PRIVATE
        -L${SGEMM_CUDA_LIBRARY_DIR}
    )

    # WMMA target definition if needed
    if(ARG_REQUIRES_WMMA)
        target_compile_definitions(${ARG_NAME} PRIVATE
            SGEMM_HAS_WMMA_TARGET=${SGEMM_HAS_WMMA_TARGET}
        )
    endif()

    # Default CUDA libraries
    set(DEFAULT_CUDA_LIBS CUDA::cudart CUDA::cublas)

    target_link_libraries(${ARG_NAME} PRIVATE
        GTest::gtest_main
        ${DEFAULT_CUDA_LIBS}
        ${ARG_CUDA_LIBRARIES}
        ${ARG_LIBRARIES}
    )

    # Register with CTest
    gtest_discover_tests(${ARG_NAME}
        PROPERTIES
            LABELS "cuda"
        DISCOVERY_MODE PRE_TEST
    )
endfunction()

# sgemm_add_cuda_perf_test(
#   NAME <test_name>
#   SOURCES <source1> [<source2> ...]
#   [LIBRARIES <lib1> <lib2> ...]
# )
#
# Creates a CUDA performance test target.
# Same as sgemm_add_cuda_test but with additional labels for performance testing.
function(sgemm_add_cuda_perf_test)
    set(options "REQUIRES_WMMA")
    set(oneValueArgs NAME)
    set(multiValueArgs SOURCES LIBRARIES CUDA_LIBRARIES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_NAME)
        message(FATAL_ERROR "sgemm_add_cuda_perf_test: NAME is required")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "sgemm_add_cuda_perf_test: SOURCES is required")
    endif()

    add_executable(${ARG_NAME} ${ARG_SOURCES})

    target_include_directories(${ARG_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    target_compile_options(${ARG_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )

    target_link_options(${ARG_NAME} PRIVATE
        -L${SGEMM_CUDA_LIBRARY_DIR}
    )

    if(ARG_REQUIRES_WMMA)
        target_compile_definitions(${ARG_NAME} PRIVATE
            SGEMM_HAS_WMMA_TARGET=${SGEMM_HAS_WMMA_TARGET}
        )
    endif()

    set(DEFAULT_CUDA_LIBS CUDA::cudart CUDA::cublas CUDA::curand)

    target_link_libraries(${ARG_NAME} PRIVATE
        GTest::gtest_main
        ${DEFAULT_CUDA_LIBS}
        ${ARG_CUDA_LIBRARIES}
        ${ARG_LIBRARIES}
    )

    # Register with CTest with both cuda and performance labels
    gtest_discover_tests(${ARG_NAME}
        PROPERTIES
            LABELS "cuda;performance"
        DISCOVERY_MODE PRE_TEST
    )
endfunction()
