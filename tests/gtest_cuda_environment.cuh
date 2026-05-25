#pragma once

#include <cstring>
#include <gtest/gtest.h>

#include "utils/cuda_utils.cuh"

inline bool isGtestListMode(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gtest_list_tests") == 0) {
            return true;
        }
    }
    return false;
}

class CudaTestEnvironment : public ::testing::Environment {
  public:
    void SetUp() override {
        if (!cudaDeviceAvailable()) {
            GTEST_SKIP() << "No CUDA-capable device is detected";
        }

        printGPUInfo();
    }
};

inline int runCudaAwareTests(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    if (!isGtestListMode(argc, argv)) {
        ::testing::AddGlobalTestEnvironment(new CudaTestEnvironment());
    }

    return RUN_ALL_TESTS();
}
