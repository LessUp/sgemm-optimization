#include "../src/kernels/kernel_catalog.cuh"
#include "../src/utils/cuda_utils.cuh"
#include "../src/utils/benchmark_settings.cuh"
#include <cassert>
#include <cstdio>

void test_catalog_not_empty() {
    const auto& catalog = getKernelCatalog();
    assert(catalog.size() > 0 && "Catalog should contain at least one kernel");
    printf("✓ Catalog contains %zu kernels\n", catalog.size());
}

void test_catalog_has_standard_kernels() {
    const auto& catalog = getKernelCatalog();
    int standard_count = 0;
    for (const auto& entry : catalog) {
        if (entry.type == KernelType::Standard) {
            standard_count++;
        }
    }
    assert(standard_count >= 4 && "Should have at least 4 standard kernels (Naive, Tiled, BankConflictFree, DoubleBuffer)");
    printf("✓ Catalog contains %d standard kernels\n", standard_count);
}

void test_catalog_has_tensor_core_kernels() {
    const auto& catalog = getKernelCatalog();
    int tc_count = 0;
    for (const auto& entry : catalog) {
        if (entry.type == KernelType::TensorCore) {
            tc_count++;
        }
    }
    assert(tc_count >= 1 && "Should have at least 1 tensor core kernel (end-to-end)");
    printf("✓ Catalog contains %d tensor core kernels\n", tc_count);
}

void test_catalog_entries_have_names() {
    const auto& catalog = getKernelCatalog();
    for (const auto& entry : catalog) {
        assert(!entry.name.empty() && "All entries should have non-empty names");
        assert(entry.launcher != nullptr && "All entries should have valid launchers");
    }
    printf("✓ All catalog entries have names and launchers\n");
}

void test_catalog_launch_callable() {
    const auto& catalog = getKernelCatalog();
    if (catalog.empty()) {
        printf("⚠ Catalog is empty, skipping launch test\n");
        return;
    }
    
    // Small test: verify we can call the launcher without crashing
    const auto& entry = catalog[0];
    const int M = 64, K = 64, N = 64;
    
    DeviceMemory<float> d_A(M * K);
    DeviceMemory<float> d_B(K * N);
    DeviceMemory<float> d_C(M * N);
    
    // Initialize with zeros
    CUDA_CHECK(cudaMemset(d_A.get(), 0, M * K * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_B.get(), 0, K * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C.get(), 0, M * N * sizeof(float)));
    
    // Launch should not crash
    entry.launcher(d_A.get(), d_B.get(), d_C.get(), M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("✓ Can launch catalog entry '%s'\n", entry.name.c_str());
}

void test_catalog_preserves_order() {
    const auto& catalog = getKernelCatalog();
    assert(catalog.size() >= 5 && "Expected at least 5 kernels");
    
    // Verify the expected order: Naive, Tiled, BankConflictFree, DoubleBuffer, TensorCore end-to-end
    assert(catalog[0].name == "Naive" && "First kernel should be Naive");
    assert(catalog[1].name == "Tiled (32x32)" && "Second kernel should be Tiled");
    assert(catalog[2].name == "Bank Conflict Free" && "Third kernel should be BankConflictFree");
    assert(catalog[3].name == "Double Buffer" && "Fourth kernel should be DoubleBuffer");
    assert(catalog[4].name == "Tensor Core (WMMA end-to-end)" && "Fifth kernel should be Tensor Core end-to-end");
    
    printf("✓ Catalog preserves expected kernel order\n");
}

int main() {
    printf("Running Kernel Catalog Tests...\n\n");
    
    try {
        test_catalog_not_empty();
        test_catalog_has_standard_kernels();
        test_catalog_has_tensor_core_kernels();
        test_catalog_entries_have_names();
        test_catalog_launch_callable();
        test_catalog_preserves_order();
        
        printf("\n✅ All kernel catalog tests passed!\n");
        return 0;
    } catch (const std::exception& e) {
        printf("\n❌ Test failed with exception: %s\n", e.what());
        return 1;
    }
}
