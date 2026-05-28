/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * Unit test for shmem UDMA API
 *
 * 测试目标：验证 aclshmemx_get_udma_info() API 的基本功能
 */

#include <iostream>
#include <cstdlib>
#include <cstring>
#include "acl/acl.h"
#include "host/init/shmem_host_init.h"

using namespace std;

// 测试结果统计
struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
};

TestStats g_stats;

// 测试辅助宏
#define TEST_ASSERT(condition, msg) \
    do { \
        g_stats.total++; \
        if (condition) { \
            cout << "[PASS] " << msg << endl; \
            g_stats.passed++; \
        } else { \
            cout << "[FAIL] " << msg << endl; \
            g_stats.failed++; \
        } \
    } while(0)

#define TEST_CASE(name) \
    cout << "\n=== Test Case: " << name << " ===" << endl

// 测试 1: API 参数验证
void test_api_parameter_validation() {
    TEST_CASE("API Parameter Validation");

    void* ptr = nullptr;
    size_t size = 0;

    // 测试 NULL 指针参数
    int ret = aclshmemx_get_udma_info(nullptr, &size);
    TEST_ASSERT(ret == ACLSHMEM_INVALID_PARAM,
                "Should return INVALID_PARAM when udma_info_ptr is NULL");

    ret = aclshmemx_get_udma_info(&ptr, nullptr);
    TEST_ASSERT(ret == ACLSHMEM_INVALID_PARAM,
                "Should return INVALID_PARAM when udma_info_size is NULL");

    ret = aclshmemx_get_udma_info(nullptr, nullptr);
    TEST_ASSERT(ret == ACLSHMEM_INVALID_PARAM,
                "Should return INVALID_PARAM when both parameters are NULL");
}

// 测试 2: 未初始化状态
void test_uninitialized_state() {
    TEST_CASE("Uninitialized State");

    void* ptr = nullptr;
    size_t size = 0;

    // 在 shmem 初始化之前调用
    int ret = aclshmemx_get_udma_info(&ptr, &size);
    TEST_ASSERT(ret == ACLSHMEM_INNER_ERROR,
                "Should return INNER_ERROR when shmem not initialized");
}

// 测试 3: 完整初始化流程
void test_full_initialization() {
    TEST_CASE("Full Initialization Flow");

    int rank = 0;
    int rankSize = 1;

    // Step 1: 设置设备
    int deviceId = 0;
    int ret = aclrtSetDevice(deviceId);
    TEST_ASSERT(ret == ACL_SUCCESS, "aclrtSetDevice should succeed");

    // Step 2: 获取 unique ID
    aclshmemx_uniqueid_t shmemUid;
    ret = aclshmemx_get_uniqueid(&shmemUid);
    TEST_ASSERT(ret == ACLSHMEM_SUCCESS, "aclshmemx_get_uniqueid should succeed");

    // Step 3: 设置属性
    aclshmemx_init_attr_t shmemAttr = {};
    ret = aclshmemx_set_attr_uniqueid_args(
        rank, rankSize, 100 * 1024 * 1024,
        &shmemUid, &shmemAttr
    );
    TEST_ASSERT(ret == ACLSHMEM_SUCCESS, "aclshmemx_set_attr_uniqueid_args should succeed");

    // Step 4: 启用 UDMA
    shmemAttr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;

    // Step 5: 初始化 shmem
    ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &shmemAttr);
    TEST_ASSERT(ret == ACLSHMEM_SUCCESS, "aclshmemx_init_attr should succeed");

    // Step 6: 获取 UDMA 信息
    void* udmaInfoPtr = nullptr;
    size_t udmaInfoSize = 0;
    ret = aclshmemx_get_udma_info(&udmaInfoPtr, &udmaInfoSize);
    if (rankSize == 1) {
        TEST_ASSERT(ret == ACLSHMEM_INNER_ERROR,
                    "aclshmemx_get_udma_info should report unavailable for single rank");
        TEST_ASSERT(udmaInfoPtr == nullptr, "UDMA info pointer should be NULL for single rank");
        TEST_ASSERT(udmaInfoSize == 0, "UDMA info size should be 0 for single rank");
    } else {
        TEST_ASSERT(ret == ACLSHMEM_SUCCESS, "aclshmemx_get_udma_info should succeed");
        TEST_ASSERT(udmaInfoPtr != nullptr, "UDMA info pointer should not be NULL");
        TEST_ASSERT(udmaInfoSize > 0, "UDMA info size should be greater than 0");
    }

    cout << "UDMA Info Pointer: " << udmaInfoPtr << endl;
    cout << "UDMA Info Size: " << udmaInfoSize << " bytes" << endl;

    // Step 7: 验证指针是设备内存
    aclrtPtrAttributes attr = {};
    if (udmaInfoPtr != nullptr) {
        ret = aclrtPointerGetAttributes(udmaInfoPtr, &attr);
        if (ret == ACL_SUCCESS) {
            TEST_ASSERT(attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE,
                        "UDMA info should be in device memory");
            cout << "Memory Location Type: " << attr.location.type << " (1=DEVICE)" << endl;
        } else {
            cout << "[WARN] aclrtPointerGetAttributes failed: " << ret << endl;
        }
    }

    // Step 8: 清理
    ret = aclshmem_finalize();
    TEST_ASSERT(ret == ACLSHMEM_SUCCESS, "aclshmem_finalize should succeed");

    aclrtResetDevice(deviceId);
}

// 测试 4: 多次调用一致性
void test_multiple_calls_consistency() {
    TEST_CASE("Multiple Calls Consistency");

    int rank = 0;
    int rankSize = 1;
    int deviceId = 0;

    aclrtSetDevice(deviceId);

    // 初始化 shmem
    aclshmemx_uniqueid_t shmemUid;
    aclshmemx_get_uniqueid(&shmemUid);

    aclshmemx_init_attr_t shmemAttr = {};
    aclshmemx_set_attr_uniqueid_args(rank, rankSize, 100 * 1024 * 1024, &shmemUid, &shmemAttr);
    shmemAttr.option_attr.data_op_engine_type = ACLSHMEM_DATA_OP_UDMA;
    aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_UNIQUEID, &shmemAttr);

    // 多次调用 API
    void* ptr1 = nullptr;
    size_t size1 = 0;
    int ret1 = aclshmemx_get_udma_info(&ptr1, &size1);

    void* ptr2 = nullptr;
    size_t size2 = 0;
    int ret2 = aclshmemx_get_udma_info(&ptr2, &size2);

    void* ptr3 = nullptr;
    size_t size3 = 0;
    int ret3 = aclshmemx_get_udma_info(&ptr3, &size3);

    if (rankSize == 1) {
        TEST_ASSERT(ret1 == ACLSHMEM_INNER_ERROR && ret2 == ACLSHMEM_INNER_ERROR && ret3 == ACLSHMEM_INNER_ERROR,
                    "All calls should report unavailable for single rank");
    } else {
        TEST_ASSERT(ret1 == ACLSHMEM_SUCCESS && ret2 == ACLSHMEM_SUCCESS && ret3 == ACLSHMEM_SUCCESS,
                    "All calls should succeed");
    }
    TEST_ASSERT(ptr1 == ptr2 && ptr2 == ptr3,
                "All calls should return the same pointer");
    TEST_ASSERT(size1 == size2 && size2 == size3,
                "All calls should return the same size");

    cout << "Pointer consistency: " << ptr1 << " == " << ptr2 << " == " << ptr3 << endl;
    cout << "Size consistency: " << size1 << " == " << size2 << " == " << size3 << endl;

    aclshmem_finalize();
    aclrtResetDevice(deviceId);
}

int main(int argc, char** argv) {
    cout << "========================================" << endl;
    cout << "  shmem UDMA API Unit Tests" << endl;
    cout << "========================================" << endl;

    // 初始化 ACL
    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        cerr << "ERROR: aclInit failed: " << ret << endl;
        return 1;
    }

    // 运行测试
    test_api_parameter_validation();
    test_uninitialized_state();
    test_full_initialization();
    test_multiple_calls_consistency();

    // 清理 ACL
    aclFinalize();

    // 输出统计
    cout << "\n========================================" << endl;
    cout << "  Test Summary" << endl;
    cout << "========================================" << endl;
    cout << "Total:  " << g_stats.total << endl;
    cout << "Passed: " << g_stats.passed << endl;
    cout << "Failed: " << g_stats.failed << endl;
    cout << "========================================" << endl;

    return (g_stats.failed == 0) ? 0 : 1;
}
