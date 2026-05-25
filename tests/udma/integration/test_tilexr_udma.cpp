/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * Integration test for TileXR UDMA integration
 *
 * 测试目标：验证 TileXR 正确集成 shmem UDMA 功能
 */

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "acl/acl.h"
#include "tilexr_api.h"

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

// 获取环境变量
int get_rank_from_env() {
    const char* rank_str = getenv("RANK");
    return rank_str ? atoi(rank_str) : 0;
}

int get_rank_size_from_env() {
    const char* size_str = getenv("RANK_SIZE");
    return size_str ? atoi(size_str) : 1;
}

// 测试 1: TileXR 基本初始化
void test_tilexr_basic_init() {
    TEST_CASE("TileXR Basic Initialization");

    int rank = get_rank_from_env();
    int rankSize = get_rank_size_from_env();

    cout << "Rank: " << rank << "/" << rankSize << endl;

    // 初始化 TileXR
    int ret = TileXRInit(rank, rankSize);
    TEST_ASSERT(ret == TILEXR_SUCCESS, "TileXRInit should succeed");

    // 同步所有 rank
    ret = TileXRSync();
    TEST_ASSERT(ret == TILEXR_SUCCESS, "TileXRSync should succeed");

    // 清理
    TileXRFinalize();
}

// 测试 2: 验证 UDMA 初始化
void test_udma_initialization() {
    TEST_CASE("UDMA Initialization");

    int rank = get_rank_from_env();
    int rankSize = get_rank_size_from_env();

    // 初始化 TileXR
    int ret = TileXRInit(rank, rankSize);
    TEST_ASSERT(ret == TILEXR_SUCCESS, "TileXRInit should succeed");

    // 获取 CommArgs 指针
    const void* commArgsPtr = TileXRGetCommArgs();
    TEST_ASSERT(commArgsPtr != nullptr, "CommArgs pointer should not be NULL");

    if (commArgsPtr != nullptr) {
        // 注意：这里需要访问 CommArgs 结构，但它在设备内存中
        // 实际测试中需要从设备拷贝到主机
        cout << "CommArgs pointer: " << commArgsPtr << endl;

        // TODO: 如果需要验证 UDMA 字段，需要：
        // 1. 定义 CommArgs 结构（或包含 comm_args.h）
        // 2. 从设备内存拷贝到主机
        // 3. 检查 udmaInfoPtr 和 extraFlag
    }

    TileXRSync();
    TileXRFinalize();
}

// 测试 3: 多 rank 初始化（需要 mpirun）
void test_multi_rank_init() {
    TEST_CASE("Multi-Rank Initialization");

    int rank = get_rank_from_env();
    int rankSize = get_rank_size_from_env();

    if (rankSize < 2) {
        cout << "[SKIP] This test requires at least 2 ranks" << endl;
        return;
    }

    cout << "Rank " << rank << "/" << rankSize << " starting..." << endl;

    // 初始化
    int ret = TileXRInit(rank, rankSize);
    TEST_ASSERT(ret == TILEXR_SUCCESS, "TileXRInit should succeed on rank " + to_string(rank));

    // 同步点 1
    ret = TileXRSync();
    TEST_ASSERT(ret == TILEXR_SUCCESS, "First sync should succeed on rank " + to_string(rank));

    cout << "Rank " << rank << " passed first sync" << endl;

    // 同步点 2
    ret = TileXRSync();
    TEST_ASSERT(ret == TILEXR_SUCCESS, "Second sync should succeed on rank " + to_string(rank));

    cout << "Rank " << rank << " passed second sync" << endl;

    // 清理
    TileXRFinalize();
    cout << "Rank " << rank << " finalized" << endl;
}

// 测试 4: 获取共享内存缓冲区
void test_shared_memory_buffers() {
    TEST_CASE("Shared Memory Buffers");

    int rank = get_rank_from_env();
    int rankSize = get_rank_size_from_env();

    int ret = TileXRInit(rank, rankSize);
    TEST_ASSERT(ret == TILEXR_SUCCESS, "TileXRInit should succeed");

    // 获取发送缓冲区
    void* sendBuf = TileXRGetSendBuf();
    TEST_ASSERT(sendBuf != nullptr, "Send buffer should not be NULL");
    cout << "Send buffer: " << sendBuf << endl;

    // 获取接收缓冲区
    void* recvBuf = TileXRGetRecvBuf();
    TEST_ASSERT(recvBuf != nullptr, "Recv buffer should not be NULL");
    cout << "Recv buffer: " << recvBuf << endl;

    // 验证缓冲区不同
    TEST_ASSERT(sendBuf != recvBuf, "Send and recv buffers should be different");

    TileXRSync();
    TileXRFinalize();
}

// 测试 5: 压力测试 - 多次初始化/清理
void test_stress_init_finalize() {
    TEST_CASE("Stress Test - Multiple Init/Finalize");

    int rank = get_rank_from_env();
    int rankSize = get_rank_size_from_env();

    const int iterations = 5;
    int successCount = 0;

    for (int i = 0; i < iterations; i++) {
        cout << "Iteration " << (i + 1) << "/" << iterations << endl;

        int ret = TileXRInit(rank, rankSize);
        if (ret == TILEXR_SUCCESS) {
            ret = TileXRSync();
            if (ret == TILEXR_SUCCESS) {
                successCount++;
            }
            TileXRFinalize();
        }

        // 短暂延迟
        usleep(100000); // 100ms
    }

    TEST_ASSERT(successCount == iterations,
                "All iterations should succeed (" + to_string(successCount) + "/" + to_string(iterations) + ")");
}

int main(int argc, char** argv) {
    cout << "========================================" << endl;
    cout << "  TileXR UDMA Integration Tests" << endl;
    cout << "========================================" << endl;

    int rank = get_rank_from_env();
    int rankSize = get_rank_size_from_env();

    cout << "Environment:" << endl;
    cout << "  RANK: " << rank << endl;
    cout << "  RANK_SIZE: " << rankSize << endl;
    cout << "  PID: " << getpid() << endl;

    // 初始化 ACL
    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        cerr << "ERROR: aclInit failed: " << ret << endl;
        return 1;
    }

    // 设置设备
    int deviceId = rank % 8; // 假设最多 8 卡
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        cerr << "ERROR: aclrtSetDevice(" << deviceId << ") failed: " << ret << endl;
        aclFinalize();
        return 1;
    }

    cout << "Using device: " << deviceId << endl;

    // 运行测试
    test_tilexr_basic_init();
    test_udma_initialization();
    test_multi_rank_init();
    test_shared_memory_buffers();
    test_stress_init_finalize();

    // 清理
    aclrtResetDevice(deviceId);
    aclFinalize();

    // 输出统计
    cout << "\n========================================" << endl;
    cout << "  Test Summary (Rank " << rank << ")" << endl;
    cout << "========================================" << endl;
    cout << "Total:  " << g_stats.total << endl;
    cout << "Passed: " << g_stats.passed << endl;
    cout << "Failed: " << g_stats.failed << endl;
    cout << "========================================" << endl;

    return (g_stats.failed == 0) ? 0 : 1;
}
