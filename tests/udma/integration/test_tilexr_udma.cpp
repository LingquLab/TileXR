/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * Integration test for TileXR UDMA integration
 *
 * 测试目标：验证 TileXR UDMA communicator metadata and registered-memory setup hooks
 */

#include <cstdlib>
#include <iostream>
#include <unistd.h>

#include "acl/acl.h"
#include "comm_args.h"
#include "tilexr_api.h"
#include "tilexr_types.h"

using namespace std;

struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
};

TestStats g_stats;

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
    } while (0)

#define TEST_CASE(name) \
    cout << "\n=== Test Case: " << name << " ===" << endl

int get_rank_from_env()
{
    const char* rankStr = getenv("PMI_RANK");
    if (rankStr == nullptr) {
        rankStr = getenv("OMPI_COMM_WORLD_RANK");
    }
    if (rankStr == nullptr) {
        rankStr = getenv("MV2_COMM_WORLD_RANK");
    }
    if (rankStr == nullptr) {
        rankStr = getenv("RANK");
    }
    return rankStr ? atoi(rankStr) : 0;
}

int get_rank_size_from_env()
{
    const char* sizeStr = getenv("PMI_SIZE");
    if (sizeStr == nullptr) {
        sizeStr = getenv("OMPI_COMM_WORLD_SIZE");
    }
    if (sizeStr == nullptr) {
        sizeStr = getenv("MV2_COMM_WORLD_SIZE");
    }
    if (sizeStr == nullptr) {
        sizeStr = getenv("RANK_SIZE");
    }
    return sizeStr ? atoi(sizeStr) : 1;
}

int get_device_id_from_env(int rank)
{
    const char* devices = getenv("TILEXR_TEST_DEVICES");
    if (devices != nullptr && devices[0] != '\0') {
        std::string list(devices);
        size_t start = 0;
        int index = 0;
        while (start <= list.size()) {
            const size_t comma = list.find(',', start);
            const size_t end = comma == std::string::npos ? list.size() : comma;
            if (index == rank && end > start) {
                return atoi(list.substr(start, end - start).c_str());
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
            ++index;
        }
    }

    const char* firstNpu = getenv("TILEXR_TEST_FIRST_NPU");
    const int firstDevice = firstNpu == nullptr ? 0 : atoi(firstNpu);
    return firstDevice + rank;
}

TileXRCommPtr init_comm(int rank, int rankSize)
{
    TileXRCommPtr comm = nullptr;
    int ret = TileXRCommInitRankLocal(rankSize, rank, &comm);
    TEST_ASSERT(ret == TileXR::TILEXR_SUCCESS, "TileXRCommInitRankLocal should succeed");
    TEST_ASSERT(comm != nullptr, "TileXRComm pointer should not be NULL");
    return comm;
}

void destroy_comm(TileXRCommPtr comm)
{
    if (comm == nullptr) {
        return;
    }
    int ret = TileXRCommDestroy(comm);
    TEST_ASSERT(ret == TileXR::TILEXR_SUCCESS, "TileXRCommDestroy should succeed");
}

TileXR::CommArgs* get_host_comm_args(TileXRCommPtr comm)
{
    TileXR::CommArgs* commArgs = nullptr;
    int ret = TileXRGetCommArgsHost(comm, commArgs);
    TEST_ASSERT(ret == TileXR::TILEXR_SUCCESS, "TileXRGetCommArgsHost should succeed");
    TEST_ASSERT(commArgs != nullptr, "Host CommArgs pointer should not be NULL");
    return commArgs;
}

void test_comm_args_initialization(TileXRCommPtr comm, int rank, int rankSize)
{
    TEST_CASE("CommArgs Initialization");

    TileXR::CommArgs* commArgs = get_host_comm_args(comm);

    if (commArgs != nullptr) {
        TEST_ASSERT(commArgs->rank == rank, "CommArgs rank should match environment rank");
        TEST_ASSERT(commArgs->rankSize == rankSize, "CommArgs rankSize should match environment rank size");
        TEST_ASSERT(commArgs->peerMems[rank] != nullptr, "Local peer memory should not be NULL");

        bool udmaEnabled = (commArgs->extraFlag & TileXR::ExtraFlag::UDMA) != 0;
        if (udmaEnabled) {
            TEST_ASSERT(commArgs->udmaInfoPtr != nullptr, "UDMA info pointer should not be NULL when UDMA is enabled");
        } else {
            cout << "[INFO] UDMA not enabled in this environment; IPC path remains available" << endl;
        }

        cout << "CommArgs host pointer: " << commArgs << endl;
        cout << "Local peer memory: " << static_cast<void*>(commArgs->peerMems[rank]) << endl;
        cout << "UDMA info pointer: " << static_cast<void*>(commArgs->udmaInfoPtr) << endl;
    }
}

void test_comm_args_device_pointer(TileXRCommPtr comm)
{
    TEST_CASE("CommArgs Device Pointer");

    GM_ADDR commArgsDev = nullptr;
    int ret = TileXRGetCommArgsDev(comm, commArgsDev);
    TEST_ASSERT(ret == TileXR::TILEXR_SUCCESS, "TileXRGetCommArgsDev should succeed");
    TEST_ASSERT(commArgsDev != nullptr, "Device CommArgs pointer should not be NULL");
    cout << "CommArgs device pointer: " << static_cast<void*>(commArgsDev) << endl;
}

void test_multi_rank_peer_visibility(TileXRCommPtr comm, int rankSize)
{
    TEST_CASE("Multi-Rank Peer Visibility");

    if (rankSize < 2) {
        cout << "[SKIP] This test requires at least 2 ranks" << endl;
        return;
    }

    TileXR::CommArgs* commArgs = get_host_comm_args(comm);

    if (commArgs != nullptr) {
        int nonNullPeerCount = 0;
        for (int i = 0; i < rankSize; ++i) {
            if (commArgs->peerMems[i] != nullptr) {
                nonNullPeerCount++;
            }
        }
        TEST_ASSERT(nonNullPeerCount == rankSize, "All rank peer memories should be visible");
    }
}

int main(int argc, char** argv)
{
    cout << "========================================" << endl;
    cout << "  TileXR UDMA Integration Tests" << endl;
    cout << "========================================" << endl;

    int rank = get_rank_from_env();
    int rankSize = get_rank_size_from_env();

    cout << "Environment:" << endl;
    cout << "  RANK: " << rank << endl;
    cout << "  RANK_SIZE: " << rankSize << endl;
    cout << "  PID: " << getpid() << endl;

    int ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        cerr << "ERROR: aclInit failed: " << ret << endl;
        return 1;
    }

    int deviceId = get_device_id_from_env(rank);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        cerr << "ERROR: aclrtSetDevice(" << deviceId << ") failed: " << ret << endl;
        aclFinalize();
        return 1;
    }

    cout << "Using device: " << deviceId << endl;

    TEST_CASE("TileXR Basic Initialization");
    cout << "Rank: " << rank << "/" << rankSize << endl;
    TileXRCommPtr comm = init_comm(rank, rankSize);
    if (comm != nullptr) {
        test_comm_args_initialization(comm, rank, rankSize);
        test_comm_args_device_pointer(comm);
        test_multi_rank_peer_visibility(comm, rankSize);
    }
    destroy_comm(comm);

    aclrtResetDevice(deviceId);
    aclFinalize();

    cout << "\n========================================" << endl;
    cout << "  Test Summary (Rank " << rank << ")" << endl;
    cout << "========================================" << endl;
    cout << "Total:  " << g_stats.total << endl;
    cout << "Passed: " << g_stats.passed << endl;
    cout << "Failed: " << g_stats.failed << endl;
    cout << "========================================" << endl;

    return (g_stats.failed == 0) ? 0 : 1;
}
