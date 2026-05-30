/*
 * Copyright (c) 2024-2026 TileXR Project
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "tilexr_collectives.h"
#include "../common/int32_pattern.h"

namespace {

enum class CollectiveOp {
    ALLGATHER,
    ALLTOALL,
    BOTH,
};

struct Options {
    int rankSize = 2;
    int rank = 0;
    int64_t count = 16;
    int firstNpu = 0;
    CollectiveOp op = CollectiveOp::BOTH;
};

using TileXRCollectivesTest::CanUseCollisionFreeInt32Pattern;
using TileXRCollectivesTest::ExpectedAllGatherValue;
using TileXRCollectivesTest::ExpectedAllToAllValue;

int GetEnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    return (value == nullptr || value[0] == '\0') ? fallback : std::atoi(value);
}

int64_t GetEnvInt64(const char *name, int64_t fallback)
{
    const char *value = std::getenv(name);
    return (value == nullptr || value[0] == '\0') ? fallback : std::atoll(value);
}

void PrintUsage(const char *program)
{
    std::cerr << "Usage: " << program
              << " --rank-size N --rank R --count C --first-npu D [--op allgather|alltoall|both]\n"
              << "Environment fallbacks: TILEXR_RANK_SIZE, TILEXR_RANK, TILEXR_COUNT, TILEXR_FIRST_NPU, TILEXR_OP"
              << std::endl;
}

bool ParseOp(const std::string &value, CollectiveOp &op)
{
    if (value == "allgather") {
        op = CollectiveOp::ALLGATHER;
        return true;
    }
    if (value == "alltoall") {
        op = CollectiveOp::ALLTOALL;
        return true;
    }
    if (value == "both") {
        op = CollectiveOp::BOTH;
        return true;
    }
    return false;
}

bool ParseOptions(int argc, char **argv, Options &options)
{
    options.rankSize = GetEnvInt("TILEXR_RANK_SIZE", options.rankSize);
    options.rank = GetEnvInt("TILEXR_RANK", options.rank);
    options.count = GetEnvInt64("TILEXR_COUNT", options.count);
    options.firstNpu = GetEnvInt("TILEXR_FIRST_NPU", options.firstNpu);
    const char *envOp = std::getenv("TILEXR_OP");
    if (envOp != nullptr && envOp[0] != '\0' && !ParseOp(envOp, options.op)) {
        std::cerr << "ERROR: invalid TILEXR_OP=" << envOp << std::endl;
        return false;
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const std::string &name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "ERROR: missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--rank-size") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            options.rankSize = std::atoi(value);
        } else if (arg == "--rank") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            options.rank = std::atoi(value);
        } else if (arg == "--count") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            options.count = std::atoll(value);
        } else if (arg == "--first-npu") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            options.firstNpu = std::atoi(value);
        } else if (arg == "--op") {
            const char *value = requireValue(arg);
            if (value == nullptr || !ParseOp(value, options.op)) {
                std::cerr << "ERROR: --op must be allgather, alltoall, or both" << std::endl;
                return false;
            }
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "ERROR: unknown argument " << arg << std::endl;
            return false;
        }
    }

    if (options.rankSize <= 0 || options.rank < 0 || options.rank >= options.rankSize ||
        options.count <= 0 || options.firstNpu < 0) {
        std::cerr << "ERROR: invalid rank-size/rank/count/first-npu" << std::endl;
        return false;
    }
    if (options.count > std::numeric_limits<int64_t>::max() / std::max(1, options.rankSize)) {
        std::cerr << "ERROR: count is too large" << std::endl;
        return false;
    }
    if (!CanUseCollisionFreeInt32Pattern(options.rankSize, options.count)) {
        std::cerr << "ERROR: count is too large for collision-free INT32 validation" << std::endl;
        return false;
    }
    return true;
}

bool CheckAcl(int rank, const std::string &step, aclError ret)
{
    if (ret == ACL_SUCCESS) {
        return true;
    }
    std::cerr << "[rank " << rank << "] ERROR: " << step << " failed, ret=" << ret << std::endl;
    return false;
}

bool CheckTileXR(int rank, const std::string &step, int ret)
{
    if (ret == TileXR::TILEXR_SUCCESS) {
        return true;
    }
    std::cerr << "[rank " << rank << "] ERROR: " << step << " failed, ret=" << ret << std::endl;
    return false;
}

bool AllocDeviceInt32(int rank, const std::string &name, int64_t count, int32_t **ptr)
{
    if (count <= 0 || count > static_cast<int64_t>(std::numeric_limits<size_t>::max() / sizeof(int32_t))) {
        std::cerr << "[rank " << rank << "] ERROR: invalid allocation count for " << name << std::endl;
        return false;
    }
    return CheckAcl(rank, "aclrtMalloc " + name,
        aclrtMalloc(reinterpret_cast<void **>(ptr), static_cast<size_t>(count) * sizeof(int32_t),
            ACL_MEM_MALLOC_HUGE_FIRST));
}

void FreeDevice(void *ptr)
{
    if (ptr != nullptr) {
        aclrtFree(ptr);
    }
}

bool CopyHostToDevice(int rank, void *dst, size_t dstSize, const void *src, size_t srcSize, const std::string &name)
{
    return CheckAcl(rank, "aclrtMemcpy H2D " + name,
        aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_HOST_TO_DEVICE));
}

bool CopyDeviceToHost(int rank, void *dst, size_t dstSize, const void *src, size_t srcSize, const std::string &name)
{
    return CheckAcl(rank, "aclrtMemcpy D2H " + name,
        aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_DEVICE_TO_HOST));
}

bool RunAllGather(const Options &options, TileXRCommPtr comm, aclrtStream stream)
{
    const int64_t sendCount = options.count;
    const int64_t recvCount = sendCount * options.rankSize;
    std::vector<int32_t> hostSend(static_cast<size_t>(sendCount));
    std::vector<int32_t> hostRecv(static_cast<size_t>(recvCount), -1);
    for (int64_t i = 0; i < sendCount; ++i) {
        hostSend[static_cast<size_t>(i)] = ExpectedAllGatherValue(options.rankSize, options.rank, i);
    }

    int32_t *devSend = nullptr;
    int32_t *devRecv = nullptr;
    const size_t sendBytes = hostSend.size() * sizeof(int32_t);
    const size_t recvBytes = hostRecv.size() * sizeof(int32_t);
    bool ok = AllocDeviceInt32(options.rank, "allgather send", sendCount, &devSend) &&
        AllocDeviceInt32(options.rank, "allgather recv", recvCount, &devRecv) &&
        CopyHostToDevice(options.rank, devSend, sendBytes, hostSend.data(), sendBytes, "allgather send") &&
        CopyHostToDevice(options.rank, devRecv, recvBytes, hostRecv.data(), recvBytes, "allgather recv") &&
        CheckTileXR(options.rank, "TileXRAllGather",
            TileXRAllGather(devSend, devRecv, sendCount, TileXR::TILEXR_DATA_TYPE_INT32, comm, stream)) &&
        CheckAcl(options.rank, "aclrtSynchronizeStream allgather", aclrtSynchronizeStream(stream)) &&
        CopyDeviceToHost(options.rank, hostRecv.data(), recvBytes, devRecv, recvBytes, "allgather result");

    for (int src = 0; ok && src < options.rankSize; ++src) {
        for (int64_t i = 0; i < sendCount; ++i) {
            const int64_t index = static_cast<int64_t>(src) * sendCount + i;
            const int32_t expected = ExpectedAllGatherValue(options.rankSize, src, i);
            const int32_t actual = hostRecv[static_cast<size_t>(index)];
            if (actual != expected) {
                std::cerr << "[rank " << options.rank << "] AllGather mismatch src=" << src
                          << " i=" << i << " expected=" << expected << " actual=" << actual << std::endl;
                ok = false;
                break;
            }
        }
    }

    FreeDevice(devSend);
    FreeDevice(devRecv);
    return ok;
}

bool RunAllToAll(const Options &options, TileXRCommPtr comm, aclrtStream stream)
{
    const int64_t countPerPeer = options.count;
    const int64_t totalCount = countPerPeer * options.rankSize;
    std::vector<int32_t> hostSend(static_cast<size_t>(totalCount));
    std::vector<int32_t> hostRecv(static_cast<size_t>(totalCount), -1);
    for (int dst = 0; dst < options.rankSize; ++dst) {
        for (int64_t i = 0; i < countPerPeer; ++i) {
            const int64_t index = static_cast<int64_t>(dst) * countPerPeer + i;
            hostSend[static_cast<size_t>(index)] = ExpectedAllToAllValue(options.rankSize, options.rank, dst, i);
        }
    }

    int32_t *devSend = nullptr;
    int32_t *devRecv = nullptr;
    const size_t bytes = hostSend.size() * sizeof(int32_t);
    bool ok = AllocDeviceInt32(options.rank, "alltoall send", totalCount, &devSend) &&
        AllocDeviceInt32(options.rank, "alltoall recv", totalCount, &devRecv) &&
        CopyHostToDevice(options.rank, devSend, bytes, hostSend.data(), bytes, "alltoall send") &&
        CopyHostToDevice(options.rank, devRecv, bytes, hostRecv.data(), bytes, "alltoall recv") &&
        CheckTileXR(options.rank, "TileXRAllToAll",
            TileXRAllToAll(devSend, devRecv, countPerPeer, TileXR::TILEXR_DATA_TYPE_INT32, comm, stream)) &&
        CheckAcl(options.rank, "aclrtSynchronizeStream alltoall", aclrtSynchronizeStream(stream)) &&
        CopyDeviceToHost(options.rank, hostRecv.data(), bytes, devRecv, bytes, "alltoall result");

    for (int src = 0; ok && src < options.rankSize; ++src) {
        for (int64_t i = 0; i < countPerPeer; ++i) {
            const int64_t index = static_cast<int64_t>(src) * countPerPeer + i;
            const int32_t expected = ExpectedAllToAllValue(options.rankSize, src, options.rank, i);
            const int32_t actual = hostRecv[static_cast<size_t>(index)];
            if (actual != expected) {
                std::cerr << "[rank " << options.rank << "] AllToAll mismatch src=" << src
                          << " i=" << i << " expected=" << expected << " actual=" << actual << std::endl;
                ok = false;
                break;
            }
        }
    }

    FreeDevice(devSend);
    FreeDevice(devRecv);
    return ok;
}

void Cleanup(TileXRCommPtr comm, aclrtStream stream, int deviceId, bool deviceSet)
{
    if (comm != nullptr) {
        TileXRCommDestroy(comm);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        aclrtResetDevice(deviceId);
    }
    aclFinalize();
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    if (!ParseOptions(argc, argv, options)) {
        PrintUsage(argv[0]);
        return 2;
    }

    const int deviceId = options.firstNpu + options.rank;
    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    bool deviceSet = false;

    if (!CheckAcl(options.rank, "aclInit", aclInit(nullptr))) {
        return 1;
    }
    if (!CheckAcl(options.rank, "aclrtSetDevice", aclrtSetDevice(deviceId))) {
        Cleanup(comm, stream, deviceId, deviceSet);
        return 1;
    }
    deviceSet = true;
    if (!CheckAcl(options.rank, "aclrtCreateStream", aclrtCreateStream(&stream)) ||
        !CheckTileXR(options.rank, "TileXRCommInitRankLocal",
            TileXRCommInitRankLocal(options.rankSize, options.rank, &comm))) {
        Cleanup(comm, stream, deviceId, deviceSet);
        return 1;
    }

    bool ok = true;
    if (options.op == CollectiveOp::ALLGATHER || options.op == CollectiveOp::BOTH) {
        ok = RunAllGather(options, comm, stream) && ok;
    }
    if (options.op == CollectiveOp::ALLTOALL || options.op == CollectiveOp::BOTH) {
        ok = RunAllToAll(options, comm, stream) && ok;
    }

    Cleanup(comm, stream, deviceId, deviceSet);
    if (ok) {
        std::cout << "[rank " << options.rank << "] collectives correctness passed" << std::endl;
    }
    return ok ? 0 : 1;
}
