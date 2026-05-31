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
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "tilexr_collectives.h"
#include "tilexr_collectives_perf.h"
#include "../common/int32_pattern.h"

namespace {

constexpr int64_t kMaxHostBufferBytes = 1LL << 30;

enum class CollectiveOp {
    ALLGATHER,
    ALLTOALL,
};

struct DataTypeInfo {
    TileXR::TileXRDataType type = TileXR::TILEXR_DATA_TYPE_INT32;
    std::string name = "int32";
    size_t bytes = sizeof(int32_t);
};

struct Options {
    CollectiveOp op = CollectiveOp::ALLGATHER;
    int64_t minBytes = 4;
    int64_t maxBytes = 4096;
    double stepFactor = 2.0;
    int iters = 20;
    int warmupIters = 5;
    DataTypeInfo dtype;
    int rankSize = 2;
    int rank = 0;
    int firstNpu = 0;
    bool check = true;
    std::string csvPath;
    double minAlgBw = -1.0;
    double maxLatencyUs = -1.0;
    bool profile = false;
    std::string profileDir;
    bool profileAiPrompt = false;
    int profileSampleEvery = 1;
};

struct Measurement {
    double avgUs = 0.0;
    double minUs = 0.0;
    double maxUs = 0.0;
};

struct Row {
    std::string op;
    std::string dtype;
    int ranks = 0;
    int64_t bytes = 0;
    int64_t count = 0;
    int iters = 0;
    double algBw = 0.0;
    double busBw = 0.0;
    double avgUs = 0.0;
    double minUs = 0.0;
    double maxUs = 0.0;
    int errors = 0;
};

using TileXRCollectivesTest::CanUseCollisionFreeInt32Pattern;
using TileXRCollectivesTest::ExpectedAllGatherValue;
using TileXRCollectivesTest::ExpectedAllToAllValue;

uint8_t PatternByte(int rank, int peer, int64_t byteIndex)
{
    return static_cast<uint8_t>((rank * 37 + peer * 13 + byteIndex) & 0xff);
}

void StoreInt32(std::vector<uint8_t> &buffer, int64_t index, int32_t value)
{
    std::memcpy(buffer.data() + static_cast<size_t>(index) * sizeof(int32_t), &value, sizeof(value));
}

int32_t LoadInt32(const std::vector<uint8_t> &buffer, int64_t index)
{
    int32_t value = 0;
    std::memcpy(&value, buffer.data() + static_cast<size_t>(index) * sizeof(int32_t), sizeof(value));
    return value;
}

void PrintUsage(const char *program)
{
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "  --op allgather|alltoall\n"
        << "  --min-bytes N --max-bytes N --step-factor F\n"
        << "  --iters N --warmup-iters N\n"
        << "  --datatype int8|int16|int32|int64|fp16|fp32|bf16\n"
        << "  --rank-size N --rank R --first-npu D\n"
        << "  --check 0|1 [--csv path]\n"
        << "  [--min-algbw GB/s] [--max-latency-us us]\n"
        << "  [--profile 0|1] [--profile-dir path]\n"
        << "  [--profile-ai-prompt 0|1] [--profile-sample-every N]\n";
}

bool ParseBool(const std::string &value, bool &out)
{
    if (value == "1" || value == "true" || value == "yes") {
        out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "no") {
        out = false;
        return true;
    }
    return false;
}

bool ParseDataType(const std::string &value, DataTypeInfo &info)
{
    if (value == "int8") {
        info = DataTypeInfo{TileXR::TILEXR_DATA_TYPE_INT8, "int8", 1};
    } else if (value == "int16") {
        info = DataTypeInfo{TileXR::TILEXR_DATA_TYPE_INT16, "int16", 2};
    } else if (value == "int32") {
        info = DataTypeInfo{TileXR::TILEXR_DATA_TYPE_INT32, "int32", 4};
    } else if (value == "int64") {
        info = DataTypeInfo{TileXR::TILEXR_DATA_TYPE_INT64, "int64", 8};
    } else if (value == "fp16") {
        info = DataTypeInfo{TileXR::TILEXR_DATA_TYPE_FP16, "fp16", 2};
    } else if (value == "fp32") {
        info = DataTypeInfo{TileXR::TILEXR_DATA_TYPE_FP32, "fp32", 4};
    } else if (value == "bf16" || value == "bfp16") {
        info = DataTypeInfo{TileXR::TILEXR_DATA_TYPE_BFP16, "bf16", 2};
    } else {
        return false;
    }
    return true;
}

bool ParseInt64(const std::string &text, int64_t &out)
{
    char *end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0') {
        return false;
    }
    out = static_cast<int64_t>(value);
    return true;
}

bool ParseInt(const std::string &text, int &out)
{
    int64_t value = 0;
    if (!ParseInt64(text, value) || value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool ParseDouble(const std::string &text, double &out)
{
    char *end = nullptr;
    errno = 0;
    const double value = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

bool CheckedMulInt64(int64_t lhs, int64_t rhs, int64_t &out)
{
    if (lhs < 0 || rhs < 0) {
        return false;
    }
    if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool CheckedBytesForElements(int64_t elements, size_t elementBytes, int64_t &bytes)
{
    if (elementBytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    return CheckedMulInt64(elements, static_cast<int64_t>(elementBytes), bytes) &&
        static_cast<uint64_t>(bytes) <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

bool FitsHostBufferLimit(int64_t bytes)
{
    return bytes > 0 && bytes <= kMaxHostBufferBytes;
}

bool ComputeMessageSizes(const Options &options, int64_t count, int64_t &sendElements, int64_t &recvElements,
                         int64_t &sendBytes, int64_t &recvBytes)
{
    if (count <= 0) {
        return false;
    }
    const int64_t rankSize = static_cast<int64_t>(options.rankSize);
    if (!CheckedMulInt64(count, rankSize, recvElements)) {
        return false;
    }
    sendElements = count;
    if (options.op == CollectiveOp::ALLTOALL && !CheckedMulInt64(count, rankSize, sendElements)) {
        return false;
    }
    return CheckedBytesForElements(sendElements, options.dtype.bytes, sendBytes) &&
        CheckedBytesForElements(recvElements, options.dtype.bytes, recvBytes) &&
        FitsHostBufferLimit(sendBytes) && FitsHostBufferLimit(recvBytes);
}

bool ValidateMaxMessageSize(const Options &options)
{
    int64_t count = options.maxBytes / static_cast<int64_t>(options.dtype.bytes);
    if (count <= 0) {
        count = 1;
    }
    if (options.check && options.dtype.name == "int32" &&
        !CanUseCollisionFreeInt32Pattern(options.rankSize, count)) {
        std::cerr << "ERROR: message size is too large for collision-free INT32 validation" << std::endl;
        return false;
    }
    int64_t sendElements = 0;
    int64_t recvElements = 0;
    int64_t sendBytes = 0;
    int64_t recvBytes = 0;
    if (!ComputeMessageSizes(options, count, sendElements, recvElements, sendBytes, recvBytes)) {
        std::cerr << "ERROR: message size too large" << std::endl;
        return false;
    }
    (void)sendElements;
    (void)recvElements;
    (void)sendBytes;
    (void)recvBytes;
    return true;
}

bool AdvanceBytes(int64_t current, double factor, int64_t maxBytes, int64_t &next)
{
    const double scaled = static_cast<double>(current) * factor;
    if (!std::isfinite(scaled) || scaled > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    const int64_t candidate = static_cast<int64_t>(scaled);
    if (candidate <= current) {
        if (current == std::numeric_limits<int64_t>::max()) {
            return false;
        }
        next = current + 1;
    } else {
        next = candidate;
    }
    if (next < current) {
        return false;
    }
    if (next > maxBytes) {
        next = maxBytes;
    }
    return true;
}

bool ParseOptions(int argc, char **argv, Options &options)
{
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const std::string &name) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "ERROR: missing value for " << name << std::endl;
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--op") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            const std::string op(value);
            if (op == "allgather") {
                options.op = CollectiveOp::ALLGATHER;
            } else if (op == "alltoall") {
                options.op = CollectiveOp::ALLTOALL;
            } else {
                std::cerr << "ERROR: --op must be allgather or alltoall" << std::endl;
                return false;
            }
        } else if (arg == "--min-bytes") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseInt64(value, options.minBytes)) {
                std::cerr << "ERROR: invalid --min-bytes" << std::endl;
                return false;
            }
        } else if (arg == "--max-bytes") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseInt64(value, options.maxBytes)) {
                std::cerr << "ERROR: invalid --max-bytes" << std::endl;
                return false;
            }
        } else if (arg == "--step-factor") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseDouble(value, options.stepFactor)) {
                std::cerr << "ERROR: invalid --step-factor" << std::endl;
                return false;
            }
        } else if (arg == "--iters") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseInt(value, options.iters)) {
                std::cerr << "ERROR: invalid --iters" << std::endl;
                return false;
            }
        } else if (arg == "--warmup-iters") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseInt(value, options.warmupIters)) {
                std::cerr << "ERROR: invalid --warmup-iters" << std::endl;
                return false;
            }
        } else if (arg == "--datatype") {
            const char *value = requireValue(arg);
            if (value == nullptr || !ParseDataType(value, options.dtype)) {
                std::cerr << "ERROR: unsupported --datatype" << std::endl;
                return false;
            }
        } else if (arg == "--rank-size") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseInt(value, options.rankSize)) {
                std::cerr << "ERROR: invalid --rank-size" << std::endl;
                return false;
            }
        } else if (arg == "--rank") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseInt(value, options.rank)) {
                std::cerr << "ERROR: invalid --rank" << std::endl;
                return false;
            }
        } else if (arg == "--first-npu") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseInt(value, options.firstNpu)) {
                std::cerr << "ERROR: invalid --first-npu" << std::endl;
                return false;
            }
        } else if (arg == "--check") {
            const char *value = requireValue(arg);
            if (value == nullptr || !ParseBool(value, options.check)) {
                std::cerr << "ERROR: --check must be 0 or 1" << std::endl;
                return false;
            }
        } else if (arg == "--csv") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            options.csvPath = value;
        } else if (arg == "--min-algbw") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseDouble(value, options.minAlgBw)) {
                std::cerr << "ERROR: invalid --min-algbw" << std::endl;
                return false;
            }
        } else if (arg == "--max-latency-us") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            if (!ParseDouble(value, options.maxLatencyUs)) {
                std::cerr << "ERROR: invalid --max-latency-us" << std::endl;
                return false;
            }
        } else if (arg == "--profile") {
            const char *value = requireValue(arg);
            if (value == nullptr || !ParseBool(value, options.profile)) {
                std::cerr << "ERROR: --profile must be 0 or 1" << std::endl;
                return false;
            }
        } else if (arg == "--profile-dir") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            options.profileDir = value;
        } else if (arg == "--profile-ai-prompt") {
            const char *value = requireValue(arg);
            if (value == nullptr || !ParseBool(value, options.profileAiPrompt)) {
                std::cerr << "ERROR: --profile-ai-prompt must be 0 or 1" << std::endl;
                return false;
            }
        } else if (arg == "--profile-sample-every") {
            const char *value = requireValue(arg);
            if (value == nullptr || !ParseInt(value, options.profileSampleEvery)) {
                std::cerr << "ERROR: invalid --profile-sample-every" << std::endl;
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

    if (options.minBytes <= 0 || options.maxBytes < options.minBytes || options.stepFactor <= 1.0 ||
        options.iters <= 0 || options.warmupIters < 0 || options.rankSize <= 0 ||
        options.rank < 0 || options.rank >= options.rankSize || options.firstNpu < 0) {
        std::cerr << "ERROR: invalid option value" << std::endl;
        return false;
    }
    if (options.profileSampleEvery <= 0) {
        std::cerr << "ERROR: --profile-sample-every must be positive" << std::endl;
        return false;
    }
    if (!ValidateMaxMessageSize(options)) {
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

bool CallCollective(const Options &options, void *sendBuf, void *recvBuf, int64_t count, TileXRCommPtr comm,
                    aclrtStream stream)
{
    if (options.op == CollectiveOp::ALLGATHER) {
        return CheckTileXR(options.rank, "TileXRAllGather",
            TileXRAllGather(sendBuf, recvBuf, count, options.dtype.type, comm, stream));
    }
    return CheckTileXR(options.rank, "TileXRAllToAll",
        TileXRAllToAll(sendBuf, recvBuf, count, options.dtype.type, comm, stream));
}

bool FillPattern(const Options &options, int64_t count, std::vector<uint8_t> &send, std::vector<uint8_t> &recv)
{
    if (options.dtype.name == "int32") {
        if (options.op == CollectiveOp::ALLGATHER) {
            for (int64_t i = 0; i < count; ++i) {
                StoreInt32(send, i, ExpectedAllGatherValue(options.rankSize, options.rank, i));
            }
        } else {
            for (int dst = 0; dst < options.rankSize; ++dst) {
                for (int64_t i = 0; i < count; ++i) {
                    StoreInt32(send, static_cast<int64_t>(dst) * count + i,
                        ExpectedAllToAllValue(options.rankSize, options.rank, dst, i));
                }
            }
        }
        std::fill(recv.begin(), recv.end(), 0xff);
        return true;
    }

    if (options.op == CollectiveOp::ALLGATHER) {
        for (size_t i = 0; i < send.size(); ++i) {
            send[i] = PatternByte(options.rank, 0, static_cast<int64_t>(i));
        }
    } else {
        const size_t bytesPerPeer = static_cast<size_t>(count) * options.dtype.bytes;
        for (int dst = 0; dst < options.rankSize; ++dst) {
            for (size_t i = 0; i < bytesPerPeer; ++i) {
                send[static_cast<size_t>(dst) * bytesPerPeer + i] =
                    PatternByte(options.rank, dst, static_cast<int64_t>(i));
            }
        }
    }
    std::fill(recv.begin(), recv.end(), 0xff);
    return true;
}

int ValidateInt32(const Options &options, int64_t count, const std::vector<uint8_t> &recv)
{
    int errors = 0;
    if (options.op == CollectiveOp::ALLGATHER) {
        for (int src = 0; src < options.rankSize; ++src) {
            for (int64_t i = 0; i < count; ++i) {
                const int64_t index = static_cast<int64_t>(src) * count + i;
                const int32_t expected = ExpectedAllGatherValue(options.rankSize, src, i);
                const int32_t actual = LoadInt32(recv, index);
                if (actual != expected) {
                    if (errors < 8) {
                        std::cerr << "[rank " << options.rank << "] allgather mismatch src=" << src
                                  << " i=" << i << " expected=" << expected
                                  << " actual=" << actual << std::endl;
                    }
                    ++errors;
                }
            }
        }
    } else {
        for (int src = 0; src < options.rankSize; ++src) {
            for (int64_t i = 0; i < count; ++i) {
                const int64_t index = static_cast<int64_t>(src) * count + i;
                const int32_t expected = ExpectedAllToAllValue(options.rankSize, src, options.rank, i);
                const int32_t actual = LoadInt32(recv, index);
                if (actual != expected) {
                    if (errors < 8) {
                        std::cerr << "[rank " << options.rank << "] alltoall mismatch src=" << src
                                  << " i=" << i << " expected=" << expected
                                  << " actual=" << actual << std::endl;
                    }
                    ++errors;
                }
            }
        }
    }
    return errors;
}

int ValidateBytePattern(const Options &options, int64_t count, const std::vector<uint8_t> &recv)
{
    int errors = 0;
    const size_t bytesPerPeer = static_cast<size_t>(count) * options.dtype.bytes;
    if (options.op == CollectiveOp::ALLGATHER) {
        for (int src = 0; src < options.rankSize; ++src) {
            for (size_t i = 0; i < bytesPerPeer; ++i) {
                const uint8_t expected = PatternByte(src, 0, static_cast<int64_t>(i));
                const uint8_t actual = recv[static_cast<size_t>(src) * bytesPerPeer + i];
                if (actual != expected) {
                    if (errors < 8) {
                        std::cerr << "[rank " << options.rank << "] byte allgather mismatch src=" << src
                                  << " byte=" << i << " expected=" << static_cast<int>(expected)
                                  << " actual=" << static_cast<int>(actual) << std::endl;
                    }
                    ++errors;
                }
            }
        }
    } else {
        for (int src = 0; src < options.rankSize; ++src) {
            for (size_t i = 0; i < bytesPerPeer; ++i) {
                const uint8_t expected = PatternByte(src, options.rank, static_cast<int64_t>(i));
                const uint8_t actual = recv[static_cast<size_t>(src) * bytesPerPeer + i];
                if (actual != expected) {
                    if (errors < 8) {
                        std::cerr << "[rank " << options.rank << "] byte alltoall mismatch src=" << src
                                  << " byte=" << i << " expected=" << static_cast<int>(expected)
                                  << " actual=" << static_cast<int>(actual) << std::endl;
                    }
                    ++errors;
                }
            }
        }
    }
    return errors;
}

double ComputeAlgBandwidthGbps(int64_t outputBytesPerRank, double avgUs)
{
    if (avgUs <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(outputBytesPerRank) / avgUs / 1000.0;
}

double ComputeBusBandwidthGbps(CollectiveOp op, int rankSize, double algBwGbps)
{
    (void)op;
    if (rankSize <= 0) {
        return 0.0;
    }
    return algBwGbps * static_cast<double>(rankSize - 1) / static_cast<double>(rankSize);
}

bool MeasureOnce(const Options &options, void *devSend, void *devRecv, int64_t count, TileXRCommPtr comm,
                 aclrtStream stream, double &us)
{
    aclrtEvent start = nullptr;
    aclrtEvent stop = nullptr;
    if (!CheckAcl(options.rank, "aclrtCreateEvent start", aclrtCreateEvent(&start)) ||
        !CheckAcl(options.rank, "aclrtCreateEvent stop", aclrtCreateEvent(&stop)) ||
        !CheckAcl(options.rank, "aclrtRecordEvent start", aclrtRecordEvent(start, stream))) {
        if (start != nullptr) {
            aclrtDestroyEvent(start);
        }
        if (stop != nullptr) {
            aclrtDestroyEvent(stop);
        }
        return false;
    }
    if (!CallCollective(options, devSend, devRecv, count, comm, stream) ||
        !CheckAcl(options.rank, "aclrtRecordEvent stop", aclrtRecordEvent(stop, stream)) ||
        !CheckAcl(options.rank, "aclrtSynchronizeEvent stop", aclrtSynchronizeEvent(stop))) {
        aclrtDestroyEvent(start);
        aclrtDestroyEvent(stop);
        return false;
    }

    float elapsedMs = 0.0f;
    const bool ok = CheckAcl(options.rank, "aclrtEventElapsedTime",
        aclrtEventElapsedTime(&elapsedMs, start, stop));
    aclrtDestroyEvent(start);
    aclrtDestroyEvent(stop);
    us = static_cast<double>(elapsedMs) * 1000.0;
    return ok;
}

bool Measure(const Options &options, void *devSend, void *devRecv, int64_t count, TileXRCommPtr comm,
             aclrtStream stream, Measurement &measurement)
{
    for (int i = 0; i < options.warmupIters; ++i) {
        if (!CallCollective(options, devSend, devRecv, count, comm, stream)) {
            return false;
        }
    }
    if (!CheckAcl(options.rank, "aclrtSynchronizeStream warmup", aclrtSynchronizeStream(stream))) {
        return false;
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(options.iters));
    for (int i = 0; i < options.iters; ++i) {
        double us = 0.0;
        if (!MeasureOnce(options, devSend, devRecv, count, comm, stream, us)) {
            return false;
        }
        samples.push_back(us);
    }
    const auto minmax = std::minmax_element(samples.begin(), samples.end());
    double sum = 0.0;
    for (double value : samples) {
        sum += value;
    }
    measurement.avgUs = sum / static_cast<double>(samples.size());
    measurement.minUs = *minmax.first;
    measurement.maxUs = *minmax.second;
    return true;
}

void PrintHeader()
{
    std::cout << "op dtype ranks bytes count iters algbw(GB/s) busbw(GB/s) avg(us) min(us) max(us) errors"
              << std::endl;
}

void PrintRow(const Row &row)
{
    std::cout << row.op << ' ' << row.dtype << ' ' << row.ranks << ' ' << row.bytes << ' '
              << row.count << ' ' << row.iters << ' '
              << std::fixed << std::setprecision(3)
              << row.algBw << ' ' << row.busBw << ' ' << row.avgUs << ' '
              << row.minUs << ' ' << row.maxUs << ' ' << row.errors << std::endl;
}

bool AppendCsv(const std::string &path, const Row &row)
{
    if (path.empty()) {
        return true;
    }
    std::ifstream existing(path.c_str());
    const bool needsHeader = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
    existing.close();

    std::ofstream out(path.c_str(), std::ios::app);
    if (!out.is_open()) {
        std::cerr << "ERROR: failed to open CSV " << path << std::endl;
        return false;
    }
    if (needsHeader) {
        out << "op,dtype,ranks,bytes,count,iters,algbw(GB/s),busbw(GB/s),avg(us),min(us),max(us),errors\n";
    }
    out << row.op << ',' << row.dtype << ',' << row.ranks << ',' << row.bytes << ','
        << row.count << ',' << row.iters << ',' << std::fixed << std::setprecision(6)
        << row.algBw << ',' << row.busBw << ',' << row.avgUs << ','
        << row.minUs << ',' << row.maxUs << ',' << row.errors << '\n';
    return true;
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

void FinishPerfSession(TileXRCollectivePerfSession &perfSession, const Options &options, int &totalErrors)
{
    if (perfSession == nullptr) {
        return;
    }
    TileXRCollectivePerfSetActiveSession(nullptr);
    if (TileXRCollectivePerfWriteReport(perfSession) != TileXR::TILEXR_SUCCESS) {
        std::cerr << "[rank " << options.rank << "] ERROR: TileXRCollectivePerfWriteReport failed" << std::endl;
        totalErrors += 1;
    }
    TileXRCollectivePerfSessionDestroy(perfSession);
    perfSession = nullptr;
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

    int totalErrors = 0;
    TileXRCollectivePerfSession perfSession = nullptr;
    if (options.profile) {
        const std::string defaultDir = options.profileDir.empty() ?
            ("run/prof/collectives/rank" + std::to_string(options.rank)) : options.profileDir;
        TileXRCollectivePerfConfig perfConfig {};
        perfConfig.enabled = 1;
        perfConfig.outputDir = defaultDir.c_str();
        perfConfig.emitAiPrompt = options.profileAiPrompt ? 1 : 0;
        perfConfig.sampleEveryN = static_cast<unsigned int>(options.profileSampleEvery);
        if (!CheckTileXR(options.rank, "TileXRCollectivePerfSessionCreate",
                TileXRCollectivePerfSessionCreate(&perfConfig, &perfSession)) ||
            !CheckTileXR(options.rank, "TileXRCollectivePerfSetActiveSession",
                TileXRCollectivePerfSetActiveSession(perfSession))) {
            FinishPerfSession(perfSession, options, totalErrors);
            Cleanup(comm, stream, deviceId, deviceSet);
            return 1;
        }
    }

    if (options.rank == 0) {
        PrintHeader();
    }

    for (int64_t bytes = options.minBytes;;) {
        int64_t count = bytes / static_cast<int64_t>(options.dtype.bytes);
        if (count <= 0) {
            count = 1;
        }
        int64_t sendElements = 0;
        int64_t recvElements = 0;
        int64_t sendBytes = 0;
        int64_t recvBytes = 0;
        if (!ComputeMessageSizes(options, count, sendElements, recvElements, sendBytes, recvBytes)) {
            std::cerr << "ERROR: message size too large" << std::endl;
            FinishPerfSession(perfSession, options, totalErrors);
            Cleanup(comm, stream, deviceId, deviceSet);
            return 1;
        }
        const int64_t actualSendBytesPerRank = sendBytes;

        std::vector<uint8_t> hostSend(static_cast<size_t>(sendBytes));
        std::vector<uint8_t> hostRecv(static_cast<size_t>(recvBytes));
        FillPattern(options, count, hostSend, hostRecv);

        void *devSend = nullptr;
        void *devRecv = nullptr;
        bool ok = CheckAcl(options.rank, "aclrtMalloc send",
            aclrtMalloc(&devSend, static_cast<size_t>(sendBytes), ACL_MEM_MALLOC_HUGE_FIRST)) &&
            CheckAcl(options.rank, "aclrtMalloc recv",
                aclrtMalloc(&devRecv, static_cast<size_t>(recvBytes), ACL_MEM_MALLOC_HUGE_FIRST)) &&
            CheckAcl(options.rank, "aclrtMemcpy H2D send",
                aclrtMemcpy(devSend, static_cast<size_t>(sendBytes), hostSend.data(),
                    static_cast<size_t>(sendBytes), ACL_MEMCPY_HOST_TO_DEVICE)) &&
            CheckAcl(options.rank, "aclrtMemcpy H2D recv",
                aclrtMemcpy(devRecv, static_cast<size_t>(recvBytes), hostRecv.data(),
                    static_cast<size_t>(recvBytes), ACL_MEMCPY_HOST_TO_DEVICE));

        Measurement measurement;
        if (ok) {
            ok = Measure(options, devSend, devRecv, count, comm, stream, measurement);
        }

        int errors = 0;
        if (ok && options.check) {
            std::fill(hostRecv.begin(), hostRecv.end(), 0xff);
            ok = CheckAcl(options.rank, "aclrtMemcpy H2D devRecv sentinel",
                    aclrtMemcpy(devRecv, static_cast<size_t>(recvBytes), hostRecv.data(),
                        static_cast<size_t>(recvBytes), ACL_MEMCPY_HOST_TO_DEVICE)) &&
                CallCollective(options, devSend, devRecv, count, comm, stream) &&
                CheckAcl(options.rank, "aclrtSynchronizeStream check", aclrtSynchronizeStream(stream)) &&
                CheckAcl(options.rank, "aclrtMemcpy D2H recv",
                    aclrtMemcpy(hostRecv.data(), static_cast<size_t>(recvBytes), devRecv,
                        static_cast<size_t>(recvBytes), ACL_MEMCPY_DEVICE_TO_HOST));
            if (ok) {
                errors = options.dtype.name == "int32" ?
                    ValidateInt32(options, count, hostRecv) : ValidateBytePattern(options, count, hostRecv);
            } else {
                errors = 1;
            }
        }

        if (devSend != nullptr) {
            aclrtFree(devSend);
        }
        if (devRecv != nullptr) {
            aclrtFree(devRecv);
        }
        if (!ok) {
            FinishPerfSession(perfSession, options, totalErrors);
            Cleanup(comm, stream, deviceId, deviceSet);
            return 1;
        }

        const std::string opName = options.op == CollectiveOp::ALLGATHER ? "allgather" : "alltoall";
        const int64_t outputBytesPerRank = recvBytes;
        const double algBw = ComputeAlgBandwidthGbps(outputBytesPerRank, measurement.avgUs);
        const double busBw = ComputeBusBandwidthGbps(options.op, options.rankSize, algBw);
        if (options.minAlgBw >= 0.0 && algBw < options.minAlgBw) {
            ++errors;
        }
        if (options.maxLatencyUs >= 0.0 && measurement.avgUs > options.maxLatencyUs) {
            ++errors;
        }
        totalErrors += errors;

        Row row{opName, options.dtype.name, options.rankSize, actualSendBytesPerRank, count, options.iters,
            algBw, busBw, measurement.avgUs, measurement.minUs, measurement.maxUs, errors};
        if (options.rank == 0) {
            PrintRow(row);
            if (!AppendCsv(options.csvPath, row)) {
                FinishPerfSession(perfSession, options, totalErrors);
                Cleanup(comm, stream, deviceId, deviceSet);
                return 1;
            }
        }

        int64_t nextBytes = 0;
        if (!AdvanceBytes(bytes, options.stepFactor, options.maxBytes, nextBytes)) {
            std::cerr << "ERROR: message size step overflow" << std::endl;
            FinishPerfSession(perfSession, options, totalErrors);
            Cleanup(comm, stream, deviceId, deviceSet);
            return 1;
        }
        if (nextBytes <= bytes) {
            break;
        }
        bytes = nextBytes;
    }

    FinishPerfSession(perfSession, options, totalErrors);
    Cleanup(comm, stream, deviceId, deviceSet);
    return totalErrors == 0 ? 0 : 1;
}
