/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "acl/acl.h"
#include "tilexr_api.h"
#include "tilexr_types.h"

extern void launch_tilexr_udma_all_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR debug, int32_t elementsPerRank);
extern void launch_tilexr_udma_put_signal(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR signals, GM_ADDR debug,
    int32_t elementsPerRank, uint64_t signalByteOffset, uint64_t signal);

namespace {
constexpr int32_t kDefaultElementsPerRank = 16;
constexpr uint64_t kSignalValue = 1000;
constexpr int32_t kDemoMagic = 0x5444554d;
constexpr size_t kDebugQpCountWord = 5;
constexpr size_t kDebugQpStatusBaseWord = 6;
constexpr int kDefaultCommPort = 10067;
constexpr int kDemoBarrierPortOffset = 97;
constexpr size_t kDefaultRegisteredBytes = 2097152;
constexpr uint32_t kMaxExpectedQpCount = 48;
constexpr size_t kDebugWords = kDebugQpStatusBaseWord + kMaxExpectedQpCount;
constexpr int kConnectRetryCount = 500;
constexpr int kConnectRetrySleepMs = 10;

struct BarrierEndpoint {
    std::string host;
    uint16_t port;
    bool valid;
};

int GetEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    return value == nullptr ? defaultValue : std::atoi(value);
}

bool ParseUnsignedEnv(const char* name, uint64_t defaultValue, uint64_t maxValue, uint64_t& result)
{
    const char* value = std::getenv(name);
    if (value == nullptr) {
        result = defaultValue;
        return true;
    }
    if (value[0] == '\0') {
        std::cerr << "ERROR: " << name << " must be a non-empty decimal integer" << std::endl;
        return false;
    }
    for (const char* cursor = value; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            std::cerr << "ERROR: " << name << " must be a decimal integer, got '" << value << "'" << std::endl;
            return false;
        }
    }

    errno = 0;
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' || parsed > maxValue) {
        std::cerr << "ERROR: " << name << " is out of range, got '" << value << "'" << std::endl;
        return false;
    }
    result = static_cast<uint64_t>(parsed);
    return true;
}

bool CheckedMultiply(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAdd(size_t lhs, size_t rhs, size_t& result)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

int GetDeviceIdFromEnv(int rank, int npuCount, int firstNpu)
{
    const char* devices = std::getenv("TILEXR_DEMO_DEVICES");
    if (devices != nullptr && devices[0] != '\0') {
        std::string list(devices);
        size_t start = 0;
        int index = 0;
        while (start <= list.size()) {
            const size_t comma = list.find(',', start);
            const size_t end = comma == std::string::npos ? list.size() : comma;
            if (index == rank && end > start) {
                return std::atoi(list.substr(start, end - start).c_str());
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
            ++index;
        }
    }
    return rank % npuCount + firstNpu;
}

int GetRankFromEnv()
{
    const char* names[] = {"PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "RANK"};
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr) {
            return std::atoi(value);
        }
    }
    return 0;
}

int GetRankSizeFromEnv()
{
    const char* names[] = {"PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "RANK_SIZE"};
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr) {
            return std::atoi(value);
        }
    }
    return 1;
}

void PrintStatus(int rank, const std::string& message)
{
    std::cout << "[rank " << rank << "] " << message << std::endl;
}

bool CheckAcl(int rank, const std::string& step, int ret)
{
    if (ret == ACL_SUCCESS) {
        PrintStatus(rank, step + " success");
        return true;
    }
    std::cerr << "[rank " << rank << "] ERROR: " << step << " failed, ret=" << ret << std::endl;
    return false;
}

bool CheckTileXR(int rank, const std::string& step, int ret)
{
    if (ret == TileXR::TILEXR_SUCCESS) {
        PrintStatus(rank, step + " success");
        return true;
    }
    std::cerr << "[rank " << rank << "] ERROR: " << step << " failed, ret=" << ret << std::endl;
    return false;
}

void PrintCommArgs(int rank, const TileXR::CommArgs& args, GM_ADDR commArgsDev)
{
    std::cout << "[rank " << rank << "] CommArgs host fields:" << std::endl;
    std::cout << "  commArgsDev=" << static_cast<void*>(commArgsDev) << std::endl;
    std::cout << "  rank=" << args.rank << " rankSize=" << args.rankSize
              << " localRank=" << args.localRank << " localRankSize=" << args.localRankSize << std::endl;
    std::cout << "  extraFlag=0x" << std::hex << args.extraFlag << std::dec
              << " UDMA=" << (((args.extraFlag & TileXR::ExtraFlag::UDMA) != 0) ? "enabled" : "disabled")
              << std::endl;
    std::cout << "  udmaInfoPtr=" << static_cast<void*>(args.udmaInfoPtr)
              << " udmaRegistryPtr=" << static_cast<void*>(args.udmaRegistryPtr)
              << " dumpAddr=" << static_cast<void*>(args.dumpAddr) << std::endl;
    for (int i = 0; i < args.rankSize; ++i) {
        std::cout << "  peerMems[" << i << "]=" << static_cast<void*>(args.peerMems[i]) << std::endl;
    }
}

bool CopyHostToDevice(int rank, void* dst, size_t dstSize, const void* src, size_t srcSize, const std::string& name)
{
    int ret = aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_HOST_TO_DEVICE);
    return CheckAcl(rank, "aclrtMemcpy H2D " + name, ret);
}

bool CopyDeviceToHost(int rank, void* dst, size_t dstSize, const void* src, size_t srcSize, const std::string& name)
{
    int ret = aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_DEVICE_TO_HOST);
    return CheckAcl(rank, "aclrtMemcpy D2H " + name, ret);
}

BarrierEndpoint GetBarrierEndpoint()
{
    std::string host = "127.0.0.1";
    int basePort = kDefaultCommPort;
    const char* configured = std::getenv("TILEXR_DEMO_BARRIER_ADDR");
    if (configured == nullptr || configured[0] == '\0') {
        configured = std::getenv("TILEXR_COMM_ID");
    }
    if (configured != nullptr && configured[0] != '\0') {
        std::string value(configured);
        size_t colon = value.rfind(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 >= value.size()) {
            return BarrierEndpoint{"", 0, false};
        }
        host = value.substr(0, colon);
        in_addr parsedAddr {};
        if (inet_pton(AF_INET, host.c_str(), &parsedAddr) != 1) {
            return BarrierEndpoint{"", 0, false};
        }
        if (colon + 1 < value.size()) {
            basePort = std::atoi(value.c_str() + colon + 1);
        }
    }
    int barrierPort = basePort + kDemoBarrierPortOffset;
    if (barrierPort <= 0 || barrierPort > 65535) {
        barrierPort = kDefaultCommPort + kDemoBarrierPortOffset;
    }
    return BarrierEndpoint{host, static_cast<uint16_t>(barrierPort), true};
}

BarrierEndpoint GetPerfBarrierEndpoint()
{
    const char* configured = std::getenv("TILEXR_DEMO_PERF_BARRIER_ADDR");
    if (configured == nullptr || configured[0] == '\0') {
        return BarrierEndpoint{"", 0, false};
    }
    std::string value(configured);
    const size_t colon = value.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= value.size()) {
        return BarrierEndpoint{"", 0, false};
    }
    const std::string host = value.substr(0, colon);
    in_addr parsedAddr {};
    if (inet_pton(AF_INET, host.c_str(), &parsedAddr) != 1) {
        return BarrierEndpoint{"", 0, false};
    }
    errno = 0;
    char* end = nullptr;
    const long port = std::strtol(value.c_str() + colon + 1, &end, 10);
    if (errno == ERANGE || end == value.c_str() + colon + 1 || *end != '\0' || port <= 0 || port > 65535) {
        return BarrierEndpoint{"", 0, false};
    }
    return BarrierEndpoint{host, static_cast<uint16_t>(port), true};
}

bool SendAll(int fd, const void* data, size_t bytes)
{
    const auto* ptr = static_cast<const uint8_t*>(data);
    while (bytes > 0) {
        ssize_t sent = send(fd, ptr, bytes, 0);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (sent == 0) {
            return false;
        }
        ptr += sent;
        bytes -= static_cast<size_t>(sent);
    }
    return true;
}

bool RecvAll(int fd, void* data, size_t bytes)
{
    auto* ptr = static_cast<uint8_t*>(data);
    while (bytes > 0) {
        ssize_t received = recv(fd, ptr, bytes, 0);
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (received == 0) {
            return false;
        }
        ptr += received;
        bytes -= static_cast<size_t>(received);
    }
    return true;
}

int CreateBarrierServer(uint16_t port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    int reuse = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
        close(fd);
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        listen(fd, SOMAXCONN) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int ConnectBarrierServer(const std::string& host, uint16_t port)
{
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        return -1;
    }
    addr.sin_port = htons(port);

    for (int attempt = 0; attempt < kConnectRetryCount; ++attempt) {
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return -1;
        }
        if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            return fd;
        }
        close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(kConnectRetrySleepMs));
    }
    return -1;
}

bool DemoBarrierAll(
    int rank, int rankSize, const std::string& step, bool localSuccess = true, bool* allSucceeded = nullptr,
    const BarrierEndpoint* endpointOverride = nullptr, int participantRank = -1, int participantCount = -1)
{
    const int barrierRank = participantRank >= 0 ? participantRank : rank;
    const int barrierSize = participantCount >= 0 ? participantCount : rankSize;
    if (barrierRank < 0 || barrierSize <= 0 || barrierRank >= barrierSize) {
        std::cerr << "[rank " << rank << "] ERROR: invalid demo barrier participant "
                  << barrierRank << "/" << barrierSize << std::endl;
        return false;
    }
    if (barrierSize <= 1) {
        if (allSucceeded != nullptr) {
            *allSucceeded = localSuccess;
        }
        return true;
    }

    const BarrierEndpoint endpoint = endpointOverride == nullptr ? GetBarrierEndpoint() : *endpointOverride;
    if (!endpoint.valid) {
        std::cerr << "[rank " << rank << "] ERROR: invalid TILEXR_DEMO_BARRIER_ADDR" << std::endl;
        return false;
    }
    PrintStatus(rank, "demo tcp barrier begin: " + step + " endpoint=" + endpoint.host + ":" +
        std::to_string(endpoint.port));
    constexpr uint8_t kArriveFailure = 0;
    constexpr uint8_t kArriveSuccess = 1;
    constexpr uint8_t kReleaseSuccess = 2;
    constexpr uint8_t kReleaseFailure = 3;
    bool globalSuccess = localSuccess;

    if (barrierRank == 0) {
        int serverFd = CreateBarrierServer(endpoint.port);
        if (serverFd < 0) {
            std::cerr << "[rank " << rank << "] ERROR: failed to create demo barrier server on 0.0.0.0:"
                      << endpoint.port << ", errno=" << errno << std::endl;
            return false;
        }
        std::vector<int> clients;
        clients.reserve(static_cast<size_t>(barrierSize - 1));
        bool ok = true;
        for (int i = 1; i < barrierSize; ++i) {
            int clientFd = accept(serverFd, nullptr, nullptr);
            if (clientFd < 0) {
                ok = false;
                break;
            }
            uint8_t token = 0;
            if (!RecvAll(clientFd, &token, sizeof(token)) ||
                (token != kArriveSuccess && token != kArriveFailure)) {
                close(clientFd);
                ok = false;
                break;
            }
            globalSuccess = globalSuccess && token == kArriveSuccess;
            clients.push_back(clientFd);
        }
        const uint8_t release = globalSuccess ? kReleaseSuccess : kReleaseFailure;
        for (int clientFd : clients) {
            ok = SendAll(clientFd, &release, sizeof(release)) && ok;
            close(clientFd);
        }
        close(serverFd);
        if (!ok) {
            std::cerr << "[rank " << rank << "] ERROR: demo barrier failed at " << step << std::endl;
            return false;
        }
    } else {
        int fd = ConnectBarrierServer(endpoint.host, endpoint.port);
        if (fd < 0) {
            std::cerr << "[rank " << rank << "] ERROR: failed to connect demo barrier on " << endpoint.host << ":"
                      << endpoint.port << std::endl;
            return false;
        }
        uint8_t release = 0;
        const uint8_t arrive = localSuccess ? kArriveSuccess : kArriveFailure;
        bool ok = SendAll(fd, &arrive, sizeof(arrive)) && RecvAll(fd, &release, sizeof(release)) &&
            (release == kReleaseSuccess || release == kReleaseFailure);
        close(fd);
        if (!ok) {
            std::cerr << "[rank " << rank << "] ERROR: demo barrier failed at " << step << std::endl;
            return false;
        }
        globalSuccess = release == kReleaseSuccess;
    }
    if (allSucceeded != nullptr) {
        *allSucceeded = globalSuccess;
    }
    PrintStatus(rank, "demo tcp barrier end: " + step);
    return true;
}

int32_t ExpectedDataValue(uint32_t qpIdx, int srcRank)
{
    return static_cast<int32_t>(100000U * qpIdx + 1000U + static_cast<uint32_t>(srcRank));
}

bool ValidateData(
    int rank, int rankSize, uint32_t qpCount, const std::vector<int32_t>& data, int32_t elementsPerRank)
{
    bool ok = true;
    const size_t elementsPerQp = static_cast<size_t>(rankSize) * elementsPerRank;
    for (uint32_t qpIdx = 0; qpIdx < qpCount; ++qpIdx) {
        for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
            const int32_t expected = ExpectedDataValue(qpIdx, srcRank);
            for (int32_t i = 0; i < elementsPerRank; ++i) {
                const size_t offset = static_cast<size_t>(qpIdx) * elementsPerQp +
                    static_cast<size_t>(srcRank) * elementsPerRank + i;
                if (data[offset] != expected) {
                    std::cerr << "[rank " << rank << "] DATA MISMATCH at qp=" << qpIdx
                              << " segment=" << srcRank << " elem=" << i << " offset=" << offset
                              << " got=" << data[offset] << " expected=" << expected << std::endl;
                    ok = false;
                    break;
                }
            }
        }
    }

    std::cout << "[rank " << rank << "] result sample:";
    for (uint32_t qpIdx = 0; qpIdx < qpCount; ++qpIdx) {
        for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
            const size_t offset = static_cast<size_t>(qpIdx) * elementsPerQp +
                static_cast<size_t>(srcRank) * elementsPerRank;
            std::cout << " qp" << qpIdx << "/seg" << srcRank << "=" << data[offset];
        }
    }
    std::cout << std::endl;
    return ok;
}

bool ValidateKernelDebug(
    int rank, int rankSize, uint32_t qpCount, int32_t elementsPerRank, const std::vector<int32_t>& debug)
{
    if (debug.size() < kDebugQpStatusBaseWord + qpCount) {
        std::cerr << "[rank " << rank << "] ERROR: debug buffer is too small for " << qpCount << " QPs"
                  << std::endl;
        return false;
    }
    bool ok = debug[0] == kDemoMagic && debug[1] == rank && debug[2] == rankSize && debug[3] == 1 &&
        debug[4] == elementsPerRank && debug[kDebugQpCountWord] == static_cast<int32_t>(qpCount);
    std::cout << "[rank " << rank << "] per-QP completion status:";
    for (uint32_t qpIdx = 0; qpIdx < qpCount; ++qpIdx) {
        const uint32_t status = static_cast<uint32_t>(debug[kDebugQpStatusBaseWord + qpIdx]);
        std::cout << " qp" << qpIdx << "=" << status;
        ok = status == 0U && ok;
    }
    std::cout << std::endl;
    if (!ok) {
        std::cerr << "[rank " << rank << "] ERROR: invalid kernel metadata or per-QP completion status"
                  << std::endl;
    }
    return ok;
}

bool ValidateSignals(int rank, int rankSize, const std::vector<uint64_t>& signals)
{
    bool ok = true;
    std::cout << "[rank " << rank << "] signal values:";
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        std::cout << " [" << srcRank << "]=" << signals[srcRank];
        if (srcRank != rank && signals[srcRank] != kSignalValue) {
            ok = false;
        }
    }
    std::cout << std::endl;
    if (!ok) {
        std::cerr << "[rank " << rank << "] ERROR: expected non-local signals to equal "
                  << kSignalValue << std::endl;
    }
    return ok;
}

void LaunchDemoKernel(int testType, aclrtStream stream, GM_ADDR commArgsDev, int32_t* data,
    uint64_t* signals, int32_t* debug, int32_t elementsPerRank, size_t signalOffset)
{
    if (testType == 1) {
        launch_tilexr_udma_put_signal(
            1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(data), reinterpret_cast<GM_ADDR>(signals),
            reinterpret_cast<GM_ADDR>(debug), elementsPerRank, signalOffset, kSignalValue);
    } else {
        launch_tilexr_udma_all_gather(
            1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(data), reinterpret_cast<GM_ADDR>(debug),
            elementsPerRank);
    }
}

void Cleanup(
    TileXRCommPtr comm, aclrtStream stream, void* registeredMemory, int32_t* debug, int rank, int deviceId)
{
    if (comm != nullptr) {
        CheckTileXR(rank, "TileXRCommDestroy", TileXRCommDestroy(comm));
    }
    if (registeredMemory != nullptr) {
        PrintStatus(rank, "aclrtFree registered memory");
        aclrtFree(registeredMemory);
    }
    if (debug != nullptr) {
        PrintStatus(rank, "aclrtFree debug");
        aclrtFree(debug);
    }
    if (stream != nullptr) {
        CheckAcl(rank, "aclrtDestroyStream", aclrtDestroyStream(stream));
    }
    CheckAcl(rank, "aclrtResetDevice", aclrtResetDevice(deviceId));
    CheckAcl(rank, "aclFinalize", aclFinalize());
}
} // namespace

int main(int argc, char** argv)
{
    int argIndex = 1;
    int rankSize = argc > argIndex ? std::atoi(argv[argIndex++]) : GetRankSizeFromEnv();
    int rank = argc > argIndex ? std::atoi(argv[argIndex++]) : GetRankFromEnv();
    int testType = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_TEST_TYPE", 0);
    int32_t elementsPerRank = argc > argIndex ? std::atoi(argv[argIndex++]) :
        GetEnvInt("TILEXR_DEMO_ELEMENTS_PER_RANK", kDefaultElementsPerRank);
    int npuCount = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_NPUS", 8);
    int firstNpu = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_FIRST_NPU", 0);

    uint64_t expectUdmaValue = 0;
    uint64_t expectedQpCountValue = 0;
    uint64_t registeredBytesValue = 0;
    uint64_t reregisterValue = 0;
    uint64_t warmupItersValue = 0;
    uint64_t timedItersValue = 0;
    uint64_t perfBarrierRankBaseValue = 0;
    uint64_t perfBarrierSizeValue = 0;
    uint64_t sharedQpDomainValue = 0;
    if (!ParseUnsignedEnv("TILEXR_DEMO_EXPECT_UDMA", 1, 1, expectUdmaValue) ||
        !ParseUnsignedEnv("TILEXR_DEMO_EXPECT_QP_COUNT", expectUdmaValue == 0 ? 0 : 1,
            kMaxExpectedQpCount, expectedQpCountValue) ||
        !ParseUnsignedEnv("TILEXR_DEMO_REGISTERED_BYTES", kDefaultRegisteredBytes,
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()), registeredBytesValue) ||
        !ParseUnsignedEnv("TILEXR_DEMO_REREGISTER", 1, 1, reregisterValue) ||
        !ParseUnsignedEnv("TILEXR_DEMO_WARMUP_ITERS", 0, std::numeric_limits<uint32_t>::max(),
            warmupItersValue) ||
        !ParseUnsignedEnv("TILEXR_DEMO_TIMED_ITERS", 1, std::numeric_limits<uint32_t>::max(),
            timedItersValue) ||
        !ParseUnsignedEnv("TILEXR_DEMO_PERF_BARRIER_RANK_BASE", 0,
            static_cast<uint64_t>(std::numeric_limits<int>::max()), perfBarrierRankBaseValue) ||
        !ParseUnsignedEnv("TILEXR_DEMO_PERF_BARRIER_SIZE", 0,
            static_cast<uint64_t>(std::numeric_limits<int>::max()), perfBarrierSizeValue) ||
        !ParseUnsignedEnv("TILEXR_DEMO_SHARED_QP_DOMAIN", 0,
            static_cast<uint64_t>(std::numeric_limits<int>::max()), sharedQpDomainValue)) {
        return 2;
    }
    const bool expectUdma = expectUdmaValue != 0;
    const uint32_t expectedQpCount = static_cast<uint32_t>(expectedQpCountValue);
    const size_t registeredBytes = static_cast<size_t>(registeredBytesValue);
    const bool reregister = reregisterValue != 0;
    const uint32_t warmupIters = static_cast<uint32_t>(warmupItersValue);
    const uint32_t timedIters = static_cast<uint32_t>(timedItersValue);
    const int perfBarrierRankBase = static_cast<int>(perfBarrierRankBaseValue);
    const int perfBarrierSize = static_cast<int>(perfBarrierSizeValue);
    const int sharedQpDomain = static_cast<int>(sharedQpDomainValue);
    if (timedIters == 0U) {
        std::cerr << "ERROR: TILEXR_DEMO_TIMED_ITERS must be positive" << std::endl;
        return 2;
    }
    const bool perfEnabled = warmupIters > 0U || timedIters > 1U;
    if ((expectUdma && expectedQpCount == 0) || (!expectUdma && expectedQpCount != 0)) {
        std::cerr << "ERROR: TILEXR_DEMO_EXPECT_QP_COUNT must be positive when UDMA is expected and zero "
                  << "when UDMA is expected to be unavailable" << std::endl;
        return 2;
    }
    if (registeredBytes == 0) {
        std::cerr << "ERROR: TILEXR_DEMO_REGISTERED_BYTES must be positive" << std::endl;
        return 2;
    }
    if (rankSize <= 0 || rank < 0 || rank >= rankSize || elementsPerRank <= 0 || npuCount <= 0 || firstNpu < 0 ||
        testType < 0 || testType > 1) {
        std::cerr << "ERROR: invalid rank, rank size, test type, element count, or NPU count" << std::endl;
        return 2;
    }
    const char* perfBarrierAddrValue = std::getenv("TILEXR_DEMO_PERF_BARRIER_ADDR");
    const bool perfBarrierAddrSet = perfBarrierAddrValue != nullptr && perfBarrierAddrValue[0] != '\0';
    const bool perfBarrierEnabled = perfBarrierSize > 0;
    if (perfBarrierAddrSet != perfBarrierEnabled || (!perfBarrierEnabled && perfBarrierRankBase != 0) ||
        (perfBarrierEnabled && (!perfEnabled || perfBarrierRankBase > perfBarrierSize - rankSize))) {
        std::cerr << "ERROR: performance barrier requires an address, performance mode, and a participant range "
                  << "within TILEXR_DEMO_PERF_BARRIER_SIZE" << std::endl;
        return 2;
    }
    const BarrierEndpoint perfBarrierEndpoint = perfBarrierEnabled ? GetPerfBarrierEndpoint() :
        BarrierEndpoint{"", 0, false};
    if (perfBarrierEnabled && !perfBarrierEndpoint.valid) {
        std::cerr << "ERROR: invalid TILEXR_DEMO_PERF_BARRIER_ADDR" << std::endl;
        return 2;
    }
    int deviceId = GetDeviceIdFromEnv(rank, npuCount, firstNpu);
    if (deviceId < 0) {
        std::cerr << "ERROR: resolved NPU device id must be non-negative" << std::endl;
        return 2;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  TileXR UDMA Communication Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[rank " << rank << "] argv: rankSize=" << rankSize << " rank=" << rank
              << " testType=" << testType << " elementsPerRank=" << elementsPerRank
              << " npuCount=" << npuCount << " firstNpu=" << firstNpu << std::endl;
    std::cout << "[rank " << rank << "] validation: expectUDMA=" << expectUdma
              << " expectedQpCount=" << expectedQpCount << " registeredBytes=" << registeredBytes
              << " reregister=" << reregister << " sharedQpDomain=" << sharedQpDomain << std::endl;
    std::cout << "[rank " << rank << "] perf: enabled=" << perfEnabled
              << " warmupIters=" << warmupIters << " timedIters=" << timedIters
              << " barrierEnabled=" << perfBarrierEnabled;
    if (perfBarrierEnabled) {
        std::cout << " barrierParticipant=" << perfBarrierRankBase + rank << "/" << perfBarrierSize
                  << " barrierEndpoint=" << perfBarrierEndpoint.host << ":" << perfBarrierEndpoint.port;
    }
    std::cout << std::endl;
    std::cout << "[rank " << rank << "] PID=" << getpid()
              << " TILEXR_COMM_ID=" << (std::getenv("TILEXR_COMM_ID") ? std::getenv("TILEXR_COMM_ID") : "<unset>")
              << " LD_LIBRARY_PATH=" << (std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "<unset>")
              << std::endl;

    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    void* registeredMemory = nullptr;
    int32_t* debug = nullptr;
    TileXRUDMAMemHandle udmaHandle = 0;
    bool udmaRegistered = false;

    if (!CheckAcl(rank, "aclInit", aclInit(nullptr))) {
        return 1;
    }
    if (!CheckAcl(rank, "aclrtSetDevice(" + std::to_string(deviceId) + ")", aclrtSetDevice(deviceId))) {
        aclFinalize();
        return 1;
    }
    if (!CheckAcl(rank, "aclrtCreateStream", aclrtCreateStream(&stream))) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    const int commInitRet = sharedQpDomain == 0 ?
        TileXRCommInitRankLocal(rankSize, rank, &comm) :
        TileXRCommInitRankWithSharedQpDomain(sharedQpDomain, rankSize, rank, &comm);
    if (!CheckTileXR(rank, sharedQpDomain == 0 ? "TileXRCommInitRankLocal" :
            "TileXRCommInitRankWithSharedQpDomain", commInitRet)) {
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    TileXR::CommArgs* commArgsHost = nullptr;
    GM_ADDR commArgsDev = nullptr;
    if (!CheckTileXR(rank, "TileXRGetCommArgsHost", TileXRGetCommArgsHost(comm, commArgsHost)) ||
        !CheckTileXR(rank, "TileXRGetCommArgsDev", TileXRGetCommArgsDev(comm, commArgsDev)) ||
        commArgsHost == nullptr || commArgsDev == nullptr) {
        std::cerr << "[rank " << rank << "] ERROR: failed to get TileXR CommArgs" << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    PrintCommArgs(rank, *commArgsHost, commArgsDev);

    uint32_t actualQpCount = std::numeric_limits<uint32_t>::max();
    int qpCountRet = TileXRUDMAGetQpCount(comm, &actualQpCount);
    const bool udmaFlagPublished = (commArgsHost->extraFlag & TileXR::ExtraFlag::UDMA) != 0;
    const bool sharedQpFlagPublished =
        (commArgsHost->extraFlag & TileXR::ExtraFlag::UDMA_SHARED_QP) != 0;
    const bool udmaInfoPublished = commArgsHost->udmaInfoPtr != nullptr;
    std::cout << "[rank " << rank << "] TileXRUDMAGetQpCount ret=" << qpCountRet
              << " qpCount=" << actualQpCount << " flagPublished=" << udmaFlagPublished
              << " sharedQpFlagPublished=" << sharedQpFlagPublished
              << " infoPublished=" << udmaInfoPublished << std::endl;

    if (!expectUdma) {
        if (udmaFlagPublished || udmaInfoPublished || qpCountRet != TileXR::TILEXR_ERROR_NOT_SUPPORT ||
            actualQpCount != 0) {
            std::cerr << "[rank " << rank << "] ERROR: expected UDMA to be unavailable without a published "
                      << "capability, ret=" << qpCountRet << " qpCount=" << actualQpCount << std::endl;
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        std::cout << "[rank " << rank
                  << "] TileXR UDMA unavailable as expected: capability unpublished, not-supported, qpCount=0"
                  << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 0;
    }

    if (!udmaFlagPublished || !udmaInfoPublished) {
        std::cerr << "[rank " << rank << "] ERROR: TileXR UDMA is not enabled. "
                  << "Check A5/Ascend950 hardware support, CANN/driver setup, and LD_LIBRARY_PATH." << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if ((sharedQpDomain != 0) != sharedQpFlagPublished) {
        std::cerr << "[rank " << rank << "] ERROR: shared-QP capability mismatch, requested="
                  << (sharedQpDomain != 0) << " published=" << sharedQpFlagPublished << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (qpCountRet != TileXR::TILEXR_SUCCESS || actualQpCount != expectedQpCount) {
        std::cerr << "[rank " << rank << "] ERROR: unexpected UDMA QP count, ret=" << qpCountRet
                  << " got=" << actualQpCount << " expected=" << expectedQpCount << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    PrintStatus(rank, "TileXR UDMA QP count matched expected value " + std::to_string(expectedQpCount));

    size_t elementsPerQp = 0;
    size_t dataCount = 0;
    size_t segmentBytes = 0;
    size_t dataBytes = 0;
    size_t signalBytes = 0;
    size_t signalOffset = 0;
    size_t payloadBytes = 0;
    if (!CheckedMultiply(static_cast<size_t>(rankSize), static_cast<size_t>(elementsPerRank), elementsPerQp) ||
        !CheckedMultiply(static_cast<size_t>(actualQpCount), elementsPerQp, dataCount) ||
        !CheckedMultiply(static_cast<size_t>(elementsPerRank), sizeof(int32_t), segmentBytes) ||
        segmentBytes > std::numeric_limits<uint32_t>::max() ||
        !CheckedMultiply(dataCount, sizeof(int32_t), dataBytes) ||
        !CheckedMultiply(static_cast<size_t>(rankSize), sizeof(uint64_t), signalBytes) ||
        !CheckedAdd(dataBytes, (alignof(uint64_t) - dataBytes % alignof(uint64_t)) % alignof(uint64_t),
            signalOffset) ||
        !CheckedAdd(signalOffset, signalBytes, payloadBytes)) {
        std::cerr << "[rank " << rank << "] ERROR: demo payload size overflow" << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (registeredBytes < payloadBytes) {
        std::cerr << "[rank " << rank << "] ERROR: TILEXR_DEMO_REGISTERED_BYTES=" << registeredBytes
                  << " is smaller than payloadBytes=" << payloadBytes << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (!CheckAcl(rank, "aclrtMalloc debug", aclrtMalloc(reinterpret_cast<void**>(&debug),
            kDebugWords * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc registered memory", aclrtMalloc(&registeredMemory,
            registeredBytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    auto data = static_cast<int32_t*>(registeredMemory);
    auto signals = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(registeredMemory) + signalOffset);
    if (!CheckTileXR(rank, "TileXRUDMARegister",
            TileXRUDMARegister(comm, static_cast<GM_ADDR>(registeredMemory), registeredBytes, &udmaHandle))) {
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    udmaRegistered = true;
    PrintStatus(rank, "registered UDMA memory base=" + std::to_string(reinterpret_cast<uintptr_t>(registeredMemory)) +
        " bytes=" + std::to_string(registeredBytes) +
        " dataOffset=0 signalOffset=" + std::to_string(signalOffset));
    PrintCommArgs(rank, *commArgsHost, commArgsDev);

    std::vector<int32_t> hostData(dataCount, -1);
    for (uint32_t qpIdx = 0; qpIdx < actualQpCount; ++qpIdx) {
        const size_t begin = static_cast<size_t>(qpIdx) * elementsPerQp +
            static_cast<size_t>(rank) * elementsPerRank;
        std::fill(hostData.begin() + begin, hostData.begin() + begin + elementsPerRank,
            ExpectedDataValue(qpIdx, rank));
    }
    std::vector<uint64_t> hostSignals(static_cast<size_t>(rankSize), 0);
    std::vector<int32_t> hostDebug(kDebugWords, 0);

    if (!CopyHostToDevice(rank, data, dataBytes, hostData.data(), dataBytes, "data") ||
        !CopyHostToDevice(rank, signals, signalBytes, hostSignals.data(), signalBytes, "signals") ||
        !CopyHostToDevice(rank, debug, hostDebug.size() * sizeof(int32_t),
            hostDebug.data(), hostDebug.size() * sizeof(int32_t), "debug")) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (!DemoBarrierAll(rank, rankSize, "all ranks registered and initialized demo buffers")) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    bool kernelRunOk = true;
    if (!perfEnabled) {
        PrintStatus(rank, testType == 1 ? "launch put-signal kernel" : "launch all-gather kernel");
        LaunchDemoKernel(testType, stream, commArgsDev, data, signals, debug, elementsPerRank, signalOffset);
        kernelRunOk = CheckAcl(rank, "aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
    } else {
        PrintStatus(rank, "launch perf warmup kernels: iterations=" + std::to_string(warmupIters));
        for (uint32_t iter = 0; iter < warmupIters; ++iter) {
            LaunchDemoKernel(testType, stream, commArgsDev, data, signals, debug, elementsPerRank, signalOffset);
        }
        const bool warmupOk = CheckAcl(rank, "aclrtSynchronizeStream perf warmup", aclrtSynchronizeStream(stream));
        bool allRanksWarmedUp = false;
        if (!DemoBarrierAll(rank, rankSize, "all ranks completed perf warmup", warmupOk, &allRanksWarmedUp) ||
            !allRanksWarmedUp) {
            if (udmaRegistered) {
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
            }
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }

        aclrtEvent startEvent = nullptr;
        aclrtEvent stopEvent = nullptr;
        bool perfSetupOk = CheckAcl(rank, "aclrtCreateEvent perf start", aclrtCreateEvent(&startEvent));
        if (perfSetupOk) {
            perfSetupOk = CheckAcl(rank, "aclrtCreateEvent perf stop", aclrtCreateEvent(&stopEvent));
        }
        bool allRanksReadyForPerf = false;
        if (!DemoBarrierAll(rank, rankSize, "all ranks ready for timed perf", perfSetupOk,
                &allRanksReadyForPerf) || !allRanksReadyForPerf) {
            if (startEvent != nullptr) {
                (void)aclrtDestroyEvent(startEvent);
            }
            if (stopEvent != nullptr) {
                (void)aclrtDestroyEvent(stopEvent);
            }
            if (udmaRegistered) {
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
            }
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        if (perfBarrierEnabled &&
            !DemoBarrierAll(rank, rankSize, "all communicators ready for timed perf", true, nullptr,
                &perfBarrierEndpoint, perfBarrierRankBase + rank, perfBarrierSize)) {
            (void)aclrtDestroyEvent(startEvent);
            (void)aclrtDestroyEvent(stopEvent);
            if (udmaRegistered) {
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
            }
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }

        float elapsedMs = 0.0F;
        const auto wallStart = std::chrono::steady_clock::now();
        const auto wallStartNs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(wallStart.time_since_epoch()).count();
        bool perfRunOk = CheckAcl(rank, "aclrtRecordEvent perf start", aclrtRecordEvent(startEvent, stream));
        if (perfRunOk) {
            for (uint32_t iter = 0; iter < timedIters; ++iter) {
                LaunchDemoKernel(testType, stream, commArgsDev, data, signals, debug, elementsPerRank, signalOffset);
            }
            perfRunOk = CheckAcl(rank, "aclrtRecordEvent perf stop", aclrtRecordEvent(stopEvent, stream));
        }
        if (perfRunOk) {
            perfRunOk = CheckAcl(rank, "aclrtSynchronizeEvent perf stop", aclrtSynchronizeEvent(stopEvent));
        }
        const auto wallStop = std::chrono::steady_clock::now();
        const auto wallStopNs =
            std::chrono::duration_cast<std::chrono::nanoseconds>(wallStop.time_since_epoch()).count();
        const double wallElapsedMs =
            std::chrono::duration<double, std::milli>(wallStop - wallStart).count();
        if (!perfRunOk) {
            (void)CheckAcl(rank, "aclrtSynchronizeStream perf fallback", aclrtSynchronizeStream(stream));
        }
        if (perfRunOk) {
            perfRunOk = CheckAcl(rank, "aclrtEventElapsedTime perf",
                aclrtEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        }
        if (perfRunOk && elapsedMs <= 0.0F) {
            std::cerr << "[rank " << rank << "] ERROR: perf elapsed time must be positive, got "
                      << elapsedMs << " ms" << std::endl;
            perfRunOk = false;
        }
        if (perfRunOk && wallElapsedMs <= 0.0) {
            std::cerr << "[rank " << rank << "] ERROR: perf wall time must be positive, got "
                      << wallElapsedMs << " ms" << std::endl;
            perfRunOk = false;
        }

        bool eventCleanupOk = true;
        if (startEvent != nullptr) {
            eventCleanupOk = CheckAcl(rank, "aclrtDestroyEvent perf start", aclrtDestroyEvent(startEvent)) &&
                eventCleanupOk;
        }
        if (stopEvent != nullptr) {
            eventCleanupOk = CheckAcl(rank, "aclrtDestroyEvent perf stop", aclrtDestroyEvent(stopEvent)) &&
                eventCleanupOk;
        }

        size_t bytesPerPeer = 0;
        size_t bytesPerIter = 0;
        const size_t transfersPerPeer = static_cast<size_t>(actualQpCount) + (testType == 1 ? 1U : 0U);
        if (perfRunOk &&
            (!CheckedMultiply(static_cast<size_t>(rankSize - 1), segmentBytes, bytesPerPeer) ||
             !CheckedMultiply(bytesPerPeer, transfersPerPeer, bytesPerIter))) {
            std::cerr << "[rank " << rank << "] ERROR: perf byte count overflow" << std::endl;
            perfRunOk = false;
        }
        if (perfRunOk) {
            const double txGBps = static_cast<double>(bytesPerIter) * static_cast<double>(timedIters) /
                (static_cast<double>(elapsedMs) * 1.0e6);
            const double wallTxGBps = static_cast<double>(bytesPerIter) * static_cast<double>(timedIters) /
                (wallElapsedMs * 1.0e6);
            std::cout << "TILEXR_UDMA_PERF"
                      << " rank=" << rank
                      << " device=" << deviceId
                      << " qp_count=" << actualQpCount
                      << " bytes_per_iter=" << bytesPerIter
                      << " iterations=" << timedIters
                      << " elapsed_ms=" << std::fixed << std::setprecision(3) << elapsedMs
                      << " tx_GBps=" << std::setprecision(6) << txGBps
                      << " wall_start_ns=" << wallStartNs
                      << " wall_stop_ns=" << wallStopNs
                      << " wall_elapsed_ms=" << std::setprecision(3) << wallElapsedMs
                      << " wall_tx_GBps=" << std::setprecision(6) << wallTxGBps
                      << std::defaultfloat << std::endl;
        }
        kernelRunOk = perfRunOk && eventCleanupOk;
    }

    bool allRanksCompletedKernels = false;
    if (!DemoBarrierAll(rank, rankSize, "all ranks completed demo kernels", kernelRunOk,
            &allRanksCompletedKernels)) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (!allRanksCompletedKernels) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    if (!CopyDeviceToHost(rank, hostData.data(), dataBytes, data, dataBytes, "data") ||
        !CopyDeviceToHost(rank, hostSignals.data(), signalBytes, signals, signalBytes, "signals") ||
        !CopyDeviceToHost(rank, hostDebug.data(), hostDebug.size() * sizeof(int32_t),
            debug, hostDebug.size() * sizeof(int32_t), "debug")) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    std::cout << "[rank " << rank << "] debug words:";
    for (size_t i = 0; i < std::min(kDebugQpStatusBaseWord + actualQpCount, hostDebug.size()); ++i) {
        std::cout << " d" << i << "=" << hostDebug[i];
    }
    std::cout << std::endl;

    bool ok = ValidateKernelDebug(rank, rankSize, actualQpCount, elementsPerRank, hostDebug);
    ok = ValidateData(rank, rankSize, actualQpCount, hostData, elementsPerRank) && ok;
    if (testType == 1) {
        ok = ValidateSignals(rank, rankSize, hostSignals) && ok;
    }

    bool allRanksValidated = false;
    if (!DemoBarrierAll(rank, rankSize, "all ranks validated demo data", ok, &allRanksValidated)) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    ok = allRanksValidated;

    if (udmaRegistered) {
        int unregisterRet = TileXRUDMAUnregister(comm, udmaHandle);
        bool unregisterOk = CheckTileXR(rank, "TileXRUDMAUnregister", unregisterRet);
        udmaRegistered = !unregisterOk;
        ok = unregisterOk && ok;
    }
    bool allRanksReadyForReregister = false;
    if (reregister &&
        !DemoBarrierAll(rank, rankSize, "all ranks completed initial unregister", ok, &allRanksReadyForReregister)) {
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (reregister) {
        ok = allRanksReadyForReregister;
    }
    if (ok && reregister) {
        int registerRet = TileXRUDMARegister(
            comm, static_cast<GM_ADDR>(registeredMemory), registeredBytes, &udmaHandle);
        bool registerOk = CheckTileXR(rank, "TileXRUDMARegister(re-register)", registerRet);
        udmaRegistered = registerOk;
        ok = registerOk && ok;
        if (udmaRegistered) {
            int unregisterRet = TileXRUDMAUnregister(comm, udmaHandle);
            bool unregisterOk = CheckTileXR(rank, "TileXRUDMAUnregister(re-register)", unregisterRet);
            udmaRegistered = !unregisterOk;
            ok = unregisterOk && ok;
        }
        if (ok) {
            PrintStatus(rank, "TileXR UDMA re-register lifecycle success");
        }
    }
    bool allRanksPassedLifecycle = false;
    if (reregister && allRanksReadyForReregister &&
        !DemoBarrierAll(rank, rankSize, "all ranks completed re-register lifecycle", ok, &allRanksPassedLifecycle)) {
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (reregister && allRanksReadyForReregister) {
        ok = allRanksPassedLifecycle;
    }
    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
    if (!ok) {
        std::cerr << "[rank " << rank << "] TileXR UDMA demo failed" << std::endl;
        return 1;
    }
    std::cout << "[rank " << rank << "] TileXR UDMA demo success" << std::endl;
    return 0;
}
