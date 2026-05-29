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
#include <iostream>
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

extern void launch_tilexr_memory_all_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output, GM_ADDR debug,
    int32_t elementsPerRank);

namespace {
constexpr int32_t kDefaultElementsPerRank = 16;
constexpr size_t kDebugWords = 16;
constexpr int kDefaultCommPort = 10067;
constexpr int kDemoBarrierPortOffset = 113;
constexpr int kConnectRetryCount = 500;
constexpr int kConnectRetrySleepMs = 10;
constexpr int kDefaultBlockDim = 8;
constexpr int kDataCopyAlignmentBytes = 32;

struct BarrierEndpoint {
    uint16_t port;
};

int GetEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    return value == nullptr ? defaultValue : std::atoi(value);
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

bool CopyHostToDevice(int rank, void* dst, size_t dstSize, const void* src, size_t srcSize, const std::string& name)
{
    return CheckAcl(rank, "aclrtMemcpy H2D " + name,
        aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_HOST_TO_DEVICE));
}

bool CopyDeviceToHost(int rank, void* dst, size_t dstSize, const void* src, size_t srcSize, const std::string& name)
{
    return CheckAcl(rank, "aclrtMemcpy D2H " + name,
        aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_DEVICE_TO_HOST));
}

BarrierEndpoint GetBarrierEndpoint()
{
    int basePort = kDefaultCommPort;
    const char* commId = std::getenv("TILEXR_COMM_ID");
    if (commId != nullptr) {
        std::string value(commId);
        size_t colon = value.rfind(':');
        if (colon != std::string::npos && colon + 1 < value.size()) {
            basePort = std::atoi(value.c_str() + colon + 1);
        }
    }
    int barrierPort = basePort + kDemoBarrierPortOffset;
    if (barrierPort <= 0 || barrierPort > 65535) {
        barrierPort = kDefaultCommPort + kDemoBarrierPortOffset;
    }
    return BarrierEndpoint{static_cast<uint16_t>(barrierPort)};
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
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 ||
        listen(fd, SOMAXCONN) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int ConnectBarrierServer(uint16_t port)
{
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
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

bool DemoBarrierAll(int rank, int rankSize, const std::string& step)
{
    if (rankSize <= 1) {
        return true;
    }

    BarrierEndpoint endpoint = GetBarrierEndpoint();
    PrintStatus(rank, "demo tcp barrier begin: " + step + " port=" + std::to_string(endpoint.port));
    constexpr uint8_t kArrive = 1;
    constexpr uint8_t kRelease = 2;

    if (rank == 0) {
        int serverFd = CreateBarrierServer(endpoint.port);
        if (serverFd < 0) {
            std::cerr << "[rank " << rank << "] ERROR: failed to create demo barrier server on 127.0.0.1:"
                      << endpoint.port << ", errno=" << errno << std::endl;
            return false;
        }
        std::vector<int> clients;
        clients.reserve(static_cast<size_t>(rankSize - 1));
        bool ok = true;
        for (int i = 1; i < rankSize; ++i) {
            int clientFd = accept(serverFd, nullptr, nullptr);
            if (clientFd < 0) {
                ok = false;
                break;
            }
            uint8_t token = 0;
            if (!RecvAll(clientFd, &token, sizeof(token)) || token != kArrive) {
                close(clientFd);
                ok = false;
                break;
            }
            clients.push_back(clientFd);
        }
        for (int clientFd : clients) {
            ok = SendAll(clientFd, &kRelease, sizeof(kRelease)) && ok;
            close(clientFd);
        }
        close(serverFd);
        if (!ok) {
            std::cerr << "[rank " << rank << "] ERROR: demo barrier failed at " << step << std::endl;
            return false;
        }
    } else {
        int fd = ConnectBarrierServer(endpoint.port);
        if (fd < 0) {
            std::cerr << "[rank " << rank << "] ERROR: failed to connect demo barrier on 127.0.0.1:"
                      << endpoint.port << std::endl;
            return false;
        }
        uint8_t release = 0;
        bool ok = SendAll(fd, &kArrive, sizeof(kArrive)) &&
            RecvAll(fd, &release, sizeof(release)) && release == kRelease;
        close(fd);
        if (!ok) {
            std::cerr << "[rank " << rank << "] ERROR: demo barrier failed at " << step << std::endl;
            return false;
        }
    }
    PrintStatus(rank, "demo tcp barrier end: " + step);
    return true;
}

void PrintCommArgs(int rank, const TileXR::CommArgs& args, GM_ADDR commArgsDev)
{
    std::cout << "[rank " << rank << "] CommArgs host fields:" << std::endl;
    std::cout << "  commArgsDev=" << static_cast<void*>(commArgsDev) << std::endl;
    std::cout << "  rank=" << args.rank << " rankSize=" << args.rankSize
              << " localRank=" << args.localRank << " localRankSize=" << args.localRankSize << std::endl;
    for (int i = 0; i < args.rankSize; ++i) {
        std::cout << "  peerMems[" << i << "]=" << static_cast<void*>(args.peerMems[i]) << std::endl;
    }
}

bool ValidateData(int rank, int rankSize, const std::vector<int32_t>& output, int32_t elementsPerRank)
{
    bool ok = true;
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        int32_t expected = 2000 + srcRank;
        for (int32_t i = 0; i < elementsPerRank; ++i) {
            size_t offset = static_cast<size_t>(srcRank) * elementsPerRank + i;
            if (output[offset] != expected) {
                std::cerr << "[rank " << rank << "] DATA MISMATCH at segment=" << srcRank
                          << " elem=" << i << " got=" << output[offset]
                          << " expected=" << expected << std::endl;
                ok = false;
                break;
            }
        }
    }

    std::cout << "[rank " << rank << "] result sample:";
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        std::cout << " seg" << srcRank << "="
                  << output[static_cast<size_t>(srcRank) * elementsPerRank];
    }
    std::cout << std::endl;
    return ok;
}

void Cleanup(TileXRCommPtr comm, aclrtStream stream, int32_t* input, int32_t* output, int32_t* debug, int rank,
    int deviceId)
{
    if (input != nullptr) {
        PrintStatus(rank, "aclrtFree input");
        aclrtFree(input);
    }
    if (output != nullptr) {
        PrintStatus(rank, "aclrtFree output");
        aclrtFree(output);
    }
    if (debug != nullptr) {
        PrintStatus(rank, "aclrtFree debug");
        aclrtFree(debug);
    }
    if (comm != nullptr) {
        CheckTileXR(rank, "TileXRCommDestroy", TileXRCommDestroy(comm));
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
    int rankSize = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("RANK_SIZE", 1);
    int rank = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("RANK", 0);
    int32_t elementsPerRank = argc > argIndex ? std::atoi(argv[argIndex++]) : kDefaultElementsPerRank;
    int npuCount = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_NPUS", 8);
    int firstNpu = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_FIRST_NPU", 0);
    int deviceId = GetDeviceIdFromEnv(rank, npuCount, firstNpu);

    if (rankSize <= 0 || rank < 0 || rank >= rankSize || elementsPerRank <= 0) {
        std::cerr << "ERROR: invalid rank/rankSize/elementsPerRank" << std::endl;
        return 1;
    }
    if ((elementsPerRank * static_cast<int32_t>(sizeof(int32_t))) % kDataCopyAlignmentBytes != 0) {
        std::cerr << "ERROR: elementsPerRank must make each segment 32-byte aligned for this DataCopy example"
                  << std::endl;
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  TileXR Peer Memory DataCopy Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[rank " << rank << "] argv: rankSize=" << rankSize << " rank=" << rank
              << " elementsPerRank=" << elementsPerRank << " npuCount=" << npuCount
              << " firstNpu=" << firstNpu << " deviceId=" << deviceId << std::endl;

    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    int32_t* input = nullptr;
    int32_t* output = nullptr;
    int32_t* debug = nullptr;

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

    if (!CheckTileXR(rank, "TileXRCommInitRankLocal", TileXRCommInitRankLocal(rankSize, rank, &comm))) {
        Cleanup(comm, stream, input, output, debug, rank, deviceId);
        return 1;
    }

    TileXR::CommArgs* commArgsHost = nullptr;
    GM_ADDR commArgsDev = nullptr;
    if (!CheckTileXR(rank, "TileXRGetCommArgsHost", TileXRGetCommArgsHost(comm, commArgsHost)) ||
        !CheckTileXR(rank, "TileXRGetCommArgsDev", TileXRGetCommArgsDev(comm, commArgsDev)) ||
        commArgsHost == nullptr || commArgsDev == nullptr) {
        std::cerr << "[rank " << rank << "] ERROR: failed to get TileXR CommArgs" << std::endl;
        Cleanup(comm, stream, input, output, debug, rank, deviceId);
        return 1;
    }
    PrintCommArgs(rank, *commArgsHost, commArgsDev);

    size_t inputCount = static_cast<size_t>(elementsPerRank);
    size_t outputCount = static_cast<size_t>(rankSize) * elementsPerRank;
    if (!CheckAcl(rank, "aclrtMalloc input",
            aclrtMalloc(reinterpret_cast<void**>(&input), inputCount * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc output",
            aclrtMalloc(reinterpret_cast<void**>(&output), outputCount * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc debug",
            aclrtMalloc(reinterpret_cast<void**>(&debug), kDebugWords * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST))) {
        Cleanup(comm, stream, input, output, debug, rank, deviceId);
        return 1;
    }

    std::vector<int32_t> hostInput(inputCount, 2000 + rank);
    std::vector<int32_t> hostOutput(outputCount, -1);
    std::vector<int32_t> hostDebug(kDebugWords, 0);
    if (!CopyHostToDevice(rank, input, inputCount * sizeof(int32_t), hostInput.data(),
            hostInput.size() * sizeof(int32_t), "input") ||
        !CopyHostToDevice(rank, output, outputCount * sizeof(int32_t), hostOutput.data(),
            hostOutput.size() * sizeof(int32_t), "output") ||
        !CopyHostToDevice(rank, debug, kDebugWords * sizeof(int32_t), hostDebug.data(),
            hostDebug.size() * sizeof(int32_t), "debug")) {
        Cleanup(comm, stream, input, output, debug, rank, deviceId);
        return 1;
    }

    if (!DemoBarrierAll(rank, rankSize, "all ranks initialized input/output")) {
        Cleanup(comm, stream, input, output, debug, rank, deviceId);
        return 1;
    }

    uint32_t blockDim = static_cast<uint32_t>(std::max(kDefaultBlockDim, rankSize));
    PrintStatus(rank, "launch memory all-gather kernel blockDim=" + std::to_string(blockDim));
    launch_tilexr_memory_all_gather(blockDim, stream, commArgsDev, reinterpret_cast<GM_ADDR>(input),
        reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(debug), elementsPerRank);
    if (!CheckAcl(rank, "aclrtSynchronizeStream", aclrtSynchronizeStream(stream))) {
        Cleanup(comm, stream, input, output, debug, rank, deviceId);
        return 1;
    }

    if (!DemoBarrierAll(rank, rankSize, "all ranks completed memory kernels")) {
        Cleanup(comm, stream, input, output, debug, rank, deviceId);
        return 1;
    }

    if (!CopyDeviceToHost(rank, hostOutput.data(), hostOutput.size() * sizeof(int32_t), output,
            outputCount * sizeof(int32_t), "output") ||
        !CopyDeviceToHost(rank, hostDebug.data(), hostDebug.size() * sizeof(int32_t), debug,
            kDebugWords * sizeof(int32_t), "debug")) {
        Cleanup(comm, stream, input, output, debug, rank, deviceId);
        return 1;
    }

    std::cout << "[rank " << rank << "] debug words:";
    for (size_t i = 0; i < std::min<size_t>(5, hostDebug.size()); ++i) {
        std::cout << " d" << i << "=" << hostDebug[i];
    }
    std::cout << std::endl;

    bool ok = ValidateData(rank, rankSize, hostOutput, elementsPerRank);
    Cleanup(comm, stream, input, output, debug, rank, deviceId);
    if (!ok) {
        std::cerr << "[rank " << rank << "] TileXR peer memory DataCopy demo failed" << std::endl;
        return 1;
    }
    std::cout << "[rank " << rank << "] TileXR peer memory DataCopy demo success" << std::endl;
    return 0;
}
