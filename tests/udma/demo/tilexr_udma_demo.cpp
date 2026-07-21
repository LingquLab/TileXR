/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
#include "tilexr_data_as_flag.h"
#include "tilexr_types.h"
#include "tilexr_udma_allreduce_layout.h"
#include "tilexr_udma_alltoall_group_layout.h"
#include "tilexr_udma_alltoall_group_route.h"
#include "tilexr_udma_alltoall_group_trace.h"
#include "tilexr_udma_alltoall_layout.h"
#include "tilexr_udma_fullmesh_trace.h"

extern void launch_tilexr_udma_all_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR debug, int32_t elementsPerRank);
extern void launch_tilexr_udma_put_signal(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR signals, GM_ADDR debug,
    int32_t elementsPerRank, uint64_t signal);
extern void launch_tilexr_udma_all_to_all(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset,
    int32_t chunkElements);
extern void launch_tilexr_udma_p2p_latency(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, uint64_t outputByteOffset, int32_t inputElementOffset,
    int32_t chunkElements);
extern void launch_tilexr_datacopy_latency(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, int32_t chunkElements);
extern void launch_tilexr_udma_all_to_all_fused(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR udmaMem, GM_ADDR signal, GM_ADDR debug, int32_t elementsPerPeer,
    uint64_t udmaMemByteOffset, uint64_t signalByteOffsetBase,
    int32_t chunkElements, uint32_t passCount, uint32_t loopCount);
extern void launch_tilexr_udma_all_to_all_group(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR registeredMemory, GM_ADDR debug, uint32_t invocationId,
    int32_t elementsPerPeer, int32_t chunkElements,
    uint32_t passCount, uint32_t groupCount,
    uint64_t payloadOffset0, uint64_t payloadOffset1,
    uint64_t signalOffset0, uint64_t signalOffset1,
    GM_ADDR groupTrace, uint32_t traceIteration,
    uint32_t copyoutWorkers, uint32_t routeStage, uint32_t waitPollCycles);
extern void launch_tilexr_udma_all_to_all_bigdata(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR udmaMem, GM_ADDR debug, GM_ADDR fullmeshTrace, uint32_t fullmeshTraceIteration,
    uint32_t isolatedTask,
    int32_t elementsPerPeer,
    uint64_t dataOffset, uint64_t copyDoneOffset,
    uint64_t recvCopyDoneOffset, uint64_t remoteSendDoneOffset,
    uint64_t readySignalOffset, uint64_t ackSignalOffset,
    int32_t chunkElements, uint32_t passCount, uint32_t loopCount, uint64_t kernelLoopBase,
    uint32_t profileStage, uint32_t force35Core);
extern void launch_tilexr_all_to_all_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer);
extern void launch_tilexr_all_to_all_ipc_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer);
extern void launch_tilexr_all_to_all_plain_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer);
extern void launch_tilexr_all_to_all_plain_ipc_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer);
extern void launch_tilexr_all_to_all_fused_ipc(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR output,
    GM_ADDR debug, int32_t elementsPerPeer, int32_t round);
extern void launch_tilexr_all_to_all_ipc_scatter_dma(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerPeer);
extern void launch_tilexr_all_to_all_ipc_gather_dma(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerPeer);
extern void launch_tilexr_all_reduce_ipc_scatter(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR input, GM_ADDR debug, int32_t elementsPerRank);
extern void launch_tilexr_all_reduce_ipc_sum(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR output, GM_ADDR debug, int32_t elementsPerRank);

namespace {
constexpr int32_t kDefaultElementsPerRank = 16;
constexpr uint64_t kSignalValue = 1000;
constexpr int kDebugUdmaStatusBase = 6;
constexpr int kDebugIpcScatter = kDebugUdmaStatusBase + TileXR::TILEXR_MAX_RANK_SIZE;
constexpr int kDebugIpcGather = kDebugIpcScatter + 1;
constexpr int kDebugAllReduceScatter = kDebugIpcGather + 1;
constexpr int kDebugAllReduceSum = kDebugAllReduceScatter + 1;
constexpr int kDebugRecvSlotSampleBase = kDebugUdmaStatusBase + 160;
constexpr int kDebugReadySeenBase = kDebugUdmaStatusBase + 208;
constexpr int kDebugAckSeenBase = kDebugReadySeenBase + TileXR::TILEXR_MAX_RANK_SIZE;
constexpr size_t kDebugWords = kDebugAckSeenBase + TileXR::TILEXR_MAX_RANK_SIZE;
constexpr int kDefaultCommPort = 10067;
constexpr int kDemoBarrierPortOffset = 97;
constexpr int kConnectRetryCount = 500;
constexpr int kConnectRetrySleepMs = 10;
constexpr int kBigDataProfileStageFull = 8;

struct BarrierEndpoint {
    std::string host;
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

bool WriteFullmeshTraceBinary(int rank, const std::string& directory, const std::vector<uint8_t>& data)
{
    const std::string path = directory + "/tilexr_fullmesh_trace_rank_" +
        std::to_string(rank) + ".bin";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        std::cerr << "[rank " << rank << "] ERROR: open fullmesh trace output failed path="
                  << path << std::endl;
        return false;
    }
    output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!output.good()) {
        std::cerr << "[rank " << rank << "] ERROR: write fullmesh trace output failed path="
                  << path << std::endl;
        return false;
    }
    PrintStatus(rank, "fullmesh trace output=" + path + " bytes=" + std::to_string(data.size()));
    return true;
}

bool WriteGroupTraceBinary(
    int rank, const std::string& directory, const std::string& stageName,
    const std::vector<uint8_t>& data)
{
    const std::string path = stageName.empty() ?
        directory + "/tilexr_group_trace_rank_" + std::to_string(rank) + ".bin" :
        directory + "/tilexr_group_trace_" + stageName + "_rank_" +
        std::to_string(rank) + ".bin";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        std::cerr << "[rank " << rank << "] ERROR: open grouped trace output failed path="
                  << path << std::endl;
        return false;
    }
    output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!output.good()) {
        std::cerr << "[rank " << rank << "] ERROR: write grouped trace output failed path="
                  << path << std::endl;
        return false;
    }
    PrintStatus(rank, "grouped trace output=" + path + " bytes=" + std::to_string(data.size()));
    return true;
}

BarrierEndpoint GetBarrierEndpoint()
{
    std::string host = "127.0.0.1";
    int basePort = kDefaultCommPort;
    const char* barrierHost = std::getenv("TILEXR_DEMO_BARRIER_HOST");
    if (barrierHost != nullptr && barrierHost[0] != '\0') {
        host = barrierHost;
    }
    const char* commId = std::getenv("TILEXR_COMM_ID");
    if (commId != nullptr) {
        std::string value(commId);
        size_t colon = value.rfind(':');
        if ((barrierHost == nullptr || barrierHost[0] == '\0') && colon != std::string::npos && colon > 0) {
            host = value.substr(0, colon);
        }
        if (colon != std::string::npos && colon + 1 < value.size()) {
            basePort = std::atoi(value.c_str() + colon + 1);
        }
    }
    int barrierPort = basePort + kDemoBarrierPortOffset;
    if (barrierPort <= 0 || barrierPort > 65535) {
        barrierPort = kDefaultCommPort + kDemoBarrierPortOffset;
    }
    return BarrierEndpoint{host, static_cast<uint16_t>(barrierPort)};
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

bool DemoBarrierAll(int rank, int rankSize, const std::string& step)
{
    if (rankSize <= 1) {
        return true;
    }

    BarrierEndpoint endpoint = GetBarrierEndpoint();
    PrintStatus(rank, "demo tcp barrier begin: " + step +
        " host=" + endpoint.host + " port=" + std::to_string(endpoint.port));
    constexpr uint8_t kArrive = 1;
    constexpr uint8_t kRelease = 2;

    if (rank == 0) {
        int serverFd = CreateBarrierServer(endpoint.port);
        if (serverFd < 0) {
            std::cerr << "[rank " << rank << "] ERROR: failed to create demo barrier server on 0.0.0.0:"
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
        int fd = ConnectBarrierServer(endpoint.host, endpoint.port);
        if (fd < 0) {
            std::cerr << "[rank " << rank << "] ERROR: failed to connect demo barrier on "
                      << endpoint.host << ":" << endpoint.port << std::endl;
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

bool ValidateData(int rank, int rankSize, const std::vector<int32_t>& data, int32_t elementsPerRank)
{
    bool ok = true;
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        int32_t expected = 1000 + srcRank;
        for (int32_t i = 0; i < elementsPerRank; ++i) {
            size_t offset = static_cast<size_t>(srcRank) * elementsPerRank + i;
            if (data[offset] != expected) {
                std::cerr << "[rank " << rank << "] DATA MISMATCH at segment=" << srcRank
                          << " elem=" << i << " offset=" << offset
                          << " got=" << data[offset] << " expected=" << expected << std::endl;
                ok = false;
                break;
            }
        }
    }

    std::cout << "[rank " << rank << "] result sample:";
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        size_t offset = static_cast<size_t>(srcRank) * elementsPerRank;
        std::cout << " seg" << srcRank << "=" << data[offset];
    }
    std::cout << std::endl;
    return ok;
}

bool ValidateAllToAllData(
    int rank, int rankSize, const std::vector<int32_t>& output, int32_t elementsPerPeer)
{
    bool ok = true;
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        int32_t expected = TileXR::Demo::AllToAllValue(srcRank, rank);
        for (int32_t i = 0; i < elementsPerPeer; ++i) {
            size_t offset = static_cast<size_t>(srcRank) * elementsPerPeer + i;
            if (output[offset] != expected) {
                std::cerr << "[rank " << rank << "] ALLTOALL MISMATCH at src=" << srcRank
                          << " elem=" << i << " offset=" << offset
                          << " got=" << output[offset] << " expected=" << expected << std::endl;
                ok = false;
                break;
            }
        }
    }

    std::cout << "[rank " << rank << "] alltoall output sample:";
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        size_t offset = static_cast<size_t>(srcRank) * elementsPerPeer;
        std::cout << " from" << srcRank << "=" << output[offset];
    }
    std::cout << std::endl;
    return TileXR::Demo::ValidateAllToAllOutput(output, rank, rankSize, elementsPerPeer) && ok;
}

bool ValidateAllReduceData(
    int rank, int rankSize, const std::vector<int32_t>& output, int32_t elementsPerRank)
{
    bool ok = true;
    const int32_t expected = TileXR::Demo::AllReduceExpectedSum(rankSize);
    for (int32_t i = 0; i < elementsPerRank; ++i) {
        if (output[static_cast<size_t>(i)] != expected) {
            std::cerr << "[rank " << rank << "] ALLREDUCE MISMATCH at elem=" << i
                      << " got=" << output[static_cast<size_t>(i)]
                      << " expected=" << expected << std::endl;
            ok = false;
            break;
        }
    }

    std::cout << "[rank " << rank << "] allreduce output sample:";
    for (int32_t i = 0; i < std::min<int32_t>(elementsPerRank, 8); ++i) {
        std::cout << " elem" << i << "=" << output[static_cast<size_t>(i)];
    }
    std::cout << std::endl;
    return TileXR::Demo::ValidateAllReduceOutput(output, rankSize, elementsPerRank) && ok;
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

bool AllToAllUdmaComplete(int rankSize, const std::vector<int32_t>& debug)
{
    for (int peer = 0; peer < rankSize; ++peer) {
        if (debug[kDebugUdmaStatusBase + peer] != 0) {
            return false;
        }
    }
    return true;
}

void PrintAllToAllUdmaDebug(int rank, int rankSize, const std::vector<int32_t>& debug)
{
    constexpr int rangeBase = kDebugUdmaStatusBase + 16;
    constexpr int wqeBeforeBase = kDebugUdmaStatusBase + 32;
    constexpr int wqeAfterBase = kDebugUdmaStatusBase + 48;
    constexpr int localTokenBase = kDebugUdmaStatusBase + 64;
    constexpr int remoteBaseLowBase = kDebugUdmaStatusBase + 80;
    constexpr int memAddrLowBase = kDebugUdmaStatusBase + 96;
    constexpr int tpnBase = kDebugUdmaStatusBase + 112;
    constexpr int sendSampleBase = kDebugUdmaStatusBase + 128;
    constexpr int recvSampleBase = kDebugUdmaStatusBase + 144;
    constexpr int recvSlotSampleBase = kDebugUdmaStatusBase + 160;
    constexpr int remoteDataOffsetBase = kDebugUdmaStatusBase + 176;
    constexpr int remoteReadyOffsetBase = kDebugUdmaStatusBase + 192;
    std::cout << "[rank " << rank << "] alltoall udma peer debug:";
    for (int peer = 0; peer < rankSize && peer < 16; ++peer) {
        std::cout << " peer" << peer
                  << "{status=" << debug[kDebugUdmaStatusBase + peer]
                  << ",range=" << debug[rangeBase + peer]
                  << ",wqe=" << debug[wqeBeforeBase + peer] << "->" << debug[wqeAfterBase + peer]
                  << ",token=" << debug[localTokenBase + peer]
                  << ",regLo=" << debug[remoteBaseLowBase + peer]
                  << ",memLo=" << debug[memAddrLowBase + peer]
                  << ",tpn=" << debug[tpnBase + peer]
                  << ",send0=" << debug[sendSampleBase + peer]
                  << ",recv0=" << debug[recvSampleBase + peer]
                  << ",slot0=" << debug[recvSlotSampleBase + peer]
                  << ",rDataOff=" << debug[remoteDataOffsetBase + peer]
                  << ",rReadyOff=" << debug[remoteReadyOffsetBase + peer]
                  << ",ready=" << debug[kDebugReadySeenBase + peer]
                  << ",ack=" << debug[kDebugAckSeenBase + peer]
                  << "}";
    }
    std::cout << std::endl;
}

size_t AllToAllDataAsFlagStagingBytes(int rankSize, int32_t elementsPerPeer)
{
    const uint64_t payloadBytes = static_cast<uint64_t>(elementsPerPeer) * sizeof(int32_t);
    const uint64_t blocks = TileXR::DataAsFlagBlockCountForPayloadBytes(payloadBytes);
    return static_cast<size_t>(static_cast<uint64_t>(rankSize) * blocks * TileXR::DATA_AS_FLAG_BLOCK_BYTES);
}

size_t AllToAllPlainIpcStagingBytes(int rankSize, int32_t elementsPerPeer)
{
    return static_cast<size_t>(rankSize) * static_cast<size_t>(rankSize) *
        static_cast<size_t>(elementsPerPeer) * sizeof(int32_t);
}

bool CopyChunkHostToDevice(
    int rank, int32_t* chunkInput, int32_t* chunkOutput, int rankSize, int32_t elementsPerRank,
    int32_t chunkOffset, int32_t chunkElements, const std::vector<int32_t>& hostInput,
    std::vector<int32_t>& hostOutput)
{
    std::vector<int32_t> inputChunk(static_cast<size_t>(rankSize) * chunkElements, 0);
    std::vector<int32_t> outputChunk(static_cast<size_t>(rankSize) * chunkElements, -1);
    for (int peer = 0; peer < rankSize; ++peer) {
        const size_t srcBase = static_cast<size_t>(peer) * elementsPerRank + chunkOffset;
        const size_t dstBase = static_cast<size_t>(peer) * chunkElements;
        std::copy(hostInput.begin() + srcBase,
                  hostInput.begin() + srcBase + chunkElements,
                  inputChunk.begin() + dstBase);
        std::copy(hostOutput.begin() + srcBase,
                  hostOutput.begin() + srcBase + chunkElements,
                  outputChunk.begin() + dstBase);
    }
    return CopyHostToDevice(rank, chunkInput, inputChunk.size() * sizeof(int32_t),
               inputChunk.data(), inputChunk.size() * sizeof(int32_t), "registered input chunk") &&
        CopyHostToDevice(rank, chunkOutput, outputChunk.size() * sizeof(int32_t),
            outputChunk.data(), outputChunk.size() * sizeof(int32_t), "registered output chunk");
}

bool CopyChunkDeviceToHost(
    int rank, const int32_t* chunkOutput, int rankSize, int32_t elementsPerRank,
    int32_t chunkOffset, int32_t chunkElements, std::vector<int32_t>& hostOutput)
{
    std::vector<int32_t> outputChunk(static_cast<size_t>(rankSize) * chunkElements, -1);
    if (!CopyDeviceToHost(rank, outputChunk.data(), outputChunk.size() * sizeof(int32_t),
            chunkOutput, outputChunk.size() * sizeof(int32_t), "registered output chunk")) {
        return false;
    }
    for (int peer = 0; peer < rankSize; ++peer) {
        const size_t dstBase = static_cast<size_t>(peer) * elementsPerRank + chunkOffset;
        const size_t srcBase = static_cast<size_t>(peer) * chunkElements;
        std::copy(outputChunk.begin() + srcBase,
                  outputChunk.begin() + srcBase + chunkElements,
                  hostOutput.begin() + dstBase);
    }
    return true;
}

bool RunGroupedAllToAll(
    int rank, int rankSize, int32_t elementsPerPeer,
    TileXRCommPtr comm, aclrtStream stream, GM_ADDR commArgsDev)
{
    constexpr uint32_t kErrorWordsPerCore = 12U;
    constexpr uint32_t kErrorCoreCount = TileXR::Demo::kAllToAllGroupBlockDim;
    const int32_t requestedChunkElements = std::max(
        1, GetEnvInt("TILEXR_DEMO_ALLTOALL_GROUP_CHUNK_ELEMENTS", elementsPerPeer));
    const auto plan = TileXR::Demo::PlanAllToAllGroup(
        rankSize, elementsPerPeer, requestedChunkElements);
    if (!plan.valid) {
        std::cerr << "[rank " << rank << "] ERROR: invalid grouped alltoall plan"
                  << " rankSize=" << rankSize
                  << " elementsPerPeer=" << elementsPerPeer
                  << " chunkElements=" << requestedChunkElements << std::endl;
        return false;
    }
    const int copyoutWorkersValue = GetEnvInt(
        "TILEXR_DEMO_ALLTOALL_GROUP_COPYOUT_WORKERS", 16);
    if (copyoutWorkersValue < 0 || !TileXR::Demo::AllToAllGroupValidCopyoutWorkers(
            static_cast<uint32_t>(copyoutWorkersValue))) {
        std::cerr << "[rank " << rank
                  << "] ERROR: TILEXR_DEMO_ALLTOALL_GROUP_COPYOUT_WORKERS must be 8 or 16, got "
                  << copyoutWorkersValue << std::endl;
        return false;
    }
    const uint32_t copyoutWorkers = static_cast<uint32_t>(copyoutWorkersValue);
    const uint32_t groupBlockDim = TileXR::Demo::AllToAllGroupBlockDim(copyoutWorkers);
    const int waitPollCyclesValue = GetEnvInt(
        "TILEXR_DEMO_ALLTOALL_GROUP_WAIT_POLL_CYCLES", 0);
    if (waitPollCyclesValue < 0 || waitPollCyclesValue > 1000000) {
        std::cerr << "[rank " << rank
                  << "] ERROR: TILEXR_DEMO_ALLTOALL_GROUP_WAIT_POLL_CYCLES must be "
                  << "between 0 and 1000000, got " << waitPollCyclesValue << std::endl;
        return false;
    }
    const uint32_t waitPollCycles = static_cast<uint32_t>(waitPollCyclesValue);
    const int routeStagesValue = GetEnvInt(
        "TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES", 0);
    if (routeStagesValue != 0 && routeStagesValue != 1) {
        std::cerr << "[rank " << rank
                  << "] ERROR: TILEXR_DEMO_ALLTOALL_GROUP_ROUTE_STAGES must be 0 or 1, got "
                  << routeStagesValue << std::endl;
        return false;
    }
    const bool routeStages = routeStagesValue == 1;
    constexpr size_t kRouteStageCount = 10U;
    const std::array<TileXR::Demo::AllToAllGroupRouteStage, kRouteStageCount>
        stagedRouteStages {{
            TileXR::Demo::AllToAllGroupRouteStage::kLocalSend,
            TileXR::Demo::AllToAllGroupRouteStage::kLocalCopy,
            TileXR::Demo::AllToAllGroupRouteStage::kRemoteSend,
            TileXR::Demo::AllToAllGroupRouteStage::kAllSend,
            TileXR::Demo::AllToAllGroupRouteStage::kRemoteWait,
            TileXR::Demo::AllToAllGroupRouteStage::kRemoteCopy,
            TileXR::Demo::AllToAllGroupRouteStage::kNoCopy,
            TileXR::Demo::AllToAllGroupRouteStage::kPrimary,
            TileXR::Demo::AllToAllGroupRouteStage::kSecondary,
            TileXR::Demo::AllToAllGroupRouteStage::kCombined,
    }};
    const std::array<const char*, kRouteStageCount> stageNames {{
        "local-send", "local-copy", "remote-send", "all-send", "remote-wait",
        "remote-copy", "no-copy", "primary", "secondary", "combined"
    }};
    const int warmup = std::max(0, GetEnvInt("TILEXR_DEMO_ALLTOALL_WARMUP", 0));
    const int repeat = std::max(1, GetEnvInt("TILEXR_DEMO_ALLTOALL_REPEAT", 1));
    const bool traceEnabled = GetEnvInt("TILEXR_UDMA_GROUP_TRACE", 0) != 0;
    const char* traceDirEnv = std::getenv("TILEXR_UDMA_GROUP_TRACE_DIR");
    const std::string traceDir = traceDirEnv != nullptr && traceDirEnv[0] != '\0' ?
        traceDirEnv : ".";
    if (traceEnabled && !TileXR::Demo::AllToAllGroupTraceLayoutFits(
            static_cast<uint32_t>(repeat), plan.groupCount, plan.passCount)) {
        std::cerr << "[rank " << rank << "] ERROR: grouped trace dimensions exceed capacity"
                  << " repeat=" << repeat
                  << " groupCount=" << plan.groupCount
                  << " passCount=" << plan.passCount
                  << " requiredBytes=" << TileXR::Demo::AllToAllGroupTraceLayoutBytes(
                      static_cast<uint32_t>(repeat), plan.groupCount, plan.passCount)
                  << " capacityBytes=" << TileXR::Demo::kAllToAllGroupTraceBytes << std::endl;
        return false;
    }

    const size_t elementCount = static_cast<size_t>(rankSize) * elementsPerPeer;
    const size_t dataBytes = elementCount * sizeof(int32_t);
    std::vector<int32_t> hostInput(elementCount, 0);
    std::vector<int32_t> hostOutput(elementCount, -1);
    TileXR::Demo::FillAllToAllInput(hostInput, rank, rankSize, elementsPerPeer);

    int32_t* input = nullptr;
    int32_t* output = nullptr;
    void* registeredMemory = nullptr;
    std::array<void*, kRouteStageCount> groupTraceDevices {};
    std::array<std::vector<uint8_t>, kRouteStageCount> hostGroupTraces;
    aclrtEvent stageStartEvent = nullptr;
    aclrtEvent stageEndEvent = nullptr;
    TileXRUDMAMemHandle handle = 0;
    bool registered = false;
    auto release = [&]() {
        if (registered) {
            CheckTileXR(rank, "TileXRUDMAUnregister grouped alltoall",
                TileXRUDMAUnregister(comm, handle));
            registered = false;
        }
        if (registeredMemory != nullptr) {
            aclrtFree(registeredMemory);
            registeredMemory = nullptr;
        }
        if (stageEndEvent != nullptr) {
            aclrtDestroyEvent(stageEndEvent);
            stageEndEvent = nullptr;
        }
        if (stageStartEvent != nullptr) {
            aclrtDestroyEvent(stageStartEvent);
            stageStartEvent = nullptr;
        }
        for (void*& groupTraceDevice : groupTraceDevices) {
            if (groupTraceDevice != nullptr) {
                aclrtFree(groupTraceDevice);
                groupTraceDevice = nullptr;
            }
        }
        if (output != nullptr) {
            aclrtFree(output);
            output = nullptr;
        }
        if (input != nullptr) {
            aclrtFree(input);
            input = nullptr;
        }
    };

    if (!CheckAcl(rank, "aclrtMalloc grouped input",
            aclrtMalloc(reinterpret_cast<void**>(&input), dataBytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc grouped output",
            aclrtMalloc(reinterpret_cast<void**>(&output), dataBytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc grouped registered memory",
            aclrtMalloc(&registeredMemory, plan.registeredBytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CopyHostToDevice(rank, input, dataBytes, hostInput.data(), dataBytes, "grouped input") ||
        !CopyHostToDevice(rank, output, dataBytes, hostOutput.data(), dataBytes, "grouped output init") ||
        !CheckAcl(rank, "aclrtMemset grouped registered memory",
            aclrtMemset(registeredMemory, plan.registeredBytes, 0, plan.registeredBytes))) {
        release();
        return false;
    }

    if (traceEnabled) {
        const size_t traceCount = routeStages ? kRouteStageCount : 1U;
        for (size_t traceIndex = 0U; traceIndex < traceCount; ++traceIndex) {
            auto& hostGroupTrace = hostGroupTraces[traceIndex];
            hostGroupTrace.assign(TileXR::Demo::kAllToAllGroupTraceBytes, 0U);
            TileXR::Demo::AllToAllGroupTraceHeader header {};
            header.magic = TileXR::Demo::kAllToAllGroupTraceMagic;
            header.version = TileXR::Demo::kAllToAllGroupTraceVersion;
            header.rank = static_cast<uint32_t>(rank);
            header.iterationCount = static_cast<uint32_t>(repeat);
            header.groupCount = plan.groupCount;
            header.passCount = plan.passCount;
            header.coreCount = TileXR::Demo::kAllToAllGroupTraceCoreCount;
            header.phaseCount = TileXR::Demo::kAllToAllGroupTracePhaseCount;
            header.cyclesPerUs = TileXR::Demo::kAllToAllGroupTraceCyclesPerUs;
            header.traceBytes = TileXR::Demo::kAllToAllGroupTraceBytes;
            header.kernelSpanOffset = TileXR::Demo::kAllToAllGroupTraceHeaderBytes;
            header.taskSpanOffset = TileXR::Demo::AllToAllGroupTraceTaskSpanBaseOffset();
            std::memcpy(hostGroupTrace.data(), &header, sizeof(header));
            if (!CheckAcl(rank, "aclrtMalloc grouped trace",
                    aclrtMalloc(&groupTraceDevices[traceIndex],
                        TileXR::Demo::kAllToAllGroupTraceBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST)) ||
                !CopyHostToDevice(rank, groupTraceDevices[traceIndex],
                    TileXR::Demo::kAllToAllGroupTraceBytes,
                    hostGroupTrace.data(), hostGroupTrace.size(), "grouped trace")) {
                release();
                return false;
            }
        }
    }

    const int registerRet = TileXRUDMARegister(
        comm, static_cast<GM_ADDR>(registeredMemory), plan.registeredBytes, &handle);
    if (!CheckTileXR(rank, "TileXRUDMARegister grouped alltoall", registerRet)) {
        release();
        return false;
    }
    registered = true;

    if (routeStages &&
        (!CheckAcl(rank, "aclrtCreateEvent grouped stage start",
            aclrtCreateEvent(&stageStartEvent)) ||
         !CheckAcl(rank, "aclrtCreateEvent grouped stage end",
            aclrtCreateEvent(&stageEndEvent)))) {
        release();
        return false;
    }

    auto debug = reinterpret_cast<int32_t*>(
        static_cast<uint8_t*>(registeredMemory) + plan.controlOffset);
    PrintStatus(rank, "grouped alltoall registeredBytes=" + std::to_string(plan.registeredBytes) +
        " payloadPlaneBytes=" + std::to_string(plan.payloadPlaneBytes) +
        " payloadOffset0=" + std::to_string(plan.payloadOffset[0]) +
        " payloadOffset1=" + std::to_string(plan.payloadOffset[1]) +
        " signalPlaneBytes=" + std::to_string(plan.signalPlaneBytes) +
        " signalOffset0=" + std::to_string(plan.signalOffset[0]) +
        " signalOffset1=" + std::to_string(plan.signalOffset[1]) +
        " controlOffset=" + std::to_string(plan.controlOffset) +
        " groups=" + std::to_string(plan.groupCount) +
        " passes=" + std::to_string(plan.passCount));
    PrintStatus(rank, "grouped alltoall warmup=" + std::to_string(warmup) +
        " repeat=" + std::to_string(repeat) +
        " copyoutWorkers=" + std::to_string(copyoutWorkers) +
        " waitPollCycles=" + std::to_string(waitPollCycles) +
        " blockDim=" + std::to_string(groupBlockDim) +
        " routeStages=" + std::to_string(routeStagesValue));

    if (routeStages &&
        !DemoBarrierAll(rank, rankSize, "grouped route stages ready")) {
        release();
        return false;
    }

    uint32_t invocationId = 0U;
    auto launchGroupStage = [&](TileXR::Demo::AllToAllGroupRouteStage routeStage,
                                void* trace, uint32_t traceIteration) {
        launch_tilexr_udma_all_to_all_group(
            groupBlockDim, stream, commArgsDev,
            reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
            reinterpret_cast<GM_ADDR>(registeredMemory), reinterpret_cast<GM_ADDR>(debug),
            invocationId, elementsPerPeer, plan.chunkElements,
            plan.passCount, plan.groupCount,
            plan.payloadOffset[0], plan.payloadOffset[1],
            plan.signalOffset[0], plan.signalOffset[1],
            reinterpret_cast<GM_ADDR>(trace), traceIteration, copyoutWorkers,
            static_cast<uint32_t>(routeStage), waitPollCycles);
    };

    double totalUs = 0.0;
    std::array<double, kRouteStageCount> stageTotalUs {};
    if (!routeStages) {
        for (int iter = 0; iter < warmup; ++iter, ++invocationId) {
            launchGroupStage(TileXR::Demo::AllToAllGroupRouteStage::kCombined, nullptr, 0U);
        }
        if (!CheckAcl(rank, "aclrtSynchronizeStream grouped warmup",
                aclrtSynchronizeStream(stream))) {
            release();
            return false;
        }

        const auto begin = std::chrono::steady_clock::now();
        for (int iter = 0; iter < repeat; ++iter, ++invocationId) {
            launchGroupStage(TileXR::Demo::AllToAllGroupRouteStage::kCombined,
                groupTraceDevices[0], static_cast<uint32_t>(iter));
        }
        if (!CheckAcl(rank, "aclrtSynchronizeStream grouped measured",
                aclrtSynchronizeStream(stream))) {
            release();
            return false;
        }
        const auto end = std::chrono::steady_clock::now();
        totalUs = std::chrono::duration<double, std::micro>(end - begin).count();
    } else {
        auto runStageBatch = [&](size_t stageIndex) -> bool {
            for (int iter = 0; iter < warmup; ++iter, ++invocationId) {
                launchGroupStage(stagedRouteStages[stageIndex], nullptr, 0U);
            }
            if (!CheckAcl(rank, "aclrtSynchronizeStream grouped stage warmup",
                    aclrtSynchronizeStream(stream)) ||
                !CheckAcl(rank, "aclrtRecordEvent grouped stage start",
                    aclrtRecordEvent(stageStartEvent, stream))) {
                return false;
            }
            for (int iter = 0; iter < repeat; ++iter, ++invocationId) {
                launchGroupStage(stagedRouteStages[stageIndex],
                    groupTraceDevices[stageIndex], static_cast<uint32_t>(iter));
            }
            if (!CheckAcl(rank, "aclrtRecordEvent grouped stage end",
                    aclrtRecordEvent(stageEndEvent, stream)) ||
                !CheckAcl(rank, "aclrtSynchronizeStream grouped stage measured",
                    aclrtSynchronizeStream(stream))) {
                return false;
            }
            float elapsedMs = 0.0F;
            if (!CheckAcl(rank, "aclrtEventElapsedTime grouped stage",
                    aclrtEventElapsedTime(&elapsedMs, stageStartEvent, stageEndEvent))) {
                return false;
            }
            stageTotalUs[stageIndex] = static_cast<double>(elapsedMs) * 1000.0;
            const std::string barrierStep = "grouped route stage " +
                std::string(stageNames[stageIndex]) + " complete";
            return DemoBarrierAll(rank, rankSize, barrierStep);
        };

        for (size_t stageIndex = 0U; stageIndex < kRouteStageCount; ++stageIndex) {
            if (!runStageBatch(stageIndex)) {
                release();
                return false;
            }
        }
        for (double stageUs : stageTotalUs) {
            totalUs += stageUs;
        }
    }

    std::vector<int32_t> hostDebug(kErrorWordsPerCore * kErrorCoreCount, 0);
    const size_t debugBytes = hostDebug.size() * sizeof(int32_t);
    bool copyOk = CopyDeviceToHost(
        rank, hostOutput.data(), dataBytes, output, dataBytes, "grouped alltoall output") &&
        CopyDeviceToHost(rank, hostDebug.data(), debugBytes, debug, debugBytes,
            "grouped alltoall debug");
    if (traceEnabled) {
        const size_t traceCount = routeStages ? kRouteStageCount : 1U;
        for (size_t traceIndex = 0U; traceIndex < traceCount; ++traceIndex) {
            auto& hostGroupTrace = hostGroupTraces[traceIndex];
            const std::string stageName = routeStages ? stageNames[traceIndex] : "";
            copyOk = CopyDeviceToHost(
                rank, hostGroupTrace.data(), hostGroupTrace.size(),
                groupTraceDevices[traceIndex], TileXR::Demo::kAllToAllGroupTraceBytes,
                "grouped trace " + stageName) &&
                WriteGroupTraceBinary(rank, traceDir, stageName, hostGroupTrace) && copyOk;
        }
    }
    bool debugOk = true;
    for (uint32_t core = 0; core < kErrorCoreCount; ++core) {
        const size_t base = static_cast<size_t>(core) * kErrorWordsPerCore;
        if (hostDebug[base] == 0) {
            continue;
        }
        debugOk = false;
        const uint64_t expected = static_cast<uint32_t>(hostDebug[base + 8]) |
            (static_cast<uint64_t>(static_cast<uint32_t>(hostDebug[base + 9])) << 32U);
        const uint64_t observed = static_cast<uint32_t>(hostDebug[base + 10]) |
            (static_cast<uint64_t>(static_cast<uint32_t>(hostDebug[base + 11])) << 32U);
        std::cerr << "[rank " << rank << "] ERROR: grouped core=" << core
                  << " stage=" << hostDebug[base + 1]
                  << " group=" << hostDebug[base + 2]
                  << " pass=" << hostDebug[base + 3]
                  << " peer=" << hostDebug[base + 4]
                  << " qp=" << hostDebug[base + 5]
                  << " quiet=" << hostDebug[base + 6]
                  << " expected=" << expected
                  << " observed=" << observed << std::endl;
    }

    const double perIterUs = totalUs / static_cast<double>(repeat);
    const double bandwidthGbs = static_cast<double>(dataBytes) / (perIterUs * 1.0e3);
    std::cout << "[rank " << rank << "] grouped alltoall " << repeat
              << " iters total=" << totalUs / 1000.0
              << " ms perIter=" << perIterUs
              << " us payload=" << dataBytes
              << " bytes bw=" << bandwidthGbs << " GB/s" << std::endl;
    if (routeStages) {
        for (size_t stageIndex = 0U; stageIndex < kRouteStageCount; ++stageIndex) {
            std::cout << "[rank " << rank << "] grouped route stage "
                      << stageNames[stageIndex] << " mean="
                      << stageTotalUs[stageIndex] / static_cast<double>(repeat)
                      << " us" << std::endl;
        }
    }

    const bool valid = copyOk && debugOk &&
        ValidateAllToAllData(rank, rankSize, hostOutput, elementsPerPeer);
    release();
    return valid;
}

void Cleanup(
    TileXRCommPtr comm, aclrtStream stream, void* registeredMemory, int32_t* debug, int rank, int deviceId)
{
    if (registeredMemory != nullptr) {
        PrintStatus(rank, "aclrtFree registered memory");
        aclrtFree(registeredMemory);
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
    int rankSize = argc > argIndex ? std::atoi(argv[argIndex++]) : GetRankSizeFromEnv();
    int rank = argc > argIndex ? std::atoi(argv[argIndex++]) : GetRankFromEnv();
    int testType = argc > argIndex ? std::atoi(argv[argIndex++]) : 0;
    int32_t elementsPerRank = argc > argIndex ? std::atoi(argv[argIndex++]) : kDefaultElementsPerRank;
    int npuCount = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_NPUS", 8);
    int firstNpu = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_FIRST_NPU", 0);
    int deviceId = GetDeviceIdFromEnv(rank, npuCount, firstNpu);

    std::cout << "========================================" << std::endl;
    std::cout << "  TileXR UDMA Communication Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[rank " << rank << "] argv: rankSize=" << rankSize << " rank=" << rank
              << " testType=" << testType << " elementsPerRank=" << elementsPerRank
              << " npuCount=" << npuCount << " firstNpu=" << firstNpu << std::endl;
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

    if (!CheckTileXR(rank, "TileXRCommInitRankLocal", TileXRCommInitRankLocal(rankSize, rank, &comm))) {
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

    if ((commArgsHost->extraFlag & TileXR::ExtraFlag::UDMA) == 0 || commArgsHost->udmaInfoPtr == nullptr) {
        std::cerr << "[rank " << rank << "] ERROR: TileXR UDMA is not enabled. "
                  << "Check A5/Ascend950 hardware support, CANN/driver setup, and LD_LIBRARY_PATH." << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    if (testType == 8) {
        const bool ok = RunGroupedAllToAll(
            rank, rankSize, elementsPerRank, comm, stream, commArgsDev);
        Cleanup(comm, stream, nullptr, nullptr, rank, deviceId);
        if (!ok) {
            std::cerr << "[rank " << rank << "] TileXR grouped alltoall demo failed" << std::endl;
            return 1;
        }
        std::cout << "[rank " << rank << "] TileXR grouped alltoall demo success" << std::endl;
        return 0;
    }

    bool isAllToAll = testType == 2 || testType == 4 || testType == 5 || testType == 6 || testType == 7;
    bool isAllReduce = testType == 3;
    bool strictAllToAllUdma =
        isAllToAll && (testType == 4 || testType == 7 || GetEnvInt("TILEXR_DEMO_ALLTOALL_USE_UDMA", 0) != 0);
    int allToAllRepeat = isAllToAll ? std::max(1, GetEnvInt("TILEXR_DEMO_ALLTOALL_REPEAT", 1)) : 1;
    int allToAllWarmup = isAllToAll ? std::max(0, GetEnvInt("TILEXR_DEMO_ALLTOALL_WARMUP", 0)) : 0;
    int bigDataProfileStage = testType == 7 ?
        GetEnvInt("TILEXR_DEMO_BIGDATA_PROFILE_STAGE", kBigDataProfileStageFull) :
        kBigDataProfileStageFull;
    bigDataProfileStage = std::max(0, std::min(bigDataProfileStage, kBigDataProfileStageFull));
    const bool forceBigData35Core =
        testType == 7 && GetEnvInt("TILEXR_DEMO_BIGDATA_FORCE_35CORE", 0) != 0;
    const bool bigDataRemotePutOnly =
        testType == 7 && GetEnvInt("TILEXR_DEMO_BIGDATA_REMOTE_PUT_ONLY", 0) != 0;
    const int bigDataIsolatedTask =
        testType == 7 ? GetEnvInt("TILEXR_DEMO_BIGDATA_ISOLATED_TASK", 0) : 0;
    if (bigDataIsolatedTask < 0 || bigDataIsolatedTask > 11) {
        std::cerr << "[rank " << rank << "] ERROR: TILEXR_DEMO_BIGDATA_ISOLATED_TASK must be 0..11"
                  << ", got " << bigDataIsolatedTask << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    bool bigDataProfilePartial = testType == 7 && bigDataProfileStage < kBigDataProfileStageFull;
    bigDataProfilePartial = bigDataProfilePartial || bigDataIsolatedTask != 0;
    bool syncAllToAllAtEnd =
        isAllToAll && GetEnvInt("TILEXR_DEMO_ALLTOALL_SYNC_AT_END", 0) != 0;
    bool useAllToAllPlainIpc =
        isAllToAll && testType != 7 && GetEnvInt("TILEXR_DEMO_ALLTOALL_PLAIN_IPC", 0) != 0;
    bool useAllToAllFusedIpc =
        isAllToAll && testType != 7 && GetEnvInt("TILEXR_DEMO_ALLTOALL_FUSED_IPC", 0) != 0;
    bool dumpAllToAllOnStrictFail = isAllToAll && GetEnvInt("TILEXR_DEMO_ALLTOALL_DUMP_ON_STRICT_FAIL", 0) != 0;
    bool useAllToAllDataAsFlagIpc =
        isAllToAll && testType != 6 && testType != 7 &&
        !strictAllToAllUdma && !useAllToAllPlainIpc && !useAllToAllFusedIpc;
    const char* allToAllIpcFallbackLabel =
        useAllToAllFusedIpc ? "fused IPC" :
        (useAllToAllPlainIpc ? "plain IPC fallback" : "data-as-flag IPC fallback");
    bool forceAllToAllIpcFallback = false;
    bool hasOutput = isAllToAll || isAllReduce;
    size_t dataCount = static_cast<size_t>(rankSize) * elementsPerRank;
    size_t dataBytes = dataCount * sizeof(int32_t);
    const TileXR::Demo::AllToAllChunkPlan chunkPlan =
        isAllToAll ? TileXR::Demo::PlanAllToAllUdmaChunks(rankSize, elementsPerRank) :
            TileXR::Demo::AllToAllChunkPlan {};
    const int32_t bigDataRanksPerNode = std::max(1, npuCount);
    const TileXR::Demo::AllToAllBigDataPlan bigDataPlan =
        isAllToAll ? TileXR::Demo::PlanAllToAllBigDataUdma(
            rankSize, elementsPerRank, forceBigData35Core, bigDataRanksPerNode) :
            TileXR::Demo::AllToAllBigDataPlan {};
    const bool fullmeshTraceRequested =
        testType == 7 && GetEnvInt("TILEXR_UDMA_FULLMESH_TRACE", 0) != 0;
    const bool fullmeshTraceEnabled = fullmeshTraceRequested &&
        !bigDataRemotePutOnly && TileXR::Demo::AllToAllBigDataIsMultiNode(
            rankSize, bigDataRanksPerNode);
    const char* fullmeshTraceDirEnv = std::getenv("TILEXR_UDMA_FULLMESH_TRACE_DIR");
    const std::string fullmeshTraceDir =
        fullmeshTraceDirEnv != nullptr && fullmeshTraceDirEnv[0] != '\0' ?
        fullmeshTraceDirEnv : ".";
    if (testType == 7 && !TileXR::Demo::AllToAllBigDataValidTopology(rankSize, bigDataRanksPerNode)) {
        std::cerr << "[rank " << rank
                  << "] ERROR: bigdata alltoall multi-node requires rankSize multiple of ranksPerNode"
                  << " when rankSize > ranksPerNode"
                  << ", rankSize=" << rankSize
                  << " ranksPerNode=" << bigDataRanksPerNode << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (isAllToAll) {
        const size_t dataAsFlagStagingBytes = AllToAllDataAsFlagStagingBytes(rankSize, elementsPerRank);
        const size_t plainIpcStagingBytes = AllToAllPlainIpcStagingBytes(rankSize, elementsPerRank);
        const size_t selectedIpcStagingBytes = useAllToAllPlainIpc ? plainIpcStagingBytes : dataAsFlagStagingBytes;
        if (useAllToAllPlainIpc && plainIpcStagingBytes > static_cast<size_t>(TileXR::IPC_BUFF_MAX_SIZE)) {
            std::cerr << "[rank " << rank << "] ERROR: alltoall plain IPC fallback staging requires "
                      << plainIpcStagingBytes << " bytes, exceeds IPC data capacity "
                      << TileXR::IPC_BUFF_MAX_SIZE << std::endl;
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        if (useAllToAllDataAsFlagIpc && dataAsFlagStagingBytes > static_cast<size_t>(TileXR::IPC_BUFF_MAX_SIZE)) {
            std::cerr << "[rank " << rank << "] ERROR: alltoall data-as-flag IPC fallback staging requires "
                      << dataAsFlagStagingBytes << " bytes, exceeds IPC data capacity "
                      << TileXR::IPC_BUFF_MAX_SIZE << std::endl;
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        PrintStatus(rank, "alltoall plain IPC staging bytes=" + std::to_string(plainIpcStagingBytes));
        PrintStatus(rank, "alltoall data-as-flag staging bytes=" + std::to_string(dataAsFlagStagingBytes));
        if (strictAllToAllUdma && selectedIpcStagingBytes > static_cast<size_t>(TileXR::IPC_BUFF_MAX_SIZE)) {
            PrintStatus(rank, "skip IPC staging capacity guard for strict UDMA alltoall");
        }
        PrintStatus(rank, "alltoall repeat=" + std::to_string(allToAllRepeat) +
            " syncAtEnd=" + std::string(syncAllToAllAtEnd ? "true" : "false"));
        if (testType == 7) {
            PrintStatus(rank, "bigdata profile stage=" + std::to_string(bigDataProfileStage) +
                " fullStage=" + std::to_string(kBigDataProfileStageFull));
            PrintStatus(rank, "bigdata multinode mode=" +
                std::string(TileXR::Demo::AllToAllBigDataIsMultiNode(
                    rankSize, bigDataRanksPerNode) ? "true" : "false") +
                " force35Core=" + std::string(forceBigData35Core ? "true" : "false") +
                " remotePutOnly=" + std::string(bigDataRemotePutOnly ? "true" : "false") +
                " isolatedTask=" + std::to_string(bigDataIsolatedTask) +
                " ranksPerNode=" + std::to_string(bigDataRanksPerNode) +
                " blockDim=" + std::to_string(TileXR::Demo::AllToAllBigDataBlockDim(
                    rankSize, forceBigData35Core, bigDataRemotePutOnly, bigDataRanksPerNode)) +
                " shards=" + std::to_string(TileXR::Demo::AllToAllBigDataShardCount(
                    rankSize, forceBigData35Core, bigDataRanksPerNode)));
        }
        PrintStatus(rank, "alltoall UDMA chunk plan: passCount=" + std::to_string(chunkPlan.passCount) +
            " chunkElements=" + std::to_string(chunkPlan.chunkElements) +
            " registeredBytes=" + std::to_string(chunkPlan.registeredBytes));
        if (testType == 7) {
            PrintStatus(rank, "alltoall bigdata UDMA plan: passCount=" + std::to_string(bigDataPlan.passCount) +
                " chunkElements=" + std::to_string(bigDataPlan.chunkElements) +
                " dataBytes=" + std::to_string(bigDataPlan.dataBytes) +
                " copyDoneOffset=" + std::to_string(bigDataPlan.copyDoneOffset) +
                " recvCopyDoneOffset=" + std::to_string(bigDataPlan.recvCopyDoneOffset) +
                " remoteSendDoneOffset=" + std::to_string(bigDataPlan.remoteSendDoneOffset) +
                " readySignalOffset=" + std::to_string(bigDataPlan.readySignalOffset) +
                " ackSignalOffset=" + std::to_string(bigDataPlan.ackSignalOffset) +
                " registeredBytes=" + std::to_string(bigDataPlan.registeredBytes));
        }
    }
    size_t inputOffset = 0;
    const size_t activeBytesPerRank =
        isAllToAll && testType == 7 ? 0 :
        isAllToAll && (strictAllToAllUdma || testType == 6) ?
            chunkPlan.chunkBytesPerRank : dataBytes;
    size_t outputOffset = hasOutput ? activeBytesPerRank : 0;
    size_t signalBytes = static_cast<size_t>(rankSize) * sizeof(uint64_t);
    size_t signalOffset = testType == 7 ? bigDataPlan.readySignalOffset :
        (hasOutput ? (outputOffset + activeBytesPerRank) : activeBytesPerRank);
    size_t payloadBytes = testType == 7 ? bigDataPlan.registeredBytes : (signalOffset + signalBytes);
    size_t allocBytes = payloadBytes;
    size_t registeredBytes = allocBytes;
    if (isAllToAll && testType == 7) {
        registeredBytes = bigDataPlan.registeredBytes;
    } else if (isAllToAll && (strictAllToAllUdma || testType == 6)) {
        registeredBytes = chunkPlan.registeredBytes;
    }
    if (allocBytes < registeredBytes) {
        allocBytes = registeredBytes;
    }
    if (!CheckAcl(rank, "aclrtMalloc debug", aclrtMalloc(reinterpret_cast<void**>(&debug),
            kDebugWords * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl(rank, "aclrtMalloc registered memory", aclrtMalloc(&registeredMemory,
            allocBytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    auto data = static_cast<int32_t*>(registeredMemory);
    auto input = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(registeredMemory) + inputOffset);
    auto output = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(registeredMemory) + outputOffset);
    auto signals = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(registeredMemory) + signalOffset);
    const bool chunkedStrictAllToAll =
        isAllToAll && testType != 7 && strictAllToAllUdma && chunkPlan.passCount > 1;
    if (useAllToAllFusedIpc) {
        PrintStatus(rank, "skip TileXRUDMARegister for alltoall fused IPC path");
        forceAllToAllIpcFallback = true;
    } else if (useAllToAllPlainIpc) {
        PrintStatus(rank, "skip TileXRUDMARegister for alltoall plain IPC path");
        forceAllToAllIpcFallback = true;
    } else if (useAllToAllDataAsFlagIpc) {
        PrintStatus(rank, "skip TileXRUDMARegister for alltoall data-as-flag IPC path");
        forceAllToAllIpcFallback = true;
    } else if (chunkedStrictAllToAll) {
        PrintStatus(rank, "defer TileXRUDMARegister to per-pass registered output chunk");
    } else if (testType == 6) {
        PrintStatus(rank, "defer TileXRUDMARegister to fused alltoall relay chunk");
    } else if (testType == 7) {
        PrintStatus(rank, "defer TileXRUDMARegister to bigdata alltoall relay buffer");
    } else {
        int registerRet =
            TileXRUDMARegister(comm, static_cast<GM_ADDR>(registeredMemory), registeredBytes, &udmaHandle);
        if (registerRet != TileXR::TILEXR_SUCCESS) {
            if (!isAllToAll || strictAllToAllUdma) {
                if (strictAllToAllUdma) {
                    std::cerr << "[rank " << rank << "] ERROR: strict alltoall UDMA registration failed"
                              << " ret=" << registerRet << std::endl;
                }
                CheckTileXR(rank, "TileXRUDMARegister", registerRet);
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }
            std::cerr << "[rank " << rank
                      << "] WARNING: TileXRUDMARegister failed; use alltoall data-as-flag IPC fallback"
                      << " ret=" << registerRet << std::endl;
            forceAllToAllIpcFallback = true;
        } else {
            udmaRegistered = true;
        }
    }
    PrintStatus(rank, "registered UDMA memory base=" + std::to_string(reinterpret_cast<uintptr_t>(registeredMemory)) +
        " bytes=" + std::to_string(registeredBytes) +
        " inputOffset=" + std::to_string(inputOffset) +
        " outputOffset=" + std::to_string(outputOffset) +
        " signalOffset=" + std::to_string(signalOffset));
    PrintCommArgs(rank, *commArgsHost, commArgsDev);

    std::vector<int32_t> hostData(dataCount, -1);
    std::vector<int32_t> hostOutput(dataCount, -1);
    if (isAllToAll) {
        TileXR::Demo::FillAllToAllInput(hostData, rank, rankSize, elementsPerRank);
    } else if (isAllReduce) {
        TileXR::Demo::FillAllReduceInput(hostData, rank, elementsPerRank);
    } else {
        std::fill(hostData.begin() + static_cast<size_t>(rank) * elementsPerRank,
                  hostData.begin() + static_cast<size_t>(rank + 1) * elementsPerRank,
                  1000 + rank);
    }
    std::vector<uint64_t> hostSignals(static_cast<size_t>(rankSize), 0);
    std::vector<int32_t> hostDebug(kDebugWords, 0);

    const char* inputName = isAllToAll ? "alltoall input" : (isAllReduce ? "allreduce input" : "data");
    bool initOk = true;
    if (!chunkedStrictAllToAll && testType != 7) {
        initOk = CopyHostToDevice(rank, input, dataCount * sizeof(int32_t),
            hostData.data(), dataCount * sizeof(int32_t), inputName);
        if (hasOutput) {
            const char* outputName = isAllToAll ? "alltoall output" : "allreduce output";
            initOk = CopyHostToDevice(rank, output, dataCount * sizeof(int32_t),
                hostOutput.data(), dataCount * sizeof(int32_t), outputName) && initOk;
        }
    } else {
        std::fill(hostOutput.begin(), hostOutput.end(), -1);
    }
    if (!initOk ||
        !CopyHostToDevice(rank, signals, hostSignals.size() * sizeof(uint64_t),
            hostSignals.data(), hostSignals.size() * sizeof(uint64_t), "signals") ||
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

    if (testType == 7) {
        void* bigInput = nullptr;
        void* bigOutput = nullptr;
        void* fullmeshTraceDevice = nullptr;
        std::vector<uint8_t> hostFullmeshTrace;
        auto freeFullmeshTrace = [&]() {
            if (fullmeshTraceDevice != nullptr) {
                aclrtFree(fullmeshTraceDevice);
                fullmeshTraceDevice = nullptr;
            }
        };
        const size_t bigDataBytes = dataBytes;
        if (!CheckAcl(rank, "aclrtMalloc bigdata input",
                aclrtMalloc(&bigInput, bigDataBytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
            !CheckAcl(rank, "aclrtMalloc bigdata output",
                aclrtMalloc(&bigOutput, bigDataBytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
            aclrtFree(bigInput);
            aclrtFree(bigOutput);
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        if (!CopyHostToDevice(rank, bigInput, dataCount * sizeof(int32_t),
                hostData.data(), dataCount * sizeof(int32_t), "bigdata alltoall input") ||
            !CopyHostToDevice(rank, bigOutput, dataCount * sizeof(int32_t),
                hostOutput.data(), dataCount * sizeof(int32_t), "bigdata alltoall output")) {
            aclrtFree(bigInput);
            aclrtFree(bigOutput);
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        std::vector<uint8_t> zeroBigControl(bigDataPlan.controlBytes + bigDataPlan.signalBytes, 0);
        if (!CopyHostToDevice(rank, static_cast<uint8_t*>(registeredMemory) + bigDataPlan.copyDoneOffset,
                bigDataPlan.controlBytes + bigDataPlan.signalBytes,
                zeroBigControl.data(), zeroBigControl.size(),
                "bigdata copy/ready/ack payload+signals zero")) {
            aclrtFree(bigInput);
            aclrtFree(bigOutput);
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        if (!udmaRegistered) {
            int registerRet = TileXRUDMARegister(comm, static_cast<GM_ADDR>(registeredMemory),
                registeredBytes, &udmaHandle);
            if (registerRet != TileXR::TILEXR_SUCCESS) {
                std::cerr << "[rank " << rank << "] ERROR: bigdata alltoall UDMA registration failed"
                          << " ret=" << registerRet << " regBytes=" << registeredBytes << std::endl;
                aclrtFree(bigInput);
                aclrtFree(bigOutput);
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }
            udmaRegistered = true;
        }
        PrintStatus(rank, "bigdata alltoall registered dataBytes=" + std::to_string(bigDataPlan.dataBytes) +
            " copyDoneOffset=" + std::to_string(bigDataPlan.copyDoneOffset) +
            " recvCopyDoneOffset=" + std::to_string(bigDataPlan.recvCopyDoneOffset) +
            " remoteSendDoneOffset=" + std::to_string(bigDataPlan.remoteSendDoneOffset) +
            " readySignalOffset=" + std::to_string(bigDataPlan.readySignalOffset) +
            " ackSignalOffset=" + std::to_string(bigDataPlan.ackSignalOffset) +
            " regBytes=" + std::to_string(registeredBytes) +
            " passCount=" + std::to_string(bigDataPlan.passCount) +
            " chunkElements=" + std::to_string(bigDataPlan.chunkElements) +
            " repeat=" + std::to_string(allToAllRepeat) +
            " profileStage=" + std::to_string(bigDataProfileStage));
        if (fullmeshTraceRequested && !fullmeshTraceEnabled) {
            std::cerr << "[rank " << rank << "] ERROR: fullmesh trace requires multi-node "
                      << "non-remote-put-only bigdata mode" << std::endl;
            if (udmaRegistered) {
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                udmaRegistered = false;
            }
            aclrtFree(bigInput);
            aclrtFree(bigOutput);
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        if (fullmeshTraceEnabled) {
            if (!TileXR::Demo::FullmeshTraceLayoutFits(
                    static_cast<uint32_t>(allToAllRepeat), bigDataPlan.passCount,
                    static_cast<uint32_t>(rankSize))) {
                std::cerr << "[rank " << rank << "] ERROR: fullmesh trace dimensions exceed capacity"
                          << " repeat=" << allToAllRepeat
                          << " passCount=" << bigDataPlan.passCount
                          << " rankSize=" << rankSize
                          << " requiredBytes=" << TileXR::Demo::FullmeshTraceLayoutBytes(
                              static_cast<uint32_t>(allToAllRepeat), bigDataPlan.passCount,
                              static_cast<uint32_t>(rankSize))
                          << " capacityBytes=" << TileXR::Demo::kFullmeshTraceBytes << std::endl;
                if (udmaRegistered) {
                    CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    udmaRegistered = false;
                }
                aclrtFree(bigInput);
                aclrtFree(bigOutput);
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }
            hostFullmeshTrace.assign(TileXR::Demo::kFullmeshTraceBytes, 0U);
            TileXR::Demo::FullmeshTraceHeader header {};
            header.magic = TileXR::Demo::kFullmeshTraceMagic;
            header.version = TileXR::Demo::kFullmeshTraceVersion;
            header.rank = static_cast<uint32_t>(rank);
            header.iterationCount = static_cast<uint32_t>(allToAllRepeat);
            header.passCount = bigDataPlan.passCount;
            header.coreCount = TileXR::Demo::kFullmeshTraceMaxCores;
            header.rankSize = static_cast<uint32_t>(rankSize);
            header.phaseCount = TileXR::Demo::kFullmeshTracePhaseCount;
            header.cyclesPerUs = TileXR::Demo::kFullmeshTraceCyclesPerUs;
            header.traceBytes = TileXR::Demo::kFullmeshTraceBytes;
            header.kernelSpanOffset = TileXR::Demo::kFullmeshTraceHeaderBytes;
            header.taskSpanOffset = TileXR::Demo::FullmeshTraceTaskSpanBaseOffset();
            std::memcpy(hostFullmeshTrace.data(), &header, sizeof(header));
            if (!CheckAcl(rank, "aclrtMalloc fullmesh trace",
                    aclrtMalloc(&fullmeshTraceDevice, TileXR::Demo::kFullmeshTraceBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST)) ||
                !CopyHostToDevice(rank, fullmeshTraceDevice, TileXR::Demo::kFullmeshTraceBytes,
                    hostFullmeshTrace.data(), hostFullmeshTrace.size(), "fullmesh trace")) {
                freeFullmeshTrace();
                if (udmaRegistered) {
                    CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    udmaRegistered = false;
                }
                aclrtFree(bigInput);
                aclrtFree(bigOutput);
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }
        }
        if (!CheckAcl(rank, "aclrtSynchronizeStream bigdata prime", aclrtSynchronizeStream(stream)) ||
            !DemoBarrierAll(rank, rankSize, "all ranks bigdata prime")) {
            if (udmaRegistered) {
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                udmaRegistered = false;
            }
            aclrtFree(bigInput);
            aclrtFree(bigOutput);
            freeFullmeshTrace();
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }

        const uint32_t bigDataBlockDim = TileXR::Demo::AllToAllBigDataBlockDim(
            rankSize, forceBigData35Core, bigDataRemotePutOnly, bigDataRanksPerNode);
        const uint32_t bigDataModeFlags =
            (forceBigData35Core ? 1U : 0U) | (bigDataRemotePutOnly ? 2U : 0U) |
            (static_cast<uint32_t>(bigDataRanksPerNode) << 8U);
        auto a2aStart = std::chrono::steady_clock::now();
        for (int iter = 0; iter < allToAllRepeat; ++iter) {
            const uint64_t kernelLoopBase = static_cast<uint64_t>(iter);
            const uint32_t fullmeshTraceIteration = static_cast<uint32_t>(iter);
            launch_tilexr_udma_all_to_all_bigdata(
                bigDataBlockDim, stream, commArgsDev,
                reinterpret_cast<GM_ADDR>(bigInput), reinterpret_cast<GM_ADDR>(bigOutput),
                reinterpret_cast<GM_ADDR>(registeredMemory), reinterpret_cast<GM_ADDR>(debug),
                reinterpret_cast<GM_ADDR>(fullmeshTraceDevice), fullmeshTraceIteration,
                static_cast<uint32_t>(bigDataIsolatedTask),
                elementsPerRank, 0, bigDataPlan.copyDoneOffset,
                bigDataPlan.recvCopyDoneOffset,
                bigDataPlan.remoteSendDoneOffset,
                bigDataPlan.readySignalOffset, bigDataPlan.ackSignalOffset,
                bigDataPlan.chunkElements, bigDataPlan.passCount, 1, kernelLoopBase,
                static_cast<uint32_t>(bigDataProfileStage),
                bigDataModeFlags);
        }
        if (!CheckAcl(rank, "aclrtSynchronizeStream post-alltoall", aclrtSynchronizeStream(stream))) {
            if (udmaRegistered) {
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                udmaRegistered = false;
            }
            aclrtFree(bigInput);
            aclrtFree(bigOutput);
            freeFullmeshTrace();
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        auto a2aEnd = std::chrono::steady_clock::now();
        double a2aMs = std::chrono::duration<double, std::milli>(a2aEnd - a2aStart).count();
        double a2aPerIterUs = (allToAllRepeat > 0) ?
            (a2aMs * 1000.0 / static_cast<double>(allToAllRepeat)) : 0.0;
        double payload = static_cast<double>(rankSize) * static_cast<double>(elementsPerRank) * sizeof(int32_t);
        double bwGbs = (a2aPerIterUs > 0.0) ? (payload / (a2aPerIterUs * 1e3)) : 0.0;
        std::cout << "[rank " << rank << "] alltoall udma-bigdata " << allToAllRepeat
                  << " iters(total=" << bigDataPlan.passCount << " pass/iter) total=" << a2aMs
                  << " ms perIter=" << a2aPerIterUs
                  << " us payload=" << payload << " bytes bw=" << bwGbs << " GB/s" << std::endl;

        bool bigDataCopyBackOk = true;
        if (!bigDataProfilePartial && !bigDataRemotePutOnly) {
            bigDataCopyBackOk = CopyDeviceToHost(rank, hostOutput.data(), dataCount * sizeof(int32_t),
                bigOutput, dataCount * sizeof(int32_t), "bigdata alltoall output");
        }
        bigDataCopyBackOk = CopyDeviceToHost(rank, hostDebug.data(), hostDebug.size() * sizeof(int32_t),
            debug, hostDebug.size() * sizeof(int32_t), "debug after bigdata alltoall") && bigDataCopyBackOk;
        if (fullmeshTraceEnabled) {
            const bool traceCopyOk = CopyDeviceToHost(
                rank, hostFullmeshTrace.data(), hostFullmeshTrace.size(), fullmeshTraceDevice,
                TileXR::Demo::kFullmeshTraceBytes, "fullmesh trace");
            bigDataCopyBackOk = traceCopyOk &&
                WriteFullmeshTraceBinary(rank, fullmeshTraceDir, hostFullmeshTrace) && bigDataCopyBackOk;
        }
        if (!bigDataCopyBackOk) {
            if (udmaRegistered) {
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                udmaRegistered = false;
            }
            aclrtFree(bigInput);
            aclrtFree(bigOutput);
            freeFullmeshTrace();
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        if (bigDataRemotePutOnly && !AllToAllUdmaComplete(rankSize, hostDebug)) {
            std::cerr << "[rank " << rank << "] ERROR: bigdata remote-put-only incomplete:";
            for (int peer = 0; peer < rankSize; ++peer) {
                std::cerr << " peer" << peer << "=" << hostDebug[kDebugUdmaStatusBase + peer];
            }
            std::cerr << std::endl;
            PrintAllToAllUdmaDebug(rank, rankSize, hostDebug);
            if (udmaRegistered) {
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                udmaRegistered = false;
            }
            aclrtFree(bigInput);
            aclrtFree(bigOutput);
            freeFullmeshTrace();
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
            udmaRegistered = false;
        }
        aclrtFree(bigInput);
        aclrtFree(bigOutput);
        freeFullmeshTrace();
    } else if (testType == 6) {
        // Forced UDMA alltoall (no IPC fallback). Single kernel launch loops
        // REPEAT times internally; stream sync only after all loops.
        // input/output: independent full-size GM, NOT registered.
        // udmaMem+signals: registered chunk-sized relay, reused per pass.
        void* fusedInput = nullptr;
        void* fusedOutput = nullptr;
        const size_t fusedDataBytes = dataBytes;
        if (!CheckAcl(rank, "aclrtMalloc fused input", aclrtMalloc(&fusedInput, fusedDataBytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
            !CheckAcl(rank, "aclrtMalloc fused output", aclrtMalloc(&fusedOutput, fusedDataBytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
            aclrtFree(fusedInput); aclrtFree(fusedOutput);
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        if (!CopyHostToDevice(rank, fusedInput, dataCount * sizeof(int32_t), hostData.data(), dataCount * sizeof(int32_t), "fused alltoall input")) {
            aclrtFree(fusedInput); aclrtFree(fusedOutput);
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        // Registered relay region: [udmaMem chunk | signals].
        const size_t fusedChunkBytes = chunkPlan.chunkBytesPerRank;
        const size_t fusedSignalBytes = static_cast<size_t>(rankSize) * sizeof(uint64_t);
        const size_t fusedRegBytes = chunkPlan.registeredBytes;
        if (!udmaRegistered) {
            int registerRet = TileXRUDMARegister(comm, static_cast<GM_ADDR>(registeredMemory), fusedRegBytes, &udmaHandle);
            if (registerRet != TileXR::TILEXR_SUCCESS) {
                std::cerr << "[rank " << rank << "] ERROR: fused alltoall UDMA registration failed"
                          << " ret=" << registerRet << " regBytes=" << fusedRegBytes << std::endl;
                aclrtFree(fusedInput); aclrtFree(fusedOutput);
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }
            udmaRegistered = true;
        }
        auto fusedUdmaMem = reinterpret_cast<int32_t*>(registeredMemory);
        auto fusedSignals = reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(registeredMemory) + fusedChunkBytes);
        const uint64_t fusedUdmaOffset = 0;
        const uint64_t fusedSignalOffset = fusedChunkBytes;
        std::vector<uint64_t> zeroSignals(static_cast<size_t>(rankSize), 0);
        CopyHostToDevice(rank, fusedSignals, zeroSignals.size() * sizeof(uint64_t), zeroSignals.data(), zeroSignals.size() * sizeof(uint64_t), "fused signals zero");
        PrintStatus(rank, "fused alltoall registered chunkBytes=" + std::to_string(fusedChunkBytes) +
            " regBytes=" + std::to_string(fusedRegBytes) + " passCount=" + std::to_string(chunkPlan.passCount) +
            " chunkElements=" + std::to_string(chunkPlan.chunkElements) + " repeat=" + std::to_string(allToAllRepeat));
        if (!CheckAcl(rank, "aclrtSynchronizeStream fused prime", aclrtSynchronizeStream(stream)) ||
            !DemoBarrierAll(rank, rankSize, "all ranks fused prime")) {
            if (udmaRegistered) { CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle)); udmaRegistered = false; }
            aclrtFree(fusedInput); aclrtFree(fusedOutput);
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        // Warmup: separate launches (loop=1 each) with per-iter sync+barrier.
        for (int witer = 0; witer < allToAllWarmup; ++witer) {
            if (witer == 0) PrintStatus(rank, "warmup fused all-to-all warmup=" + std::to_string(allToAllWarmup));
            launch_tilexr_udma_all_to_all_fused(
                static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(fusedInput), reinterpret_cast<GM_ADDR>(fusedOutput),
                reinterpret_cast<GM_ADDR>(fusedUdmaMem), reinterpret_cast<GM_ADDR>(fusedSignals), reinterpret_cast<GM_ADDR>(debug),
                elementsPerRank, fusedUdmaOffset, fusedSignalOffset, chunkPlan.chunkElements, chunkPlan.passCount, 1);
            if (!CheckAcl(rank, "aclrtSynchronizeStream warmup", aclrtSynchronizeStream(stream)) ||
                !DemoBarrierAll(rank, rankSize, "all ranks completed warmup")) {
                if (udmaRegistered) { CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle)); udmaRegistered = false; }
                aclrtFree(fusedInput); aclrtFree(fusedOutput);
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }
        }
        // Timed run: single launch, loop=REPEAT inside kernel, sync only at end.
        auto a2aStart = std::chrono::steady_clock::now();
        launch_tilexr_udma_all_to_all_fused(
            static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(fusedInput), reinterpret_cast<GM_ADDR>(fusedOutput),
            reinterpret_cast<GM_ADDR>(fusedUdmaMem), reinterpret_cast<GM_ADDR>(fusedSignals), reinterpret_cast<GM_ADDR>(debug),
            elementsPerRank, fusedUdmaOffset, fusedSignalOffset, chunkPlan.chunkElements, chunkPlan.passCount, static_cast<uint32_t>(allToAllRepeat));
        if (!CheckAcl(rank, "aclrtSynchronizeStream post-alltoall", aclrtSynchronizeStream(stream))) {
            if (udmaRegistered) { CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle)); udmaRegistered = false; }
            aclrtFree(fusedInput); aclrtFree(fusedOutput);
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        auto a2aEnd = std::chrono::steady_clock::now();
        {
            double a2aMs = std::chrono::duration<double, std::milli>(a2aEnd - a2aStart).count();
            double a2aPerIterUs = (allToAllRepeat > 0) ? (a2aMs * 1000.0 / static_cast<double>(allToAllRepeat)) : 0.0;
            double payloadBytes = static_cast<double>(rankSize) * static_cast<double>(elementsPerRank) * sizeof(int32_t);
            double bwGbs = (a2aPerIterUs > 0.0) ? (payloadBytes / (a2aPerIterUs * 1e3)) : 0.0;
            std::cout << "[rank " << rank << "] alltoall udma-fused " << allToAllRepeat
                      << " iters(total=" << chunkPlan.passCount << " pass/iter) total=" << a2aMs << " ms perIter=" << a2aPerIterUs
                      << " us payload=" << payloadBytes << " bytes bw=" << bwGbs << " GB/s" << std::endl;
        }
        if (!CopyDeviceToHost(rank, hostOutput.data(), dataCount * sizeof(int32_t),
                fusedOutput, dataCount * sizeof(int32_t), "fused alltoall output")) {
            if (udmaRegistered) { CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle)); udmaRegistered = false; }
            aclrtFree(fusedInput);
            aclrtFree(fusedOutput);
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
        if (udmaRegistered) { CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle)); udmaRegistered = false; }
        aclrtFree(fusedInput);
        aclrtFree(fusedOutput);
    } else if (isAllToAll) {
        if (forceAllToAllIpcFallback) {            PrintStatus(rank, std::string("skip all-to-all UDMA kernel; use ") + allToAllIpcFallbackLabel);
        } else if (chunkedStrictAllToAll) {
            for (uint32_t pass = 0; pass < chunkPlan.passCount; ++pass) {
                const int32_t chunkOffset = static_cast<int32_t>(pass) * chunkPlan.chunkElements;
                const int32_t chunkElements = std::min(
                    chunkPlan.chunkElements, elementsPerRank - chunkOffset);
                if (!CopyChunkHostToDevice(rank, input, output, rankSize, elementsPerRank,
                        chunkOffset, chunkElements, hostData, hostOutput)) {
                    if (udmaRegistered) {
                        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    }
                    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                    return 1;
                }
                if (!udmaRegistered) {
                    int registerRet =
                        TileXRUDMARegister(comm, static_cast<GM_ADDR>(registeredMemory), registeredBytes, &udmaHandle);
                    if (registerRet != TileXR::TILEXR_SUCCESS) {
                        std::cerr << "[rank " << rank << "] ERROR: strict alltoall UDMA registration failed"
                                  << " ret=" << registerRet << std::endl;
                        CheckTileXR(rank, "TileXRUDMARegister", registerRet);
                        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                        return 1;
                    }
                    udmaRegistered = true;
                }
                PrintStatus(rank, "launch all-to-all kernel pass=" + std::to_string(pass));
                if (testType == 4) {
                    launch_tilexr_udma_p2p_latency(
                        static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), chunkElements, static_cast<uint64_t>(outputOffset),
                        0, chunkElements);
                } else if (testType == 5) {
                    launch_tilexr_datacopy_latency(
                        static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), chunkElements, chunkElements);
                } else {
                    launch_tilexr_udma_all_to_all(
                        static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), chunkElements, static_cast<uint64_t>(outputOffset),
                        0, chunkElements);
                }
                if (!CheckAcl(rank, "aclrtSynchronizeStream", aclrtSynchronizeStream(stream)) ||
                    !DemoBarrierAll(rank, rankSize, "all ranks completed demo kernels")) {
                    if (udmaRegistered) {
                        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    }
                    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                    return 1;
                }
                if (!CopyDeviceToHost(rank, hostDebug.data(), hostDebug.size() * sizeof(int32_t),
                        debug, hostDebug.size() * sizeof(int32_t), "debug after alltoall udma")) {
                    if (udmaRegistered) {
                        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    }
                    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                    return 1;
                }
                if (!AllToAllUdmaComplete(rankSize, hostDebug)) {
                    std::cerr << "[rank " << rank << "] ERROR: strict alltoall UDMA CQ incomplete:" << std::endl;
                    PrintAllToAllUdmaDebug(rank, rankSize, hostDebug);
                    if (udmaRegistered) {
                        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    }
                    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                    return 1;
                }
                if (!CopyChunkDeviceToHost(rank, output, rankSize, elementsPerRank,
                        chunkOffset, chunkElements, hostOutput)) {
                    if (udmaRegistered) {
                        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    }
                    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                    return 1;
                }
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                udmaRegistered = false;
            }
        } else {
            if (!CheckAcl(rank, "aclrtSynchronizeStream pre-alltoall", aclrtSynchronizeStream(stream))) {
                if (udmaRegistered) {
                    CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                }
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }
            auto a2aStart = std::chrono::steady_clock::now();
            for (int iter = 0; iter < allToAllRepeat; ++iter) {
                if (iter == 0) {
                    PrintStatus(rank, "launch all-to-all kernel repeat=" + std::to_string(allToAllRepeat));
                }
                if (testType == 4) {
                    launch_tilexr_udma_p2p_latency(
                        static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank, static_cast<uint64_t>(outputOffset), 0,
                        elementsPerRank);
                } else if (testType == 5) {
                    launch_tilexr_datacopy_latency(
                        static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank, elementsPerRank);
                } else {
                    launch_tilexr_udma_all_to_all(
                        static_cast<uint32_t>(rankSize), stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank, static_cast<uint64_t>(outputOffset), 0,
                        elementsPerRank);
                }
                if (!syncAllToAllAtEnd &&
                    !CheckAcl(rank, "aclrtSynchronizeStream", aclrtSynchronizeStream(stream))) {
                    if (udmaRegistered) {
                        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    }
                    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                    return 1;
                }
            }
            if (syncAllToAllAtEnd &&
                !CheckAcl(rank, "aclrtSynchronizeStream post-alltoall", aclrtSynchronizeStream(stream))) {
                if (udmaRegistered) {
                    CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                }
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }
            auto a2aEnd = std::chrono::steady_clock::now();
            double a2aMs = std::chrono::duration<double, std::milli>(a2aEnd - a2aStart).count();
            double a2aPerIterUs = (allToAllRepeat > 0) ? (a2aMs * 1000.0 / static_cast<double>(allToAllRepeat)) : 0.0;
            double payloadBytes = static_cast<double>(rankSize) * static_cast<double>(elementsPerRank) * sizeof(int32_t);
            double bwGbs = (a2aPerIterUs > 0.0) ? (payloadBytes / (a2aPerIterUs * 1e3)) : 0.0;
            std::cout << "[rank " << rank << "] alltoall udma " << allToAllRepeat
                      << " iters total=" << a2aMs << " ms perIter=" << a2aPerIterUs
                      << " us payload=" << payloadBytes << " bytes bw=" << bwGbs << " GB/s" << std::endl;
        }
    } else if (testType == 3) {
        PrintStatus(rank, "launch all-reduce IPC scatter kernel");
        launch_tilexr_all_reduce_ipc_scatter(
            1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(input), reinterpret_cast<GM_ADDR>(debug),
            elementsPerRank);
    } else if (testType == 1) {
        PrintStatus(rank, "launch put-signal kernel");
        launch_tilexr_udma_put_signal(
            1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(data), reinterpret_cast<GM_ADDR>(signals),
            reinterpret_cast<GM_ADDR>(debug), elementsPerRank, kSignalValue);
    } else {
        PrintStatus(rank, "launch all-gather kernel");
        launch_tilexr_udma_all_gather(
            1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(data), reinterpret_cast<GM_ADDR>(debug),
            elementsPerRank);
    }
    if (!CheckAcl(rank, "aclrtSynchronizeStream", aclrtSynchronizeStream(stream))) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (!DemoBarrierAll(rank, rankSize, "all ranks completed demo kernels")) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    if (isAllReduce) {
        PrintStatus(rank, "launch all-reduce IPC sum kernel");
        launch_tilexr_all_reduce_ipc_sum(
            1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(debug),
            elementsPerRank);
        if (!CheckAcl(rank, "aclrtSynchronizeStream allreduce ipc sum", aclrtSynchronizeStream(stream)) ||
            !DemoBarrierAll(rank, rankSize, "all ranks completed allreduce ipc sum")) {
            if (udmaRegistered) {
                CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
            }
            Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
            return 1;
        }
    }

    if (isAllToAll && testType != 7 &&
        !CopyDeviceToHost(rank, hostDebug.data(), hostDebug.size() * sizeof(int32_t),
            debug, hostDebug.size() * sizeof(int32_t), "debug after alltoall udma")) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    bool usedIpcFallback = false;
    bool allToAllUdmaComplete =
        !isAllToAll || bigDataRemotePutOnly || bigDataProfilePartial || AllToAllUdmaComplete(rankSize, hostDebug);
    if (isAllToAll && strictAllToAllUdma && !allToAllUdmaComplete) {
        std::cerr << "[rank " << rank << "] ERROR: strict alltoall UDMA CQ incomplete:";
        for (int peer = 0; peer < rankSize; ++peer) {
            std::cerr << " peer" << peer << "=" << hostDebug[kDebugUdmaStatusBase + peer];
        }
        std::cerr << std::endl;
        PrintAllToAllUdmaDebug(rank, rankSize, hostDebug);
        if (dumpAllToAllOnStrictFail) {
            std::vector<int32_t> strictFailOutput(dataCount, 0);
            if (CopyDeviceToHost(rank, strictFailOutput.data(), dataBytes,
                    output, dataBytes, "alltoall output after strict UDMA fail")) {
                (void)ValidateAllToAllData(rank, rankSize, strictFailOutput, elementsPerRank);
            }
        }
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    if (isAllToAll && (forceAllToAllIpcFallback || !allToAllUdmaComplete)) {
        usedIpcFallback = true;
        std::cout << "[rank " << rank << "] alltoall use " << allToAllIpcFallbackLabel;
        if (forceAllToAllIpcFallback) {
            if (useAllToAllDataAsFlagIpc) {
                std::cout << " by default";
            } else if (useAllToAllPlainIpc) {
                std::cout << " by request";
            } else if (useAllToAllFusedIpc) {
                std::cout << " by request";
            } else {
                std::cout << " after UDMA registration failure";
            }
        } else {
            std::cout << " after UDMA CQ incomplete";
        }
        std::cout << ":";
        for (int peer = 0; peer < rankSize; ++peer) {
            std::cout << " peer" << peer << "=" << hostDebug[kDebugUdmaStatusBase + peer];
        }
        std::cout << std::endl;

        if (useAllToAllFusedIpc) {
            PrintStatus(rank, "alltoall fused IPC: single kernel send+flag+recv");
            if (syncAllToAllAtEnd) {
                for (int iter = 0; iter < allToAllRepeat; ++iter) {
                    launch_tilexr_all_to_all_fused_ipc(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(input),
                        reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(debug),
                        elementsPerRank, iter + 1);
                }
                if (!CheckAcl(rank, "aclrtSynchronizeStream alltoall fused ipc", aclrtSynchronizeStream(stream)) ||
                    !DemoBarrierAll(rank, rankSize, "all ranks completed alltoall fused ipc")) {
                    if (udmaRegistered) {
                        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    }
                    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                    return 1;
                }
            } else {
                for (int iter = 0; iter < allToAllRepeat; ++iter) {
                    launch_tilexr_all_to_all_fused_ipc(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(input),
                        reinterpret_cast<GM_ADDR>(output), reinterpret_cast<GM_ADDR>(debug),
                        elementsPerRank, iter + 1);
                    if (!CheckAcl(rank, "aclrtSynchronizeStream alltoall fused ipc", aclrtSynchronizeStream(stream)) ||
                        !DemoBarrierAll(rank, rankSize, "all ranks completed alltoall fused ipc")) {
                        if (udmaRegistered) {
                            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                        }
                        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                        return 1;
                    }
                }
            }
        } else if (syncAllToAllAtEnd) {
            for (int iter = 0; iter < allToAllRepeat; ++iter) {
                if (useAllToAllPlainIpc) {
                    launch_tilexr_all_to_all_plain_ipc_scatter(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(input),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank);
                } else {
                    launch_tilexr_all_to_all_ipc_scatter(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(input),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank);
                }
            }
            if (!CheckAcl(rank, "aclrtSynchronizeStream alltoall ipc scatter", aclrtSynchronizeStream(stream)) ||
                !DemoBarrierAll(rank, rankSize, "all ranks completed alltoall ipc scatter")) {
                if (udmaRegistered) {
                    CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                }
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }

            for (int iter = 0; iter < allToAllRepeat; ++iter) {
                if (useAllToAllPlainIpc) {
                    launch_tilexr_all_to_all_plain_ipc_gather(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank);
                } else {
                    launch_tilexr_all_to_all_ipc_gather(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank);
                }
            }
            if (!CheckAcl(rank, "aclrtSynchronizeStream alltoall ipc gather", aclrtSynchronizeStream(stream)) ||
                !DemoBarrierAll(rank, rankSize, "all ranks completed alltoall ipc gather")) {
                if (udmaRegistered) {
                    CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                }
                Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                return 1;
            }
        } else {
            for (int iter = 0; iter < allToAllRepeat; ++iter) {
                if (useAllToAllPlainIpc) {
                    launch_tilexr_all_to_all_plain_ipc_scatter(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(input),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank);
                } else {
                    launch_tilexr_all_to_all_ipc_scatter(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(input),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank);
                }
                if (!CheckAcl(rank, "aclrtSynchronizeStream alltoall ipc scatter", aclrtSynchronizeStream(stream)) ||
                    !DemoBarrierAll(rank, rankSize, "all ranks completed alltoall ipc scatter")) {
                    if (udmaRegistered) {
                        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    }
                    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                    return 1;
                }

                if (useAllToAllPlainIpc) {
                    launch_tilexr_all_to_all_plain_ipc_gather(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank);
                } else {
                    launch_tilexr_all_to_all_ipc_gather(
                        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(output),
                        reinterpret_cast<GM_ADDR>(debug), elementsPerRank);
                }
                if (!CheckAcl(rank, "aclrtSynchronizeStream alltoall ipc gather", aclrtSynchronizeStream(stream)) ||
                    !DemoBarrierAll(rank, rankSize, "all ranks completed alltoall ipc gather")) {
                    if (udmaRegistered) {
                        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
                    }
                    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
                    return 1;
                }
            }
        }
    }

    bool copyBackOk = true;
    if (!chunkedStrictAllToAll && testType != 7) {
        copyBackOk = CopyDeviceToHost(rank, hostData.data(), dataCount * sizeof(int32_t),
            data, dataCount * sizeof(int32_t), "data");
    }
    if (hasOutput && !chunkedStrictAllToAll && testType != 6 && testType != 7) {
        const char* outputName = isAllToAll ? "alltoall output" : "allreduce output";
        copyBackOk = CopyDeviceToHost(rank, hostOutput.data(), dataCount * sizeof(int32_t),
            output, dataCount * sizeof(int32_t), outputName) && copyBackOk;
    }
    bool signalsCopyBackOk = true;
    if (testType != 7) {
        signalsCopyBackOk = CopyDeviceToHost(rank, hostSignals.data(), hostSignals.size() * sizeof(uint64_t),
            signals, hostSignals.size() * sizeof(uint64_t), "signals");
    }
    bool debugCopyBackOk = CopyDeviceToHost(rank, hostDebug.data(), hostDebug.size() * sizeof(int32_t),
        debug, hostDebug.size() * sizeof(int32_t), "debug");
    if (!copyBackOk || !signalsCopyBackOk || !debugCopyBackOk) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    std::cout << "[rank " << rank << "] debug words:";
    for (size_t i = 0; i < std::min<size_t>(10, hostDebug.size()); ++i) {
        std::cout << " d" << i << "=" << hostDebug[i];
    }
    std::cout << std::endl;
    if (isAllToAll) {
        PrintAllToAllUdmaDebug(rank, rankSize, hostDebug);
    }
    if (usedIpcFallback) {
        std::cout << "[rank " << rank << "] alltoall IPC fallback completed"
                  << " scatter=" << hostDebug[kDebugIpcScatter]
                  << " gather=" << hostDebug[kDebugIpcGather] << std::endl;
    }
    if (isAllReduce) {
        std::cout << "[rank " << rank << "] allreduce IPC completed"
                  << " scatter=" << hostDebug[kDebugAllReduceScatter]
                  << " sum=" << hostDebug[kDebugAllReduceSum] << std::endl;
    }

    bool ok = false;
    if (bigDataRemotePutOnly) {
        PrintStatus(rank, "skip result validation for bigdata remote-put-only profile");
        ok = true;
    } else if (bigDataProfilePartial) {
        PrintStatus(rank, "skip result validation for bigdata profile stage=" +
            std::to_string(bigDataProfileStage));
        ok = true;
    } else if (isAllToAll) {
        ok = ValidateAllToAllData(rank, rankSize, hostOutput, elementsPerRank);
    } else if (isAllReduce) {
        ok = ValidateAllReduceData(rank, rankSize, hostOutput, elementsPerRank);
    } else {
        ok = ValidateData(rank, rankSize, hostData, elementsPerRank);
    }
    if (testType == 1) {
        ok = ValidateSignals(rank, rankSize, hostSignals) && ok;
    }

    if (udmaRegistered) {
        CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        udmaRegistered = false;
    }
    Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
    if (!ok) {
        std::cerr << "[rank " << rank << "] TileXR UDMA demo failed" << std::endl;
        return 1;
    }
    std::cout << "[rank " << rank << "] TileXR UDMA demo success" << std::endl;
    return 0;
}
