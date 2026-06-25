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
#include <fstream>
#include <iostream>
#include <sstream>
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
#include "tilexr_udma_p2p_perf_config.h"

extern void launch_tilexr_udma_all_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR debug, int32_t elementsPerRank);
extern void launch_tilexr_udma_put_signal(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR signals, GM_ADDR debug,
    int32_t elementsPerRank, uint64_t signal);
extern void launch_tilexr_udma_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern, int32_t traffic);
extern void launch_tilexr_memory_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern, int32_t traffic);
extern void launch_tilexr_data_as_flag_p2p_perf(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR src, GM_ADDR dst, GM_ADDR debug,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset, uint32_t bytes, uint32_t pattern, int32_t traffic);

namespace {
constexpr int32_t kDefaultElementsPerRank = 16;
constexpr uint64_t kSignalValue = 1000;
constexpr size_t kDebugWords = 16;
constexpr int kDefaultCommPort = 10067;
constexpr int kDemoBarrierPortOffset = 97;
constexpr size_t kUdmaRegistrationAlignment = 2 * 1024 * 1024;
constexpr int kConnectRetryCount = 500;
constexpr int kConnectRetrySleepMs = 10;
constexpr size_t kP2PDebugWords = 32;

struct BarrierEndpoint {
    uint16_t port;
};

int GetEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    return value == nullptr ? defaultValue : std::atoi(value);
}

const char* GetEnvString(const char* name, const char* defaultValue)
{
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? defaultValue : value;
}

bool GetEnvFlag(const char* name, bool defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    return std::atoi(value) != 0;
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

uint64_t RoundUpToAlignment(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

bool WriteTextFile(const std::string& path, const std::string& text)
{
    std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
    if (!out) {
        return false;
    }
    out << text;
    return static_cast<bool>(out);
}

bool ReadP2PStatusFile(const std::string& path, TileXR::Demo::P2PRankStatus* status)
{
    std::ifstream in(path.c_str());
    if (!in) {
        return false;
    }
    in >> status->status >> status->errors;
    if (!in) {
        return false;
    }
    if (in >> status->elapsedMs) {
        return true;
    }
    return in.eof();
}

bool AppendP2PPerfCsvRow(const TileXR::Demo::P2PPerfOptions& options, const TileXR::Demo::P2PPerfRow& row)
{
    if (options.csvPath.empty()) {
        return true;
    }
    std::ifstream check(options.csvPath.c_str());
    bool exists = static_cast<bool>(check);
    check.close();
    std::ofstream out(options.csvPath.c_str(), std::ios::out | std::ios::app);
    if (!out) {
        return false;
    }
    if (!exists) {
        out << TileXR::Demo::P2PPerfCsvHeader();
    }
    out << TileXR::Demo::FormatP2PPerfCsvRow(row);
    return static_cast<bool>(out);
}

bool CheckPeerMemsReady(int rank, int rankSize, const TileXR::CommArgs& args)
{
    for (int peer = 0; peer < rankSize; ++peer) {
        if (args.peerMems[peer] == nullptr) {
            std::cerr << "[rank " << rank << "] ERROR: peerMems[" << peer << "] is null" << std::endl;
            return false;
        }
    }
    return true;
}

bool IsP2PSourceRank(int rank, const TileXR::Demo::P2PPerfOptions& options)
{
    return options.traffic == TileXR::Demo::P2PTraffic::BiDir ?
        (rank == options.srcRank || rank == options.dstRank) : rank == options.srcRank;
}

bool IsP2PReceiveRank(int rank, const TileXR::Demo::P2PPerfOptions& options)
{
    return options.traffic == TileXR::Demo::P2PTraffic::BiDir ?
        (rank == options.srcRank || rank == options.dstRank) : rank == options.dstRank;
}

uint32_t FoldP2PDebugStatus(const std::vector<uint32_t>& debug, uint32_t blockDim)
{
    uint32_t status = debug.size() > 5 ? debug[5] : 0xffffffffu;
    const uint32_t limit = std::min<uint32_t>(blockDim, 16U);
    for (uint32_t i = 0; i < limit && 8U + i < debug.size(); ++i) {
        status |= debug[8U + i];
    }
    return status;
}

void PrintP2PDebugSummary(int rank, const std::vector<uint32_t>& debug, uint32_t blockDim)
{
    std::ostringstream out;
    out << "p2p debug:"
        << " magic=0x" << std::hex << (debug.size() > 0 ? debug[0] : 0)
        << std::dec
        << " kernelRank=" << (debug.size() > 1 ? debug[1] : 0)
        << " enabled=" << (debug.size() > 2 ? debug[2] : 0)
        << " bytes=" << (debug.size() > 3 ? debug[3] : 0)
        << " status=" << (debug.size() > 5 ? debug[5] : 0xffffffffu)
        << " blockNum=" << (debug.size() > 6 ? debug[6] : 0)
        << " qpNum=" << (debug.size() > 7 ? debug[7] : 0)
        << " jettyCount=" << (debug.size() > 15 ? debug[15] : 0);
    const uint32_t limit = std::min<uint32_t>(blockDim, 8U);
    for (uint32_t i = 0; i < limit; ++i) {
        out << " b" << i
            << "{status=" << (8U + i < debug.size() ? debug[8U + i] : 0xffffffffu)
            << ",offset=" << (16U + i < debug.size() ? debug[16U + i] : 0)
            << ",bytes=" << (24U + i < debug.size() ? debug[24U + i] : 0)
            << "}";
    }
    PrintStatus(rank, out.str());
}

bool ClearLocalPeerWindow(
    int rank, const TileXR::CommArgs& commArgsHost, uint64_t offset, uint64_t bytes, const std::string& name)
{
    if (bytes == 0) {
        return true;
    }
    void* localWindow = reinterpret_cast<void*>(commArgsHost.peerMems[rank] + offset);
    return CheckAcl(rank, "aclrtMemset " + name, aclrtMemset(localWindow, static_cast<size_t>(bytes), 0,
        static_cast<size_t>(bytes)));
}

void LaunchP2PKernel(
    aclrtStream stream, GM_ADDR commArgsDev, GM_ADDR srcDev, GM_ADDR dstDev, GM_ADDR debugDev,
    const TileXR::Demo::P2PPerfOptions& options, uint64_t dstOffset, uint32_t transferBytes, uint32_t pattern)
{
    const int32_t traffic = options.traffic == TileXR::Demo::P2PTraffic::BiDir ? 1 : 0;
    if (options.transport == TileXR::Demo::P2PTransport::Memory) {
        launch_tilexr_memory_p2p_perf(options.blockDim, stream, commArgsDev, srcDev, debugDev,
            options.srcRank, options.dstRank, dstOffset, transferBytes, pattern, traffic);
        return;
    }
    if (options.transport == TileXR::Demo::P2PTransport::DataAsFlag) {
        launch_tilexr_data_as_flag_p2p_perf(options.blockDim, stream, commArgsDev, srcDev, dstDev, debugDev,
            options.srcRank, options.dstRank, dstOffset, transferBytes, pattern, traffic);
        return;
    }
    launch_tilexr_udma_p2p_perf(options.blockDim, stream, commArgsDev, srcDev, debugDev,
        options.srcRank, options.dstRank, dstOffset, transferBytes, pattern, traffic);
}

bool RunP2PPerfMode(
    int rank, int rankSize, TileXRCommPtr comm, const TileXR::CommArgs& commArgsHost,
    GM_ADDR commArgsDev, aclrtStream stream,
    const TileXR::Demo::P2PPerfOptions& options)
{
    std::string error;
    if (!TileXR::Demo::ValidateP2PPerfOptions(options, rankSize, &error)) {
        std::cerr << "[rank " << rank << "] ERROR: " << error << std::endl;
        return false;
    }

    void* registeredMemory = nullptr;
    uint32_t* debug = nullptr;
    TileXRUDMAMemHandle udmaHandle = 0;
    bool udmaRegistered = false;
    aclrtEvent startEvent = nullptr;
    aclrtEvent stopEvent = nullptr;

    const bool useMemoryTransport = options.transport == TileXR::Demo::P2PTransport::Memory;
    const bool useDataAsFlagTransport = options.transport == TileXR::Demo::P2PTransport::DataAsFlag;
    const bool useIpcTransport = useMemoryTransport || useDataAsFlagTransport;
    const uint64_t srcOffset = 0;
    const uint64_t dstWindowBytes =
        TileXR::Demo::P2PTransportWindowBytes(options.transport, options.maxBytes, options.blockDim);
    const uint64_t dstOffset = useIpcTransport ? TileXR::IPC_DATA_OFFSET : dstWindowBytes;
    const uint64_t localDstOffset = dstWindowBytes;
    const uint64_t debugOffset = useIpcTransport ? localDstOffset + dstWindowBytes : dstOffset + dstWindowBytes;
    const uint64_t payloadBytes = debugOffset + kP2PDebugWords * sizeof(uint32_t);
    const uint64_t registeredPayloadBytes = payloadBytes;
    const uint64_t registeredBytes = RoundUpToAlignment(registeredPayloadBytes, kUdmaRegistrationAlignment);

    auto cleanup = [&]() {
        if (startEvent != nullptr) {
            CheckAcl(rank, "aclrtDestroyEvent start", aclrtDestroyEvent(startEvent));
        }
        if (stopEvent != nullptr) {
            CheckAcl(rank, "aclrtDestroyEvent stop", aclrtDestroyEvent(stopEvent));
        }
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        if (registeredMemory != nullptr) {
            PrintStatus(rank, "aclrtFree p2p registered memory");
            aclrtFree(registeredMemory);
        }
    };

    const std::string memoryName = useMemoryTransport ? "p2p memory local scratch" : "p2p registered memory";
    if (!CheckAcl(rank, "aclrtMalloc " + memoryName,
            aclrtMalloc(&registeredMemory, registeredBytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
        cleanup();
        return false;
    }
    auto base = static_cast<uint8_t*>(registeredMemory);
    auto srcDev = base + srcOffset;
    auto dstDev = base + localDstOffset;
    debug = reinterpret_cast<uint32_t*>(base + debugOffset);

    if (!useIpcTransport) {
        if (!CheckTileXR(rank, "TileXRUDMARegister p2p",
                TileXRUDMARegister(comm, static_cast<GM_ADDR>(registeredMemory), registeredBytes, &udmaHandle))) {
            cleanup();
            return false;
        }
        udmaRegistered = true;
    }
    PrintStatus(rank, std::string("p2p transport=") + TileXR::Demo::P2PTransportName(options.transport) +
        " traffic=" + TileXR::Demo::P2PTrafficName(options.traffic) +
        " blockDim=" + std::to_string(options.blockDim) +
        " memory base=" + std::to_string(reinterpret_cast<uintptr_t>(registeredMemory)) +
        " bytes=" + std::to_string(registeredBytes) +
        " srcOffset=" + std::to_string(srcOffset) +
        " dstOffset=" + std::to_string(dstOffset) +
        " debugOffset=" + std::to_string(debugOffset));

    if (!CheckAcl(rank, "aclrtCreateEvent start", aclrtCreateEvent(&startEvent)) ||
        !CheckAcl(rank, "aclrtCreateEvent stop", aclrtCreateEvent(&stopEvent))) {
        cleanup();
        return false;
    }

    bool ok = true;
    for (uint64_t bytes : TileXR::Demo::BuildP2PPerfSizeSweep(options)) {
        const uint32_t transferBytes = static_cast<uint32_t>(bytes);
        const uint32_t pattern = TileXR::Demo::P2PPattern(options.srcRank, options.dstRank, bytes);
        const uint64_t transferWindowBytes =
            TileXR::Demo::P2PTransportWindowBytes(options.transport, bytes, options.blockDim);
        std::vector<uint8_t> hostSrc(static_cast<size_t>(transferWindowBytes), 0);
        std::vector<uint8_t> hostDst(static_cast<size_t>(transferWindowBytes), 0);
        std::vector<uint32_t> hostDebug(kP2PDebugWords, 0);
        TileXR::Demo::FillP2PPattern(hostSrc, pattern);

        const bool initSrc = true;
        const bool initDst = true;
        if ((initSrc && !CopyHostToDevice(rank, srcDev, static_cast<size_t>(transferWindowBytes), hostSrc.data(),
                static_cast<size_t>(transferWindowBytes), "p2p src")) ||
            (initDst && !CopyHostToDevice(rank, dstDev, static_cast<size_t>(transferWindowBytes), hostDst.data(),
                static_cast<size_t>(transferWindowBytes), "p2p dst")) ||
            !CopyHostToDevice(rank, debug, hostDebug.size() * sizeof(uint32_t), hostDebug.data(),
                hostDebug.size() * sizeof(uint32_t), "p2p debug")) {
            ok = false;
            break;
        }
        if (useIpcTransport && IsP2PReceiveRank(rank, options)) {
            const uint64_t clearBytes = useDataAsFlagTransport ?
                TileXR::Demo::P2PTransportWindowBytes(options.transport, bytes) : bytes;
            if (!ClearLocalPeerWindow(rank, commArgsHost, dstOffset, clearBytes, "p2p ipc dst window")) {
                ok = false;
                break;
            }
        }
        if (!DemoBarrierAll(rank, rankSize, "p2p initialized bytes=" + std::to_string(bytes))) {
            ok = false;
            break;
        }

        for (int i = 0; i < options.warmupIters; ++i) {
            if (useDataAsFlagTransport && IsP2PReceiveRank(rank, options) &&
                !ClearLocalPeerWindow(rank, commArgsHost, dstOffset,
                    TileXR::Demo::P2PTransportWindowBytes(options.transport, bytes), "p2p data_as_flag warmup window")) {
                ok = false;
                break;
            }
            if (useDataAsFlagTransport &&
                !DemoBarrierAll(rank, rankSize,
                    "p2p data_as_flag warmup clear bytes=" + std::to_string(bytes) +
                    " iter=" + std::to_string(i))) {
                ok = false;
                break;
            }
            LaunchP2PKernel(stream, commArgsDev, reinterpret_cast<GM_ADDR>(srcDev), reinterpret_cast<GM_ADDR>(dstDev),
                reinterpret_cast<GM_ADDR>(debug), options, dstOffset, transferBytes, pattern);
            if (!CheckAcl(rank, "aclrtSynchronizeStream p2p warmup", aclrtSynchronizeStream(stream))) {
                ok = false;
                break;
            }
        }
        if (!ok || !DemoBarrierAll(rank, rankSize, "p2p warmup done bytes=" + std::to_string(bytes))) {
            ok = false;
            break;
        }

        if (useDataAsFlagTransport && IsP2PReceiveRank(rank, options) &&
            !ClearLocalPeerWindow(rank, commArgsHost, dstOffset,
                TileXR::Demo::P2PTransportWindowBytes(options.transport, bytes), "p2p data_as_flag measured window")) {
            ok = false;
            break;
        }
        if (useDataAsFlagTransport &&
            !DemoBarrierAll(rank, rankSize,
                "p2p data_as_flag measured clear bytes=" + std::to_string(bytes))) {
            ok = false;
            break;
        }

        if (!CheckAcl(rank, "aclrtRecordEvent start", aclrtRecordEvent(startEvent, stream))) {
            ok = false;
            break;
        }
        for (int i = 0; i < options.iters; ++i) {
            LaunchP2PKernel(stream, commArgsDev, reinterpret_cast<GM_ADDR>(srcDev), reinterpret_cast<GM_ADDR>(dstDev),
                reinterpret_cast<GM_ADDR>(debug), options, dstOffset, transferBytes, pattern);
        }
        if (!ok ||
            !CheckAcl(rank, "aclrtRecordEvent stop", aclrtRecordEvent(stopEvent, stream)) ||
            !CheckAcl(rank, "aclrtSynchronizeStream p2p measured", aclrtSynchronizeStream(stream))) {
            ok = false;
            break;
        }

        float elapsedMs = 0.0f;
        if (!CheckAcl(rank, "aclrtEventElapsedTime p2p", aclrtEventElapsedTime(&elapsedMs, startEvent, stopEvent))) {
            ok = false;
            break;
        }
        if (!DemoBarrierAll(rank, rankSize, "p2p measured done bytes=" + std::to_string(bytes))) {
            ok = false;
            break;
        }

        uint64_t errors = 0;
        if (IsP2PReceiveRank(rank, options) && options.check) {
            const void* validateSrc = useMemoryTransport ?
                reinterpret_cast<void*>(commArgsHost.peerMems[rank] + TileXR::IPC_DATA_OFFSET) :
                static_cast<const void*>(dstDev);
            if (!CopyDeviceToHost(rank, hostDst.data(), static_cast<size_t>(transferWindowBytes), validateSrc,
                    static_cast<size_t>(transferWindowBytes), "p2p dst")) {
                ok = false;
                break;
            }
            errors = TileXR::Demo::CountP2PTransportMismatches(
                hostDst, pattern, bytes, options.transport, options.blockDim);
        }
        if (IsP2PSourceRank(rank, options) || useDataAsFlagTransport) {
            if (!CopyDeviceToHost(rank, hostDebug.data(), hostDebug.size() * sizeof(uint32_t), debug,
                    hostDebug.size() * sizeof(uint32_t), "p2p debug")) {
                ok = false;
                break;
            }
        }

        TileXR::Demo::P2PRankStatus localStatus;
        localStatus.status = (IsP2PSourceRank(rank, options) || useDataAsFlagTransport) ?
            FoldP2PDebugStatus(hostDebug, options.blockDim) : 0;
        localStatus.errors = IsP2PReceiveRank(rank, options) ? errors : 0;
        localStatus.elapsedMs = elapsedMs;
        if (IsP2PSourceRank(rank, options) &&
            (GetEnvFlag("TILEXR_P2P_DEBUG_SUMMARY", false) || localStatus.status != 0 || errors != 0)) {
            PrintP2PDebugSummary(rank, hostDebug, options.blockDim);
        }
        if (!options.logDir.empty()) {
            std::string statusPath = options.logDir + "/p2p_rank_" + std::to_string(rank) +
                "_" + std::to_string(bytes) + ".status";
            std::ostringstream statusText;
            statusText << localStatus.status << ' ' << localStatus.errors << ' ' << localStatus.elapsedMs << '\n';
            if (!WriteTextFile(statusPath, statusText.str())) {
                std::cerr << "[rank " << rank << "] ERROR: failed to write " << statusPath << std::endl;
                ok = false;
                break;
            }
        }
        if (!DemoBarrierAll(rank, rankSize, "p2p status files bytes=" + std::to_string(bytes))) {
            ok = false;
            break;
        }

        if (rank == 0) {
            TileXR::Demo::P2PRankStatus srcStatus = localStatus;
            TileXR::Demo::P2PRankStatus dstStatus = localStatus;
            if (options.srcRank != 0) {
                std::string srcPath = options.logDir + "/p2p_rank_" + std::to_string(options.srcRank) +
                    "_" + std::to_string(bytes) + ".status";
                if (!ReadP2PStatusFile(srcPath, &srcStatus)) {
                    std::cerr << "[rank " << rank << "] ERROR: failed to read " << srcPath << std::endl;
                    ok = false;
                    break;
                }
            }
            if (options.dstRank != 0) {
                std::string dstPath = options.logDir + "/p2p_rank_" + std::to_string(options.dstRank) +
                    "_" + std::to_string(bytes) + ".status";
                if (!ReadP2PStatusFile(dstPath, &dstStatus)) {
                    std::cerr << "[rank " << rank << "] ERROR: failed to read " << dstPath << std::endl;
                    ok = false;
                    break;
                }
            }
            TileXR::Demo::P2PPerfRow row =
                TileXR::Demo::BuildP2PPerfRow(options, rankSize, bytes, srcStatus, dstStatus);
            if (!AppendP2PPerfCsvRow(options, row)) {
                std::cerr << "[rank " << rank << "] ERROR: failed to append CSV " << options.csvPath << std::endl;
                ok = false;
                break;
            }
            std::cout << "[rank " << rank << "] p2p row: "
                      << TileXR::Demo::FormatP2PPerfCsvRow(row);
            if (row.status != 0 || row.errors != 0) {
                ok = false;
            }
        }
        if (!DemoBarrierAll(rank, rankSize, "p2p csv written bytes=" + std::to_string(bytes))) {
            ok = false;
            break;
        }
    }

    cleanup();
    return ok;
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
    TileXR::Demo::P2PPerfOptions p2pOptions;
    if (testType == 4) {
        p2pOptions.srcRank = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_P2P_SRC_RANK", 0);
        p2pOptions.dstRank = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_P2P_DST_RANK", 1);
        p2pOptions.minBytes = argc > argIndex ? std::strtoull(argv[argIndex++], nullptr, 10) :
            static_cast<uint64_t>(GetEnvInt("TILEXR_P2P_MIN_BYTES", 4096));
        p2pOptions.maxBytes = argc > argIndex ? std::strtoull(argv[argIndex++], nullptr, 10) :
            static_cast<uint64_t>(GetEnvInt("TILEXR_P2P_MAX_BYTES", 4096));
        p2pOptions.stepFactor = argc > argIndex ? std::strtoull(argv[argIndex++], nullptr, 10) :
            static_cast<uint64_t>(GetEnvInt("TILEXR_P2P_STEP_FACTOR", 2));
        p2pOptions.iters = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_P2P_ITERS", 100);
        p2pOptions.warmupIters = argc > argIndex ? std::atoi(argv[argIndex++]) :
            GetEnvInt("TILEXR_P2P_WARMUP_ITERS", 10);
        p2pOptions.check = (argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_P2P_CHECK", 1)) != 0;
        p2pOptions.csvPath = argc > argIndex ? argv[argIndex++] : GetEnvString("TILEXR_P2P_CSV", "");
        p2pOptions.logDir = argc > argIndex ? argv[argIndex++] : GetEnvString("TILEXR_P2P_LOG_DIR", "");
        std::string transportName = argc > argIndex ? argv[argIndex++] : GetEnvString("TILEXR_P2P_TRANSPORT", "direct_urma");
        p2pOptions.transport = TileXR::Demo::ParseP2PTransport(transportName);
        p2pOptions.blockDim = argc > argIndex ? static_cast<uint32_t>(std::strtoul(argv[argIndex++], nullptr, 10)) :
            static_cast<uint32_t>(GetEnvInt("TILEXR_P2P_BLOCK_DIM", 1));
        std::string trafficName = argc > argIndex ? argv[argIndex++] : GetEnvString("TILEXR_P2P_TRAFFIC", "unidir");
        p2pOptions.traffic = TileXR::Demo::ParseP2PTraffic(trafficName);
    }
    int deviceId = GetDeviceIdFromEnv(rank, npuCount, firstNpu);

    std::cout << "========================================" << std::endl;
    std::cout << "  TileXR UDMA Communication Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[rank " << rank << "] argv: rankSize=" << rankSize << " rank=" << rank
              << " testType=" << testType << " elementsPerRank=" << elementsPerRank
              << " npuCount=" << npuCount << " firstNpu=" << firstNpu << std::endl;
    if (testType == 4) {
        std::cout << "[rank " << rank << "] p2p: src=" << p2pOptions.srcRank
                  << " dst=" << p2pOptions.dstRank
                  << " minBytes=" << p2pOptions.minBytes
                  << " maxBytes=" << p2pOptions.maxBytes
                  << " stepFactor=" << p2pOptions.stepFactor
                  << " iters=" << p2pOptions.iters
                  << " warmupIters=" << p2pOptions.warmupIters
                  << " check=" << (p2pOptions.check ? 1 : 0)
                  << " transport=" << TileXR::Demo::P2PTransportName(p2pOptions.transport)
                  << " blockDim=" << p2pOptions.blockDim
                  << " traffic=" << TileXR::Demo::P2PTrafficName(p2pOptions.traffic)
                  << " csv=" << p2pOptions.csvPath
                  << " logDir=" << p2pOptions.logDir << std::endl;
    }
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

    if (testType == 4 &&
        (p2pOptions.transport == TileXR::Demo::P2PTransport::Memory ||
            p2pOptions.transport == TileXR::Demo::P2PTransport::DataAsFlag) &&
        !CheckPeerMemsReady(rank, rankSize, *commArgsHost)) {
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    if ((testType != 4 ||
            p2pOptions.transport == TileXR::Demo::P2PTransport::DirectUrma) &&
        ((commArgsHost->extraFlag & TileXR::ExtraFlag::UDMA) == 0 || commArgsHost->udmaInfoPtr == nullptr)) {
        std::cerr << "[rank " << rank << "] ERROR: TileXR UDMA is not enabled. "
                  << "Check A5/Ascend950 hardware support, CANN/driver setup, and LD_LIBRARY_PATH." << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    if (testType == 4) {
        bool ok = RunP2PPerfMode(rank, rankSize, comm, *commArgsHost, commArgsDev, stream, p2pOptions);
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        if (!ok) {
            std::cerr << "[rank " << rank << "] TileXR UDMA P2P perf failed" << std::endl;
            return 1;
        }
        std::cout << "[rank " << rank << "] TileXR UDMA P2P perf success" << std::endl;
        return 0;
    }

    size_t dataCount = static_cast<size_t>(rankSize) * elementsPerRank;
    size_t dataBytes = dataCount * sizeof(int32_t);
    size_t signalBytes = static_cast<size_t>(rankSize) * sizeof(uint64_t);
    size_t signalOffset = dataBytes;
    size_t payloadBytes = dataBytes + signalBytes;
    size_t registeredBytes = ((payloadBytes + kUdmaRegistrationAlignment - 1) / kUdmaRegistrationAlignment) *
        kUdmaRegistrationAlignment;
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
    std::fill(hostData.begin() + static_cast<size_t>(rank) * elementsPerRank,
              hostData.begin() + static_cast<size_t>(rank + 1) * elementsPerRank,
              1000 + rank);
    std::vector<uint64_t> hostSignals(static_cast<size_t>(rankSize), 0);
    std::vector<int32_t> hostDebug(kDebugWords, 0);

    if (!CopyHostToDevice(rank, data, dataCount * sizeof(int32_t),
            hostData.data(), dataCount * sizeof(int32_t), "data") ||
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

    PrintStatus(rank, testType == 1 ? "launch put-signal kernel" : "launch all-gather kernel");
    if (testType == 1) {
        launch_tilexr_udma_put_signal(
            1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(data), reinterpret_cast<GM_ADDR>(signals),
            reinterpret_cast<GM_ADDR>(debug), elementsPerRank, kSignalValue);
    } else {
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

    if (!CopyDeviceToHost(rank, hostData.data(), dataCount * sizeof(int32_t),
            data, dataCount * sizeof(int32_t), "data") ||
        !CopyDeviceToHost(rank, hostSignals.data(), hostSignals.size() * sizeof(uint64_t),
            signals, hostSignals.size() * sizeof(uint64_t), "signals") ||
        !CopyDeviceToHost(rank, hostDebug.data(), hostDebug.size() * sizeof(int32_t),
            debug, hostDebug.size() * sizeof(int32_t), "debug")) {
        if (udmaRegistered) {
            CheckTileXR(rank, "TileXRUDMAUnregister", TileXRUDMAUnregister(comm, udmaHandle));
        }
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    std::cout << "[rank " << rank << "] debug words:";
    for (size_t i = 0; i < std::min<size_t>(5, hostDebug.size()); ++i) {
        std::cout << " d" << i << "=" << hostDebug[i];
    }
    std::cout << std::endl;

    bool ok = ValidateData(rank, rankSize, hostData, elementsPerRank);
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
