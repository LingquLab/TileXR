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

extern void launch_tilexr_udma_all_gather(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR debug, int32_t elementsPerRank);
extern void launch_tilexr_udma_put_signal(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR signals, GM_ADDR debug,
    int32_t elementsPerRank, uint64_t signal);
extern void launch_tilexr_udma_slot_signal_get_probe(
    uint32_t blockDim, void* stream, GM_ADDR commArgs, GM_ADDR data, GM_ADDR signals, GM_ADDR debug,
    int32_t elementsPerRank, uint64_t signal);

namespace {
constexpr int32_t kDefaultElementsPerRank = 16;
constexpr uint64_t kSignalValue = 1000;
constexpr size_t kDebugWords = 16;
constexpr int kDefaultCommPort = 10067;
constexpr int kDemoBarrierPortOffset = 97;
constexpr size_t kUdmaRegistrationAlignment = 2 * 1024 * 1024;
constexpr int kConnectRetryCount = 500;
constexpr int kConnectRetrySleepMs = 10;

struct BarrierEndpoint {
    std::string host;
    uint16_t port;
};

int GetEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? defaultValue : std::atoi(value);
}

bool EnvEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && (std::string(value) == "1" || std::string(value) == "true" ||
        std::string(value) == "TRUE" || std::string(value) == "on" || std::string(value) == "ON");
}

std::vector<int> ParseDeviceList(const char* devices)
{
    std::vector<int> parsed;
    if (devices == nullptr || devices[0] == '\0') {
        return parsed;
    }
    std::string list(devices);
    size_t start = 0;
    while (start <= list.size()) {
        const size_t comma = list.find(',', start);
        const size_t end = comma == std::string::npos ? list.size() : comma;
        if (end > start) {
            parsed.push_back(std::atoi(list.substr(start, end - start).c_str()));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return parsed;
}

int GetDeviceIdFromEnv(int rank, int npuCount, int firstNpu)
{
    const std::vector<int> devices = ParseDeviceList(std::getenv("TILEXR_DEMO_DEVICES"));
    if (!devices.empty()) {
        return devices[rank % static_cast<int>(devices.size())];
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
    const char* commId = std::getenv("TILEXR_COMM_ID");
    if (commId != nullptr && commId[0] != '\0') {
        std::string value(commId);
        size_t colon = value.rfind(':');
        if (colon != std::string::npos) {
            host = value.substr(0, colon);
            if (colon + 1 < value.size()) {
                basePort = std::atoi(value.c_str() + colon + 1);
            }
        }
    }
    const char* barrierAddr = std::getenv("TILEXR_DEMO_BARRIER_ADDR");
    if (barrierAddr != nullptr && barrierAddr[0] != '\0') {
        std::string value(barrierAddr);
        size_t colon = value.rfind(':');
        if (colon == std::string::npos) {
            host = value;
        } else {
            host = value.substr(0, colon);
            if (colon + 1 < value.size()) {
                basePort = std::atoi(value.c_str() + colon + 1) - kDemoBarrierPortOffset;
            }
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
    PrintStatus(rank, "demo tcp barrier begin: " + step + " host=" + endpoint.host +
        " port=" + std::to_string(endpoint.port));
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
    int testType = argc > argIndex ? std::atoi(argv[argIndex++]) : GetEnvInt("TILEXR_DEMO_TEST_TYPE", 0);
    int32_t elementsPerRank = argc > argIndex ? std::atoi(argv[argIndex++]) :
        GetEnvInt("TILEXR_DEMO_ELEMENTS_PER_RANK", kDefaultElementsPerRank);
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

    bool sdmaAvailable = false;
    GM_ADDR sdmaWorkspace = nullptr;
    if (!CheckTileXR(rank, "TileXRSDMAAvailable", TileXRSDMAAvailable(comm, &sdmaAvailable)) ||
        !CheckTileXR(rank, "TileXRGetSDMAWorkspaceDev", TileXRGetSDMAWorkspaceDev(comm, &sdmaWorkspace))) {
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }
    std::cout << "[rank " << rank << "] SDMA available=" << (sdmaAvailable ? "true" : "false")
              << " sdmaWorkspacePtr=" << static_cast<void*>(sdmaWorkspace) << std::endl;
    if (EnvEnabled("TILEXR_DEMO_REQUIRE_SDMA") && (!sdmaAvailable || sdmaWorkspace == nullptr)) {
        std::cerr << "[rank " << rank << "] ERROR: TileXR SDMA is required but unavailable" << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
    }

    if ((commArgsHost->extraFlag & TileXR::ExtraFlag::UDMA) == 0 || commArgsHost->udmaInfoPtr == nullptr) {
        std::cerr << "[rank " << rank << "] ERROR: TileXR UDMA is not enabled. "
                  << "Check A5/Ascend950 hardware support, CANN/driver setup, and LD_LIBRARY_PATH." << std::endl;
        Cleanup(comm, stream, registeredMemory, debug, rank, deviceId);
        return 1;
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

    PrintStatus(rank, testType == 1 ? "launch put-signal kernel" :
            (testType == 2 ? "launch slot signal-get probe kernel" : "launch all-gather kernel"));
    if (testType == 1) {
        launch_tilexr_udma_put_signal(
            1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(data), reinterpret_cast<GM_ADDR>(signals),
            reinterpret_cast<GM_ADDR>(debug), elementsPerRank, kSignalValue);
    } else if (testType == 2) {
        launch_tilexr_udma_slot_signal_get_probe(
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
    if (testType == 1 || testType == 2) {
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
