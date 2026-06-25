#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
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
#include "tilexr_ep.h"
#include "tilexr_types.h"

namespace {

constexpr int64_t kBs = 4;
constexpr int64_t kH = 8;
constexpr int64_t kTopK = 2;
constexpr int64_t kRoutes = kBs * kTopK;
constexpr int64_t kXElements = kBs * kH;
constexpr int64_t kAssistInts = 4;
constexpr uint16_t kFp16One = 0x3c00;
constexpr uint16_t kFp16Two = 0x4000;
constexpr std::size_t kUdmaRegistrationAlignment = 2 * 1024 * 1024;

struct DemoConfig {
    int64_t moeExpertNum = 8;
    int64_t sharedExpertNum = 0;
    int64_t sharedExpertRankNum = 0;
    int64_t tpWorldSize = 0;
    int64_t tpRankId = 0;

    int64_t maxRoutesPerRank() const
    {
        return kBs * (kTopK + sharedExpertNum);
    }

    int64_t effectiveTpWorldSize() const
    {
        return tpWorldSize == 0 ? 1 : tpWorldSize;
    }

    int64_t expandedElements() const
    {
        return maxRoutesPerRank() * effectiveTpWorldSize() * kH;
    }
};

struct HostPort {
    std::string host;
    int port;
};

bool EnvEnabled(const char *name)
{
    const char *value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

int GetEnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

bool ParseHostPort(const std::string &text, HostPort *out)
{
    const std::size_t pos = text.rfind(':');
    if (out == nullptr || pos == std::string::npos || pos == 0 || pos + 1 >= text.size()) {
        return false;
    }
    const int port = std::atoi(text.substr(pos + 1).c_str());
    if (port <= 0 || port > 65535) {
        return false;
    }
    out->host = text.substr(0, pos);
    out->port = port;
    return true;
}

HostPort GetBarrierEndpoint()
{
    HostPort endpoint {"127.0.0.1", 10174};
    const char *barrier = std::getenv("TILEXR_DEMO_BARRIER_ADDR");
    if (barrier != nullptr && barrier[0] != '\0' && ParseHostPort(barrier, &endpoint)) {
        return endpoint;
    }

    const char *commId = std::getenv("TILEXR_COMM_ID");
    HostPort commEndpoint {};
    if (commId != nullptr && ParseHostPort(commId, &commEndpoint)) {
        endpoint.host = commEndpoint.host;
        endpoint.port = commEndpoint.port + 97;
        if (endpoint.port > 65535) {
            endpoint.port = commEndpoint.port - 97;
        }
    }
    return endpoint;
}

bool SendAll(int fd, const void *data, std::size_t bytes)
{
    const char *ptr = static_cast<const char *>(data);
    std::size_t sent = 0;
    while (sent < bytes) {
        const ssize_t ret = send(fd, ptr + sent, bytes - sent, 0);
        if (ret <= 0) {
            return false;
        }
        sent += static_cast<std::size_t>(ret);
    }
    return true;
}

bool RecvAll(int fd, void *data, std::size_t bytes)
{
    char *ptr = static_cast<char *>(data);
    std::size_t received = 0;
    while (received < bytes) {
        const ssize_t ret = recv(fd, ptr + received, bytes - received, MSG_WAITALL);
        if (ret <= 0) {
            return false;
        }
        received += static_cast<std::size_t>(ret);
    }
    return true;
}

bool DemoBarrierServer(int rankSize, const HostPort &endpoint)
{
    const int listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd < 0) {
        std::cerr << "demo barrier server socket failed: " << std::strerror(errno) << std::endl;
        return false;
    }

    int reuse = 1;
    (void)setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(endpoint.port));
    if (bind(listenFd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
        listen(listenFd, rankSize) != 0) {
        std::cerr << "demo barrier server listen failed on port " << endpoint.port << ": "
                  << std::strerror(errno) << std::endl;
        close(listenFd);
        return false;
    }

    std::vector<int> clients;
    for (int peer = 1; peer < rankSize; ++peer) {
        const int fd = accept(listenFd, nullptr, nullptr);
        if (fd < 0) {
            std::cerr << "demo barrier accept failed: " << std::strerror(errno) << std::endl;
            close(listenFd);
            return false;
        }
        char byte = 0;
        if (!RecvAll(fd, &byte, 1)) {
            std::cerr << "demo barrier recv failed" << std::endl;
            close(fd);
            close(listenFd);
            return false;
        }
        clients.push_back(fd);
    }

    const char release = 1;
    for (int fd : clients) {
        if (!SendAll(fd, &release, 1)) {
            std::cerr << "demo barrier release failed" << std::endl;
            close(fd);
            close(listenFd);
            return false;
        }
        close(fd);
    }
    close(listenFd);
    return true;
}

bool DemoBarrierClient(const HostPort &endpoint)
{
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(endpoint.port));
    if (inet_pton(AF_INET, endpoint.host.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "demo barrier invalid host: " << endpoint.host << std::endl;
        return false;
    }

    int fd = -1;
    for (int attempt = 0; attempt < 600; ++attempt) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return false;
        }
        if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
            break;
        }
        close(fd);
        fd = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (fd < 0) {
        std::cerr << "demo barrier connect failed to " << endpoint.host << ":" << endpoint.port << std::endl;
        return false;
    }

    const char arrived = 1;
    char release = 0;
    const bool ok = SendAll(fd, &arrived, 1) && RecvAll(fd, &release, 1);
    close(fd);
    if (!ok) {
        std::cerr << "demo barrier client exchange failed" << std::endl;
    }
    return ok;
}

bool DemoBarrierAll(int rank, int rankSize, const std::string &step)
{
    if (rankSize <= 1) {
        return true;
    }
    const HostPort endpoint = GetBarrierEndpoint();
    const bool ok = rank == 0 ? DemoBarrierServer(rankSize, endpoint) : DemoBarrierClient(endpoint);
    if (!ok) {
        std::cerr << "rank " << rank << " demo barrier failed after " << step << std::endl;
    }
    return ok;
}

std::vector<int> ParseDeviceList(const char *value)
{
    std::vector<int> devices;
    if (value == nullptr || value[0] == '\0') {
        return devices;
    }

    std::string text(value);
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string part = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!part.empty()) {
            devices.push_back(std::atoi(part.c_str()));
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return devices;
}

int GetDeviceIdFromEnv(int rank, int npuCount, int firstNpu)
{
    const std::vector<int> devices = ParseDeviceList(std::getenv("TILEXR_DEMO_DEVICES"));
    if (!devices.empty()) {
        return devices[rank % static_cast<int>(devices.size())];
    }
    return firstNpu + rank % std::max(npuCount, 1);
}

bool CheckAcl(aclError ret, const std::string &what)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << what << " failed, acl ret=" << ret << std::endl;
        return false;
    }
    return true;
}

bool CheckTileXR(int ret, const std::string &what)
{
    if (ret != TileXR::TILEXR_SUCCESS) {
        std::cerr << what << " failed, TileXR ret=" << ret << std::endl;
        return false;
    }
    return true;
}

std::uintptr_t AlignAddress(std::uintptr_t value, std::size_t alignment)
{
    const std::uintptr_t mask = static_cast<std::uintptr_t>(alignment - 1);
    return (value + mask) & ~mask;
}

std::size_t AlignSize(std::size_t value, std::size_t alignment)
{
    const std::size_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

uint16_t XValue(int rank, int64_t token, int64_t h)
{
    return static_cast<uint16_t>(0x3c00 + rank * 0x0400 + token * 0x0100 + h * 0x0010);
}

float HalfBitsToFloat(uint16_t bits)
{
    const int sign = (bits & 0x8000U) != 0 ? -1 : 1;
    const int exponent = static_cast<int>((bits >> 10) & 0x1fU);
    const int mantissa = static_cast<int>(bits & 0x03ffU);
    if (exponent == 0 && mantissa == 0) {
        return sign < 0 ? -0.0f : 0.0f;
    }
    if (exponent == 0) {
        float value = static_cast<float>(mantissa) / 1024.0f;
        for (int shift = 0; shift < 14; ++shift) {
            value *= 0.5f;
        }
        return static_cast<float>(sign) * value;
    }
    float value = 1.0f + static_cast<float>(mantissa) / 1024.0f;
    const int power = exponent - 15;
    if (power >= 0) {
        for (int shift = 0; shift < power; ++shift) {
            value *= 2.0f;
        }
    } else {
        for (int shift = 0; shift < -power; ++shift) {
            value *= 0.5f;
        }
    }
    return static_cast<float>(sign) * value;
}

int8_t QuantizedXValue(int rank, int64_t token, int64_t h, float scale)
{
    const float value = HalfBitsToFloat(XValue(rank, token, h)) * scale;
    int rounded = static_cast<int>(value >= 0.0f ? value + 0.5f : value - 0.5f);
    if (rounded > 127) {
        rounded = 127;
    } else if (rounded < -128) {
        rounded = -128;
    }
    return static_cast<int8_t>(rounded);
}

float DynamicScaleForXValue(int rank, int64_t token)
{
    float maxAbs = 0.0f;
    for (int64_t h = 0; h < kH; ++h) {
        const float value = std::fabs(HalfBitsToFloat(XValue(rank, token, h)));
        maxAbs = std::max(maxAbs, value);
    }
    return maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
}

int8_t DynamicQuantizedXValue(int rank, int64_t token, int64_t h)
{
    const float scale = DynamicScaleForXValue(rank, token);
    return QuantizedXValue(rank, token, h, scale > 0.0f ? 1.0f / scale : 1.0f);
}

std::vector<int32_t> ExpertIds(const DemoConfig &config)
{
    std::vector<int32_t> expertIds(kRoutes);
    for (int64_t route = 0; route < kRoutes; ++route) {
        expertIds[route] = static_cast<int32_t>(route % config.moeExpertNum);
    }
    return expertIds;
}

std::vector<uint8_t> ActiveMask(bool enabled)
{
    std::vector<uint8_t> mask(kBs, 1);
    if (enabled && kBs > 0) {
        mask[kBs - 1] = 0;
    }
    return mask;
}

int64_t LocalExpertNum(int rankSize, const DemoConfig &config)
{
    const int64_t expertRankSize = static_cast<int64_t>(rankSize) / config.effectiveTpWorldSize();
    const int64_t moeRankNum = expertRankSize - config.sharedExpertRankNum;
    if (moeRankNum <= 0) {
        return 0;
    }
    return config.moeExpertNum / moeRankNum;
}

int ExpertRankForRank(int rank, const DemoConfig &config)
{
    return static_cast<int>(rank / config.effectiveTpWorldSize());
}

int64_t DstRankForExpert(int32_t globalExpertId, int rankSize, const DemoConfig &config)
{
    const int64_t localExpertNum = LocalExpertNum(rankSize, config);
    if (globalExpertId < 0 || localExpertNum <= 0) {
        return -1;
    }
    const int64_t expertRankSize = static_cast<int64_t>(rankSize) / config.effectiveTpWorldSize();
    if (globalExpertId < config.sharedExpertNum) {
        return globalExpertId < config.sharedExpertRankNum ? globalExpertId : -1;
    }
    const int64_t moeExpertId = static_cast<int64_t>(globalExpertId) - config.sharedExpertNum;
    const int64_t dstRank = config.sharedExpertRankNum + moeExpertId / localExpertNum;
    return dstRank < expertRankSize ? dstRank : -1;
}

int64_t LocalExpertForExpert(int32_t globalExpertId, int rankSize, const DemoConfig &config)
{
    const int64_t localExpertNum = LocalExpertNum(rankSize, config);
    if (globalExpertId < 0 || localExpertNum <= 0) {
        return -1;
    }
    if (globalExpertId < config.sharedExpertNum) {
        return globalExpertId < config.sharedExpertRankNum ? globalExpertId : -1;
    }
    return (static_cast<int64_t>(globalExpertId) - config.sharedExpertNum) % localExpertNum;
}

bool RouteBelongsToRank(int32_t globalExpertId, int rank, int rankSize, const DemoConfig &config)
{
    return DstRankForExpert(globalExpertId, rankSize, config) == ExpertRankForRank(rank, config);
}

struct ExpectedRoute {
    int32_t srcRank;
    int32_t tokenId;
    int32_t topKId;
    int32_t expertId;
};

std::vector<ExpectedRoute> BuildExpectedRoutes(
    int rank, int rankSize, const DemoConfig &config, const std::vector<uint8_t> &activeMask)
{
    const std::vector<int32_t> expertIds = ExpertIds(config);
    std::vector<ExpectedRoute> expected;
    const int64_t effectiveTpWorldSize = config.effectiveTpWorldSize();
    const int targetTpRankId = static_cast<int>(rank % effectiveTpWorldSize);
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        if (effectiveTpWorldSize > 1 && srcRank % effectiveTpWorldSize != targetTpRankId) {
            continue;
        }
        for (int64_t token = 0; token < kBs; ++token) {
            if (!activeMask.empty() && activeMask[token] == 0) {
                continue;
            }
            for (int64_t sharedExpertId = 0; sharedExpertId < config.sharedExpertNum; ++sharedExpertId) {
                if (RouteBelongsToRank(static_cast<int32_t>(sharedExpertId), rank, rankSize, config)) {
                    expected.push_back(ExpectedRoute {srcRank, static_cast<int32_t>(token),
                        static_cast<int32_t>(kTopK + sharedExpertId), static_cast<int32_t>(sharedExpertId)});
                }
            }
            for (int64_t topKId = 0; topKId < kTopK; ++topKId) {
                const int64_t route = token * kTopK + topKId;
                const int32_t expertId = static_cast<int32_t>(config.sharedExpertNum) + expertIds[route];
                if (RouteBelongsToRank(expertId, rank, rankSize, config)) {
                    expected.push_back(ExpectedRoute {srcRank, static_cast<int32_t>(token),
                        static_cast<int32_t>(topKId), expertId});
                }
            }
        }
    }
    return expected;
}

std::vector<ExpectedRoute> BuildExpectedTpRoutes(
    int rank, int rankSize, const DemoConfig &config, const std::vector<uint8_t> &activeMask)
{
    std::vector<ExpectedRoute> expected = BuildExpectedRoutes(rank, rankSize, config, activeMask);
    const int64_t effectiveTpWorldSize = config.effectiveTpWorldSize();
    if (effectiveTpWorldSize <= 1) {
        return expected;
    }
    const int tpGroupStartRank = ExpertRankForRank(rank, config) * static_cast<int>(effectiveTpWorldSize);
    for (int64_t tpLane = 0; tpLane < effectiveTpWorldSize; ++tpLane) {
        const int tpPeerRank = tpGroupStartRank + static_cast<int>(tpLane);
        if (tpPeerRank == rank || tpPeerRank < 0 || tpPeerRank >= rankSize) {
            continue;
        }
        const std::vector<ExpectedRoute> peerExpected = BuildExpectedRoutes(tpPeerRank, rankSize, config, activeMask);
        expected.insert(expected.end(), peerExpected.begin(), peerExpected.end());
    }
    return expected;
}

bool ValidateOutputs(int rank, int rankSize, const DemoConfig &config, const std::vector<uint8_t> &expandX,
    const std::vector<int64_t> &expertTokenNums, const std::vector<int32_t> &recvCounts,
    const std::vector<int32_t> &assist, const std::vector<float> &dynamicScalesOut,
    const std::vector<uint8_t> &activeMask, int expertTokenNumsType, bool useStaticQuant,
    bool usePerTokenDynamicQuant, float staticQuantScale)
{
    const int64_t localExpertNum = LocalExpertNum(rankSize, config);
    const std::vector<ExpectedRoute> expected = BuildExpectedTpRoutes(rank, rankSize, config, activeMask);
    const std::vector<ExpectedRoute> localExpected = BuildExpectedRoutes(rank, rankSize, config, activeMask);
    std::vector<int64_t> expectedRecv(rankSize, 0);
    std::vector<int64_t> expectedExpertCounts(localExpertNum, 0);
    for (const ExpectedRoute &route : localExpected) {
        ++expectedRecv[route.srcRank];
    }
    for (const ExpectedRoute &route : expected) {
        const int64_t localExpert = LocalExpertForExpert(route.expertId, rankSize, config);
        if (localExpert >= 0 && localExpert < localExpertNum) {
            ++expectedExpertCounts[localExpert];
        }
    }
    if (expertTokenNumsType == 0) {
        int64_t running = 0;
        for (int64_t localExpert = 0; localExpert < localExpertNum; ++localExpert) {
            running += expectedExpertCounts[localExpert];
            expectedExpertCounts[localExpert] = running;
        }
    }

    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        if (recvCounts[srcRank] != expectedRecv[srcRank]) {
            std::cerr << "rank " << rank << " recvCounts[" << srcRank << "] expected "
                      << expectedRecv[srcRank] << " got " << recvCounts[srcRank] << std::endl;
            return false;
        }
    }

    for (int64_t localExpert = 0; localExpert < localExpertNum; ++localExpert) {
        if (expertTokenNums[localExpert] != expectedExpertCounts[localExpert]) {
            std::cerr << "rank " << rank << " expertTokenNums[" << localExpert << "] expected "
                      << expectedExpertCounts[localExpert] << " got " << expertTokenNums[localExpert] << std::endl;
            return false;
        }
    }

    for (std::size_t row = 0; row < expected.size(); ++row) {
        const ExpectedRoute &route = expected[row];
        const std::size_t assistOffset = row * kAssistInts;
        if (assist[assistOffset] != route.srcRank || assist[assistOffset + 1] != route.tokenId ||
            assist[assistOffset + 2] != route.topKId || assist[assistOffset + 3] != route.expertId) {
            std::cerr << "rank " << rank << " assist row " << row << " mismatch: got {"
                      << assist[assistOffset] << ", " << assist[assistOffset + 1] << ", "
                      << assist[assistOffset + 2] << ", " << assist[assistOffset + 3] << "}" << std::endl;
            return false;
        }

        if (usePerTokenDynamicQuant) {
            const float expectedScale = DynamicScaleForXValue(route.srcRank, route.tokenId);
            const float actualScale = dynamicScalesOut[row];
            if (std::fabs(actualScale - expectedScale) > 1.0e-5f) {
                std::cerr << "rank " << rank << " dynamicScalesOut[" << row << "] expected "
                          << expectedScale << " got " << actualScale << std::endl;
                return false;
            }
        }

        for (int64_t h = 0; h < kH; ++h) {
            const bool useInt8Output = useStaticQuant || usePerTokenDynamicQuant;
            const std::size_t byteOffset = row * kH * (useInt8Output ? sizeof(int8_t) : sizeof(uint16_t)) +
                h * (useInt8Output ? sizeof(int8_t) : sizeof(uint16_t));
            const int expectedValue = useStaticQuant ?
                static_cast<int>(QuantizedXValue(route.srcRank, route.tokenId, h, staticQuantScale)) :
                (usePerTokenDynamicQuant ?
                    static_cast<int>(DynamicQuantizedXValue(route.srcRank, route.tokenId, h)) :
                    static_cast<int>(XValue(route.srcRank, route.tokenId, h)));
            const int actualValue = useInt8Output ?
                static_cast<int>(*reinterpret_cast<const int8_t *>(&expandX[byteOffset])) :
                static_cast<int>(*reinterpret_cast<const uint16_t *>(&expandX[byteOffset]));
            if (actualValue != expectedValue) {
                std::cerr << "rank " << rank << " expandX[" << row << "][" << h << "] expected 0x"
                          << std::hex << expectedValue << " got 0x" << actualValue << std::dec << std::endl;
                return false;
            }
        }
    }

    return true;
}

bool ValidateCombineOutputs(int rank, const std::vector<uint16_t> &yOut)
{
    for (int64_t token = 0; token < kBs; ++token) {
        for (int64_t h = 0; h < kH; ++h) {
            const uint16_t actualValue = yOut[token * kH + h];
            if (actualValue != kFp16Two) {
                std::cerr << "rank " << rank << " yOut[" << token << "][" << h << "] expected 0x"
                          << std::hex << kFp16Two << " got 0x" << actualValue << std::dec << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool ValidateTpRecvCounts(int rank, int rankSize, const DemoConfig &config, const std::vector<uint8_t> &activeMask,
    const std::vector<int32_t> &recvCounts, const std::vector<int32_t> &tpRecvCounts)
{
    if (config.effectiveTpWorldSize() == 1) {
        if (recvCounts.size() != tpRecvCounts.size()) {
            std::cerr << "rank " << rank << " tpRecvCounts size mismatch" << std::endl;
            return false;
        }
        for (std::size_t index = 0; index < recvCounts.size(); ++index) {
            if (tpRecvCounts[index] != recvCounts[index]) {
                std::cerr << "rank " << rank << " tpRecvCounts[" << index << "] expected "
                          << recvCounts[index] << " got " << tpRecvCounts[index] << std::endl;
                return false;
            }
        }
        return true;
    }
    const int64_t effectiveTpWorldSize = config.effectiveTpWorldSize();
    if (effectiveTpWorldSize <= 1 || tpRecvCounts.size() < static_cast<std::size_t>(effectiveTpWorldSize)) {
        std::cerr << "rank " << rank << " tpRecvCounts size mismatch" << std::endl;
        return false;
    }
    const int tpGroupStartRank = ExpertRankForRank(rank, config) * static_cast<int>(effectiveTpWorldSize);
    for (int64_t tpLane = 0; tpLane < effectiveTpWorldSize; ++tpLane) {
        const int tpPeerRank = tpGroupStartRank + static_cast<int>(tpLane);
        const int64_t expectedCount = tpPeerRank >= 0 && tpPeerRank < rankSize ?
            static_cast<int64_t>(BuildExpectedRoutes(tpPeerRank, rankSize, config, activeMask).size()) : 0;
        if (tpRecvCounts[tpLane] != expectedCount) {
            std::cerr << "rank " << rank << " tpRecvCounts[" << tpLane << "] expected "
                      << expectedCount << " got " << tpRecvCounts[tpLane] << std::endl;
            return false;
        }
    }
    return true;
}

void Cleanup(TileXRCommPtr comm, aclrtStream stream, int deviceId, bool deviceSet, bool aclReady,
    std::vector<void *> &buffers)
{
    for (void *buffer : buffers) {
        if (buffer != nullptr) {
            (void)aclrtFree(buffer);
        }
    }
    if (comm != nullptr) {
        (void)TileXRCommDestroy(comm);
    }
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        (void)aclrtResetDevice(deviceId);
    }
    if (aclReady) {
        (void)aclFinalize();
    }
}

} // namespace

int main(int argc, char **argv)
{
    const int rankSize = argc > 1 ? std::atoi(argv[1]) : GetEnvInt("RANK_SIZE", 2);
    const int rank = argc > 2 ? std::atoi(argv[2]) : GetEnvInt("RANK", 0);
    const int npuCount = argc > 3 ? std::atoi(argv[3]) : GetEnvInt("TILEXR_DEMO_NPUS", rankSize);
    const int firstNpu = argc > 4 ? std::atoi(argv[4]) : GetEnvInt("TILEXR_DEMO_FIRST_NPU", 0);
    const bool dispatchOnly = argc > 5 ? std::atoi(argv[5]) != 0 : GetEnvInt("TILEXR_DEMO_DISPATCH_ONLY", 0) != 0;
    const bool useActiveMask = EnvEnabled("TILEXR_EP_DEMO_ACTIVE_MASK");
    const bool requestedTpRecvCounts = EnvEnabled("TILEXR_EP_DEMO_TP_RECV_COUNTS");
    const int expertTokenNumsType = GetEnvInt("TILEXR_EP_DEMO_EXPERT_TOKEN_NUMS_TYPE", 1);
    const int quantMode = GetEnvInt("TILEXR_EP_DEMO_QUANT_MODE", 0);
    const bool useStaticQuant = quantMode == 1;
    const bool usePerTokenDynamicQuant = quantMode == 2;
    const float staticQuantScale = static_cast<float>(GetEnvInt("TILEXR_EP_DEMO_STATIC_QUANT_SCALE", 1));
    DemoConfig config {};
    config.moeExpertNum = GetEnvInt("TILEXR_EP_DEMO_MOE_EXPERT_NUM", static_cast<int>(config.moeExpertNum));
    config.sharedExpertNum = GetEnvInt("TILEXR_EP_DEMO_SHARED_EXPERT_NUM", 0);
    config.sharedExpertRankNum = GetEnvInt("TILEXR_EP_DEMO_SHARED_EXPERT_RANK_NUM", 0);
    config.tpWorldSize = GetEnvInt("TILEXR_EP_DEMO_TP_WORLD_SIZE", 0);
    config.tpRankId = GetEnvInt("TILEXR_EP_DEMO_TP_RANK_ID",
        config.effectiveTpWorldSize() > 1 ? rank % config.effectiveTpWorldSize() : 0);
    const bool useTpRecvCounts = requestedTpRecvCounts || config.effectiveTpWorldSize() != 1;

    const int64_t expertRankSize = static_cast<int64_t>(rankSize) / config.effectiveTpWorldSize();
    const int64_t moeRankNum = expertRankSize - config.sharedExpertRankNum;
    if (rankSize <= 0 || rank < 0 || rank >= rankSize || config.effectiveTpWorldSize() <= 0 ||
        rankSize % config.effectiveTpWorldSize() != 0 || moeRankNum <= 0 ||
        config.moeExpertNum <= 0 || config.moeExpertNum % moeRankNum != 0 ||
        config.sharedExpertNum < 0 || config.sharedExpertRankNum < 0 ||
        config.sharedExpertNum != config.sharedExpertRankNum ||
        (config.effectiveTpWorldSize() > 1 && config.tpRankId != rank % config.effectiveTpWorldSize()) ||
        (expertTokenNumsType != 0 && expertTokenNumsType != 1) ||
        (quantMode != 0 && quantMode != 1 && quantMode != 2) ||
        ((useStaticQuant || usePerTokenDynamicQuant) && !dispatchOnly)) {
        std::cerr << "This demo expects a valid rank and moeExpertNum divisible by MoE rank num, got moeExpertNum="
                  << config.moeExpertNum << " rankSize=" << rankSize
                  << " sharedExpertNum=" << config.sharedExpertNum
                  << " sharedExpertRankNum=" << config.sharedExpertRankNum
                  << " tpWorldSize=" << config.tpWorldSize
                  << " tpRankId=" << config.tpRankId
                  << ", and expertTokenNumsType 0 or 1, got rankSize=" << rankSize << " rank=" << rank
                  << " expertTokenNumsType=" << expertTokenNumsType
                  << " quantMode=" << quantMode << std::endl;
        return 2;
    }

    const int64_t localExpertNum = LocalExpertNum(rankSize, config);
    const int deviceId = GetDeviceIdFromEnv(rank, npuCount, firstNpu);
    bool aclReady = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    TileXRCommPtr comm = nullptr;
    std::vector<void *> buffers;

    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    aclReady = true;
    if (!CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    deviceSet = true;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    if (!CheckTileXR(TileXRCommInitRankLocal(rankSize, rank, &comm), "TileXRCommInitRankLocal")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    std::vector<uint16_t> hostX(kXElements);
    for (int64_t token = 0; token < kBs; ++token) {
        for (int64_t h = 0; h < kH; ++h) {
            hostX[token * kH + h] = XValue(rank, token, h);
        }
    }
    const std::vector<int32_t> hostExpertIds = ExpertIds(config);
    const std::vector<uint8_t> hostActiveMask = ActiveMask(useActiveMask);
    const std::size_t expectedRouteCount =
        BuildExpectedTpRoutes(rank, rankSize, config, hostActiveMask).size();

    void *xDev = nullptr;
    void *expertIdsDev = nullptr;
    void *scalesDev = nullptr;
    void *xActiveMaskDev = nullptr;
    void *expandXDev = nullptr;
    void *dynamicScalesDev = nullptr;
    void *expertTokenNumsDev = nullptr;
    void *recvCountsDev = nullptr;
    void *tpRecvCountsDev = nullptr;
    void *assistDev = nullptr;
    void *expertOutDev = nullptr;
    void *yOutDev = nullptr;
    void *workspaceDev = nullptr;
    void *rawWorkspaceDev = nullptr;
    TileXRUDMAMemHandle workspaceHandle = 0;

    const std::size_t xBytes = hostX.size() * sizeof(uint16_t);
    const std::size_t scaleBytes = sizeof(float);
    const std::size_t expertIdsBytes = hostExpertIds.size() * sizeof(int32_t);
    const std::size_t xActiveMaskBytes = hostActiveMask.size() * sizeof(uint8_t);
    const std::size_t expandedElements = std::max(static_cast<std::size_t>(config.expandedElements()),
        expectedRouteCount * static_cast<std::size_t>(kH));
    const std::size_t maxRoutesPerRank = static_cast<std::size_t>(config.maxRoutesPerRank());
    const std::size_t expandElementBytes =
        (useStaticQuant || usePerTokenDynamicQuant) ? sizeof(int8_t) : sizeof(uint16_t);
    const std::size_t expandXBytes = expandedElements * expandElementBytes;
    const std::size_t expandedRows = expandedElements / kH;
    const std::size_t dynamicScalesBytes = expandedRows * sizeof(float);
    const std::size_t expertTokenNumsBytes = localExpertNum * sizeof(int64_t);
    const std::size_t recvCountsBytes = rankSize * sizeof(int32_t);
    const std::size_t tpRecvCountsBytes = recvCountsBytes;
    const std::size_t assistBytes = (expandedElements / kH) * kAssistInts * sizeof(int32_t);
    const std::size_t yOutBytes = kXElements * sizeof(uint16_t);
    const std::size_t payloadRowBytes = kH * expandElementBytes;
    const std::size_t payloadScaleBytes = usePerTokenDynamicQuant ? maxRoutesPerRank * sizeof(float) : 0;
    const std::size_t epWindowBytes = 64 + static_cast<std::size_t>(rankSize) *
        ((64 + maxRoutesPerRank * payloadRowBytes + payloadScaleBytes + 31) / 32 * 32 +
            (maxRoutesPerRank * kAssistInts * sizeof(int32_t) + 31) / 32 * 32 + 64);
    const std::size_t workspacePayloadBytes = AlignSize(epWindowBytes, 32) *
        static_cast<std::size_t>(config.effectiveTpWorldSize() + 2) +
        static_cast<std::size_t>(rankSize) * sizeof(uint64_t);
    const std::size_t workspaceBytes = ((workspacePayloadBytes + kUdmaRegistrationAlignment - 1) /
        kUdmaRegistrationAlignment) * kUdmaRegistrationAlignment;

    if (!CheckAcl(aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc x") ||
        !CheckAcl(aclrtMalloc(&expertIdsDev, expertIdsBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc expertIds") ||
        (useStaticQuant && !CheckAcl(aclrtMalloc(&scalesDev, scaleBytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc scales")) ||
        (useActiveMask && !CheckAcl(aclrtMalloc(&xActiveMaskDev, xActiveMaskBytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc xActiveMask")) ||
        !CheckAcl(aclrtMalloc(&expandXDev, expandXBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc expandX") ||
        (usePerTokenDynamicQuant && !CheckAcl(aclrtMalloc(&dynamicScalesDev, dynamicScalesBytes,
            ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc dynamicScales")) ||
        !CheckAcl(aclrtMalloc(&expertTokenNumsDev, expertTokenNumsBytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc expertTokenNums") ||
        !CheckAcl(aclrtMalloc(&recvCountsDev, recvCountsBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc recvCounts") ||
        (useTpRecvCounts && !CheckAcl(aclrtMalloc(&tpRecvCountsDev, tpRecvCountsBytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc tpRecvCounts")) ||
        !CheckAcl(aclrtMalloc(&assistDev, assistBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc assist") ||
        !CheckAcl(aclrtMalloc(&expertOutDev, expandXBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc expertOut") ||
        !CheckAcl(aclrtMalloc(&yOutDev, yOutBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc yOut") ||
        !CheckAcl(aclrtMalloc(&rawWorkspaceDev, workspaceBytes + kUdmaRegistrationAlignment - 1,
            ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace")) {
        workspaceDev = rawWorkspaceDev;
        buffers = {xDev, expertIdsDev, scalesDev, xActiveMaskDev, expandXDev, dynamicScalesDev,
            expertTokenNumsDev, recvCountsDev, tpRecvCountsDev, assistDev, expertOutDev, yOutDev, workspaceDev};
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    workspaceDev = reinterpret_cast<void *>(AlignAddress(reinterpret_cast<std::uintptr_t>(rawWorkspaceDev),
        kUdmaRegistrationAlignment));
    buffers = {xDev, expertIdsDev, scalesDev, xActiveMaskDev, expandXDev, dynamicScalesDev, expertTokenNumsDev,
        recvCountsDev, tpRecvCountsDev, assistDev, expertOutDev, yOutDev, rawWorkspaceDev};

    if (!CheckAcl(aclrtMemcpy(xDev, xBytes, hostX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy x") ||
        !CheckAcl(aclrtMemcpy(expertIdsDev, expertIdsBytes, hostExpertIds.data(), expertIdsBytes,
            ACL_MEMCPY_HOST_TO_DEVICE), "copy expertIds") ||
        (useStaticQuant && !CheckAcl(aclrtMemcpy(scalesDev, scaleBytes, &staticQuantScale, scaleBytes,
            ACL_MEMCPY_HOST_TO_DEVICE), "copy scales")) ||
        (useActiveMask && !CheckAcl(aclrtMemcpy(xActiveMaskDev, xActiveMaskBytes, hostActiveMask.data(),
            xActiveMaskBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy xActiveMask")) ||
        !CheckAcl(aclrtMemset(expandXDev, expandXBytes, 0, expandXBytes), "memset expandX") ||
        (usePerTokenDynamicQuant && !CheckAcl(aclrtMemset(dynamicScalesDev, dynamicScalesBytes, 0,
            dynamicScalesBytes), "memset dynamicScales")) ||
        !CheckAcl(aclrtMemset(expertTokenNumsDev, expertTokenNumsBytes, 0, expertTokenNumsBytes),
            "memset expertTokenNums") ||
        !CheckAcl(aclrtMemset(recvCountsDev, recvCountsBytes, 0, recvCountsBytes), "memset recvCounts") ||
        (useTpRecvCounts && !CheckAcl(aclrtMemset(tpRecvCountsDev, tpRecvCountsBytes, 0, tpRecvCountsBytes),
            "memset tpRecvCounts")) ||
        !CheckAcl(aclrtMemset(assistDev, assistBytes, 0, assistBytes), "memset assist") ||
        !CheckAcl(aclrtMemset(yOutDev, yOutBytes, 0, yOutBytes), "memset yOut") ||
        !CheckAcl(aclrtMemset(workspaceDev, workspaceBytes, 0, workspaceBytes), "memset workspace")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    TileXR::CommArgs *commArgsHost = nullptr;
    if (!CheckTileXR(TileXRGetCommArgsHost(comm, commArgsHost), "TileXRGetCommArgsHost")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    const bool crossNode = commArgsHost != nullptr && commArgsHost->localRankSize > 0 &&
        commArgsHost->localRankSize < commArgsHost->rankSize;
    if (crossNode && !CheckTileXR(TileXRUDMARegister(comm, static_cast<GM_ADDR>(workspaceDev), workspaceBytes,
            &workspaceHandle), "TileXRUDMARegister workspace")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    const std::vector<uint16_t> hostExpertOut(expandedElements, kFp16One);
    if (!CheckAcl(aclrtMemcpy(expertOutDev, expandXBytes, hostExpertOut.data(), expandXBytes,
            ACL_MEMCPY_HOST_TO_DEVICE), "copy expertOut")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    const bool useSharedExperts = config.sharedExpertNum != 0 || config.sharedExpertRankNum != 0;
    const bool useTp = config.effectiveTpWorldSize() != 1;
    const bool useDispatchV2 = crossNode || useActiveMask || useTpRecvCounts || expertTokenNumsType != 1 ||
        useSharedExperts || useTp || useStaticQuant || usePerTokenDynamicQuant;
    const int dispatchRet = useDispatchV2 ?
        TileXRMoeEpDispatchV2(xDev, static_cast<int32_t *>(expertIdsDev), scalesDev,
            static_cast<bool *>(xActiveMaskDev), nullptr, comm, kBs, kH, kTopK, config.moeExpertNum,
            expertRankSize, ExpertRankForRank(rank, config), config.tpWorldSize, config.tpRankId, 0,
            config.sharedExpertNum, config.sharedExpertRankNum, quantMode, kBs * rankSize, expertTokenNumsType,
            expandXDev, dynamicScalesDev,
            static_cast<int32_t *>(assistDev), static_cast<int64_t *>(expertTokenNumsDev),
            static_cast<int32_t *>(recvCountsDev), static_cast<int32_t *>(tpRecvCountsDev), nullptr, workspaceDev,
            (useStaticQuant || usePerTokenDynamicQuant) ? TileXR::TILEXR_DATA_TYPE_INT8 :
                TileXR::TILEXR_DATA_TYPE_FP16,
            stream) :
        TileXRMoeEpDispatch(xDev, static_cast<int32_t *>(expertIdsDev), comm, kBs, kH, kTopK, config.moeExpertNum,
            expandXDev, static_cast<int64_t *>(expertTokenNumsDev), static_cast<int32_t *>(recvCountsDev),
            static_cast<int32_t *>(assistDev), TileXR::TILEXR_DATA_TYPE_FP16, stream);
    if (!CheckTileXR(dispatchRet, "TileXRMoeEpDispatch") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    if (!DemoBarrierAll(rank, rankSize, "dispatch synchronized")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    if (EnvEnabled("TILEXR_EP_DEMO_DUMP_WINDOW") && commArgsHost != nullptr) {
        const std::size_t rowBytes = kH * expandElementBytes;
        const std::size_t payloadBytes = AlignSize(maxRoutesPerRank * rowBytes +
            (usePerTokenDynamicQuant ? maxRoutesPerRank * sizeof(float) : 0), 32);
        const std::size_t assistWindowBytes = AlignSize(maxRoutesPerRank * kAssistInts * sizeof(int32_t), 32);
        const std::size_t slotBytes = AlignSize(64 + payloadBytes + assistWindowBytes, 32);
        const std::size_t windowBytes = AlignSize(64 + static_cast<std::size_t>(rankSize) * slotBytes, 32);
        if (crossNode) {
            for (int slotRank = 0; slotRank < rankSize; ++slotRank) {
                uint64_t slotHeader[8] = {};
                const GM_ADDR slotAddr = static_cast<GM_ADDR>(workspaceDev) + windowBytes + 64 +
                    static_cast<std::size_t>(slotRank) * slotBytes;
                if (CheckAcl(aclrtMemcpy(slotHeader, sizeof(slotHeader), slotAddr, sizeof(slotHeader),
                        ACL_MEMCPY_DEVICE_TO_HOST), "dump slot header")) {
                    const uint32_t count = static_cast<uint32_t>(slotHeader[0] & 0xffffffffULL);
                    const uint32_t slotSrc = static_cast<uint32_t>((slotHeader[0] >> 32) & 0xffffffffULL);
                    std::cerr << "rank " << rank << " dump workspace slotRank " << slotRank
                              << " count " << count << " slotSrc " << slotSrc
                              << " payloadBytes " << slotHeader[1] << " assistBytes " << slotHeader[2]
                              << " magic " << slotHeader[3]
                              << std::endl;
                }
            }
        } else {
            for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
                for (int slotRank = 0; slotRank < rankSize; ++slotRank) {
                    uint64_t slotHeader[8] = {};
                    const GM_ADDR slotAddr = commArgsHost->peerMems[srcRank] + TileXR::IPC_DATA_OFFSET + 64 +
                        static_cast<std::size_t>(slotRank) * slotBytes;
                    if (CheckAcl(aclrtMemcpy(slotHeader, sizeof(slotHeader), slotAddr, sizeof(slotHeader),
                            ACL_MEMCPY_DEVICE_TO_HOST), "dump slot header")) {
                        const uint32_t count = static_cast<uint32_t>(slotHeader[0] & 0xffffffffULL);
                        const uint32_t slotSrc = static_cast<uint32_t>((slotHeader[0] >> 32) & 0xffffffffULL);
                        std::cerr << "rank " << rank << " dump sourceWindow " << srcRank << " slotRank " << slotRank
                                  << " count " << count << " slotSrc " << slotSrc
                                  << " payloadBytes " << slotHeader[1] << " assistBytes " << slotHeader[2]
                                  << std::endl;
                    }
                }
            }
        }
    }

    std::vector<uint8_t> hostExpandX(expandXBytes);
    std::vector<int64_t> hostExpertTokenNums(localExpertNum);
    std::vector<int32_t> hostRecvCounts(rankSize);
    std::vector<int32_t> hostTpRecvCounts(rankSize);
    std::vector<int32_t> hostAssist((expandedElements / kH) * kAssistInts);
    std::vector<float> hostDynamicScales(expandedRows);

    if (!CheckAcl(aclrtMemcpy(hostExpandX.data(), expandXBytes, expandXDev, expandXBytes,
            ACL_MEMCPY_DEVICE_TO_HOST), "copy expandX") ||
        !CheckAcl(aclrtMemcpy(hostExpertTokenNums.data(), expertTokenNumsBytes, expertTokenNumsDev,
            expertTokenNumsBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy expertTokenNums") ||
        !CheckAcl(aclrtMemcpy(hostRecvCounts.data(), recvCountsBytes, recvCountsDev, recvCountsBytes,
            ACL_MEMCPY_DEVICE_TO_HOST), "copy recvCounts") ||
        !CheckAcl(aclrtMemcpy(hostAssist.data(), assistBytes, assistDev, assistBytes,
            ACL_MEMCPY_DEVICE_TO_HOST), "copy assist")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    if (useTpRecvCounts && !CheckAcl(aclrtMemcpy(hostTpRecvCounts.data(), tpRecvCountsBytes, tpRecvCountsDev,
            tpRecvCountsBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy tpRecvCounts")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    if (usePerTokenDynamicQuant && !CheckAcl(aclrtMemcpy(hostDynamicScales.data(), dynamicScalesBytes,
            dynamicScalesDev, dynamicScalesBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy dynamicScales")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    const bool dispatchOk = ValidateOutputs(rank, rankSize, config, hostExpandX, hostExpertTokenNums, hostRecvCounts,
        hostAssist, hostDynamicScales, hostActiveMask, expertTokenNumsType, useStaticQuant,
        usePerTokenDynamicQuant, staticQuantScale) &&
        (!useTpRecvCounts || ValidateTpRecvCounts(rank, rankSize, config, hostActiveMask, hostRecvCounts,
            hostTpRecvCounts));
    std::cout << "rank " << rank << " validation " << (dispatchOk ? "PASS" : "FAIL") << std::endl;
    if (!dispatchOk || dispatchOnly) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return dispatchOk ? 0 : 1;
    }

    if (!CheckTileXR(TileXRMoeEpCombine(expertOutDev, static_cast<int32_t *>(assistDev),
            static_cast<int32_t *>(recvCountsDev), comm, kBs, kH, kTopK, config.moeExpertNum, yOutDev,
            TileXR::TILEXR_DATA_TYPE_FP16, stream), "TileXRMoeEpCombine") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream combine")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    if (!DemoBarrierAll(rank, rankSize, "combine synchronized")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    std::vector<uint16_t> hostYOut(kXElements);
    if (!CheckAcl(aclrtMemcpy(hostYOut.data(), yOutBytes, yOutDev, yOutBytes,
            ACL_MEMCPY_DEVICE_TO_HOST), "copy yOut")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    const bool combineOk = ValidateCombineOutputs(rank, hostYOut);
    std::cout << "rank " << rank << " combine validation " << (combineOk ? "PASS" : "FAIL") << std::endl;
    Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
    return combineOk ? 0 : 1;
}
