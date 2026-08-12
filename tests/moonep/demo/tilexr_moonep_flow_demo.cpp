#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "acl/acl.h"
#include "tilexr_api.h"
#include "tilexr_moonep.h"
#include "tilexr_moonep_planner.h"

namespace {

constexpr int32_t kExpectedPlanningStatusSuccess = 0;
constexpr int32_t kExpectedDispatchStatusSuccess = 2000;
constexpr int32_t kExpectedCombineStatusSuccess = 3000;
constexpr int32_t kExpectedPrefetchStatusSuccess = 4000;
constexpr uint64_t kPrefetchWeightAlignment = 64;
constexpr int32_t kExpectedReduceGradStatusSuccess = 0;
constexpr int kMinReduceGradRankCount = 4;

struct Options {
    int world = 8;
    int rank = 0;
    int device = 0;
    int physicalDeviceCount = 8;
    int64_t s = 8;
    int64_t k = 2;
    int64_t experts = 16;
    int64_t hidden = 8;
    uint64_t waitIterations = 1000000;
};

struct DeviceBuffer {
    void *data = nullptr;
    uint64_t bytes = 0;
};

struct RuntimeResources {
    int rank = 0;
    int device = 0;
    bool aclInitialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    TileXRCommPtr comm = nullptr;
    bool udmaRegistered = false;
    TileXRUDMAMemHandle udmaHandle = 0;
    TileXRMoonEpReduceGradPreparedV2 reduceGradPrepared = nullptr;
    std::vector<void *> allocations;
};

bool ParseInt64(const char *text, int64_t *value)
{
    if (text == nullptr || text[0] == '\0' || value == nullptr) {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const long long parsed = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || end == nullptr || *end != '\0') {
        return false;
    }
    *value = static_cast<int64_t>(parsed);
    return true;
}

bool ParseUint64(const char *text, uint64_t *value)
{
    if (text == nullptr || text[0] == '\0' || value == nullptr || text[0] == '-') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || end == nullptr || *end != '\0') {
        return false;
    }
    *value = static_cast<uint64_t>(parsed);
    return true;
}

int64_t EnvInt64(const char *name, int64_t fallback)
{
    const char *text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        return fallback;
    }
    int64_t value = 0;
    return ParseInt64(text, &value) ? value : std::numeric_limits<int64_t>::min();
}

uint64_t EnvUint64(const char *name, uint64_t fallback)
{
    const char *text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        return fallback;
    }
    uint64_t value = 0;
    return ParseUint64(text, &value) ? value : 0;
}

bool ToInt(int64_t value, int *result)
{
    if (result == nullptr || value < INT_MIN || value > INT_MAX) {
        return false;
    }
    *result = static_cast<int>(value);
    return true;
}

bool ParseOptions(int argc, char **argv, Options *options)
{
    if (options == nullptr) {
        return false;
    }
    const int64_t rankSizeEnv = EnvInt64("RANK_SIZE", 8);
    const int64_t worldEnv = EnvInt64("WORLD_SIZE", rankSizeEnv);
    const int64_t rankEnv = EnvInt64("RANK", 0);
    const int64_t physicalFallback = worldEnv > 0 ? std::min<int64_t>(worldEnv, 8) : 0;
    const int64_t physicalEnv =
        EnvInt64("TILEXR_PHYSICAL_DEVICE_COUNT", physicalFallback);
    const int64_t deviceFallback =
        EnvInt64("LOCAL_RANK", physicalEnv > 0 ? rankEnv % physicalEnv : rankEnv);

    if (!ToInt(worldEnv, &options->world) || !ToInt(rankEnv, &options->rank) ||
        !ToInt(physicalEnv, &options->physicalDeviceCount) ||
        !ToInt(deviceFallback, &options->device)) {
        return false;
    }
    options->s = EnvInt64("TILEXR_MOONEP_FLOW_S", 8);
    options->k = EnvInt64("TILEXR_MOONEP_FLOW_K", 2);
    options->experts = EnvInt64(
        "TILEXR_MOONEP_FLOW_EXPERTS",
        worldEnv > 0 && worldEnv <= std::numeric_limits<int64_t>::max() / 2 ?
            worldEnv * 2 : 0);
    options->hidden = EnvInt64("TILEXR_MOONEP_FLOW_H", 8);
    options->waitIterations =
        EnvUint64("TILEXR_MOONEP_PLANNER_WAIT_ITERATIONS", 1000000);

    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::cout << "Usage: " << argv[0]
                  << " [world rank device [S K E H]]\n"
                  << "Environment: WORLD_SIZE/RANK_SIZE, RANK, LOCAL_RANK, "
                  << "TILEXR_PHYSICAL_DEVICE_COUNT, TILEXR_MOONEP_PLANNER_WAIT_ITERATIONS"
                  << std::endl;
        std::exit(0);
    }
    if (argc > 8) {
        return false;
    }

    int64_t parsed[7] = {};
    for (int index = 1; index < argc; ++index) {
        if (!ParseInt64(argv[index], &parsed[index - 1])) {
            return false;
        }
    }
    if (argc > 1 && !ToInt(parsed[0], &options->world)) {
        return false;
    }
    if (argc > 2 && !ToInt(parsed[1], &options->rank)) {
        return false;
    }
    if (argc > 3 && !ToInt(parsed[2], &options->device)) {
        return false;
    }
    if (argc > 4) {
        options->s = parsed[3];
    }
    if (argc > 5) {
        options->k = parsed[4];
    }
    if (argc > 6) {
        options->experts = parsed[5];
    }
    if (argc > 7) {
        options->hidden = parsed[6];
    }
    return true;
}

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t *result)
{
    if (result == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t *result)
{
    if (result == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    *result = lhs + rhs;
    return true;
}

bool CheckedAlignUp(uint64_t value, uint64_t alignment, uint64_t *result)
{
    if (result == nullptr || alignment == 0 ||
        value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
        return false;
    }
    *result = ((value + alignment - 1) / alignment) * alignment;
    return true;
}

bool CountBytes(uint64_t elements, uint64_t *bytes)
{
    return CheckedMultiply(elements, sizeof(int32_t), bytes) &&
        *bytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

template <typename T>
bool CountTypedBytes(uint64_t elements, uint64_t *bytes)
{
    return CheckedMultiply(elements, sizeof(T), bytes) &&
        *bytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

bool ValidateOptions(const Options &options, uint64_t *routeCount,
                     uint64_t *tokenHiddenCount, uint64_t *routeHiddenCount,
                     uint64_t *expertHiddenCount)
{
    if (routeCount == nullptr || tokenHiddenCount == nullptr ||
        routeHiddenCount == nullptr || expertHiddenCount == nullptr ||
        options.world < kMinReduceGradRankCount || options.world > 128 || options.rank < 0 ||
        options.rank >= options.world || options.physicalDeviceCount <= 0 ||
        options.device < 0 || options.device >= options.physicalDeviceCount ||
        options.s <= 0 || options.k < 2 || options.experts <= 0 ||
        options.hidden <= 0 || options.hidden % 32 != 0 ||
        options.experts % options.world != 0 ||
        options.k > options.experts || options.waitIterations == 0) {
        return false;
    }

    const uint64_t s = static_cast<uint64_t>(options.s);
    const uint64_t k = static_cast<uint64_t>(options.k);
    const uint64_t e = static_cast<uint64_t>(options.experts);
    const uint64_t h = static_cast<uint64_t>(options.hidden);
    uint64_t encodedCapacity = 0;
    if (!CheckedMultiply(s, k, routeCount) ||
        !CheckedMultiply(s, h, tokenHiddenCount) ||
        !CheckedMultiply(*routeCount, h, routeHiddenCount) ||
        !CheckedMultiply(e / static_cast<uint64_t>(options.world), h, expertHiddenCount) ||
        !CheckedMultiply(static_cast<uint64_t>(options.world), *routeCount, &encodedCapacity) ||
        *routeCount > static_cast<uint64_t>(INT32_MAX) ||
        encodedCapacity > static_cast<uint64_t>(INT32_MAX) + UINT64_C(1)) {
        return false;
    }
    uint64_t globalRoutes = 0;
    if (!CheckedMultiply(static_cast<uint64_t>(options.world), *routeCount,
            &globalRoutes) || globalRoutes % e != 0) {
        return false;
    }

    uint64_t ignored = 0;
    return CountBytes(*routeCount, &ignored) &&
        CountBytes(*tokenHiddenCount, &ignored) &&
        CountBytes(*routeHiddenCount, &ignored) &&
        CountBytes(*expertHiddenCount, &ignored) &&
        static_cast<uint64_t>(options.experts) <=
            static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

struct HostPort {
    std::string host;
    int port = 0;
};

bool ParseHostPort(const std::string &text, HostPort *endpoint)
{
    const size_t separator = text.rfind(':');
    if (endpoint == nullptr || separator == std::string::npos ||
        separator == 0 || separator + 1 >= text.size()) {
        return false;
    }
    int64_t port = 0;
    if (!ParseInt64(text.substr(separator + 1).c_str(), &port) ||
        port <= 0 || port > 65535) {
        return false;
    }
    endpoint->host = text.substr(0, separator);
    endpoint->port = static_cast<int>(port);
    return true;
}

HostPort GetBarrierEndpoint()
{
    HostPort endpoint {"127.0.0.1", 10314};
    const char *overrideValue = std::getenv("TILEXR_MOONEP_FLOW_BARRIER_ADDR");
    if (overrideValue != nullptr && overrideValue[0] != '\0' &&
        ParseHostPort(overrideValue, &endpoint)) {
        return endpoint;
    }

    const char *commId = std::getenv("TILEXR_COMM_ID");
    HostPort commEndpoint;
    if (commId != nullptr && ParseHostPort(commId, &commEndpoint)) {
        endpoint.host = commEndpoint.host;
        endpoint.port = commEndpoint.port + 113;
        if (endpoint.port > 65535) {
            endpoint.port = commEndpoint.port - 113;
        }
    }
    return endpoint;
}

void SetSocketTimeout(int fd)
{
    timeval timeout {};
    timeout.tv_sec = 30;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

bool SendAll(int fd, const void *data, size_t bytes)
{
    const char *cursor = static_cast<const char *>(data);
    size_t sent = 0;
    while (sent < bytes) {
        const ssize_t ret = send(fd, cursor + sent, bytes - sent, 0);
        if (ret < 0 && errno == EINTR) {
            continue;
        }
        if (ret <= 0) {
            return false;
        }
        sent += static_cast<size_t>(ret);
    }
    return true;
}

bool RecvAll(int fd, void *data, size_t bytes)
{
    char *cursor = static_cast<char *>(data);
    size_t received = 0;
    while (received < bytes) {
        const ssize_t ret = recv(fd, cursor + received, bytes - received, 0);
        if (ret < 0 && errno == EINTR) {
            continue;
        }
        if (ret <= 0) {
            return false;
        }
        received += static_cast<size_t>(ret);
    }
    return true;
}

bool WaitForConnection(int listenFd,
                       const std::chrono::steady_clock::time_point &deadline)
{
    while (true) {
        const std::chrono::steady_clock::time_point now =
            std::chrono::steady_clock::now();
        if (now >= deadline) {
            return false;
        }
        const int64_t remainingUs =
            std::chrono::duration_cast<std::chrono::microseconds>(deadline - now).count();
        timeval timeout {};
        timeout.tv_sec = static_cast<long>(remainingUs / 1000000);
        timeout.tv_usec = static_cast<long>(remainingUs % 1000000);
        fd_set readSet;
        FD_ZERO(&readSet);
        FD_SET(listenFd, &readSet);
        const int ret = select(listenFd + 1, &readSet, nullptr, nullptr, &timeout);
        if (ret < 0 && errno == EINTR) {
            continue;
        }
        return ret > 0;
    }
}

bool DemoBarrierServer(int world, const HostPort &endpoint)
{
    const int listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd < 0) {
        std::cerr << "flow barrier server socket failed: " << std::strerror(errno)
                  << std::endl;
        return false;
    }
    int reuse = 1;
    (void)setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(static_cast<uint16_t>(endpoint.port));
    if (bind(listenFd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(listenFd, world) != 0) {
        std::cerr << "flow barrier listen failed on port " << endpoint.port
                  << ": " << std::strerror(errno) << std::endl;
        close(listenFd);
        return false;
    }

    std::vector<int> clients;
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    bool ok = true;
    for (int peer = 1; peer < world && ok; ++peer) {
        if (!WaitForConnection(listenFd, deadline)) {
            std::cerr << "flow barrier timed out waiting for rank " << peer << std::endl;
            ok = false;
            break;
        }
        const int fd = accept(listenFd, nullptr, nullptr);
        if (fd < 0) {
            std::cerr << "flow barrier accept failed: " << std::strerror(errno)
                      << std::endl;
            ok = false;
            break;
        }
        SetSocketTimeout(fd);
        char arrived = 0;
        if (!RecvAll(fd, &arrived, 1)) {
            std::cerr << "flow barrier arrival receive failed" << std::endl;
            close(fd);
            ok = false;
            break;
        }
        clients.push_back(fd);
    }

    const char release = 1;
    for (std::vector<int>::const_iterator it = clients.begin();
         it != clients.end(); ++it) {
        if (ok && !SendAll(*it, &release, 1)) {
            std::cerr << "flow barrier release failed" << std::endl;
            ok = false;
        }
        close(*it);
    }
    close(listenFd);
    return ok;
}

bool DemoBarrierClient(const HostPort &endpoint)
{
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(endpoint.port));
    if (inet_pton(AF_INET, endpoint.host.c_str(), &address.sin_addr) != 1) {
        std::cerr << "flow barrier host is not an IPv4 address: "
                  << endpoint.host << std::endl;
        return false;
    }

    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    int fd = -1;
    while (std::chrono::steady_clock::now() < deadline) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return false;
        }
        SetSocketTimeout(fd);
        if (connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0) {
            break;
        }
        close(fd);
        fd = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (fd < 0) {
        std::cerr << "flow barrier connect timed out: " << endpoint.host
                  << ":" << endpoint.port << std::endl;
        return false;
    }

    const char arrived = 1;
    char release = 0;
    const bool ok = SendAll(fd, &arrived, 1) &&
        RecvAll(fd, &release, 1) && release == 1;
    close(fd);
    if (!ok) {
        std::cerr << "flow barrier client exchange failed" << std::endl;
    }
    return ok;
}

bool DemoBarrierAll(int rank, int world)
{
    if (world <= 1) {
        return true;
    }
    const HostPort endpoint = GetBarrierEndpoint();
    const bool ok = rank == 0 ?
        DemoBarrierServer(world, endpoint) : DemoBarrierClient(endpoint);
    if (!ok) {
        std::cerr << "[rank " << rank << "] completion rendezvous failed at "
                  << endpoint.host << ":" << endpoint.port << std::endl;
    }
    return ok;
}

bool CheckAcl(int rank, const std::string &step, aclError ret)
{
    if (ret == ACL_SUCCESS) {
        return true;
    }
    std::cerr << "[rank " << rank << "] " << step << " failed: " << ret << std::endl;
    const char *recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') {
        std::cerr << "[rank " << rank << "] recent ACL error: " << recent << std::endl;
    }
    return false;
}

bool CheckTileXR(int rank, const std::string &step, int ret)
{
    if (ret == TILEXR_MOONEP_SUCCESS) {
        return true;
    }
    std::cerr << "[rank " << rank << "] " << step << " failed: " << ret << std::endl;
    return false;
}

bool Allocate(RuntimeResources *resources, uint64_t bytes, const std::string &name,
              DeviceBuffer *buffer)
{
    if (resources == nullptr || buffer == nullptr || bytes == 0 ||
        bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    void *data = nullptr;
    if (!CheckAcl(resources->rank, "aclrtMalloc " + name,
            aclrtMalloc(&data, static_cast<size_t>(bytes), ACL_MEM_MALLOC_HUGE_FIRST))) {
        return false;
    }
    buffer->data = data;
    buffer->bytes = bytes;
    resources->allocations.push_back(data);
    return true;
}

bool AllocateAligned(RuntimeResources *resources, uint64_t bytes, uint64_t alignment,
                     const std::string &name, DeviceBuffer *buffer)
{
    uint64_t allocationBytes = 0;
    if (resources == nullptr || buffer == nullptr || bytes == 0 || alignment == 0 ||
        !CheckedAdd(bytes, alignment - 1, &allocationBytes)) {
        return false;
    }
    DeviceBuffer allocation;
    if (!Allocate(resources, allocationBytes, name, &allocation)) {
        return false;
    }
    const uintptr_t address = reinterpret_cast<uintptr_t>(allocation.data);
    const uint64_t remainder = static_cast<uint64_t>(address % alignment);
    const uint64_t offset = remainder == 0 ? 0 : alignment - remainder;
    buffer->data = static_cast<uint8_t *>(allocation.data) + offset;
    buffer->bytes = bytes;
    return true;
}

bool AllocateInt32(RuntimeResources *resources, uint64_t elements,
                   const std::string &name, DeviceBuffer *buffer)
{
    uint64_t bytes = 0;
    return CountBytes(elements, &bytes) && Allocate(resources, bytes, name, buffer);
}

template <typename T>
bool AllocateTyped(RuntimeResources *resources, uint64_t elements,
                   const std::string &name, DeviceBuffer *buffer)
{
    uint64_t bytes = 0;
    return CountTypedBytes<T>(elements, &bytes) && Allocate(resources, bytes, name, buffer);
}

template <typename T>
bool AllocateTypedAligned(RuntimeResources *resources, uint64_t elements,
                          uint64_t alignment, const std::string &name,
                          DeviceBuffer *buffer)
{
    uint64_t bytes = 0;
    return CountTypedBytes<T>(elements, &bytes) &&
        AllocateAligned(resources, bytes, alignment, name, buffer);
}

bool CopyHostToDevice(int rank, const DeviceBuffer &buffer,
                      const std::vector<int32_t> &values, const std::string &name)
{
    uint64_t bytes = 0;
    if (!CountBytes(values.size(), &bytes) || bytes > buffer.bytes) {
        return false;
    }
    return CheckAcl(rank, "copy H2D " + name,
        aclrtMemcpy(buffer.data, static_cast<size_t>(buffer.bytes), values.data(),
            static_cast<size_t>(bytes), ACL_MEMCPY_HOST_TO_DEVICE));
}

bool CopyDeviceToHost(int rank, std::vector<int32_t> *values,
                      const DeviceBuffer &buffer, const std::string &name)
{
    if (values == nullptr) {
        return false;
    }
    uint64_t bytes = 0;
    if (!CountBytes(values->size(), &bytes) || bytes > buffer.bytes) {
        return false;
    }
    return CheckAcl(rank, "copy D2H " + name,
        aclrtMemcpy(values->data(), static_cast<size_t>(bytes), buffer.data,
            static_cast<size_t>(bytes), ACL_MEMCPY_DEVICE_TO_HOST));
}

template <typename T>
bool CopyHostToDeviceTyped(int rank, const DeviceBuffer &buffer,
                           const std::vector<T> &values, const std::string &name)
{
    uint64_t bytes = 0;
    if (!CountTypedBytes<T>(values.size(), &bytes) || bytes > buffer.bytes) {
        return false;
    }
    return CheckAcl(rank, "copy H2D " + name,
        aclrtMemcpy(buffer.data, static_cast<size_t>(buffer.bytes), values.data(),
            static_cast<size_t>(bytes), ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename T>
bool CopyDeviceToHostTyped(int rank, std::vector<T> *values,
                           const DeviceBuffer &buffer, const std::string &name)
{
    if (values == nullptr) {
        return false;
    }
    uint64_t bytes = 0;
    if (!CountTypedBytes<T>(values->size(), &bytes) || bytes > buffer.bytes) {
        return false;
    }
    return CheckAcl(rank, "copy D2H " + name,
        aclrtMemcpy(values->data(), static_cast<size_t>(bytes), buffer.data,
            static_cast<size_t>(bytes), ACL_MEMCPY_DEVICE_TO_HOST));
}

bool Cleanup(RuntimeResources *resources)
{
    if (resources == nullptr) {
        return false;
    }
    bool ok = true;
    if (resources->stream != nullptr) {
        ok = CheckAcl(resources->rank, "cleanup stream synchronize",
            aclrtSynchronizeStream(resources->stream)) && ok;
    }
    if (resources->reduceGradPrepared != nullptr) {
        const int destroyRet = TileXRMoonEpReduceGradDestroyPreparedV2(
            resources->reduceGradPrepared);
        ok = CheckTileXR(resources->rank,
            "TileXRMoonEpReduceGradDestroyPreparedV2", destroyRet) && ok;
        if (destroyRet != TILEXR_MOONEP_SUCCESS) {
            return false;
        }
        resources->reduceGradPrepared = nullptr;
    }
    if (resources->udmaRegistered && resources->comm != nullptr) {
        const int unregisterRet =
            TileXRUDMAUnregister(resources->comm, resources->udmaHandle);
        ok = CheckTileXR(resources->rank, "TileXRUDMAUnregister prefetch arena",
            unregisterRet) && ok;
        if (unregisterRet == TILEXR_MOONEP_SUCCESS) {
            resources->udmaRegistered = false;
            resources->udmaHandle = 0;
        }
    }
    if (resources->comm != nullptr) {
        const int destroyRet = TileXRCommDestroy(resources->comm);
        ok = CheckTileXR(resources->rank, "TileXRCommDestroy", destroyRet) && ok;
        if (destroyRet == TILEXR_MOONEP_SUCCESS) {
            resources->comm = nullptr;
            resources->udmaRegistered = false;
            resources->udmaHandle = 0;
        }
    }
    if (resources->comm != nullptr) {
        return false;
    }
    for (std::vector<void *>::reverse_iterator it = resources->allocations.rbegin();
         it != resources->allocations.rend(); ++it) {
        if (*it != nullptr) {
            ok = CheckAcl(resources->rank, "aclrtFree", aclrtFree(*it)) && ok;
        }
    }
    resources->allocations.clear();
    if (resources->stream != nullptr) {
        ok = CheckAcl(resources->rank, "aclrtDestroyStream",
            aclrtDestroyStream(resources->stream)) && ok;
        resources->stream = nullptr;
    }
    if (resources->deviceSet) {
        ok = CheckAcl(resources->rank, "aclrtResetDevice",
            aclrtResetDevice(resources->device)) && ok;
        resources->deviceSet = false;
    }
    if (resources->aclInitialized) {
        ok = CheckAcl(resources->rank, "aclFinalize", aclFinalize()) && ok;
        resources->aclInitialized = false;
    }
    return ok;
}

std::vector<int32_t> MakePattern(uint64_t elements, int64_t seed)
{
    std::vector<int32_t> values(static_cast<size_t>(elements));
    for (uint64_t index = 0; index < elements; ++index) {
        const int64_t value = (seed + static_cast<int64_t>(index % 1000000) * 17) %
            INT32_MAX;
        values[static_cast<size_t>(index)] =
            static_cast<int32_t>(value == 0 ? 1 : value);
    }
    return values;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundingBias = UINT32_C(0x7fff) + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + roundingBias) >> 16);
}

float Bfloat16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<uint16_t> MakeBfloatPattern(uint64_t elements, int64_t seed)
{
    std::vector<uint16_t> values(static_cast<size_t>(elements));
    for (uint64_t index = 0; index < elements; ++index) {
        const int64_t integer = (seed + static_cast<int64_t>(index) * 17) % 29 - 14;
        values[static_cast<size_t>(index)] =
            FloatToBfloat16(static_cast<float>(integer) * 0.125f);
    }
    return values;
}

std::vector<float> MakeRouteWeights(uint64_t elements, int rank)
{
    std::vector<float> values(static_cast<size_t>(elements));
    for (uint64_t route = 0; route < elements; ++route) {
        values[static_cast<size_t>(route)] =
            static_cast<float>((rank + 1) * 1000 + static_cast<int>(route));
    }
    return values;
}

std::vector<float> MakeFloatPattern(uint64_t elements, int64_t seed)
{
    std::vector<float> values(static_cast<size_t>(elements));
    for (uint64_t index = 0; index < elements; ++index) {
        values[static_cast<size_t>(index)] = static_cast<float>(
            seed + static_cast<int64_t>(index % 17U));
    }
    return values;
}

template <typename T, typename SourceFactory>
std::vector<T> BuildExpectedDispatch(const Options &options, uint64_t routeCount,
                                     uint64_t rowElements, SourceFactory sourceFactory)
{
    const uint64_t world = static_cast<uint64_t>(options.world);
    const uint64_t expertCount = static_cast<uint64_t>(options.experts);
    const uint64_t expertsPerRank = expertCount / world;
    const uint64_t routesPerExpert = world * routeCount / expertCount;
    std::vector<T> expected(static_cast<size_t>(routeCount * rowElements), T {});
    for (int sourceRank = 0; sourceRank < options.world; ++sourceRank) {
        const std::vector<T> source = sourceFactory(sourceRank);
        for (uint64_t route = 0; route < routeCount; ++route) {
            const uint64_t globalRoute =
                static_cast<uint64_t>(sourceRank) * routeCount + route;
            const uint64_t expert = globalRoute % expertCount;
            const uint64_t destinationRank = expert / expertsPerRank;
            if (destinationRank != static_cast<uint64_t>(options.rank)) {
                continue;
            }
            const uint64_t localExpert = expert % expertsPerRank;
            const uint64_t destinationOffset =
                localExpert * routesPerExpert + globalRoute / expertCount;
            const uint64_t sourceOffset = (route / static_cast<uint64_t>(options.k)) *
                rowElements;
            std::copy(source.begin() + static_cast<ptrdiff_t>(sourceOffset),
                source.begin() + static_cast<ptrdiff_t>(sourceOffset + rowElements),
                expected.begin() + static_cast<ptrdiff_t>(destinationOffset * rowElements));
        }
    }
    return expected;
}

std::vector<uint16_t> BuildExpectedCombine(const std::vector<uint16_t> &input,
                                           int64_t topk)
{
    std::vector<uint16_t> expected(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        const float sum = Bfloat16ToFloat(input[index]) * static_cast<float>(topk);
        expected[index] = FloatToBfloat16(sum);
    }
    return expected;
}

std::vector<float> BuildExpectedRouteWeightDispatch(const Options &options,
                                                    uint64_t routeCount)
{
    const uint64_t world = static_cast<uint64_t>(options.world);
    const uint64_t expertCount = static_cast<uint64_t>(options.experts);
    const uint64_t expertsPerRank = expertCount / world;
    const uint64_t routesPerExpert = world * routeCount / expertCount;
    std::vector<float> expected(static_cast<size_t>(routeCount), 0.0f);
    for (int sourceRank = 0; sourceRank < options.world; ++sourceRank) {
        const std::vector<float> source = MakeRouteWeights(routeCount, sourceRank);
        for (uint64_t route = 0; route < routeCount; ++route) {
            const uint64_t globalRoute =
                static_cast<uint64_t>(sourceRank) * routeCount + route;
            const uint64_t expert = globalRoute % expertCount;
            if (expert / expertsPerRank != static_cast<uint64_t>(options.rank)) {
                continue;
            }
            const uint64_t destinationOffset =
                (expert % expertsPerRank) * routesPerExpert + globalRoute / expertCount;
            expected[static_cast<size_t>(destinationOffset)] =
                source[static_cast<size_t>(route)];
        }
    }
    return expected;
}

TileXRMoonEpTensorV1 Tensor1D(void *data, uint64_t elements, int64_t dim0,
                             uint32_t dtype = TILEXR_MOONEP_DTYPE_INT32)
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.structSize = sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.data = data;
    tensor.elementCount = elements;
    tensor.dtype = dtype;
    tensor.rank = 1;
    tensor.shape[0] = dim0;
    return tensor;
}

TileXRMoonEpTensorV1 Tensor2D(void *data, uint64_t elements,
                             int64_t dim0, int64_t dim1,
                             uint32_t dtype = TILEXR_MOONEP_DTYPE_INT32)
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.structSize = sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.data = data;
    tensor.elementCount = elements;
    tensor.dtype = dtype;
    tensor.rank = 2;
    tensor.shape[0] = dim0;
    tensor.shape[1] = dim1;
    return tensor;
}

TileXRMoonEpTensorV1 Tensor3D(void *data, uint64_t elements,
                             int64_t dim0, int64_t dim1, int64_t dim2,
                             uint32_t dtype)
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.structSize = sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.data = data;
    tensor.elementCount = elements;
    tensor.dtype = dtype;
    tensor.rank = 3;
    tensor.shape[0] = dim0;
    tensor.shape[1] = dim1;
    tensor.shape[2] = dim2;
    return tensor;
}

template <typename T>
bool CheckEqual(int rank, const std::string &name,
                const std::vector<T> &actual, const std::vector<T> &expected)
{
    if (actual.size() != expected.size()) {
        std::cerr << "[rank " << rank << "] " << name << " size mismatch" << std::endl;
        return false;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::cerr << "[rank " << rank << "] " << name << " mismatch index="
                      << index << " actual=" << actual[index]
                      << " expected=" << expected[index] << std::endl;
            return false;
        }
    }
    return true;
}

bool CheckStageCompletion(int rank, const char *name, int32_t expected,
                          const DeviceBuffer &status, aclrtStream stream)
{
    if (!CheckAcl(rank, std::string(name) + " synchronize",
            aclrtSynchronizeStream(stream))) {
        return false;
    }
    std::vector<int32_t> actual(1);
    if (!CopyDeviceToHost(rank, &actual, status, std::string(name) + " status")) {
        return false;
    }
    if (actual[0] != expected) {
        std::cerr << "[rank " << rank << "] " << name << " status=" << actual[0]
                  << " expected=" << expected << std::endl;
        return false;
    }
    std::cout << "[rank " << rank << "] " << name << "_status=" << actual[0]
              << std::endl;
    return true;
}

bool RunFlow(const Options &options, RuntimeResources *resources,
             uint64_t routeCount, uint64_t tokenHiddenCount,
             uint64_t routeHiddenCount, uint64_t expertHiddenCount)
{
    const int rank = options.rank;
    const int64_t b = options.experts / options.world;
    uint64_t groupCount = 0;
    uint64_t expertsToCopyCount = 0;
    if (!CheckedAdd(static_cast<uint64_t>(options.experts), static_cast<uint64_t>(b),
            &groupCount) ||
        !CheckedMultiply(static_cast<uint64_t>(options.world), static_cast<uint64_t>(b),
            &expertsToCopyCount)) {
        return false;
    }

    uint64_t nativeStages = 0;
    uint64_t stubStages = 0;
    const uint64_t expectedNative =
        static_cast<uint64_t>(TILEXR_MOONEP_STAGE_PLANNING) |
        static_cast<uint64_t>(TILEXR_MOONEP_STAGE_DISPATCH) |
        static_cast<uint64_t>(TILEXR_MOONEP_STAGE_PREFETCH_WEIGHT) |
        static_cast<uint64_t>(TILEXR_MOONEP_STAGE_COMBINE) |
        static_cast<uint64_t>(TILEXR_MOONEP_STAGE_REDUCE_GRAD);
    if (TileXRMoonEpGetAbiVersion() != TILEXR_MOONEP_ABI_VERSION_V2 ||
        !CheckTileXR(rank, "TileXRMoonEpGetCapabilitiesV2",
            TileXRMoonEpGetCapabilitiesV2(&nativeStages, &stubStages)) ||
        nativeStages != expectedNative || stubStages != 0) {
        std::cerr << "[rank " << rank << "] unexpected native/stub capabilities"
                  << " native=" << nativeStages << " stub=" << stubStages << std::endl;
        return false;
    }

    uint64_t workspaceBytes = 0;
    int64_t nvS = 0;
    if (!CheckTileXR(rank, "TileXRMoonEpPlanningGetWorkspaceSizeV1",
            TileXRMoonEpPlanningGetWorkspaceSizeV1(resources->comm, options.s,
                options.k, options.experts, b, 1, &workspaceBytes, &nvS)) ||
        nvS != static_cast<int64_t>(routeCount)) {
        std::cerr << "[rank " << rank << "] NvS mismatch actual="
                  << nvS << " expected=" << routeCount << std::endl;
        return false;
    }

    std::vector<int32_t> routing(static_cast<size_t>(routeCount));
    std::vector<int32_t> tokensPerExpert(
        static_cast<size_t>(options.experts), 0);
    for (uint64_t index = 0; index < routeCount; ++index) {
        const uint64_t globalRoute =
            static_cast<uint64_t>(rank) * routeCount + index;
        const int32_t expert = static_cast<int32_t>(
            globalRoute % static_cast<uint64_t>(options.experts));
        routing[static_cast<size_t>(index)] = expert;
        ++tokensPerExpert[static_cast<size_t>(expert)];
    }

    const std::vector<uint16_t> forwardInput =
        MakeBfloatPattern(tokenHiddenCount, static_cast<int64_t>(rank + 1) * 100003);
    const std::vector<uint16_t> backwardInput =
        MakeBfloatPattern(tokenHiddenCount, static_cast<int64_t>(rank + 1) * 300007);
    const std::vector<float> routeWeights = MakeRouteWeights(routeCount, rank);

    const uint64_t weightRows = 2U * static_cast<uint64_t>(b);
    const uint64_t fullRows = groupCount;
    const uint64_t gateRowElements = static_cast<uint64_t>(options.hidden);
    uint64_t upRowElements = 0;
    uint64_t downRowElements = 0;
    uint64_t gateFullElements = 0;
    uint64_t upFullElements = 0;
    uint64_t downFullElements = 0;
    uint64_t gateWeightElements = 0;
    uint64_t upWeightElements = 0;
    uint64_t downWeightElements = 0;
    uint64_t gateReduceElements = 0;
    uint64_t upReduceElements = 0;
    uint64_t downReduceElements = 0;
    if (!CheckedMultiply(gateRowElements, 2U, &upRowElements) ||
        !CheckedMultiply(gateRowElements, 2U, &downRowElements) ||
        upRowElements > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        !CheckedMultiply(weightRows, gateRowElements, &gateWeightElements) ||
        !CheckedMultiply(weightRows, upRowElements, &upWeightElements) ||
        !CheckedMultiply(weightRows, downRowElements, &downWeightElements) ||
        !CheckedMultiply(fullRows, gateRowElements, &gateFullElements) ||
        !CheckedMultiply(fullRows, upRowElements, &upFullElements) ||
        !CheckedMultiply(fullRows, downRowElements, &downFullElements) ||
        !CheckedMultiply(expertsToCopyCount, gateRowElements, &gateReduceElements) ||
        !CheckedMultiply(expertsToCopyCount, upRowElements, &upReduceElements) ||
        !CheckedMultiply(expertsToCopyCount, downRowElements, &downReduceElements)) {
        return false;
    }
    uint64_t gateWeightBytes = 0;
    uint64_t upWeightBytes = 0;
    uint64_t downWeightBytes = 0;
    uint64_t upWeightOffset = 0;
    uint64_t downWeightOffset = 0;
    uint64_t prefetchArenaBytes = 0;
    uint64_t gateEnd = 0;
    uint64_t upEnd = 0;
    if (!CountTypedBytes<uint16_t>(gateWeightElements, &gateWeightBytes) ||
        !CountTypedBytes<uint16_t>(upWeightElements, &upWeightBytes) ||
        !CountTypedBytes<uint16_t>(downWeightElements, &downWeightBytes) ||
        !CheckedAdd(0, gateWeightBytes, &gateEnd) ||
        !CheckedAlignUp(gateEnd, kPrefetchWeightAlignment, &upWeightOffset) ||
        !CheckedAdd(upWeightOffset, upWeightBytes, &upEnd) ||
        !CheckedAlignUp(upEnd, kPrefetchWeightAlignment, &downWeightOffset) ||
        !CheckedAdd(downWeightOffset, downWeightBytes, &prefetchArenaBytes)) {
        return false;
    }
    const std::vector<uint16_t> gateWeight = MakeBfloatPattern(
        gateWeightElements, INT64_C(200003) + rank * INT64_C(1009));
    const std::vector<uint16_t> upWeight = MakeBfloatPattern(
        upWeightElements, INT64_C(210011) + rank * INT64_C(1013));
    const std::vector<uint16_t> downWeight = MakeBfloatPattern(
        downWeightElements, INT64_C(220009) + rank * INT64_C(1019));
    std::vector<float> gateGrad = MakeFloatPattern(
        gateFullElements, INT64_C(10000) + rank * INT64_C(100));
    std::vector<float> upGrad = MakeFloatPattern(
        upFullElements, INT64_C(20000) + rank * INT64_C(100));
    std::vector<float> downGrad = MakeFloatPattern(
        downFullElements, INT64_C(30000) + rank * INT64_C(100));
    std::vector<float> gateReduce(static_cast<size_t>(gateReduceElements), -101.0F);
    std::vector<float> upReduce(static_cast<size_t>(upReduceElements), -102.0F);
    std::vector<float> downReduce(static_cast<size_t>(downReduceElements), -103.0F);
    for (int64_t slot = 0; slot < b; ++slot) {
        for (uint64_t column = 0; column < gateRowElements; ++column) {
            gateReduce[(static_cast<size_t>(rank) * static_cast<size_t>(b) +
                static_cast<size_t>(slot)) * gateRowElements + column] =
                static_cast<float>(100 + rank * 10 + slot + column % 7U);
        }
        for (uint64_t column = 0; column < upRowElements; ++column) {
            upReduce[(static_cast<size_t>(rank) * static_cast<size_t>(b) +
                static_cast<size_t>(slot)) * upRowElements + column] =
                static_cast<float>(200 + rank * 10 + slot + column % 7U);
        }
        for (uint64_t column = 0; column < downRowElements; ++column) {
            downReduce[(static_cast<size_t>(rank) * static_cast<size_t>(b) +
                static_cast<size_t>(slot)) * downRowElements + column] =
                static_cast<float>(300 + rank * 10 + slot + column % 7U);
        }
    }
    const auto packLocalReduce = [&](std::vector<float> *full,
        const std::vector<float> &reduce, uint64_t rowElements) {
        for (int64_t slot = 0; slot < b; ++slot) {
            const size_t source = (static_cast<size_t>(rank) * static_cast<size_t>(b) +
                static_cast<size_t>(slot)) * rowElements;
            const size_t destination = (static_cast<size_t>(options.experts) +
                static_cast<size_t>(slot)) * rowElements;
            std::copy(reduce.begin() + static_cast<ptrdiff_t>(source),
                reduce.begin() + static_cast<ptrdiff_t>(source + rowElements),
                full->begin() + static_cast<ptrdiff_t>(destination));
        }
    };
    packLocalReduce(&gateGrad, gateReduce, gateRowElements);
    packLocalReduce(&upGrad, upReduce, upRowElements);
    packLocalReduce(&downGrad, downReduce, downRowElements);

    DeviceBuffer topk;
    DeviceBuffer tpe;
    DeviceBuffer workspace;
    DeviceBuffer dst;
    DeviceBuffer cu;
    DeviceBuffer expertsToCopy;
    DeviceBuffer zeroFillRanges;
    DeviceBuffer remoteStats;
    DeviceBuffer dupGroups;
    DeviceBuffer dupLoffs;
    DeviceBuffer dupCounts;
    DeviceBuffer plannerStatus;
    DeviceBuffer forwardInputDev;
    DeviceBuffer forwardDispatchDev;
    DeviceBuffer prefetchWeightArena;
    DeviceBuffer gateWeightDev;
    DeviceBuffer upWeightDev;
    DeviceBuffer downWeightDev;
    DeviceBuffer forwardCombineDev;
    DeviceBuffer routeWeightsDev;
    DeviceBuffer dispatchedRouteWeightsDev;
    DeviceBuffer combinedRouteWeightsDev;
    DeviceBuffer backwardInputDev;
    DeviceBuffer backwardDispatchDev;
    DeviceBuffer backwardCombineDev;
    DeviceBuffer gateGradDev;
    DeviceBuffer upGradDev;
    DeviceBuffer downGradDev;
    DeviceBuffer reduceGradStatus;
    DeviceBuffer reduceGradWorkspace;

    if (!AllocateInt32(resources, routeCount, "topk", &topk) ||
        !AllocateInt32(resources, options.experts, "tokens per expert", &tpe) ||
        !Allocate(resources, workspaceBytes, "planner workspace", &workspace) ||
        !AllocateInt32(resources, routeCount, "dst", &dst) ||
        !AllocateInt32(resources, groupCount, "cu", &cu) ||
        !AllocateInt32(resources, expertsToCopyCount, "experts to copy", &expertsToCopy) ||
        !AllocateInt32(resources, groupCount * 2U, "zero fill ranges", &zeroFillRanges) ||
        !AllocateInt32(resources, 2, "remote stats", &remoteStats) ||
        !AllocateInt32(resources, routeCount * 3U, "duplicate groups", &dupGroups) ||
        !AllocateInt32(resources, routeCount, "duplicate offsets", &dupLoffs) ||
        !AllocateInt32(resources, 2, "duplicate counts", &dupCounts) ||
        !AllocateInt32(resources, 1, "planner status", &plannerStatus) ||
        !AllocateTyped<uint16_t>(resources, tokenHiddenCount,
            "forward input", &forwardInputDev) ||
        !AllocateTyped<uint16_t>(resources, routeHiddenCount,
            "forward dispatch", &forwardDispatchDev) ||
        !Allocate(resources, prefetchArenaBytes, "prefetch weight arena", &prefetchWeightArena) ||
        !AllocateTyped<uint16_t>(resources, tokenHiddenCount,
            "forward combine", &forwardCombineDev) ||
        !AllocateTyped<float>(resources, routeCount, "route weights", &routeWeightsDev) ||
        !AllocateTyped<float>(resources, routeCount,
            "dispatched route weights", &dispatchedRouteWeightsDev) ||
        !AllocateTyped<float>(resources, routeCount,
            "combined route weights", &combinedRouteWeightsDev) ||
        !AllocateTyped<uint16_t>(resources, tokenHiddenCount,
            "backward input", &backwardInputDev) ||
        !AllocateTyped<uint16_t>(resources, routeHiddenCount,
            "backward dispatch", &backwardDispatchDev) ||
        !AllocateTyped<uint16_t>(resources, tokenHiddenCount,
            "backward combine", &backwardCombineDev) ||
        !AllocateTypedAligned<float>(resources, gateFullElements,
            TILEXR_MOONEP_REDUCE_GRAD_WORKSPACE_ALIGNMENT,
            "gate grad", &gateGradDev) ||
        !AllocateTypedAligned<float>(resources, upFullElements,
            TILEXR_MOONEP_REDUCE_GRAD_WORKSPACE_ALIGNMENT,
            "up grad", &upGradDev) ||
        !AllocateTypedAligned<float>(resources, downFullElements,
            TILEXR_MOONEP_REDUCE_GRAD_WORKSPACE_ALIGNMENT,
            "down grad", &downGradDev) ||
        !AllocateInt32(resources, 1, "ReduceGrad status", &reduceGradStatus)) {
        return false;
    }
    gateWeightDev.data = prefetchWeightArena.data;
    gateWeightDev.bytes = gateWeightBytes;
    upWeightDev.data = static_cast<uint8_t *>(prefetchWeightArena.data) + upWeightOffset;
    upWeightDev.bytes = upWeightBytes;
    downWeightDev.data =
        static_cast<uint8_t *>(prefetchWeightArena.data) + downWeightOffset;
    downWeightDev.bytes = downWeightBytes;
    if ((reinterpret_cast<uintptr_t>(workspace.data) & static_cast<uintptr_t>(31)) != 0) {
        std::cerr << "[rank " << rank << "] planner workspace is not 32-byte aligned"
                  << std::endl;
        return false;
    }
    const std::vector<int32_t> statusSentinel(1, -1);
    if (!CopyHostToDevice(rank, topk, routing, "topk") ||
        !CopyHostToDevice(rank, tpe, tokensPerExpert, "tokens per expert") ||
        !CopyHostToDevice(rank, plannerStatus, statusSentinel, "planner status sentinel") ||
        !CopyHostToDeviceTyped(rank, forwardInputDev, forwardInput, "forward input") ||
        !CopyHostToDeviceTyped(rank, gateWeightDev, gateWeight, "gate weight") ||
        !CopyHostToDeviceTyped(rank, upWeightDev, upWeight, "up weight") ||
        !CopyHostToDeviceTyped(rank, downWeightDev, downWeight, "down weight") ||
        !CopyHostToDeviceTyped(rank, routeWeightsDev, routeWeights, "route weights") ||
        !CopyHostToDeviceTyped(rank, backwardInputDev, backwardInput, "backward input") ||
        !CopyHostToDeviceTyped(rank, gateGradDev, gateGrad, "gate grad") ||
        !CopyHostToDeviceTyped(rank, upGradDev, upGrad, "up grad") ||
        !CopyHostToDeviceTyped(rank, downGradDev, downGrad, "down grad") ||
        !CopyHostToDevice(rank, reduceGradStatus, statusSentinel,
            "ReduceGrad status sentinel")) {
        return false;
    }
    if ((reinterpret_cast<uintptr_t>(prefetchWeightArena.data) &
            static_cast<uintptr_t>(kPrefetchWeightAlignment - 1)) != 0 ||
        !CheckTileXR(rank, "TileXRUDMARegister prefetch arena",
            TileXRUDMARegister(resources->comm,
                static_cast<GM_ADDR>(prefetchWeightArena.data),
                static_cast<size_t>(prefetchWeightArena.bytes),
                &resources->udmaHandle))) {
        return false;
    }
    resources->udmaRegistered = true;

    TileXRMoonEpPlanV1 plan {};
    plan.structSize = sizeof(plan);
    plan.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    plan.n = static_cast<int64_t>(routeCount);
    plan.r = options.world;
    plan.e = options.experts;
    plan.b = b;
    plan.nvS = nvS;
    plan.k = options.k;
    plan.dst = dst.data;
    plan.expertsToCopy = expertsToCopy.data;
    plan.zeroFillRanges = zeroFillRanges.data;
    plan.remoteStats = remoteStats.data;
    plan.dupGroups = dupGroups.data;
    plan.dupLoffs = dupLoffs.data;
    plan.dupCounts = dupCounts.data;
    plan.status = plannerStatus.data;

    TileXRMoonEpTensorV1 topkTensor =
        Tensor2D(topk.data, routeCount, options.s, options.k);
    TileXRMoonEpTensorV1 tpeTensor =
        Tensor1D(tpe.data, options.experts, options.experts);
    TileXRMoonEpPlanningArgsV1 planning {};
    planning.structSize = sizeof(planning);
    planning.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    planning.comm = resources->comm;
    planning.topkExperts = &topkTensor;
    planning.tokensPerExpert = &tpeTensor;
    planning.workspace = workspace.data;
    planning.workspaceBytes = workspace.bytes;
    TileXRMoonEpTensorV1 cuTensor = Tensor1D(cu.data, groupCount, groupCount);
    planning.cuSeqlens = &cuTensor;
    planning.plan = &plan;
    planning.waitIterations = options.waitIterations;
    planning.flags = TILEXR_MOONEP_FLAG_NONE;

    TileXRMoonEpTensorV1 forwardInputTensor =
        Tensor2D(forwardInputDev.data, tokenHiddenCount, options.s, options.hidden,
            TILEXR_MOONEP_DTYPE_BFLOAT16);
    TileXRMoonEpTensorV1 forwardDispatchTensor =
        Tensor2D(forwardDispatchDev.data, routeHiddenCount,
            nvS, options.hidden, TILEXR_MOONEP_DTYPE_BFLOAT16);
    TileXRMoonEpTensorV1 routeWeightsTensor =
        Tensor2D(routeWeightsDev.data, routeCount, options.s, options.k,
            TILEXR_MOONEP_DTYPE_FLOAT32);
    TileXRMoonEpTensorV1 dispatchedRouteWeightsTensor =
        Tensor1D(dispatchedRouteWeightsDev.data, routeCount, nvS,
            TILEXR_MOONEP_DTYPE_FLOAT32);
    TileXRMoonEpDispatchArgsV1 forwardDispatch {};
    forwardDispatch.structSize = sizeof(forwardDispatch);
    forwardDispatch.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    forwardDispatch.comm = resources->comm;
    forwardDispatch.plan = &plan;
    forwardDispatch.hiddenSh = &forwardInputTensor;
    forwardDispatch.routeWeightsSk = &routeWeightsTensor;
    forwardDispatch.hiddenNvsh = &forwardDispatchTensor;
    forwardDispatch.routeWeightsNvs = &dispatchedRouteWeightsTensor;
    forwardDispatch.flags = TILEXR_MOONEP_FLAG_BUILD_DEDUP;

    TileXRMoonEpTensorV1 gateWeightTensor = Tensor3D(gateWeightDev.data,
        gateWeightElements, weightRows, options.hidden, 1,
        TILEXR_MOONEP_DTYPE_BFLOAT16);
    TileXRMoonEpTensorV1 upWeightTensor = Tensor3D(upWeightDev.data,
        upWeightElements, weightRows, options.hidden, 2,
        TILEXR_MOONEP_DTYPE_BFLOAT16);
    TileXRMoonEpTensorV1 downWeightTensor = Tensor3D(downWeightDev.data,
        downWeightElements, weightRows, static_cast<int64_t>(downRowElements), 1,
        TILEXR_MOONEP_DTYPE_BFLOAT16);
    TileXRMoonEpPrefetchWeightArgsV1 prefetch {};
    prefetch.structSize = sizeof(prefetch);
    prefetch.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    prefetch.comm = resources->comm;
    prefetch.plan = &plan;
    prefetch.gate = &gateWeightTensor;
    prefetch.up = &upWeightTensor;
    prefetch.down = &downWeightTensor;

    TileXRMoonEpTensorV1 forwardCombineTensor =
        Tensor2D(forwardCombineDev.data, tokenHiddenCount, options.s, options.hidden,
            TILEXR_MOONEP_DTYPE_BFLOAT16);
    TileXRMoonEpTensorV1 combinedRouteWeightsTensor =
        Tensor2D(combinedRouteWeightsDev.data, routeCount, options.s, options.k,
            TILEXR_MOONEP_DTYPE_FLOAT32);
    TileXRMoonEpCombineArgsV1 forwardCombine {};
    forwardCombine.structSize = sizeof(forwardCombine);
    forwardCombine.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    forwardCombine.comm = resources->comm;
    forwardCombine.plan = &plan;
    forwardCombine.hiddenNvsh = &forwardDispatchTensor;
    forwardCombine.routeWeightsNvs = &dispatchedRouteWeightsTensor;
    forwardCombine.hiddenSh = &forwardCombineTensor;
    forwardCombine.routeWeightsSk = &combinedRouteWeightsTensor;

    TileXRMoonEpTensorV1 backwardInputTensor =
        Tensor2D(backwardInputDev.data, tokenHiddenCount, options.s, options.hidden,
            TILEXR_MOONEP_DTYPE_BFLOAT16);
    TileXRMoonEpTensorV1 backwardDispatchTensor =
        Tensor2D(backwardDispatchDev.data, routeHiddenCount,
            nvS, options.hidden, TILEXR_MOONEP_DTYPE_BFLOAT16);
    TileXRMoonEpDispatchArgsV1 backwardDispatch {};
    backwardDispatch.structSize = sizeof(backwardDispatch);
    backwardDispatch.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    backwardDispatch.comm = resources->comm;
    backwardDispatch.plan = &plan;
    backwardDispatch.hiddenSh = &backwardInputTensor;
    backwardDispatch.hiddenNvsh = &backwardDispatchTensor;

    TileXRMoonEpTensorV1 backwardCombineTensor =
        Tensor2D(backwardCombineDev.data, tokenHiddenCount, options.s, options.hidden,
            TILEXR_MOONEP_DTYPE_BFLOAT16);
    TileXRMoonEpCombineArgsV1 backwardCombine {};
    backwardCombine.structSize = sizeof(backwardCombine);
    backwardCombine.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    backwardCombine.comm = resources->comm;
    backwardCombine.plan = &plan;
    backwardCombine.hiddenNvsh = &backwardDispatchTensor;
    backwardCombine.hiddenSh = &backwardCombineTensor;

    TileXRMoonEpTensorV1 gateGradTensor = Tensor3D(gateGradDev.data,
        gateFullElements, groupCount, options.hidden, 1, TILEXR_MOONEP_DTYPE_FLOAT32);
    TileXRMoonEpTensorV1 upGradTensor = Tensor3D(upGradDev.data,
        upFullElements, groupCount, options.hidden, 2, TILEXR_MOONEP_DTYPE_FLOAT32);
    TileXRMoonEpTensorV1 downGradTensor = Tensor3D(downGradDev.data,
        downFullElements, groupCount, static_cast<int64_t>(downRowElements), 1,
        TILEXR_MOONEP_DTYPE_FLOAT32);
    TileXRMoonEpTensorV1 reduceGradStatusTensor = Tensor1D(
        reduceGradStatus.data, 1, 1, TILEXR_MOONEP_DTYPE_INT32);
    TileXRMoonEpReduceGradWorkspaceQueryV2 reduceGradQuery {};
    reduceGradQuery.structSize = sizeof(reduceGradQuery);
    reduceGradQuery.abiVersion = TILEXR_MOONEP_ABI_VERSION_V2;
    reduceGradQuery.comm = resources->comm;
    reduceGradQuery.plan = &plan;
    reduceGradQuery.gate = &gateGradTensor;
    reduceGradQuery.up = &upGradTensor;
    reduceGradQuery.down = &downGradTensor;
    reduceGradQuery.flags = TILEXR_MOONEP_FLAG_NONE;
    TileXRMoonEpReduceGradWorkspaceInfoV2 reduceGradInfo {};
    reduceGradInfo.structSize = sizeof(reduceGradInfo);
    reduceGradInfo.abiVersion = TILEXR_MOONEP_ABI_VERSION_V2;
    TileXRMoonEpReduceGradArgsV2 reduceGrad {};
    reduceGrad.structSize = sizeof(reduceGrad);
    reduceGrad.abiVersion = TILEXR_MOONEP_ABI_VERSION_V2;
    reduceGrad.plan = &plan;
    reduceGrad.gate = &gateGradTensor;
    reduceGrad.up = &upGradTensor;
    reduceGrad.down = &downGradTensor;
    reduceGrad.status = &reduceGradStatusTensor;
    reduceGrad.waitIterations = options.waitIterations;
    reduceGrad.flags = TILEXR_MOONEP_FLAG_NONE;

    if (!CheckTileXR(rank, "Planning",
            TileXRMoonEpPlanningV1(&planning, resources->stream)) ||
        !CheckStageCompletion(rank, "planning", kExpectedPlanningStatusSuccess,
            plannerStatus, resources->stream)) {
        return false;
    }

    std::vector<int32_t> cuHost(static_cast<size_t>(groupCount));
    std::vector<int32_t> expertsToCopyHost(static_cast<size_t>(expertsToCopyCount));
    if (!CopyDeviceToHost(rank, &cuHost, cu, "cu") ||
        !CopyDeviceToHost(rank, &expertsToCopyHost, expertsToCopy,
            "experts to copy")) {
        return false;
    }
    if (!CheckTileXR(rank, "TileXRMoonEpReduceGradGetWorkspaceSizeV2",
            TileXRMoonEpReduceGradGetWorkspaceSizeV2(
                &reduceGradQuery, &reduceGradInfo)) ||
        reduceGradInfo.workspaceBytes == 0 || reduceGradInfo.workspaceAlignment == 0 ||
        reduceGradInfo.udmaChunkBytes == 0 || reduceGradInfo.qpCount < 3 ||
        !AllocateAligned(resources, reduceGradInfo.workspaceBytes,
            reduceGradInfo.workspaceAlignment, "ReduceGrad workspace",
            &reduceGradWorkspace)) {
        std::cerr << "[rank " << rank
                  << "] ReduceGrad owner-pull workspace preparation failed"
                  << " workspace_bytes=" << reduceGradInfo.workspaceBytes
                  << " alignment=" << reduceGradInfo.workspaceAlignment
                  << " chunk_bytes=" << reduceGradInfo.udmaChunkBytes
                  << " qp_count=" << reduceGradInfo.qpCount << std::endl;
        return false;
    }
    DeviceBuffer *gradBuffers[3] = {&gateGradDev, &upGradDev, &downGradDev};
    TileXRMoonEpReduceGradSourceSliceV2 reduceGradSources[3] {};
    for (uint32_t projection = 0; projection < 3; ++projection) {
        uint64_t sourceOffset = 0;
        uint64_t sourceBytes = 0;
        uint64_t sourceEnd = 0;
        if (!CheckedMultiply(static_cast<uint64_t>(options.experts),
                reduceGradInfo.rowBytes[projection], &sourceOffset) ||
            !CheckedMultiply(static_cast<uint64_t>(b),
                reduceGradInfo.rowBytes[projection], &sourceBytes) ||
            !CheckedAdd(sourceOffset, sourceBytes, &sourceEnd) ||
            sourceEnd > gradBuffers[projection]->bytes) {
            return false;
        }
        reduceGradSources[projection].data =
            static_cast<uint8_t *>(gradBuffers[projection]->data) + sourceOffset;
        reduceGradSources[projection].bytes = sourceBytes;
        reduceGradSources[projection].registrationBase =
            gradBuffers[projection]->data;
        reduceGradSources[projection].registrationBytes =
            gradBuffers[projection]->bytes;
        reduceGrad.sources[projection] = reduceGradSources[projection];
    }
    TileXRMoonEpReduceGradPrepareArgsV2 reduceGradPrepare {};
    reduceGradPrepare.structSize = sizeof(reduceGradPrepare);
    reduceGradPrepare.abiVersion = TILEXR_MOONEP_ABI_VERSION_V2;
    reduceGradPrepare.comm = resources->comm;
    reduceGradPrepare.plan = &plan;
    reduceGradPrepare.gate = &gateGradTensor;
    reduceGradPrepare.up = &upGradTensor;
    reduceGradPrepare.down = &downGradTensor;
    reduceGradPrepare.workspace = reduceGradWorkspace.data;
    reduceGradPrepare.workspaceBytes = reduceGradWorkspace.bytes;
    reduceGradPrepare.requestedUdmaChunkBytes = reduceGradQuery.requestedUdmaChunkBytes;
    reduceGradPrepare.flags = TILEXR_MOONEP_FLAG_NONE;
    for (uint32_t projection = 0; projection < 3; ++projection) {
        reduceGradPrepare.sources[projection] = reduceGradSources[projection];
    }
    if (!CheckTileXR(rank, "TileXRMoonEpReduceGradPrepareV2",
            TileXRMoonEpReduceGradPrepareV2(
                &reduceGradPrepare, &resources->reduceGradPrepared))) {
        return false;
    }
    reduceGrad.prepared = resources->reduceGradPrepared;
    if (!CheckTileXR(rank, "Dispatch forward",
            TileXRMoonEpDispatchV1(&forwardDispatch, resources->stream)) ||
        !CheckStageCompletion(rank, "dispatch_forward", kExpectedDispatchStatusSuccess,
            plannerStatus, resources->stream)) {
        return false;
    }
    if (!CheckTileXR(rank, "PrefetchWeight",
            TileXRMoonEpPrefetchWeightV1(&prefetch, resources->stream)) ||
        !CheckStageCompletion(rank, "prefetch_weight", kExpectedPrefetchStatusSuccess,
            plannerStatus, resources->stream) ||
        !CheckTileXR(rank, "Combine forward",
            TileXRMoonEpCombineV1(&forwardCombine, resources->stream)) ||
        !CheckStageCompletion(rank, "combine_forward", kExpectedCombineStatusSuccess,
            plannerStatus, resources->stream) ||
        !CheckTileXR(rank, "Dispatch backward",
            TileXRMoonEpDispatchV1(&backwardDispatch, resources->stream)) ||
        !CheckStageCompletion(rank, "dispatch_backward", kExpectedDispatchStatusSuccess,
            plannerStatus, resources->stream) ||
        !CheckTileXR(rank, "Combine backward",
            TileXRMoonEpCombineV1(&backwardCombine, resources->stream)) ||
        !CheckStageCompletion(rank, "combine_backward", kExpectedCombineStatusSuccess,
            plannerStatus, resources->stream) ||
        !CheckTileXR(rank, "ReduceGrad",
            TileXRMoonEpReduceGradV2(&reduceGrad, resources->stream)) ||
        !CheckStageCompletion(rank, "reduce_grad", kExpectedReduceGradStatusSuccess,
            reduceGradStatus, resources->stream)) {
        return false;
    }

    std::vector<uint16_t> forwardDispatchHost(static_cast<size_t>(routeHiddenCount));
    std::vector<uint16_t> gateWeightHost(static_cast<size_t>(gateWeightElements));
    std::vector<uint16_t> upWeightHost(static_cast<size_t>(upWeightElements));
    std::vector<uint16_t> downWeightHost(static_cast<size_t>(downWeightElements));
    std::vector<uint16_t> forwardCombineHost(static_cast<size_t>(tokenHiddenCount));
    std::vector<float> dispatchedRouteWeightsHost(static_cast<size_t>(routeCount));
    std::vector<float> combinedRouteWeightsHost(static_cast<size_t>(routeCount));
    std::vector<uint16_t> backwardDispatchHost(static_cast<size_t>(routeHiddenCount));
    std::vector<uint16_t> backwardCombineHost(static_cast<size_t>(tokenHiddenCount));
    std::vector<float> gateGradHost(static_cast<size_t>(gateFullElements));
    std::vector<float> upGradHost(static_cast<size_t>(upFullElements));
    std::vector<float> downGradHost(static_cast<size_t>(downFullElements));
    if (!CopyDeviceToHostTyped(rank, &forwardDispatchHost, forwardDispatchDev,
            "forward dispatch") ||
        !CopyDeviceToHostTyped(rank, &gateWeightHost, gateWeightDev, "gate weight") ||
        !CopyDeviceToHostTyped(rank, &upWeightHost, upWeightDev, "up weight") ||
        !CopyDeviceToHostTyped(rank, &downWeightHost, downWeightDev, "down weight") ||
        !CopyDeviceToHostTyped(rank, &forwardCombineHost, forwardCombineDev,
            "forward combine") ||
        !CopyDeviceToHostTyped(rank, &dispatchedRouteWeightsHost,
            dispatchedRouteWeightsDev, "dispatched route weights") ||
        !CopyDeviceToHostTyped(rank, &combinedRouteWeightsHost,
            combinedRouteWeightsDev, "combined route weights") ||
        !CopyDeviceToHostTyped(rank, &backwardDispatchHost, backwardDispatchDev,
            "backward dispatch") ||
        !CopyDeviceToHostTyped(rank, &backwardCombineHost, backwardCombineDev,
            "backward combine") ||
        !CopyDeviceToHostTyped(rank, &gateGradHost, gateGradDev, "gate grad") ||
        !CopyDeviceToHostTyped(rank, &upGradHost, upGradDev, "up grad") ||
        !CopyDeviceToHostTyped(rank, &downGradHost, downGradDev, "down grad")) {
        return false;
    }

    bool correct = true;
    if (cuHost.empty() || cuHost.back() != static_cast<int32_t>(routeCount)) {
        std::cerr << "[rank " << rank << "] cu last mismatch actual="
                  << (cuHost.empty() ? -1 : cuHost.back())
                  << " expected=" << routeCount << std::endl;
        correct = false;
    }
    const std::vector<uint16_t> expectedForwardDispatch =
        BuildExpectedDispatch<uint16_t>(options, routeCount,
            static_cast<uint64_t>(options.hidden),
            [&](int sourceRank) {
                return MakeBfloatPattern(tokenHiddenCount,
                    static_cast<int64_t>(sourceRank + 1) * 100003);
            });
    const std::vector<uint16_t> expectedBackwardDispatch =
        BuildExpectedDispatch<uint16_t>(options, routeCount,
            static_cast<uint64_t>(options.hidden),
            [&](int sourceRank) {
                return MakeBfloatPattern(tokenHiddenCount,
                    static_cast<int64_t>(sourceRank + 1) * 300007);
            });
    const std::vector<uint16_t> expectedForwardCombine =
        BuildExpectedCombine(forwardInput, options.k);
    const std::vector<uint16_t> expectedBackwardCombine =
        BuildExpectedCombine(backwardInput, options.k);
    const std::vector<float> expectedDispatchedRouteWeights =
        BuildExpectedRouteWeightDispatch(options, routeCount);

    const auto buildExpectedWeight = [&](const std::vector<uint16_t> &initial,
        uint64_t rowElements, int64_t seedBase, int64_t rankStride) {
        std::vector<uint16_t> expected = initial;
        for (int64_t slot = 0; slot < b; ++slot) {
            const int32_t expert = expertsToCopyHost[static_cast<size_t>(rank) *
                static_cast<size_t>(b) + static_cast<size_t>(slot)];
            if (expert < 0) {
                continue;
            }
            const int owner = expert / static_cast<int32_t>(b);
            const std::vector<uint16_t> ownerValues = MakeBfloatPattern(
                weightRows * rowElements, seedBase + owner * rankStride);
            const size_t source = static_cast<size_t>(expert % b) * rowElements;
            const size_t destination = (static_cast<size_t>(b) +
                static_cast<size_t>(slot)) * rowElements;
            std::copy(ownerValues.begin() + static_cast<ptrdiff_t>(source),
                ownerValues.begin() + static_cast<ptrdiff_t>(source + rowElements),
                expected.begin() + static_cast<ptrdiff_t>(destination));
        }
        return expected;
    };
    const std::vector<uint16_t> expectedGateWeight = buildExpectedWeight(
        gateWeight, gateRowElements, INT64_C(200003), INT64_C(1009));
    const std::vector<uint16_t> expectedUpWeight = buildExpectedWeight(
        upWeight, upRowElements, INT64_C(210011), INT64_C(1013));
    const std::vector<uint16_t> expectedDownWeight = buildExpectedWeight(
        downWeight, downRowElements, INT64_C(220009), INT64_C(1019));

    const auto buildExpectedGrad = [&](const std::vector<float> &initial,
        uint64_t rowElements, int contributionBase) {
        std::vector<float> expected = initial;
        for (int sourceRank = 0; sourceRank < options.world; ++sourceRank) {
            for (int64_t slot = 0; slot < b; ++slot) {
                const int32_t expert = expertsToCopyHost[
                    static_cast<size_t>(sourceRank) * static_cast<size_t>(b) +
                    static_cast<size_t>(slot)];
                if (expert < 0 || expert / static_cast<int32_t>(b) != rank) {
                    continue;
                }
                const size_t destination = static_cast<size_t>(expert) * rowElements;
                for (uint64_t column = 0; column < rowElements; ++column) {
                    expected[destination + column] += static_cast<float>(
                        contributionBase + sourceRank * 10 + slot + column % 7U);
                }
            }
        }
        for (int64_t slot = 0; slot < b; ++slot) {
            const size_t planIndex = static_cast<size_t>(rank) *
                static_cast<size_t>(b) + static_cast<size_t>(slot);
            if (expertsToCopyHost[planIndex] < 0) {
                continue;
            }
            const size_t begin = (static_cast<size_t>(options.experts) +
                static_cast<size_t>(slot)) * rowElements;
            std::fill(expected.begin() + static_cast<ptrdiff_t>(begin),
                expected.begin() + static_cast<ptrdiff_t>(begin + rowElements), 0.0F);
        }
        return expected;
    };
    const std::vector<float> expectedGateGrad = buildExpectedGrad(
        gateGrad, gateRowElements, 100);
    const std::vector<float> expectedUpGrad = buildExpectedGrad(
        upGrad, upRowElements, 200);
    const std::vector<float> expectedDownGrad = buildExpectedGrad(
        downGrad, downRowElements, 300);

    correct = CheckEqual(
        rank, "forward dispatch", forwardDispatchHost, expectedForwardDispatch) && correct;
    correct = CheckEqual(rank, "prefetch gate", gateWeightHost, expectedGateWeight) && correct;
    correct = CheckEqual(rank, "prefetch up", upWeightHost, expectedUpWeight) && correct;
    correct = CheckEqual(rank, "prefetch down", downWeightHost, expectedDownWeight) && correct;
    correct = CheckEqual(
        rank, "forward combine", forwardCombineHost, expectedForwardCombine) && correct;
    correct = CheckEqual(rank, "dispatched route weights", dispatchedRouteWeightsHost,
        expectedDispatchedRouteWeights) && correct;
    correct = CheckEqual(rank, "combined route weights", combinedRouteWeightsHost,
        routeWeights) && correct;
    correct = CheckEqual(
        rank, "backward dispatch", backwardDispatchHost, expectedBackwardDispatch) && correct;
    correct = CheckEqual(
        rank, "backward combine", backwardCombineHost, expectedBackwardCombine) && correct;
    correct = CheckEqual(rank, "reduce gate grad", gateGradHost, expectedGateGrad) && correct;
    correct = CheckEqual(rank, "reduce up grad", upGradHost, expectedUpGrad) && correct;
    correct = CheckEqual(rank, "reduce down grad", downGradHost, expectedDownGrad) && correct;

    if (correct) {
        const bool oversubscribed = options.world > options.physicalDeviceCount;
        const char *blockDim = std::getenv("TILEXR_MOONEP_PLANNER_BLOCK_DIM");
        std::cout << "[rank " << rank << "] native MoonEP flow smoke passed"
                  << " logical_ranks=" << options.world
                  << " physical_devices=" << options.physicalDeviceCount
                  << " device=" << options.device
                  << " oversubscribed=" << (oversubscribed ? "true" : "false")
                  << " block_dim=" << (blockDim == nullptr ? "<default>" : blockDim)
                  << " S=" << options.s << " K=" << options.k
                  << " E=" << options.experts << " H=" << options.hidden
                  << " NvS=" << nvS
                  << " planning_status=" << kExpectedPlanningStatusSuccess
                  << " dispatch_status=" << kExpectedDispatchStatusSuccess
                  << " prefetch_weight_status=" << kExpectedPrefetchStatusSuccess
                  << " combine_status=" << kExpectedCombineStatusSuccess
                  << " reduce_grad_status=" << kExpectedReduceGradStatusSuccess
                  << " reduce_grad_transport=owner_pull_udma"
                  << " cu_last=" << cuHost.back()
                  << " planning=native"
                  << " dispatch=native prefetch_weight=native_udma_registered"
                  << " combine=native reduce_grad=native_owner_pull_udma"
                  << " torch_validated=false"
                  << " transport_performance_valid=false"
                  << std::endl;
    }
    return correct;
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    if (!ParseOptions(argc, argv, &options)) {
        std::cerr << "invalid arguments; use --help" << std::endl;
        return 2;
    }
    uint64_t routeCount = 0;
    uint64_t tokenHiddenCount = 0;
    uint64_t routeHiddenCount = 0;
    uint64_t expertHiddenCount = 0;
    if (!ValidateOptions(options, &routeCount, &tokenHiddenCount,
            &routeHiddenCount, &expertHiddenCount)) {
        std::cerr << "invalid flow dimensions or rank/device mapping" << std::endl;
        return 2;
    }

    RuntimeResources resources;
    resources.rank = options.rank;
    resources.device = options.device;
    const char *commId = std::getenv("TILEXR_COMM_ID");
    std::cout << "[rank " << options.rank << "] starting native MoonEP flow"
              << " world=" << options.world << " device=" << options.device
              << " comm_id=" << (commId == nullptr ? "<unset>" : commId)
              << std::endl;

    bool ok = CheckAcl(options.rank, "aclInit", aclInit(nullptr));
    resources.aclInitialized = ok;
    if (ok) {
        ok = CheckAcl(options.rank, "aclrtSetDevice",
            aclrtSetDevice(options.device));
        resources.deviceSet = ok;
    }
    if (ok) {
        ok = CheckAcl(options.rank, "aclrtCreateStream",
            aclrtCreateStream(&resources.stream));
    }
    if (ok) {
        ok = CheckTileXR(options.rank, "TileXRCommInitRankLocal",
            TileXRCommInitRankLocal(options.world, options.rank, &resources.comm));
    }
    if (ok) {
        ok = RunFlow(options, &resources, routeCount, tokenHiddenCount,
            routeHiddenCount, expertHiddenCount);
    }

    bool localCompletionOk = true;
    if (resources.comm != nullptr && resources.stream != nullptr) {
        localCompletionOk = CheckAcl(options.rank, "local completion synchronize",
            aclrtSynchronizeStream(resources.stream));
    }
    bool rendezvousOk = true;
    if (resources.comm != nullptr) {
        rendezvousOk = DemoBarrierAll(options.rank, options.world);
    }
    ok = ok && localCompletionOk && rendezvousOk;

    const bool cleanupOk = Cleanup(&resources);
    return ok && cleanupOk ? 0 : 1;
}
