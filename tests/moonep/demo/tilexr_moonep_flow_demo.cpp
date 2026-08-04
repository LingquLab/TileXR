#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
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

bool CountBytes(uint64_t elements, uint64_t *bytes)
{
    return CheckedMultiply(elements, sizeof(int32_t), bytes) &&
        *bytes <= static_cast<uint64_t>(std::numeric_limits<size_t>::max());
}

bool ValidateOptions(const Options &options, uint64_t *routeCount,
                     uint64_t *tokenHiddenCount, uint64_t *routeHiddenCount,
                     uint64_t *expertHiddenCount)
{
    if (routeCount == nullptr || tokenHiddenCount == nullptr ||
        routeHiddenCount == nullptr || expertHiddenCount == nullptr ||
        options.world <= 0 || options.world > 128 || options.rank < 0 ||
        options.rank >= options.world || options.physicalDeviceCount <= 0 ||
        options.device < 0 || options.device >= options.physicalDeviceCount ||
        options.s <= 0 || options.k < 2 || options.experts <= 0 ||
        options.hidden <= 0 || options.experts % options.world != 0 ||
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

bool AllocateInt32(RuntimeResources *resources, uint64_t elements,
                   const std::string &name, DeviceBuffer *buffer)
{
    uint64_t bytes = 0;
    return CountBytes(elements, &bytes) && Allocate(resources, bytes, name, buffer);
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
    for (std::vector<void *>::reverse_iterator it = resources->allocations.rbegin();
         it != resources->allocations.rend(); ++it) {
        if (*it != nullptr) {
            ok = CheckAcl(resources->rank, "aclrtFree", aclrtFree(*it)) && ok;
        }
    }
    resources->allocations.clear();
    if (resources->comm != nullptr) {
        ok = CheckTileXR(resources->rank, "TileXRCommDestroy",
            TileXRCommDestroy(resources->comm)) && ok;
        resources->comm = nullptr;
    }
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

TileXRMoonEpTensorV1 Tensor1D(void *data, uint64_t elements, int64_t dim0)
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.structSize = sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.data = data;
    tensor.elementCount = elements;
    tensor.dtype = TILEXR_MOONEP_DTYPE_INT32;
    tensor.rank = 1;
    tensor.shape[0] = dim0;
    return tensor;
}

TileXRMoonEpTensorV1 Tensor2D(void *data, uint64_t elements,
                             int64_t dim0, int64_t dim1)
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.structSize = sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.data = data;
    tensor.elementCount = elements;
    tensor.dtype = TILEXR_MOONEP_DTYPE_INT32;
    tensor.rank = 2;
    tensor.shape[0] = dim0;
    tensor.shape[1] = dim1;
    return tensor;
}

template <typename Args>
Args StageArgs(TileXRCommPtr comm, const TileXRMoonEpPlanV1 *plan,
               const TileXRMoonEpTensorV1 *input, TileXRMoonEpTensorV1 *output)
{
    Args args {};
    args.structSize = sizeof(args);
    args.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    args.comm = comm;
    args.plan = plan;
    args.input = input;
    args.output = output;
    args.flags = TILEXR_MOONEP_FLAG_NONE;
    return args;
}

bool CheckEqual(int rank, const std::string &name,
                const std::vector<int32_t> &actual,
                const std::vector<int32_t> &expected)
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

bool CheckPrefixAndZeroTail(int rank, const std::string &name,
                            const std::vector<int32_t> &actual,
                            const std::vector<int32_t> &prefix)
{
    if (actual.size() <= prefix.size()) {
        std::cerr << "[rank " << rank << "] " << name
                  << " has no bounded-copy zero tail" << std::endl;
        return false;
    }
    for (size_t index = 0; index < prefix.size(); ++index) {
        if (actual[index] != prefix[index]) {
            std::cerr << "[rank " << rank << "] " << name
                      << " prefix mismatch index=" << index << std::endl;
            return false;
        }
    }
    for (size_t index = prefix.size(); index < actual.size(); ++index) {
        if (actual[index] != 0) {
            std::cerr << "[rank " << rank << "] " << name
                      << " tail not zero index=" << index
                      << " actual=" << actual[index] << std::endl;
            return false;
        }
    }
    return true;
}

bool RunFlow(const Options &options, RuntimeResources *resources,
             uint64_t routeCount, uint64_t tokenHiddenCount,
             uint64_t routeHiddenCount, uint64_t expertHiddenCount)
{
    const int rank = options.rank;
    const int64_t b = options.experts / options.world;
    const uint64_t groupCount =
        static_cast<uint64_t>(options.experts) + static_cast<uint64_t>(b);
    const uint64_t expertsToCopyCount =
        static_cast<uint64_t>(options.world) * static_cast<uint64_t>(b);

    uint64_t nativeStages = 0;
    uint64_t stubStages = 0;
    const uint64_t expectedStubs =
        static_cast<uint64_t>(TILEXR_MOONEP_STAGE_DISPATCH) |
        static_cast<uint64_t>(TILEXR_MOONEP_STAGE_PREFETCH_WEIGHT) |
        static_cast<uint64_t>(TILEXR_MOONEP_STAGE_COMBINE) |
        static_cast<uint64_t>(TILEXR_MOONEP_STAGE_REDUCE_GRAD);
    if (TileXRMoonEpGetAbiVersion() != TILEXR_MOONEP_ABI_VERSION_V1 ||
        !CheckTileXR(rank, "TileXRMoonEpGetCapabilitiesV1",
            TileXRMoonEpGetCapabilitiesV1(&nativeStages, &stubStages)) ||
        nativeStages != TILEXR_MOONEP_STAGE_PLANNING || stubStages != expectedStubs) {
        std::cerr << "[rank " << rank << "] unexpected native/stub capabilities"
                  << " native=" << nativeStages << " stub=" << stubStages << std::endl;
        return false;
    }

    uint64_t workspaceBytes = 0;
    int64_t dispatchedCapacity = 0;
    if (!CheckTileXR(rank, "TileXRMoonEpPlanningGetWorkspaceSizeV1",
            TileXRMoonEpPlanningGetWorkspaceSizeV1(resources->comm, options.s,
                options.k, options.experts, &workspaceBytes, &dispatchedCapacity)) ||
        dispatchedCapacity != static_cast<int64_t>(routeCount)) {
        std::cerr << "[rank " << rank << "] NvS mismatch actual="
                  << dispatchedCapacity << " expected=" << routeCount << std::endl;
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

    const std::vector<int32_t> forwardInput =
        MakePattern(tokenHiddenCount, static_cast<int64_t>(rank + 1) * 100003);
    const std::vector<int32_t> prefetchInput =
        MakePattern(expertHiddenCount, static_cast<int64_t>(rank + 1) * 200003);
    const std::vector<int32_t> backwardInput =
        MakePattern(tokenHiddenCount, static_cast<int64_t>(rank + 1) * 300007);
    const std::vector<int32_t> reduceInput =
        MakePattern(expertHiddenCount, static_cast<int64_t>(rank + 1) * 400009);

    DeviceBuffer topk;
    DeviceBuffer tpe;
    DeviceBuffer workspace;
    DeviceBuffer dst;
    DeviceBuffer cu;
    DeviceBuffer expertsToCopy;
    DeviceBuffer remoteStats;
    DeviceBuffer plannerStatus;
    DeviceBuffer forwardInputDev;
    DeviceBuffer forwardDispatchDev;
    DeviceBuffer prefetchInputDev;
    DeviceBuffer prefetchOutputDev;
    DeviceBuffer forwardCombineDev;
    DeviceBuffer backwardInputDev;
    DeviceBuffer backwardDispatchDev;
    DeviceBuffer backwardCombineDev;
    DeviceBuffer reduceInputDev;
    DeviceBuffer reduceOutputDev;

    if (!AllocateInt32(resources, routeCount, "topk", &topk) ||
        !AllocateInt32(resources, options.experts, "tokens per expert", &tpe) ||
        !Allocate(resources, workspaceBytes, "planner workspace", &workspace) ||
        !AllocateInt32(resources, routeCount, "dst", &dst) ||
        !AllocateInt32(resources, groupCount, "cu", &cu) ||
        !AllocateInt32(resources, expertsToCopyCount, "experts to copy", &expertsToCopy) ||
        !AllocateInt32(resources, 2, "remote stats", &remoteStats) ||
        !AllocateInt32(resources, 1, "planner status", &plannerStatus) ||
        !AllocateInt32(resources, tokenHiddenCount, "forward input", &forwardInputDev) ||
        !AllocateInt32(resources, routeHiddenCount, "forward dispatch", &forwardDispatchDev) ||
        !AllocateInt32(resources, expertHiddenCount, "prefetch input", &prefetchInputDev) ||
        !AllocateInt32(resources, expertHiddenCount, "prefetch output", &prefetchOutputDev) ||
        !AllocateInt32(resources, tokenHiddenCount, "forward combine", &forwardCombineDev) ||
        !AllocateInt32(resources, tokenHiddenCount, "backward input", &backwardInputDev) ||
        !AllocateInt32(resources, routeHiddenCount, "backward dispatch", &backwardDispatchDev) ||
        !AllocateInt32(resources, tokenHiddenCount, "backward combine", &backwardCombineDev) ||
        !AllocateInt32(resources, expertHiddenCount, "reduce input", &reduceInputDev) ||
        !AllocateInt32(resources, expertHiddenCount, "reduce output", &reduceOutputDev)) {
        return false;
    }
    if ((reinterpret_cast<uintptr_t>(workspace.data) & static_cast<uintptr_t>(31)) != 0) {
        std::cerr << "[rank " << rank << "] planner workspace is not 32-byte aligned"
                  << std::endl;
        return false;
    }

    const std::vector<int32_t> statusSentinel(1, -1);
    if (!CopyHostToDevice(rank, topk, routing, "topk") ||
        !CopyHostToDevice(rank, tpe, tokensPerExpert, "tokens per expert") ||
        !CopyHostToDevice(rank, plannerStatus, statusSentinel, "planner status sentinel") ||
        !CopyHostToDevice(rank, forwardInputDev, forwardInput, "forward input") ||
        !CopyHostToDevice(rank, prefetchInputDev, prefetchInput, "prefetch input") ||
        !CopyHostToDevice(rank, backwardInputDev, backwardInput, "backward input") ||
        !CopyHostToDevice(rank, reduceInputDev, reduceInput, "reduce input")) {
        return false;
    }

    TileXRMoonEpPlanV1 plan {};
    plan.structSize = sizeof(plan);
    plan.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    plan.s = options.s;
    plan.k = options.k;
    plan.e = options.experts;
    plan.b = b;
    plan.rank = options.rank;
    plan.world = options.world;
    plan.dispatchedCapacity = dispatchedCapacity;
    plan.dst = dst.data;
    plan.cu = cu.data;
    plan.expertsToCopy = expertsToCopy.data;
    plan.remoteStats = remoteStats.data;
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
    planning.plan = &plan;
    planning.waitIterations = options.waitIterations;
    planning.flags = TILEXR_MOONEP_FLAG_NONE;

    TileXRMoonEpTensorV1 forwardInputTensor =
        Tensor2D(forwardInputDev.data, tokenHiddenCount, options.s, options.hidden);
    TileXRMoonEpTensorV1 forwardDispatchTensor =
        Tensor2D(forwardDispatchDev.data, routeHiddenCount,
            dispatchedCapacity, options.hidden);
    TileXRMoonEpDispatchArgsV1 forwardDispatch =
        StageArgs<TileXRMoonEpDispatchArgsV1>(
            resources->comm, &plan, &forwardInputTensor, &forwardDispatchTensor);

    TileXRMoonEpTensorV1 prefetchInputTensor =
        Tensor2D(prefetchInputDev.data, expertHiddenCount, b, options.hidden);
    TileXRMoonEpTensorV1 prefetchOutputTensor =
        Tensor2D(prefetchOutputDev.data, expertHiddenCount, b, options.hidden);
    TileXRMoonEpPrefetchWeightArgsV1 prefetch =
        StageArgs<TileXRMoonEpPrefetchWeightArgsV1>(
            resources->comm, &plan, &prefetchInputTensor, &prefetchOutputTensor);

    TileXRMoonEpTensorV1 forwardCombineTensor =
        Tensor2D(forwardCombineDev.data, tokenHiddenCount, options.s, options.hidden);
    TileXRMoonEpCombineArgsV1 forwardCombine =
        StageArgs<TileXRMoonEpCombineArgsV1>(
            resources->comm, &plan, &forwardDispatchTensor, &forwardCombineTensor);

    TileXRMoonEpTensorV1 backwardInputTensor =
        Tensor2D(backwardInputDev.data, tokenHiddenCount, options.s, options.hidden);
    TileXRMoonEpTensorV1 backwardDispatchTensor =
        Tensor2D(backwardDispatchDev.data, routeHiddenCount,
            dispatchedCapacity, options.hidden);
    TileXRMoonEpDispatchArgsV1 backwardDispatch =
        StageArgs<TileXRMoonEpDispatchArgsV1>(
            resources->comm, &plan, &backwardInputTensor, &backwardDispatchTensor);

    TileXRMoonEpTensorV1 backwardCombineTensor =
        Tensor2D(backwardCombineDev.data, tokenHiddenCount, options.s, options.hidden);
    TileXRMoonEpCombineArgsV1 backwardCombine =
        StageArgs<TileXRMoonEpCombineArgsV1>(
            resources->comm, &plan, &backwardDispatchTensor, &backwardCombineTensor);

    TileXRMoonEpTensorV1 reduceInputTensor =
        Tensor2D(reduceInputDev.data, expertHiddenCount, b, options.hidden);
    TileXRMoonEpTensorV1 reduceOutputTensor =
        Tensor2D(reduceOutputDev.data, expertHiddenCount, b, options.hidden);
    TileXRMoonEpReduceGradArgsV1 reduceGrad =
        StageArgs<TileXRMoonEpReduceGradArgsV1>(
            resources->comm, &plan, &reduceInputTensor, &reduceOutputTensor);

    if (!CheckTileXR(rank, "Planning",
            TileXRMoonEpPlanningV1(&planning, resources->stream)) ||
        !CheckTileXR(rank, "Dispatch forward",
            TileXRMoonEpDispatchV1(&forwardDispatch, resources->stream)) ||
        !CheckTileXR(rank, "PrefetchWeight",
            TileXRMoonEpPrefetchWeightV1(&prefetch, resources->stream)) ||
        !CheckTileXR(rank, "Combine forward",
            TileXRMoonEpCombineV1(&forwardCombine, resources->stream)) ||
        !CheckTileXR(rank, "Dispatch backward",
            TileXRMoonEpDispatchV1(&backwardDispatch, resources->stream)) ||
        !CheckTileXR(rank, "Combine backward",
            TileXRMoonEpCombineV1(&backwardCombine, resources->stream)) ||
        !CheckTileXR(rank, "ReduceGrad",
            TileXRMoonEpReduceGradV1(&reduceGrad, resources->stream)) ||
        !CheckAcl(rank, "flow stream synchronize",
            aclrtSynchronizeStream(resources->stream))) {
        return false;
    }

    std::vector<int32_t> statusHost(1);
    std::vector<int32_t> cuHost(static_cast<size_t>(groupCount));
    std::vector<int32_t> forwardDispatchHost(static_cast<size_t>(routeHiddenCount));
    std::vector<int32_t> prefetchOutputHost(static_cast<size_t>(expertHiddenCount));
    std::vector<int32_t> forwardCombineHost(static_cast<size_t>(tokenHiddenCount));
    std::vector<int32_t> backwardDispatchHost(static_cast<size_t>(routeHiddenCount));
    std::vector<int32_t> backwardCombineHost(static_cast<size_t>(tokenHiddenCount));
    std::vector<int32_t> reduceOutputHost(static_cast<size_t>(expertHiddenCount));
    if (!CopyDeviceToHost(rank, &statusHost, plannerStatus, "planner status") ||
        !CopyDeviceToHost(rank, &cuHost, cu, "cu") ||
        !CopyDeviceToHost(rank, &forwardDispatchHost, forwardDispatchDev,
            "forward dispatch") ||
        !CopyDeviceToHost(rank, &prefetchOutputHost, prefetchOutputDev,
            "prefetch output") ||
        !CopyDeviceToHost(rank, &forwardCombineHost, forwardCombineDev,
            "forward combine") ||
        !CopyDeviceToHost(rank, &backwardDispatchHost, backwardDispatchDev,
            "backward dispatch") ||
        !CopyDeviceToHost(rank, &backwardCombineHost, backwardCombineDev,
            "backward combine") ||
        !CopyDeviceToHost(rank, &reduceOutputHost, reduceOutputDev,
            "reduce output")) {
        return false;
    }

    bool correct = true;
    if (statusHost[0] != TILEXR_MOONEP_PLANNER_STATUS_SUCCESS) {
        std::cerr << "[rank " << rank << "] planner status=" << statusHost[0];
        if (statusHost[0] >= TILEXR_MOONEP_PLANNER_STATUS_TIMEOUT_BASE) {
            std::cerr << " timeout_peer="
                      << statusHost[0] - TILEXR_MOONEP_PLANNER_STATUS_TIMEOUT_BASE;
        }
        std::cerr << std::endl;
        correct = false;
    }
    if (cuHost.empty() || cuHost.back() != static_cast<int32_t>(routeCount)) {
        std::cerr << "[rank " << rank << "] cu last mismatch actual="
                  << (cuHost.empty() ? -1 : cuHost.back())
                  << " expected=" << routeCount << std::endl;
        correct = false;
    }
    correct = CheckPrefixAndZeroTail(
        rank, "forward dispatch", forwardDispatchHost, forwardInput) && correct;
    correct = CheckEqual(
        rank, "prefetch weight", prefetchOutputHost, prefetchInput) && correct;
    correct = CheckEqual(
        rank, "forward combine", forwardCombineHost, forwardInput) && correct;
    correct = CheckPrefixAndZeroTail(
        rank, "backward dispatch", backwardDispatchHost, backwardInput) && correct;
    correct = CheckEqual(
        rank, "backward combine", backwardCombineHost, backwardInput) && correct;
    correct = CheckEqual(
        rank, "reduce grad", reduceOutputHost, reduceInput) && correct;

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
                  << " NvS=" << dispatchedCapacity
                  << " planner_status=" << statusHost[0]
                  << " cu_last=" << cuHost.back()
                  << " planning=native"
                  << " dispatch=stub prefetch_weight=stub"
                  << " combine=stub reduce_grad=stub"
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
