#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "acl/acl.h"
#include "combine_v2_profile.h"
#include "combine_v2_schedule.h"
#include "tilexr_api.h"
#include "tilexr_moonep_combine_v2.h"
#include "tilexr_types.h"

namespace {

constexpr int kDeviceCount = 8;
constexpr int kDefaultCommDomain = 141;
constexpr int64_t kH = 3584;
constexpr int64_t kTopK = 16;
constexpr int kDefaultExpertCount = 64;
constexpr uint32_t kAivCoreNum = 16;
constexpr uint32_t kQpCount = 32;
constexpr int kBarrierPortOffset = 97;
constexpr int kBarrierTimeoutSeconds = 60;
const char *const kProfileMetricNames[
    TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount] = {
    "selection_load_us",
    "selection_select_us",
    "self_route_decode_us",
    "self_copy_us",
    "remote_route_decode_us",
    "remote_descriptor_us",
    "remote_wqe_build_us",
    "remote_submit_us",
};

struct HostPort {
    std::string host;
    int port = 0;
};

struct Options {
    std::vector<int64_t> batchSizes {128};
    int warmup = 20;
    int iterations = 80;
    int experts = kDefaultExpertCount;
    int commDomain = kDefaultCommDomain;
    int rank = -1;
    int worldSize = 0;
    int device = -1;
    bool skipIterationBarriers = false;
    bool profile = false;
    bool allowSelfOnlyFailure = false;
};

struct ProfileSample {
    int iteration = 0;
    uint32_t core = 0U;
    std::array<int64_t,
        TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCount> timePoint {};
    std::array<uint64_t,
        TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount> metric {};
};

enum class OutputCheckResult {
    Passed,
    SelfOnlyFailed,
    Failed,
};

const char *OutputCheckName(OutputCheckResult result)
{
    switch (result) {
        case OutputCheckResult::Passed:
            return "passed";
        case OutputCheckResult::SelfOnlyFailed:
            return "self_only_failed";
        default:
            return "failed";
    }
}

[[noreturn]] void Abort(int rank, const std::string &step, int status)
{
    std::cerr << "[rank " << rank << "] " << step
              << " failed, status=" << status << std::endl;
    std::exit(status == 0 ? 1 : status);
}

void Usage(std::ostream &out, const char *program)
{
    out << "Usage: " << program << " [options]\n"
        << "  --bs N                 Run one batch size (default: 128)\n"
        << "  --bs-list N[,N...]     Run multiple batch sizes after one initialization\n"
        << "  --warmup N             Warmup launches per batch size (default: 20)\n"
        << "  --iterations N         Timed launches per batch size (default: 80)\n"
        << "  --experts N            Total expert count (default: 64)\n"
        << "  --comm-domain N        Shared-QP communication domain (default: 141)\n"
        << "  --rank N               Global rank (required)\n"
        << "  --world-size N         Global rank count (required)\n"
        << "  --device N             Local device id (default: rank modulo 8)\n"
        << "  --skip-iteration-barriers\n"
        << "                         Skip host barriers between warmup/timed launches\n"
        << "  --profile              Capture per-AIV kernel cycle timestamps\n"
        << "  --allow-self-only-failure\n"
        << "                         Continue timing when only Self-copy validation fails\n"
        << "  --help                 Show this help\n";
}

bool ParseInteger(const std::string &text, int64_t minValue,
    int64_t maxValue, int64_t *value)
{
    if (value == nullptr || text.empty()) {
        return false;
    }
    std::size_t consumed = 0;
    try {
        const long long parsed = std::stoll(text, &consumed, 10);
        if (consumed != text.size() || parsed < minValue || parsed > maxValue) {
            return false;
        }
        *value = static_cast<int64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool ParseBatchSizes(const std::string &text, std::vector<int64_t> *values)
{
    if (values == nullptr || text.empty()) {
        return false;
    }
    std::vector<int64_t> parsed;
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        int64_t value = 0;
        if (!ParseInteger(item, 1, std::numeric_limits<int32_t>::max(),
                &value)) {
            return false;
        }
        if (std::find(parsed.begin(), parsed.end(), value) == parsed.end()) {
            parsed.push_back(value);
        }
    }
    if (parsed.empty()) {
        return false;
    }
    *values = parsed;
    return true;
}

bool ParseOptions(int argc, char **argv, Options *options, bool *showHelp,
    std::string *error)
{
    if (options == nullptr || showHelp == nullptr || error == nullptr) {
        return false;
    }
    bool batchSizeSet = false;
    *showHelp = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help" || argument == "-h") {
            *showHelp = true;
            return true;
        }
        if (argument == "--skip-iteration-barriers") {
            options->skipIterationBarriers = true;
            continue;
        }
        if (argument == "--profile") {
            options->profile = true;
            continue;
        }
        if (argument == "--allow-self-only-failure") {
            options->allowSelfOnlyFailure = true;
            continue;
        }
        if (index + 1 >= argc) {
            *error = "missing value for " + argument;
            return false;
        }
        const std::string value = argv[++index];
        if (argument == "--bs" || argument == "--bs-list") {
            if (batchSizeSet) {
                *error = "--bs and --bs-list are mutually exclusive";
                return false;
            }
            batchSizeSet = true;
            if (!ParseBatchSizes(value, &options->batchSizes) ||
                (argument == "--bs" && options->batchSizes.size() != 1U)) {
                *error = "invalid batch size list: " + value;
                return false;
            }
        } else if (argument == "--warmup") {
            int64_t parsed = 0;
            if (!ParseInteger(value, 0, std::numeric_limits<int>::max(),
                    &parsed)) {
                *error = "invalid warmup count: " + value;
                return false;
            }
            options->warmup = static_cast<int>(parsed);
        } else if (argument == "--iterations") {
            int64_t parsed = 0;
            if (!ParseInteger(value, 1, std::numeric_limits<int>::max(),
                    &parsed)) {
                *error = "invalid iteration count: " + value;
                return false;
            }
            options->iterations = static_cast<int>(parsed);
        } else if (argument == "--experts") {
            int64_t parsed = 0;
            if (!ParseInteger(value, 1, std::numeric_limits<int>::max(),
                    &parsed)) {
                *error = "invalid expert count: " + value;
                return false;
            }
            options->experts = static_cast<int>(parsed);
        } else if (argument == "--comm-domain") {
            int64_t parsed = 0;
            if (!ParseInteger(value, 1, std::numeric_limits<int>::max(),
                    &parsed)) {
                *error = "invalid communication domain: " + value;
                return false;
            }
            options->commDomain = static_cast<int>(parsed);
        } else if (argument == "--rank") {
            int64_t parsed = 0;
            if (!ParseInteger(value, 0, std::numeric_limits<int>::max(),
                    &parsed)) {
                *error = "invalid rank: " + value;
                return false;
            }
            options->rank = static_cast<int>(parsed);
        } else if (argument == "--world-size") {
            int64_t parsed = 0;
            if (!ParseInteger(value, 1, std::numeric_limits<int>::max(),
                    &parsed)) {
                *error = "invalid world size: " + value;
                return false;
            }
            options->worldSize = static_cast<int>(parsed);
        } else if (argument == "--device") {
            int64_t parsed = 0;
            if (!ParseInteger(value, 0, kDeviceCount - 1, &parsed)) {
                *error = "invalid device id: " + value;
                return false;
            }
            options->device = static_cast<int>(parsed);
        } else {
            *error = "unknown argument: " + argument;
            return false;
        }
    }
    if (options->rank < 0 || options->worldSize <= 0 ||
        options->rank >= options->worldSize) {
        *error = "--rank and --world-size must identify a valid rank";
        return false;
    }
    return true;
}

uint16_t FloatToBfloat16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t roundToEven = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>((bits + roundToEven) >> 16U);
}

uint16_t SourceValue(int sourceRank)
{
    const float value = 1.0F + static_cast<float>(sourceRank % 16) * 0.25F +
        static_cast<float>(sourceRank / 16) * 0.0625F;
    return FloatToBfloat16(value);
}

void CheckAcl(int rank, const std::string &step, aclError status)
{
    if (status != ACL_SUCCESS) {
        Abort(rank, step, static_cast<int>(status));
    }
}

bool ParseHostPort(const std::string &text, HostPort *endpoint)
{
    const std::size_t separator = text.rfind(':');
    if (endpoint == nullptr || separator == std::string::npos ||
        separator == 0 || separator + 1 >= text.size()) {
        return false;
    }
    int64_t port = 0;
    if (!ParseInteger(text.substr(separator + 1), 1, 65535, &port)) {
        return false;
    }
    in_addr address {};
    const std::string host = text.substr(0, separator);
    if (inet_pton(AF_INET, host.c_str(), &address) != 1) {
        return false;
    }
    endpoint->host = host;
    endpoint->port = static_cast<int>(port);
    return true;
}

bool GetBarrierEndpoint(HostPort *endpoint)
{
    const char *configured = std::getenv("TILEXR_DEMO_BARRIER_ADDR");
    if (configured != nullptr && configured[0] != '\0') {
        return ParseHostPort(configured, endpoint);
    }
    const char *commId = std::getenv("TILEXR_COMM_ID");
    HostPort commEndpoint;
    if (commId == nullptr || !ParseHostPort(commId, &commEndpoint)) {
        return false;
    }
    endpoint->host = commEndpoint.host;
    endpoint->port = commEndpoint.port + kBarrierPortOffset;
    if (endpoint->port > 65535) {
        endpoint->port = commEndpoint.port - kBarrierPortOffset;
    }
    return endpoint->port > 0;
}

bool SendAll(int fd, const void *data, std::size_t bytes)
{
    const uint8_t *cursor = static_cast<const uint8_t *>(data);
    while (bytes > 0) {
        const ssize_t sent = send(fd, cursor, bytes, 0);
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        if (sent <= 0) {
            return false;
        }
        cursor += sent;
        bytes -= static_cast<std::size_t>(sent);
    }
    return true;
}

bool RecvAll(int fd, void *data, std::size_t bytes)
{
    uint8_t *cursor = static_cast<uint8_t *>(data);
    while (bytes > 0) {
        const ssize_t received = recv(fd, cursor, bytes, 0);
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received <= 0) {
            return false;
        }
        cursor += received;
        bytes -= static_cast<std::size_t>(received);
    }
    return true;
}

void SetSocketTimeout(int fd)
{
    timeval timeout {};
    timeout.tv_sec = kBarrierTimeoutSeconds;
    (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

bool BarrierServer(int world, const HostPort &endpoint, bool localSuccess)
{
    const int listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd < 0) {
        return false;
    }
    int reuse = 1;
    (void)setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    SetSocketTimeout(listenFd);

    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(static_cast<uint16_t>(endpoint.port));
    if (bind(listenFd, reinterpret_cast<sockaddr *>(&address),
            sizeof(address)) != 0 || listen(listenFd, world) != 0) {
        close(listenFd);
        return false;
    }

    bool globalSuccess = localSuccess;
    bool exchangeOk = true;
    std::vector<int> clients;
    clients.reserve(static_cast<std::size_t>(world - 1));
    for (int peer = 1; peer < world; ++peer) {
        const int client = accept(listenFd, nullptr, nullptr);
        if (client < 0) {
            exchangeOk = false;
            break;
        }
        SetSocketTimeout(client);
        uint8_t arrived = 0;
        if (!RecvAll(client, &arrived, sizeof(arrived)) || arrived > 1U) {
            exchangeOk = false;
        }
        globalSuccess = globalSuccess && arrived == 1U;
        clients.push_back(client);
        if (!exchangeOk) {
            break;
        }
    }

    const uint8_t release = exchangeOk && globalSuccess ? 1U : 0U;
    for (const int client : clients) {
        if (!SendAll(client, &release, sizeof(release))) {
            exchangeOk = false;
        }
        close(client);
    }
    close(listenFd);
    return exchangeOk && globalSuccess;
}

bool BarrierClient(const HostPort &endpoint, bool localSuccess)
{
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(endpoint.port));
    if (inet_pton(AF_INET, endpoint.host.c_str(), &address.sin_addr) != 1) {
        return false;
    }

    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(kBarrierTimeoutSeconds);
    int fd = -1;
    while (std::chrono::steady_clock::now() < deadline) {
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            return false;
        }
        SetSocketTimeout(fd);
        if (connect(fd, reinterpret_cast<sockaddr *>(&address),
                sizeof(address)) == 0) {
            break;
        }
        close(fd);
        fd = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (fd < 0) {
        return false;
    }

    const uint8_t arrived = localSuccess ? 1U : 0U;
    uint8_t release = 0;
    const bool ok = SendAll(fd, &arrived, sizeof(arrived)) &&
        RecvAll(fd, &release, sizeof(release)) && release == 1U;
    close(fd);
    return ok;
}

bool BarrierAll(int rank, int world, const std::string &step,
    bool localSuccess = true)
{
    if (world <= 1) {
        return localSuccess;
    }
    HostPort endpoint;
    if (!GetBarrierEndpoint(&endpoint)) {
        std::cerr << "[rank " << rank
                  << "] invalid TILEXR_DEMO_BARRIER_ADDR" << std::endl;
        return false;
    }
    const bool ok = rank == 0 ? BarrierServer(world, endpoint, localSuccess) :
        BarrierClient(endpoint, localSuccess);
    if (!ok) {
        std::cerr << "[rank " << rank << "] barrier failed after " << step
                  << " at " << endpoint.host << ':' << endpoint.port
                  << std::endl;
    }
    return ok;
}

void LaunchCombine(int rank, void *workspace, const int32_t *dst,
    TileXRCommPtr comm, int64_t bs, aclrtStream stream,
    uint64_t *activeOutputOffset)
{
    const int64_t slots = bs * kTopK;
    const int ret = TileXRMoonEpCombineV2(workspace, dst, comm, bs, kH,
        kTopK, slots, kAivCoreNum, activeOutputOffset,
        TILEXR_MOONEP_DTYPE_BFLOAT16, stream);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        Abort(rank, "TileXRMoonEpCombineV2", ret);
    }
}

OutputCheckResult CheckOutput(int rank, int world, int64_t bs,
    const void *workspace,
    uint64_t activeOutputOffset)
{
    const int64_t slots = bs * kTopK;
    const std::size_t outputElements = static_cast<std::size_t>(slots) *
        static_cast<std::size_t>(kH);
    const std::size_t outputBytes = outputElements * sizeof(uint16_t);
    std::vector<uint16_t> output(outputElements);
    const void *outputDevice = static_cast<const uint8_t *>(workspace) +
        activeOutputOffset;
    CheckAcl(rank, "output D2H copy", aclrtMemcpy(output.data(), outputBytes,
        outputDevice, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    const int64_t slotsPerSourceRank = slots / world;
    bool selfMismatch = false;
    for (int64_t slot = 0; slot < slots; ++slot) {
        const int sourceRank = static_cast<int>(slot / slotsPerSourceRank);
        if (sourceRank == rank && selfMismatch) {
            continue;
        }
        const uint16_t expected = SourceValue(sourceRank);
        const std::size_t rowOffset = static_cast<std::size_t>(slot) *
            static_cast<std::size_t>(kH);
        for (int64_t column = 0; column < kH; ++column) {
            const std::size_t index = rowOffset +
                static_cast<std::size_t>(column);
            if (output[index] != expected) {
                std::cerr << "[rank " << rank << "] output mismatch"
                          << " bs=" << bs
                          << " slot=" << slot
                          << " column=" << column
                          << " source_rank=" << sourceRank
                          << " got=" << output[index]
                          << " expected=" << expected << std::endl;
                if (sourceRank != rank) {
                    return OutputCheckResult::Failed;
                }
                selfMismatch = true;
                break;
            }
        }
    }
    return selfMismatch ? OutputCheckResult::SelfOnlyFailed :
        OutputCheckResult::Passed;
}

void CaptureProfileSamples(int rank, int world, int iteration,
    const void *workspace, uint64_t profileOffset,
    std::vector<ProfileSample> *samples)
{
    if (samples == nullptr) {
        Abort(rank, "profile sample destination", 1);
    }
    const uint32_t coreCount =
        TileXRMoonEp::MoonEpCombineV2ActiveCoreCount(
            static_cast<uint32_t>(world));
    std::vector<TileXRMoonEp::MoonEpCombineV2ProfileRecord> records(coreCount);
    const std::size_t profileBytes = records.size() * sizeof(records[0]);
    const void *profileDevice = static_cast<const uint8_t *>(workspace) +
        profileOffset;
    CheckAcl(rank, "profile D2H copy", aclrtMemcpy(records.data(),
        profileBytes, profileDevice, profileBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    for (uint32_t core = 0U; core < coreCount; ++core) {
        const TileXRMoonEp::MoonEpCombineV2ProfileRecord &record =
            records[core];
        if (record.marker != TileXRMoonEp::kMoonEpCombineV2ProfileMarker ||
            record.version != TileXRMoonEp::kMoonEpCombineV2ProfileVersion ||
            record.recordBytes != sizeof(record) ||
            record.rank != static_cast<uint32_t>(rank) ||
            record.core != core || record.blockDim != coreCount ||
            record.timePointCount !=
                TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCount ||
            record.metricCount !=
                TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount ||
            record.reserved != 0U) {
            Abort(rank, "profile record validation", 1);
        }
        ProfileSample sample;
        sample.iteration = iteration;
        sample.core = core;
        for (uint32_t point = 0U;
            point < TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCount;
            ++point) {
            sample.timePoint[point] = record.timePoint[point];
            if (sample.timePoint[point] <= 0 ||
                (point > 0U && sample.timePoint[point] <
                    sample.timePoint[point - 1U])) {
                Abort(rank, "profile timestamp validation", 1);
            }
        }
        for (uint32_t metric = 0U;
            metric < TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount;
            ++metric) {
            sample.metric[metric] = record.metric[metric];
        }
        samples->push_back(sample);
    }
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    bool showHelp = false;
    std::string parseError;
    if (!ParseOptions(argc, argv, &options, &showHelp, &parseError)) {
        std::cerr << parseError << '\n';
        Usage(std::cerr, argv[0]);
        return 2;
    }
    if (showHelp) {
        Usage(std::cout, argv[0]);
        return 0;
    }
    const int rank = options.rank;
    const int world = options.worldSize;
    if (!TileXRMoonEp::MoonEpCombineV2RankSizeSupported(
            static_cast<uint32_t>(world))) {
        Abort(rank, "unsupported Combine V2 world size", world);
    }
    if (options.experts % world != 0) {
        Abort(rank, "expert count must be divisible by world size",
            options.experts);
    }
    for (const int64_t bs : options.batchSizes) {
        if (bs % world != 0 || bs >
            std::numeric_limits<int32_t>::max() / kTopK) {
            Abort(rank, "batch size must be divisible by world size", 1);
        }
    }

    const int device = options.device >= 0 ? options.device :
        rank % kDeviceCount;
    CheckAcl(rank, "aclInit", aclInit(nullptr));
    CheckAcl(rank, "aclrtSetDevice", aclrtSetDevice(device));

    aclrtStream stream = nullptr;
    CheckAcl(rank, "aclrtCreateStream", aclrtCreateStream(&stream));
    aclrtEvent startEvent = nullptr;
    aclrtEvent stopEvent = nullptr;
    CheckAcl(rank, "aclrtCreateEvent start", aclrtCreateEvent(&startEvent));
    CheckAcl(rank, "aclrtCreateEvent stop", aclrtCreateEvent(&stopEvent));

    TileXRCommPtr comm = nullptr;
    int ret = TileXRCommInitRankWithSharedQpDomain(
        options.commDomain, world, rank, &comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        Abort(rank, "TileXRCommInitRankWithSharedQpDomain", ret);
    }

    TileXR::CommArgs *commArgs = nullptr;
    uint32_t qpCount = 0;
    ret = TileXRGetCommArgsHost(comm, commArgs);
    if (ret != TileXR::TILEXR_SUCCESS || commArgs == nullptr ||
        (commArgs->extraFlag & TileXR::ExtraFlag::UDMA_SHARED_QP) == 0U) {
        Abort(rank, "shared-QP CommArgs validation", ret);
    }
    ret = TileXRUDMAGetQpCount(comm, &qpCount);
    if (ret != TileXR::TILEXR_SUCCESS || qpCount != kQpCount) {
        Abort(rank, "TileXRUDMAGetQpCount", ret);
    }

    uint64_t maxWorkspaceBytes = 0;
    int64_t maxBs = 0;
    for (const int64_t bs : options.batchSizes) {
        uint64_t workspaceBytes = 0;
        uint64_t profileOffset = 0;
        uint64_t outputOffsets[2] = {};
        ret = TileXRMoonEpCombineGetWorkspaceSizeV2(bs, kH, kTopK,
            bs * kTopK, TILEXR_MOONEP_DTYPE_BFLOAT16, &workspaceBytes,
            &profileOffset, &outputOffsets[0], &outputOffsets[1]);
        if (ret != TILEXR_MOONEP_SUCCESS || workspaceBytes == 0) {
            Abort(rank, "TileXRMoonEpCombineGetWorkspaceSizeV2", ret);
        }
        maxWorkspaceBytes = std::max(maxWorkspaceBytes, workspaceBytes);
        maxBs = std::max(maxBs, bs);
    }

    void *workspace = nullptr;
    int32_t *dst = nullptr;
    const std::size_t maxDstBytes = static_cast<std::size_t>(maxBs * kTopK) *
        sizeof(int32_t);
    CheckAcl(rank, "aclrtMalloc workspace", aclrtMalloc(&workspace,
        static_cast<std::size_t>(maxWorkspaceBytes), ACL_MEM_MALLOC_HUGE_FIRST));
    CheckAcl(rank, "aclrtMalloc destinations", aclrtMalloc(
        reinterpret_cast<void **>(&dst), maxDstBytes,
        ACL_MEM_MALLOC_HUGE_FIRST));
    CheckAcl(rank, "aclrtMemset workspace", aclrtMemset(workspace,
        static_cast<std::size_t>(maxWorkspaceBytes), 0,
        static_cast<std::size_t>(maxWorkspaceBytes)));

    TileXRUDMAMemHandle handle = 0;
    ret = TileXRUDMARegister(comm, static_cast<GM_ADDR>(workspace),
        static_cast<std::size_t>(maxWorkspaceBytes), &handle);
    if (ret != TileXR::TILEXR_SUCCESS) {
        Abort(rank, "TileXRUDMARegister", ret);
    }
    if (!BarrierAll(rank, world, "workspace registration")) {
        Abort(rank, "workspace registration barrier", 1);
    }
    if (rank == 0) {
        std::cout << "COMBINE_V2_SETUP ranks=" << world
                  << " devices_per_host=" << kDeviceCount
                  << " experts=" << options.experts
                  << " k=" << kTopK
                  << " h=" << kH
                  << " dtype=bf16"
                  << " qp_count=" << qpCount
                  << " max_bs=" << maxBs
                  << " workspace_bytes=" << maxWorkspaceBytes
                  << std::endl;
    }

    bool allCasesOk = true;
    for (const int64_t bs : options.batchSizes) {
        const int64_t slots = bs * kTopK;
        uint64_t caseWorkspaceBytes = 0;
        uint64_t profileOffset = 0;
        uint64_t outputOffsets[2] = {};
        ret = TileXRMoonEpCombineGetWorkspaceSizeV2(bs, kH, kTopK, slots,
            TILEXR_MOONEP_DTYPE_BFLOAT16, &caseWorkspaceBytes,
            &profileOffset, &outputOffsets[0], &outputOffsets[1]);
        if (ret != TILEXR_MOONEP_SUCCESS || caseWorkspaceBytes == 0) {
            Abort(rank, "TileXRMoonEpCombineGetWorkspaceSizeV2 profile", ret);
        }
        const std::size_t sourceElements = static_cast<std::size_t>(slots) *
            static_cast<std::size_t>(kH);
        const std::size_t sourceBytes = sourceElements * sizeof(uint16_t);
        const std::size_t dstBytes = static_cast<std::size_t>(slots) *
            sizeof(int32_t);
        std::vector<uint16_t> source(sourceElements, SourceValue(rank));
        std::vector<int32_t> destinations(static_cast<std::size_t>(slots));
        const int64_t slotsPerRank = slots / world;
        for (int64_t slot = 0; slot < slots; ++slot) {
            const int64_t targetRank = slot % world;
            const int64_t targetSlot = static_cast<int64_t>(rank) *
                slotsPerRank + slot / world;
            destinations[static_cast<std::size_t>(slot)] =
                static_cast<int32_t>(targetRank * slots + targetSlot);
        }
        CheckAcl(rank, "input H2D copy", aclrtMemcpy(workspace,
            static_cast<std::size_t>(maxWorkspaceBytes), source.data(),
            sourceBytes, ACL_MEMCPY_HOST_TO_DEVICE));
        CheckAcl(rank, "destinations H2D copy", aclrtMemcpy(dst,
            maxDstBytes, destinations.data(), dstBytes,
            ACL_MEMCPY_HOST_TO_DEVICE));
        source.clear();
        source.shrink_to_fit();
        destinations.clear();
        destinations.shrink_to_fit();
        if (!BarrierAll(rank, world, "case inputs ready")) {
            Abort(rank, "case input barrier", 1);
        }

        uint64_t activeOutputOffset = 0;
        LaunchCombine(rank, workspace, dst, comm, bs, stream,
            &activeOutputOffset);
        CheckAcl(rank, "correctness stream synchronization",
            aclrtSynchronizeStream(stream));
        const OutputCheckResult outputResult = CheckOutput(
            rank, world, bs, workspace,
            activeOutputOffset);
        const bool validationAccepted =
            outputResult == OutputCheckResult::Passed ||
            (options.allowSelfOnlyFailure &&
                outputResult == OutputCheckResult::SelfOnlyFailed);
        if (!BarrierAll(rank, world, "correctness validation",
                validationAccepted)) {
            allCasesOk = false;
            std::cout << "COMBINE_V2_RANK_PERF bs=" << bs
                      << " rank=" << rank
                      << " correctness=failed" << std::endl;
            break;
        }

        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            LaunchCombine(rank, workspace, dst, comm, bs, stream,
                &activeOutputOffset);
            CheckAcl(rank, "warmup stream synchronization",
                aclrtSynchronizeStream(stream));
            if (!options.skipIterationBarriers &&
                !BarrierAll(rank, world, "warmup iteration")) {
                Abort(rank, "warmup barrier", 1);
            }
        }

        std::vector<float> rankSamples;
        rankSamples.reserve(static_cast<std::size_t>(options.iterations));
        std::vector<ProfileSample> profileSamples;
        if (options.profile) {
            profileSamples.reserve(static_cast<std::size_t>(options.iterations) *
                TileXRMoonEp::MoonEpCombineV2ActiveCoreCount(
                    static_cast<uint32_t>(world)));
        }
        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            CheckAcl(rank, "aclrtRecordEvent start",
                aclrtRecordEvent(startEvent, stream));
            LaunchCombine(rank, workspace, dst, comm, bs, stream,
                &activeOutputOffset);
            CheckAcl(rank, "aclrtRecordEvent stop",
                aclrtRecordEvent(stopEvent, stream));
            CheckAcl(rank, "aclrtSynchronizeEvent stop",
                aclrtSynchronizeEvent(stopEvent));
            float elapsedMs = 0.0F;
            CheckAcl(rank, "aclrtEventElapsedTime",
                aclrtEventElapsedTime(&elapsedMs, startEvent, stopEvent));
            rankSamples.push_back(elapsedMs);
            if (options.profile) {
                CaptureProfileSamples(rank, world, iteration, workspace,
                    profileOffset, &profileSamples);
            }
            if (!options.skipIterationBarriers &&
                !BarrierAll(rank, world, "timed iteration")) {
                Abort(rank, "timed iteration barrier", 1);
            }
        }

        for (int iteration = 0; iteration < options.iterations; ++iteration) {
            std::cout << std::fixed << std::setprecision(6)
                      << "COMBINE_V2_SAMPLE bs=" << bs
                      << " iteration=" << iteration
                      << " rank=" << rank
                      << " elapsed_ms=" << rankSamples[static_cast<std::size_t>(iteration)]
                      << std::endl;
        }
        for (const ProfileSample &sample : profileSamples) {
            std::cout << "COMBINE_V2_PROFILE bs=" << bs
                      << " iteration=" << sample.iteration
                      << " rank=" << rank
                      << " core=" << sample.core
                      << " cycles_per_us="
                      << TileXRMoonEp::kMoonEpCombineV2ProfileCyclesPerUs;
            for (uint32_t point = 0U;
                point < TileXRMoonEp::kMoonEpCombineV2ProfileTimePointCount;
                ++point) {
                std::cout << " t" << point << '=' << sample.timePoint[point];
            }
            std::cout << std::fixed << std::setprecision(3);
            for (uint32_t metric = 0U;
                metric < TileXRMoonEp::kMoonEpCombineV2ProfileMetricCount;
                ++metric) {
                std::cout << ' ' << kProfileMetricNames[metric] << '='
                          << static_cast<double>(sample.metric[metric]) /
                              TileXRMoonEp::kMoonEpCombineV2ProfileCyclesPerUs;
            }
            std::cout << std::endl;
        }
        const float total = std::accumulate(rankSamples.begin(),
            rankSamples.end(), 0.0F);
        std::cout << std::fixed << std::setprecision(6)
                  << "COMBINE_V2_RANK_PERF bs=" << bs
                  << " rank=" << rank
                  << " avg_ms=" << total / static_cast<float>(rankSamples.size())
                  << " correctness=" << OutputCheckName(outputResult)
                  << std::endl;
        allCasesOk = allCasesOk && validationAccepted;
    }

    const bool casesSynchronized = BarrierAll(rank, world,
        "all benchmark cases", allCasesOk);
    const int unregisterRet = TileXRUDMAUnregister(comm, handle);
    const bool unregisterSynchronized = BarrierAll(rank, world,
        "workspace unregistration",
        unregisterRet == TileXR::TILEXR_SUCCESS);
    const int destroyRet = TileXRCommDestroy(comm);
    const aclError destroyStartRet = aclrtDestroyEvent(startEvent);
    const aclError destroyStopRet = aclrtDestroyEvent(stopEvent);
    const aclError freeDstRet = aclrtFree(dst);
    const aclError freeWorkspaceRet = aclrtFree(workspace);
    const aclError destroyStreamRet = aclrtDestroyStream(stream);
    const aclError resetRet = aclrtResetDevice(device);
    const aclError finalizeRet = aclFinalize();
    const bool cleanupOk = unregisterRet == TileXR::TILEXR_SUCCESS &&
        destroyRet == TileXR::TILEXR_SUCCESS &&
        destroyStartRet == ACL_SUCCESS && destroyStopRet == ACL_SUCCESS &&
        freeDstRet == ACL_SUCCESS && freeWorkspaceRet == ACL_SUCCESS &&
        destroyStreamRet == ACL_SUCCESS && resetRet == ACL_SUCCESS &&
        finalizeRet == ACL_SUCCESS;
    return allCasesOk && casesSynchronized && unregisterSynchronized &&
        cleanupOk ? 0 : 1;
}
