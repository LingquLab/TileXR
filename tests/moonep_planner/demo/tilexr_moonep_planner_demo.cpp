#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "planner_reference.h"
#include "tilexr_api.h"
#include "tilexr_moonep_planner.h"
#include "tilexr_types.h"

namespace {

int EnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    return value == nullptr ? fallback : std::atoi(value);
}

uint64_t EnvUint64(const char *name, uint64_t fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    return end != value && end != nullptr && *end == '\0' ?
        static_cast<uint64_t>(parsed) : 0;
}

bool CheckAcl(int rank, const char *step, aclError ret)
{
    if (ret == ACL_SUCCESS) {
        return true;
    }
    std::cerr << "[rank " << rank << "] " << step << " failed: " << ret << std::endl;
    const char *recent = aclGetRecentErrMsg();
    if (recent != nullptr) {
        std::cerr << "[rank " << rank << "] recent ACL error: " << recent << std::endl;
    }
    return false;
}

bool CheckTileXR(int rank, const char *step, int ret)
{
    if (ret == TileXR::TILEXR_SUCCESS) {
        return true;
    }
    std::cerr << "[rank " << rank << "] " << step << " failed: " << ret << std::endl;
    return false;
}

template <typename T>
bool Compare(const char *name, const std::vector<T> &actual, const std::vector<T> &expected, int rank)
{
    if (actual.size() != expected.size()) {
        std::cerr << "[rank " << rank << "] " << name << " size mismatch actual="
                  << actual.size() << " expected=" << expected.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "[rank " << rank << "] " << name << " mismatch index=" << i
                      << " actual=" << actual[i] << " expected=" << expected[i] << std::endl;
            return false;
        }
    }
    return true;
}

void Cleanup(int rank, int device, TileXRCommPtr comm, aclrtStream stream,
    const std::vector<void *> &allocations)
{
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        if (*it != nullptr) {
            aclrtFree(*it);
        }
    }
    if (comm != nullptr) {
        TileXRCommDestroy(comm);
    }
    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(device);
    aclFinalize();
    std::cout << "[rank " << rank << "] cleanup complete" << std::endl;
}

} // namespace

int main(int argc, char **argv)
{
    int arg = 1;
    const int rankSize = argc > arg ? std::atoi(argv[arg++]) : EnvInt("RANK_SIZE", 8);
    const int rank = argc > arg ? std::atoi(argv[arg++]) : EnvInt("RANK", 0);
    const int device = argc > arg ? std::atoi(argv[arg++]) : rank;
    const int64_t s = argc > arg ? std::atoll(argv[arg++]) : 8192;
    const int64_t k = argc > arg ? std::atoll(argv[arg++]) : 8;
    const int64_t experts = argc > arg ? std::atoll(argv[arg++]) : 896;
    const std::string pattern = argc > arg ? argv[arg++] : "biased";
    const int warmup = argc > arg ? std::atoi(argv[arg++]) : 20;
    const int rounds = argc > arg ? std::atoi(argv[arg++]) : 100;
    const int physicalDeviceCount = EnvInt("TILEXR_PHYSICAL_DEVICE_COUNT", rankSize);
    const uint64_t waitIterations =
        EnvUint64("TILEXR_MOONEP_PLANNER_WAIT_ITERATIONS", 1000000);
    const bool oversubscribed = rankSize > physicalDeviceCount;

    if (rankSize <= 0 || rank < 0 || rank >= rankSize ||
        physicalDeviceCount <= 0 || device < 0 || device >= physicalDeviceCount ||
        experts % rankSize != 0 || waitIterations == 0) {
        std::cerr << "invalid rank or expert configuration" << std::endl;
        return 2;
    }

    const int64_t n = s * k;
    const int64_t b = experts / rankSize;
    const std::vector<int32_t> allTopk = TileXRMoonEpTest::MakeRouting(
        pattern, rankSize, s, k, experts, 20260731U);
    std::vector<int32_t> localTopk(
        allTopk.begin() + rank * n, allTopk.begin() + (rank + 1) * n);
    std::vector<int32_t> localTpe(static_cast<size_t>(experts), 0);
    for (int32_t expert : localTopk) {
        ++localTpe[static_cast<size_t>(expert)];
    }

    TileXRMoonEpTest::ReferencePlan expected;
    std::string referenceError;
    if (!TileXRMoonEpTest::BuildReferencePlan(
            rank, rankSize, s, k, experts, allTopk, &expected, &referenceError)) {
        std::cerr << "[rank " << rank << "] reference failed: " << referenceError << std::endl;
        return 2;
    }

    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    std::vector<void *> allocations;
    if (!CheckAcl(rank, "aclInit", aclInit(nullptr)) ||
        !CheckAcl(rank, "aclrtSetDevice", aclrtSetDevice(device)) ||
        !CheckAcl(rank, "aclrtCreateStream", aclrtCreateStream(&stream)) ||
        !CheckTileXR(rank, "TileXRCommInitRankLocal",
            TileXRCommInitRankLocal(rankSize, rank, &comm))) {
        Cleanup(rank, device, comm, stream, allocations);
        return 1;
    }

    uint64_t workspaceBytes = 0;
    int64_t dispatchedCapacity = 0;
    if (!CheckTileXR(rank, "workspace query", TileXRMoonEpPlannerGetWorkspaceSizeV2(
            comm, s, k, experts, &workspaceBytes, &dispatchedCapacity)) ||
        dispatchedCapacity != expected.dispatchedCapacity) {
        std::cerr << "[rank " << rank << "] capacity mismatch actual=" << dispatchedCapacity
                  << " expected=" << expected.dispatchedCapacity << std::endl;
        Cleanup(rank, device, comm, stream, allocations);
        return 1;
    }

    auto allocate = [&](void **ptr, size_t bytes, const char *name) -> bool {
        if (!CheckAcl(rank, name, aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
            return false;
        }
        allocations.push_back(*ptr);
        return true;
    };
    void *topkDev = nullptr;
    void *tpeDev = nullptr;
    void *workspaceDev = nullptr;
    void *dstDev = nullptr;
    void *cuDev = nullptr;
    void *copyDev = nullptr;
    void *statsDev = nullptr;
    void *statusDev = nullptr;
    const size_t groupCount = static_cast<size_t>(experts + b);
    if (!allocate(&topkDev, localTopk.size() * sizeof(int32_t), "malloc topk") ||
        !allocate(&tpeDev, localTpe.size() * sizeof(int32_t), "malloc tpe") ||
        !allocate(&workspaceDev, static_cast<size_t>(workspaceBytes), "malloc workspace") ||
        !allocate(&dstDev, static_cast<size_t>(n) * sizeof(int32_t), "malloc dst") ||
        !allocate(&cuDev, groupCount * sizeof(int32_t), "malloc cu") ||
        !allocate(&copyDev, static_cast<size_t>(rankSize * b) * sizeof(int32_t), "malloc copy") ||
        !allocate(&statsDev, 2 * sizeof(int32_t), "malloc stats") ||
        !allocate(&statusDev, sizeof(int32_t), "malloc status")) {
        Cleanup(rank, device, comm, stream, allocations);
        return 1;
    }
    if (!CheckAcl(rank, "copy topk", aclrtMemcpy(topkDev, localTopk.size() * sizeof(int32_t),
            localTopk.data(), localTopk.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE)) ||
        !CheckAcl(rank, "copy tpe", aclrtMemcpy(tpeDev, localTpe.size() * sizeof(int32_t),
            localTpe.data(), localTpe.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE))) {
        Cleanup(rank, device, comm, stream, allocations);
        return 1;
    }

    auto launch = [&]() -> bool {
        return CheckTileXR(rank, "TileXRMoonEpPlannerV2", TileXRMoonEpPlannerV2(
            static_cast<const int32_t *>(topkDev), static_cast<const int32_t *>(tpeDev),
            comm, s, k, experts, workspaceDev, workspaceBytes,
            static_cast<int32_t *>(dstDev), static_cast<int32_t *>(cuDev),
            static_cast<int32_t *>(copyDev), static_cast<int32_t *>(statsDev),
            static_cast<int32_t *>(statusDev), waitIterations, stream));
    };
    auto checkPlannerStatus = [&]() -> bool {
        int32_t status = TILEXR_MOONEP_PLANNER_STATUS_SUCCESS;
        if (!CheckAcl(rank, "copy planner status", aclrtMemcpy(&status, sizeof(status),
                statusDev, sizeof(status), ACL_MEMCPY_DEVICE_TO_HOST))) {
            return false;
        }
        if (status == TILEXR_MOONEP_PLANNER_STATUS_SUCCESS) {
            return true;
        }
        const int32_t peer = status - TILEXR_MOONEP_PLANNER_STATUS_TIMEOUT_BASE;
        std::cerr << "[rank " << rank << "] planner timeout status=" << status
                  << " peer=" << peer << std::endl;
        return false;
    };

    if (!launch() || !CheckAcl(rank, "initial sync", aclrtSynchronizeStream(stream)) ||
        !checkPlannerStatus()) {
        Cleanup(rank, device, comm, stream, allocations);
        return 1;
    }
    for (int i = 0; i < warmup; ++i) {
        if (!launch()) {
            Cleanup(rank, device, comm, stream, allocations);
            return 1;
        }
    }
    if (!CheckAcl(rank, "warmup sync", aclrtSynchronizeStream(stream)) ||
        !checkPlannerStatus()) {
        Cleanup(rank, device, comm, stream, allocations);
        return 1;
    }

    std::vector<double> samplesUs;
    samplesUs.reserve(static_cast<size_t>(std::max(rounds, 0)));
    for (int i = 0; i < rounds; ++i) {
        aclrtEvent start = nullptr;
        aclrtEvent stop = nullptr;
        if (!CheckAcl(rank, "create start event", aclrtCreateEvent(&start)) ||
            !CheckAcl(rank, "create stop event", aclrtCreateEvent(&stop)) ||
            !CheckAcl(rank, "record start event", aclrtRecordEvent(start, stream)) ||
            !launch() ||
            !CheckAcl(rank, "record stop event", aclrtRecordEvent(stop, stream)) ||
            !CheckAcl(rank, "synchronize stop event", aclrtSynchronizeEvent(stop)) ||
            !checkPlannerStatus()) {
            if (start != nullptr) {
                aclrtDestroyEvent(start);
            }
            if (stop != nullptr) {
                aclrtDestroyEvent(stop);
            }
            Cleanup(rank, device, comm, stream, allocations);
            return 1;
        }
        float elapsedMs = 0.0f;
        const bool measured = CheckAcl(rank, "event elapsed time",
            aclrtEventElapsedTime(&elapsedMs, start, stop));
        aclrtDestroyEvent(start);
        aclrtDestroyEvent(stop);
        if (!measured) {
            Cleanup(rank, device, comm, stream, allocations);
            return 1;
        }
        samplesUs.push_back(static_cast<double>(elapsedMs) * 1000.0);
    }

    std::vector<int32_t> actualDst(static_cast<size_t>(n));
    std::vector<int32_t> actualCu(groupCount);
    std::vector<int32_t> actualCopy(static_cast<size_t>(rankSize * b));
    std::vector<int32_t> actualStats(2);
    if (!CheckAcl(rank, "copy dst back", aclrtMemcpy(actualDst.data(), actualDst.size() * sizeof(int32_t),
            dstDev, actualDst.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST)) ||
        !CheckAcl(rank, "copy cu back", aclrtMemcpy(actualCu.data(), actualCu.size() * sizeof(int32_t),
            cuDev, actualCu.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST)) ||
        !CheckAcl(rank, "copy experts back", aclrtMemcpy(actualCopy.data(), actualCopy.size() * sizeof(int32_t),
            copyDev, actualCopy.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST)) ||
        !CheckAcl(rank, "copy stats back", aclrtMemcpy(actualStats.data(), actualStats.size() * sizeof(int32_t),
            statsDev, actualStats.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST))) {
        Cleanup(rank, device, comm, stream, allocations);
        return 1;
    }

    const bool dstCorrect = Compare("dst", actualDst, expected.dst, rank);
    const bool cuCorrect = Compare("cuSeqlens", actualCu, expected.cuSeqlens, rank);
    const bool copyCorrect = Compare("expertsToCopy", actualCopy, expected.expertsToCopy, rank);
    const bool statsCorrect = Compare("remoteStats", actualStats, expected.remoteStats, rank);
    const bool correct = dstCorrect && cuCorrect && copyCorrect && statsCorrect;
    if (correct) {
        std::vector<double> sorted = samplesUs;
        std::sort(sorted.begin(), sorted.end());
        const auto percentile = [&](int p) {
            if (sorted.empty()) {
                return 0.0;
            }
            const size_t index = static_cast<size_t>((p * sorted.size() + 99) / 100 - 1);
            return sorted[index];
        };
        double averageUs = 0.0;
        for (double sample : samplesUs) {
            averageUs += sample;
        }
        if (!samplesUs.empty()) {
            averageUs /= static_cast<double>(samplesUs.size());
        }
        const char *sampleDir = std::getenv("TILEXR_MOONEP_SAMPLE_DIR");
        if (sampleDir != nullptr && !samplesUs.empty()) {
            std::ofstream samples(std::string(sampleDir) + "/rank_" + std::to_string(rank) + ".samples");
            for (double sample : samplesUs) {
                samples << sample << '\n';
            }
        }
        std::cout << "[rank " << rank << "] validation success"
                  << " R=" << rankSize << " E=" << experts << " S=" << s
                  << " K=" << k << " pattern=" << pattern
                  << " logical_ranks=" << rankSize
                  << " physical_devices=" << physicalDeviceCount
                  << " oversubscribed=" << (oversubscribed ? "true" : "false")
                  << " planner_us_avg=" << averageUs
                  << " p50=" << percentile(50)
                  << " p90=" << percentile(90)
                  << " p99=" << percentile(99) << std::endl;
    }
    Cleanup(rank, device, comm, stream, allocations);
    return correct ? 0 : 1;
}
