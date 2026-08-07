#include <acl/acl.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "comm_args.h"
#include "ep_plan_layout.h"
#include "ep_plan_reference.h"
#include "ep_plan_types.h"
#include "runtime/kernel.h"
#include "tilexr_types.h"

using GM_ADDR = uint8_t *;

rtError_t launch_tilexr_ep_plan_kernel(uint32_t blockDim, void *stream, GM_ADDR commArgs,
    GM_ADDR topkExperts, GM_ADDR tokensPerExpert, GM_ADDR globalRankIds, GM_ADDR dst,
    GM_ADDR cuSeqlens, GM_ADDR expertsToCopy, GM_ADDR remoteExperts, GM_ADDR expertTargets,
    GM_ADDR remoteStats, GM_ADDR status,
    GM_ADDR localWorkspace, GM_ADDR metaWorkspace, int64_t rank, int64_t rankSize,
    int64_t s, int64_t topK, int64_t expertNum, int64_t prefetchSlots,
    int64_t rankTokenCapacity, int64_t nvS, int64_t tokenPadding, int64_t tokenRouteLimitPerPair,
    int32_t cardsPerServer, int32_t cardsPerCabinet, int32_t crossCandidateCount,
    uint64_t epoch, uint64_t waitIterations, int64_t magic);

namespace {

bool CheckAcl(const std::string &operation, aclError result)
{
    if (result == ACL_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed with ACL error " << result;
    const char *recentError = aclGetRecentErrMsg();
    if (recentError != nullptr && recentError[0] != '\0') {
        std::cerr << ": " << recentError;
    }
    std::cerr << std::endl;
    return false;
}

class DeviceAllocation {
public:
    DeviceAllocation() : pointer_(nullptr), bytes_(0) {}

    ~DeviceAllocation()
    {
        if (pointer_ != nullptr) {
            aclrtFree(pointer_);
        }
    }

    bool Allocate(size_t bytes, const std::string &name)
    {
        bytes_ = bytes;
        return CheckAcl("aclrtMalloc(" + name + ")",
            aclrtMalloc(&pointer_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    bool Zero(const std::string &name)
    {
        return CheckAcl("aclrtMemset(" + name + ")",
            aclrtMemset(pointer_, bytes_, 0, bytes_));
    }

    void *data() const { return pointer_; }
    GM_ADDR gm() const { return reinterpret_cast<GM_ADDR>(pointer_); }
    size_t bytes() const { return bytes_; }

private:
    void *pointer_;
    size_t bytes_;
};

template <typename T>
bool CopyHostToDevice(DeviceAllocation &destination, const std::vector<T> &source,
    const std::string &name)
{
    const size_t bytes = source.size() * sizeof(T);
    if (bytes > destination.bytes()) {
        std::cerr << name << " exceeds destination allocation" << std::endl;
        return false;
    }
    return CheckAcl("aclrtMemcpy H2D(" + name + ")",
        aclrtMemcpy(destination.data(), destination.bytes(), source.data(), bytes,
            ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename T>
bool CopyDeviceToHost(std::vector<T> *destination, const DeviceAllocation &source,
    const std::string &name)
{
    const size_t bytes = destination->size() * sizeof(T);
    if (bytes > source.bytes()) {
        std::cerr << name << " exceeds source allocation" << std::endl;
        return false;
    }
    return CheckAcl("aclrtMemcpy D2H(" + name + ")",
        aclrtMemcpy(destination->data(), bytes, source.data(), source.bytes(),
            ACL_MEMCPY_DEVICE_TO_HOST));
}

template <typename T>
bool CopyDeviceRegionToHost(T *destination, const DeviceAllocation &source,
    uint64_t offset, const std::string &name)
{
    if (offset > source.bytes() || sizeof(T) > source.bytes() - offset) {
        std::cerr << name << " is outside the source allocation" << std::endl;
        return false;
    }
    const GM_ADDR address = source.gm() + offset;
    return CheckAcl("aclrtMemcpy D2H(" + name + ")",
        aclrtMemcpy(destination, sizeof(T), address, sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST));
}

template <typename T>
bool CompareVector(const std::string &name, const std::vector<T> &actual,
    const std::vector<T> &expected)
{
    if (actual.size() != expected.size()) {
        std::cerr << name << " size mismatch: actual=" << actual.size()
                  << " expected=" << expected.size() << std::endl;
        return false;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::cerr << name << " mismatch at index " << index
                      << ": actual=" << actual[index]
                      << " expected=" << expected[index] << std::endl;
            return false;
        }
    }
    return true;
}

bool AllocateAndZero(DeviceAllocation *allocation, size_t bytes, const std::string &name)
{
    return allocation->Allocate(bytes, name) && allocation->Zero(name);
}

bool RunSingleRankParityCase(aclrtStream stream)
{
    const int64_t rank = 0;
    const int64_t rankSize = 1;
    const int64_t s = 4;
    const int64_t topK = 2;
    const int64_t expertNum = 4;
    const uint64_t epoch = 7;
    const int64_t magic = 0x4D50;
    const uint64_t waitIterations = 1ULL << 24;

    TileXRMoonEPPlanConfig config {};
    config.prefetchSlots = 2;
    config.rankTokenCapacity = s * topK;
    config.nvS = s * topK;
    config.tokenPadding = 2;
    config.tokenRouteLimitPerPair = 0;
    config.cardsPerServer = TileXREp::Plan::kPlanCardsPerServer;
    config.cardsPerCabinet = TileXREp::Plan::kPlanCardsPerCabinet;
    config.crossCandidateCount = TileXREp::Plan::kPlanCrossCandidateCount;
    config.reserved = 0;

    const std::vector<int32_t> topkExperts {0, 0, 1, 2, 3, 3, 2, 1};
    const std::vector<int32_t> tokensPerExpert {2, 2, 2, 2};
    const std::vector<int32_t> globalRankIds {669};

    TileXREp::Plan::ReferenceInput referenceInput {};
    referenceInput.rankSize = static_cast<int32_t>(rankSize);
    referenceInput.s = s;
    referenceInput.topK = topK;
    referenceInput.expertNum = expertNum;
    referenceInput.config = config;
    referenceInput.globalRankIds = globalRankIds;
    referenceInput.topkExperts = topkExperts;
    referenceInput.tokensPerExpert = tokensPerExpert;

    TileXREp::Plan::ReferenceOutput expected;
    if (TileXREp::Plan::BuildReferencePlan(referenceInput, &expected) != PLAN_OK) {
        std::cerr << "CPU reference failed to build the single-rank parity case" << std::endl;
        return false;
    }

    TileXREp::Plan::PlanWorkspaceLayout layout {};
    if (TileXREp::Plan::BuildPlanWorkspaceLayout(rankSize, s, topK, expertNum,
            config, &layout) != TileXR::TILEXR_SUCCESS) {
        std::cerr << "BuildPlanWorkspaceLayout failed" << std::endl;
        return false;
    }

    DeviceAllocation deviceTopk;
    DeviceAllocation deviceTpe;
    DeviceAllocation deviceGlobalRankIds;
    DeviceAllocation deviceDst;
    DeviceAllocation deviceCuSeqlens;
    DeviceAllocation deviceExpertsToCopy;
    DeviceAllocation deviceRemoteExperts;
    DeviceAllocation deviceExpertTargets;
    DeviceAllocation deviceRemoteStats;
    DeviceAllocation deviceStatus;
    DeviceAllocation deviceLocalWorkspace;
    DeviceAllocation deviceRegisteredMeta;
    DeviceAllocation devicePeerMemory;
    DeviceAllocation deviceCommArgs;

    bool ok = true;
    ok = AllocateAndZero(&deviceTopk, topkExperts.size() * sizeof(int32_t), "topkExperts") && ok;
    ok = AllocateAndZero(&deviceTpe, tokensPerExpert.size() * sizeof(int32_t), "tokensPerExpert") && ok;
    ok = AllocateAndZero(&deviceGlobalRankIds, globalRankIds.size() * sizeof(int32_t), "globalRankIds") && ok;
    ok = AllocateAndZero(&deviceDst, expected.dst.size() * sizeof(int32_t), "dst") && ok;
    ok = AllocateAndZero(&deviceCuSeqlens, expected.cuSeqlens.size() * sizeof(int32_t), "cuSeqlens") && ok;
    ok = AllocateAndZero(&deviceExpertsToCopy, static_cast<size_t>(config.prefetchSlots) * sizeof(int32_t), "expertsToCopy") && ok;
    ok = AllocateAndZero(&deviceRemoteExperts, expected.expertsToCopy.size() * sizeof(int32_t), "remoteExperts") && ok;
    ok = AllocateAndZero(&deviceExpertTargets, expected.expertTargets.size() * sizeof(uint64_t), "expertTargets") && ok;
    ok = AllocateAndZero(&deviceRemoteStats, 2 * sizeof(int32_t), "remoteStats") && ok;
    ok = AllocateAndZero(&deviceStatus, TileXREp::Plan::kPlanStatusWords * sizeof(int32_t), "status") && ok;
    ok = AllocateAndZero(&deviceLocalWorkspace, static_cast<size_t>(layout.local.totalBytes), "localWorkspace") && ok;
    ok = AllocateAndZero(&deviceRegisteredMeta, static_cast<size_t>(layout.registeredMeta.totalBytes),
        "metaWorkspace") && ok;
    ok = AllocateAndZero(&devicePeerMemory,
        static_cast<size_t>(TileXR::IPC_DATA_OFFSET + 4096), "peerMemory") && ok;
    ok = AllocateAndZero(&deviceCommArgs, sizeof(TileXR::CommArgs), "commArgs") && ok;
    if (!ok) {
        return false;
    }

    if (!CopyHostToDevice(deviceTopk, topkExperts, "topkExperts") ||
        !CopyHostToDevice(deviceTpe, tokensPerExpert, "tokensPerExpert") ||
        !CopyHostToDevice(deviceGlobalRankIds, globalRankIds, "globalRankIds")) {
        return false;
    }

    TileXR::CommArgs hostCommArgs {};
    hostCommArgs.rank = static_cast<int>(rank);
    hostCommArgs.localRank = 0;
    hostCommArgs.rankSize = static_cast<int>(rankSize);
    hostCommArgs.localRankSize = 1;
    hostCommArgs.peerMems[0] = devicePeerMemory.gm();
    if (!CheckAcl("aclrtMemcpy H2D(commArgs)", aclrtMemcpy(deviceCommArgs.data(),
            deviceCommArgs.bytes(), &hostCommArgs, sizeof(hostCommArgs), ACL_MEMCPY_HOST_TO_DEVICE))) {
        return false;
    }

    const rtError_t launchResult = launch_tilexr_ep_plan_kernel(1, stream, deviceCommArgs.gm(),
        deviceTopk.gm(), deviceTpe.gm(), deviceGlobalRankIds.gm(), deviceDst.gm(),
        deviceCuSeqlens.gm(), deviceExpertsToCopy.gm(), deviceRemoteExperts.gm(),
        deviceExpertTargets.gm(), deviceRemoteStats.gm(),
        deviceStatus.gm(), deviceLocalWorkspace.gm(), deviceRegisteredMeta.gm(), rank,
        rankSize, s, topK, expertNum, config.prefetchSlots, config.rankTokenCapacity,
        config.nvS, config.tokenPadding, config.tokenRouteLimitPerPair, config.cardsPerServer,
        config.cardsPerCabinet, config.crossCandidateCount, epoch, waitIterations, magic);
    if (launchResult != RT_ERROR_NONE) {
        std::cerr << "launch_tilexr_ep_plan_kernel failed with runtime error "
                  << launchResult;
        const char *recentError = aclGetRecentErrMsg();
        if (recentError != nullptr && recentError[0] != '\0') {
            std::cerr << ": " << recentError;
        }
        std::cerr << std::endl;
        return false;
    }

    if (!CheckAcl("aclrtSynchronizeStream(plan)", aclrtSynchronizeStream(stream))) {
        return false;
    }

    std::vector<int32_t> actualDst(expected.dst.size(), -999);
    std::vector<int32_t> actualCuSeqlens(expected.cuSeqlens.size(), -999);
    std::vector<int32_t> actualExpertsToCopy(static_cast<size_t>(config.prefetchSlots), -999);
    std::vector<int32_t> actualRemoteExperts(expected.expertsToCopy.size(), -999);
    std::vector<uint64_t> actualExpertTargets(expected.expertTargets.size(), UINT64_MAX);
    std::vector<int32_t> actualRemoteStats(2, -999);
    std::vector<int32_t> actualStatus(TileXREp::Plan::kPlanStatusWords, -999);

    ok = CopyDeviceToHost(&actualDst, deviceDst, "dst") && ok;
    ok = CopyDeviceToHost(&actualCuSeqlens, deviceCuSeqlens, "cuSeqlens") && ok;
    ok = CopyDeviceToHost(&actualExpertsToCopy, deviceExpertsToCopy, "expertsToCopy") && ok;
    ok = CopyDeviceToHost(&actualRemoteExperts, deviceRemoteExperts, "remoteExperts") && ok;
    ok = CopyDeviceToHost(&actualExpertTargets, deviceExpertTargets, "expertTargets") && ok;
    ok = CopyDeviceToHost(&actualRemoteStats, deviceRemoteStats, "remoteStats") && ok;
    ok = CopyDeviceToHost(&actualStatus, deviceStatus, "status") && ok;
    if (!ok) {
        return false;
    }

    const std::vector<int32_t> expectedStatus(expected.statusByRank.begin(),
        expected.statusByRank.begin() + TileXREp::Plan::kPlanStatusWords);
    const std::vector<int32_t> expectedRemoteStats(expected.remoteStats.begin(),
        expected.remoteStats.begin() + 2);

    ok = CompareVector("dst", actualDst, expected.dst) && ok;
    ok = CompareVector("cuSeqlens", actualCuSeqlens, expected.cuSeqlens) && ok;
    const std::vector<int32_t> expectedLocalExperts(expected.expertsToCopy.begin(),
        expected.expertsToCopy.begin() + config.prefetchSlots);
    ok = CompareVector("expertsToCopy", actualExpertsToCopy, expectedLocalExperts) && ok;
    ok = CompareVector("remoteExperts", actualRemoteExperts, expected.expertsToCopy) && ok;
    ok = CompareVector("expertTargets", actualExpertTargets, expected.expertTargets) && ok;
    ok = CompareVector("remoteStats", actualRemoteStats, expectedRemoteStats) && ok;
    ok = CompareVector("status", actualStatus, expectedStatus) && ok;

    TileXREp::Plan::PlanEpochState epochState {};
    TileXREp::Plan::PlanCallHeader header {};
    int32_t cachedRankId = -1;
    if (!CopyDeviceRegionToHost(&epochState, deviceRegisteredMeta,
            layout.registeredMeta.epochState.offset, "epochState") ||
        !CopyDeviceRegionToHost(&header, deviceRegisteredMeta,
            layout.registeredMeta.planCallHeaders.offset, "planCallHeader") ||
        !CopyDeviceRegionToHost(&cachedRankId, deviceRegisteredMeta,
            layout.registeredMeta.globalRankIds.offset, "cachedGlobalRankId")) {
        return false;
    }

    if (epochState.requestedEpoch != epoch || epochState.committedEpoch != epoch ||
        (epochState.reserved & TileXREp::Plan::kPlanAffinityCacheValid) == 0) {
        std::cerr << "epoch/cache state mismatch: requested=" << epochState.requestedEpoch
                  << " committed=" << epochState.committedEpoch
                  << " flags=" << epochState.reserved << std::endl;
        ok = false;
    }
    if (header.abiVersion != TileXREp::Plan::kPlanAbiVersion ||
        header.rankSize != rankSize || header.s != s || header.k != topK ||
        header.expertNum != expertNum || header.epoch != epoch) {
        std::cerr << "published PlanCallHeader mismatch" << std::endl;
        ok = false;
    }
    if (cachedRankId != globalRankIds[0]) {
        std::cerr << "meta globalRankIds cache mismatch: actual=" << cachedRankId
                  << " expected=" << globalRankIds[0] << std::endl;
        ok = false;
    }

    if (ok) {
        std::cout << "single-rank Ascend device parity passed: all Plan arrays, status, "
                     "header, epoch, and affinity-cache state match the CPU reference"
                  << std::endl;
    }
    return ok;
}

} // namespace

int main(int argc, char **argv)
{
    const int32_t deviceId = argc > 1 ? static_cast<int32_t>(std::strtol(argv[1], nullptr, 10)) : 0;
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    bool ok = false;

    if (!CheckAcl("aclInit", aclInit(nullptr))) {
        return 1;
    }
    initialized = true;
    if (!CheckAcl("aclrtSetDevice", aclrtSetDevice(deviceId))) {
        goto cleanup;
    }
    deviceSet = true;
    if (!CheckAcl("aclrtCreateStream", aclrtCreateStream(&stream))) {
        goto cleanup;
    }

    ok = RunSingleRankParityCase(stream);

cleanup:
    if (stream != nullptr) {
        CheckAcl("aclrtDestroyStream", aclrtDestroyStream(stream));
    }
    if (deviceSet) {
        CheckAcl("aclrtResetDevice", aclrtResetDevice(deviceId));
    }
    if (initialized) {
        CheckAcl("aclFinalize", aclFinalize());
    }
    return ok ? 0 : 1;
}
