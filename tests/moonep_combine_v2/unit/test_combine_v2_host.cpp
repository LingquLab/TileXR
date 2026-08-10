#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "acl/acl_rt.h"
#include "combine_v2_host.h"
#include "combine_v2_launch.h"
#include "tilexr_types.h"
#include "tilexr_udma_reg.h"

namespace {

int failures = 0;
int hostReturn = TileXR::TILEXR_SUCCESS;
int registryReturn = TileXR::TILEXR_SUCCESS;
int qpReturn = TileXR::TILEXR_SUCCESS;
int devReturn = TileXR::TILEXR_SUCCESS;
int magicReturn = TileXR::TILEXR_SUCCESS;
int launchReturn = TILEXR_MOONEP_SUCCESS;
int aclDeviceReturn = ACL_SUCCESS;
int aclInfoReturn = ACL_SUCCESS;
uint32_t qpCount = TileXRMoonEp::kMoonEpCombineV2QpCount;
int64_t vectorCoreCount = TileXRMoonEp::kMoonEpCombineV2CoreCount;
int64_t nextMagic = 17;
TileXR::CommArgs commArgs {};
TileXR::TileXRUDMARegistry registry {};
GM_ADDR devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
TileXRMoonEp::CombineV2Params launchedParams {};
TileXRMoonEp::CombineV2LaunchContext launchedContext {};

void Check(bool condition, const char *message)
{
    if (!condition) {
        std::cerr << message << '\n';
        ++failures;
    }
}

void CheckStatus(int actual, int expected, const char *message)
{
    if (actual != expected) {
        std::cerr << message << ": expected " << expected
                  << ", got " << actual << '\n';
        ++failures;
    }
}

TileXRMoonEp::CombineV2Params ValidParams()
{
    TileXRMoonEp::CombineV2Params params {};
    params.registeredWorkspace = reinterpret_cast<void *>(uintptr_t {0x100000000ULL});
    params.dstLocal = reinterpret_cast<const int32_t *>(uintptr_t {0x200000000ULL});
    params.comm = reinterpret_cast<TileXRCommPtr>(uintptr_t {0x3000});
    params.bs = TileXRMoonEp::kMoonEpCombineV2SmallBs;
    params.h = TileXRMoonEp::kMoonEpCombineV2TargetH;
    params.topK = TileXRMoonEp::kMoonEpCombineV2TargetTopK;
    params.nvS = TileXRMoonEp::kMoonEpCombineV2SmallSlots;
    params.aivCoreNum = TileXRMoonEp::kMoonEpCombineV2CoreCount;
    static uint64_t activeOutputOffset = 0;
    params.activeOutputOffset = &activeOutputOffset;
    params.dtype = TILEXR_MOONEP_DTYPE_BFLOAT16;
    params.stream = reinterpret_cast<aclrtStream>(uintptr_t {0x4000});
    return params;
}

bool RankSizeSupported(int rankSize)
{
    return (rankSize >= 2 && rankSize <= 8) || rankSize == 16 ||
        rankSize == 32 || rankSize == 64 || rankSize == 128;
}

void ConfigureRankSize(int rankSize, int rank = 0)
{
    commArgs.rank = rank;
    commArgs.rankSize = rankSize;
    commArgs.localRankSize = rankSize <= 8 ? rankSize : 8;
    commArgs.localRank = rank % commArgs.localRankSize;

    registry = TileXR::TileXRUDMARegistry {};
    registry.rankSize = rankSize;
    registry.regionCount = 1;
    const TileXRMoonEp::CombineV2Params params = ValidParams();
    for (int peer = 0; peer < rankSize; ++peer) {
        registry.regions[peer].base =
            static_cast<GM_ADDR>(params.registeredWorkspace);
        registry.regions[peer].bytes = 4194304U;
    }
}

void Reset()
{
    hostReturn = registryReturn = qpReturn = devReturn = magicReturn =
        TileXR::TILEXR_SUCCESS;
    launchReturn = TILEXR_MOONEP_SUCCESS;
    aclDeviceReturn = aclInfoReturn = ACL_SUCCESS;
    qpCount = TileXRMoonEp::kMoonEpCombineV2QpCount;
    vectorCoreCount = TileXRMoonEp::kMoonEpCombineV2CoreCount;
    nextMagic = 17;
    devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
    launchedParams = TileXRMoonEp::CombineV2Params {};
    launchedContext = TileXRMoonEp::CombineV2LaunchContext {};

    commArgs = TileXR::CommArgs {};
    commArgs.extraFlag = TileXR::ExtraFlag::UDMA | TileXR::ExtraFlag::UDMA_SHARED_QP;
    commArgs.udmaInfoPtr = reinterpret_cast<GM_ADDR>(uintptr_t {0x500000000ULL});
    commArgs.udmaRegistryPtr = reinterpret_cast<GM_ADDR>(uintptr_t {0x600000000ULL});
    ConfigureRankSize(TileXRMoonEp::kMoonEpCombineV2RankCount);
}

void TestValidLaunch()
{
    Reset();
    const TileXRMoonEp::CombineV2Params params = ValidParams();
    uint64_t workspaceBytes = 0;
    uint64_t profileOffset = 0;
    uint64_t epoch0Offset = 0;
    uint64_t epoch1Offset = 0;
    CheckStatus(TileXRMoonEpCombineGetWorkspaceSizeV2(
        params.bs, params.h, params.topK, params.nvS, params.dtype,
        &workspaceBytes, &profileOffset, &epoch0Offset, &epoch1Offset),
        TILEXR_MOONEP_SUCCESS, "public workspace query");
    Check(workspaceBytes == 4194304U && epoch0Offset < epoch1Offset,
        "public workspace layout mismatch");
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_SUCCESS, "valid V2 launch");
    Check(launchedParams.registeredWorkspace == params.registeredWorkspace &&
        launchedParams.dstLocal == params.dstLocal &&
        launchedParams.aivCoreNum == params.aivCoreNum,
        "launch parameters mismatch");
    Check(launchedContext.hostArgs == &commArgs &&
        launchedContext.devArgs == devArgs && launchedContext.magic == nextMagic,
        "launch context mismatch");
    Check(*params.activeOutputOffset == launchedContext.layout.scratchOffset[1],
        "active output epoch mismatch");
}

void TestValidation()
{
    TileXRMoonEp::CombineV2Params params = ValidParams();

    Reset();
    commArgs.extraFlag &= ~TileXR::ExtraFlag::UDMA_SHARED_QP;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_ERROR_NOT_SUPPORTED, "shared QP capability required");

    Reset();
    qpCount = 8;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_ERROR_NOT_SUPPORTED, "32 QPs required");

    Reset();
    ConfigureRankSize(12);
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT,
        "supported rank class required");

    Reset();
    registry.regions[17].bytes = 4096;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT, "workspace capacity required");

    Reset();
    vectorCoreCount = TileXRMoonEp::kMoonEpCombineV2CoreCount - 1U;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_ERROR_NOT_SUPPORTED, "16 vector cores required");

    Reset();
    nextMagic = 0;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_ERROR_INTERNAL, "positive magic required");

    Reset();
    params.bs = 16;
    params.nvS = 256;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT, "unsupported shape rejected");
}

void TestRankAndCoreGeneralization()
{
    for (int rankSize = 1; rankSize <= 128; ++rankSize) {
        Reset();
        ConfigureRankSize(rankSize);
        const int expected = RankSizeSupported(rankSize) ?
            TILEXR_MOONEP_SUCCESS : TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
        CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(ValidParams()),
            expected, "rank-size support predicate");
    }

    Reset();
    ConfigureRankSize(8);
    commArgs.localRankSize = 4;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(ValidParams()),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT,
        "small rank class requires one local domain");

    Reset();
    ConfigureRankSize(16);
    commArgs.localRankSize = 16;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(ValidParams()),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT,
        "ring rank class requires eight local ranks");

    Reset();
    ConfigureRankSize(16, 9);
    commArgs.localRank = 0;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(ValidParams()),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT,
        "local rank must match global rank mapping");

    Reset();
    TileXRMoonEp::CombineV2Params params = ValidParams();
    params.aivCoreNum = TileXRMoonEp::kMoonEpCombineV2CoreCount - 1U;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT,
        "at least sixteen requested AIVs required");

    Reset();
    params = ValidParams();
    params.aivCoreNum = TileXRMoonEp::kMoonEpCombineV2CoreCount + 1U;
    CheckStatus(TileXRMoonEp::TileXRMoonEpRunCombineV2(params),
        TILEXR_MOONEP_ERROR_NOT_SUPPORTED,
        "requested AIV count must fit the device");
}

} // namespace

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&args)
{
    args = hostReturn == TileXR::TILEXR_SUCCESS ? &commArgs : nullptr;
    return hostReturn;
}

extern "C" int TileXRGetUDMARegistryHost(
    TileXRCommPtr, const TileXR::TileXRUDMARegistry **result)
{
    if (result != nullptr) {
        *result = registryReturn == TileXR::TILEXR_SUCCESS ? &registry : nullptr;
    }
    return registryReturn;
}

extern "C" int TileXRUDMAGetQpCount(TileXRCommPtr, uint32_t *result)
{
    if (result != nullptr) {
        *result = qpReturn == TileXR::TILEXR_SUCCESS ? qpCount : 0;
    }
    return qpReturn;
}

extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &args)
{
    args = devReturn == TileXR::TILEXR_SUCCESS ? devArgs : nullptr;
    return devReturn;
}

extern "C" int TileXRCommNextMagic(TileXRCommPtr, int64_t *magic)
{
    if (magicReturn == TileXR::TILEXR_SUCCESS && magic != nullptr) {
        *magic = nextMagic;
    }
    return magicReturn;
}

extern "C" aclError aclrtGetDevice(int32_t *deviceId)
{
    if (aclDeviceReturn == ACL_SUCCESS && deviceId != nullptr) {
        *deviceId = 0;
    }
    return aclDeviceReturn;
}

extern "C" aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr, int64_t *value)
{
    if (aclInfoReturn == ACL_SUCCESS && value != nullptr) {
        *value = vectorCoreCount;
    }
    return aclInfoReturn;
}

namespace TileXRMoonEp {
int TileXRMoonEpLaunchCombineV2Kernel(
    const CombineV2Params &params,
    const CombineV2LaunchContext &context)
{
    launchedParams = params;
    launchedContext = context;
    if (launchReturn == TILEXR_MOONEP_SUCCESS) {
        *params.activeOutputOffset = context.layout.scratchOffset[
            MoonEpCombineV2Epoch(static_cast<uint64_t>(context.magic))];
    }
    return launchReturn;
}
} // namespace TileXRMoonEp

int main()
{
    TestValidLaunch();
    TestValidation();
    TestRankAndCoreGeneralization();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
