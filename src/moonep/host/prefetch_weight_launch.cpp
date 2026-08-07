#include "prefetch_weight_launch.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>

#include "acl/acl_rt.h"
#include "runtime/kernel.h"
#include "tilexr_types.h"

extern "C" {
extern const unsigned char TileXRMoonEpPrefetchWeightKernelBinaryData[];
extern const std::size_t TileXRMoonEpPrefetchWeightKernelBinarySize;
}

namespace TileXRMoonEp {
namespace {

constexpr uint32_t kAivBinaryMagic = 0x41415246U;
constexpr char kPrefetchWeightKernelName[] = "tilexr_moonep_prefetch_weight_kernel";

std::mutex gRegistrationMutex;
bool gRegistered = false;
int gRegistrationStatus = TileXR::TILEXR_ERROR_NOT_INITIALIZED;
void *gBinaryHandle = nullptr;
uint8_t gKernelStub = 0;

struct PrefetchWeightKernelArgs {
    GM_ADDR commArgs;
    GM_ADDR expertsToCopy;
    GM_ADDR gate;
    GM_ADDR up;
    GM_ADDR down;
    GM_ADDR status;
    uint64_t gateOffset;
    uint64_t upOffset;
    uint64_t downOffset;
    uint64_t gateRowBytes;
    uint64_t upRowBytes;
    uint64_t downRowBytes;
    int64_t rank;
    int64_t rankSize;
    int64_t expertsPerRank;
    uint64_t qpNum;
    uint64_t routeWeights;
};

static_assert(sizeof(PrefetchWeightKernelArgs) == 17U * sizeof(uint64_t),
    "PrefetchWeight kernel argument ABI changed");

int EnsureKernelRegistered()
{
    std::lock_guard<std::mutex> guard(gRegistrationMutex);
    if (gRegistered) {
        return gRegistrationStatus;
    }
    if (TileXRMoonEpPrefetchWeightKernelBinarySize == 0) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    rtDevBinary_t binary {};
    binary.data = TileXRMoonEpPrefetchWeightKernelBinaryData;
    binary.length = static_cast<uint32_t>(TileXRMoonEpPrefetchWeightKernelBinarySize);
    binary.magic = kAivBinaryMagic;
    binary.version = 0;

    rtError_t rtRet = rtDevBinaryRegister(&binary, &gBinaryHandle);
    if (rtRet == RT_ERROR_NONE) {
        rtRet = rtFunctionRegister(gBinaryHandle, &gKernelStub,
            kPrefetchWeightKernelName, kPrefetchWeightKernelName, 0);
    }
    if (rtRet != RT_ERROR_NONE) {
        std::cerr << "TileXR MoonEP PrefetchWeight kernel registration failed, ret="
                  << rtRet << std::endl;
        gRegistrationStatus = TileXR::TILEXR_ERROR_MKIRT;
        return gRegistrationStatus;
    }

    gRegistered = true;
    gRegistrationStatus = TileXR::TILEXR_SUCCESS;
    return gRegistrationStatus;
}

} // namespace

int LaunchPrefetchWeight(const PrefetchWeightLayout &layout, GM_ADDR commArgs,
    GM_ADDR expertsToCopy, GM_ADDR status, aclrtStream stream)
{
    const int registerRet = EnsureKernelRegistered();
    if (registerRet != TileXR::TILEXR_SUCCESS) {
        return registerRet;
    }

    PrefetchWeightKernelArgs args {
        commArgs, expertsToCopy, layout.gate.localBase, layout.up.localBase,
        layout.down.localBase, status, layout.gate.registryOffset,
        layout.up.registryOffset, layout.down.registryOffset,
        layout.gate.rowBytes, layout.up.rowBytes, layout.down.rowBytes,
        layout.rank, layout.rankSize, layout.expertsPerRank, layout.qpNum,
        layout.routeWeights
    };
    rtArgsEx_t argsInfo {};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo {};
    const rtError_t rtRet = rtKernelLaunchWithFlagV2(&gKernelStub, layout.blockDim,
        &argsInfo, nullptr, static_cast<rtStream_t>(stream), 0, &cfgInfo);
    if (rtRet != RT_ERROR_NONE) {
        std::cerr << "TileXR MoonEP PrefetchWeight kernel launch failed, ret="
                  << rtRet << std::endl;
        return TileXR::TILEXR_ERROR_MKIRT;
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXRMoonEp
