#include "reduce_grad_host.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <mutex>

#include "reduce_grad_common.h"
#include "runtime/kernel.h"
#include "tilexr_types.h"

extern "C" {
extern const unsigned char TileXRMoonEpReduceGradKernelBinaryData[];
extern const std::size_t TileXRMoonEpReduceGradKernelBinarySize;
}

namespace TileXRMoonEp {
namespace {

const char kReduceGradKernelName[] = "tilexr_moonep_reduce_grad_kernel";
std::mutex g_reduceGradRegistrationMutex;
bool g_reduceGradRegistered = false;
void *g_reduceGradBinaryHandle = nullptr;

int EnsureReduceGradKernelRegistered()
{
    std::lock_guard<std::mutex> guard(g_reduceGradRegistrationMutex);
    if (g_reduceGradRegistered) {
        return TileXR::TILEXR_SUCCESS;
    }
    if (TileXRMoonEpReduceGradKernelBinarySize == 0) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    rtDevBinary_t binary {};
    binary.data = TileXRMoonEpReduceGradKernelBinaryData;
    binary.length = static_cast<uint64_t>(TileXRMoonEpReduceGradKernelBinarySize);
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.version = 0;

    void *binaryHandle = nullptr;
    rtError_t rtRet = rtDevBinaryRegister(&binary, &binaryHandle);
    if (rtRet != RT_ERROR_NONE) {
        std::cerr << "MoonEP ReduceGrad rtDevBinaryRegister failed, ret=" << rtRet << std::endl;
        return TileXR::TILEXR_ERROR_MKIRT;
    }
    rtRet = rtFunctionRegister(binaryHandle, kReduceGradKernelName,
        kReduceGradKernelName, kReduceGradKernelName, 0);
    if (rtRet != RT_ERROR_NONE) {
        std::cerr << "MoonEP ReduceGrad rtFunctionRegister failed, ret=" << rtRet << std::endl;
        const rtError_t unregisterRet = rtDevBinaryUnRegister(binaryHandle);
        if (unregisterRet != RT_ERROR_NONE) {
            std::cerr << "MoonEP ReduceGrad rtDevBinaryUnRegister failed, ret="
                      << unregisterRet << std::endl;
        }
        return TileXR::TILEXR_ERROR_MKIRT;
    }
    g_reduceGradBinaryHandle = binaryHandle;
    g_reduceGradRegistered = true;
    return TileXR::TILEXR_SUCCESS;
}

} // namespace

int TileXRMoonEpLaunchReduceGradKernel(const ReduceGradParams &params,
    const ReduceGradLaunchContext &context)
{
    const int registerRet = EnsureReduceGradKernelRegistered();
    if (registerRet != TileXR::TILEXR_SUCCESS) {
        return registerRet;
    }

    int64_t magic = 0;
    const int ret = TileXRCommNextMagic(params.comm, &magic);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    const ReduceGradLayout &layout = context.layout;
    static_assert(sizeof(ReduceGradKernelArgs) <= std::numeric_limits<uint32_t>::max(),
        "ReduceGrad kernel argument block exceeds Runtime V2 argsSize");
    ReduceGradKernelArgs args {
        context.devArgs,
        reinterpret_cast<GM_ADDR>(params.plan->expertsToCopy),
        reinterpret_cast<GM_ADDR>(params.gradients[kReduceGradGate]->data),
        reinterpret_cast<GM_ADDR>(params.gradients[kReduceGradUp]->data),
        reinterpret_cast<GM_ADDR>(params.gradients[kReduceGradDown]->data),
        reinterpret_cast<GM_ADDR>(params.workspace),
        reinterpret_cast<GM_ADDR>(params.status->data),
        layout.rank,
        layout.rankSize,
        layout.expertCount,
        layout.expertsPerRank,
        layout.prefetchSlots,
        layout.controlBlockCount,
        layout.rowElements[kReduceGradGate],
        layout.rowElements[kReduceGradUp],
        layout.rowElements[kReduceGradDown],
        layout.rowBytes[kReduceGradGate],
        layout.rowBytes[kReduceGradUp],
        layout.rowBytes[kReduceGradDown],
        layout.transports[kReduceGradGate],
        layout.transports[kReduceGradUp],
        layout.transports[kReduceGradDown],
        layout.udmaQpCount,
        layout.peerRecordBaseOffset,
        layout.peerHalfBytes,
        layout.peerSlotStrideBytes,
        layout.peerChunkPayloadBytes,
        layout.udmaStateOffset,
        layout.udmaOutboundOffset,
        layout.udmaInboundOffset,
        layout.udmaChunkBytes,
        layout.workspaceBytes,
        params.waitIterations,
        magic,
    };

    rtArgsEx_t argsInfo {};
    argsInfo.args = &args;
    argsInfo.argsSize = static_cast<uint32_t>(sizeof(args));
    rtTaskCfgInfo_t cfgInfo {};
    cfgInfo.schemMode = 1;
    const rtError_t launchRet = rtKernelLaunchWithFlagV2(kReduceGradKernelName,
        static_cast<uint32_t>(layout.blockDim), &argsInfo, nullptr,
        static_cast<rtStream_t>(params.stream), 0, &cfgInfo);
    return launchRet == RT_ERROR_NONE ? TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_MKIRT;
}

} // namespace TileXRMoonEp
