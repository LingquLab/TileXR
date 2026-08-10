#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "acl/acl_rt.h"
#include "combine_v2_launch.h"
#include "moonep_kernel_registration.h"
#include "runtime/kernel.h"
#include "tilexr_types.h"

namespace {

constexpr int64_t kA5UbBytes = 248 * 1024;
constexpr int64_t kMinUbBytes = 216 * 1024;

int failures = 0;
aclError getDeviceReturn = ACL_SUCCESS;
aclError getDeviceInfoReturn = ACL_SUCCESS;
int64_t dynamicUbBytes = kA5UbBytes;
int registrationReturn = TileXR::TILEXR_SUCCESS;
rtError_t launchReturn = RT_ERROR_NONE;
int registrationCalls = 0;
int launchCalls = 0;
uint32_t capturedBlockDim = 0;
rtTaskCfgInfo_t capturedCfg {};

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

void Reset()
{
    getDeviceReturn = ACL_SUCCESS;
    getDeviceInfoReturn = ACL_SUCCESS;
    dynamicUbBytes = kA5UbBytes;
    registrationReturn = TileXR::TILEXR_SUCCESS;
    launchReturn = RT_ERROR_NONE;
    registrationCalls = 0;
    launchCalls = 0;
    capturedBlockDim = 0;
    capturedCfg = rtTaskCfgInfo_t {};
}

TileXRMoonEp::CombineV2Params ValidParams(uint64_t *activeOutputOffset)
{
    TileXRMoonEp::CombineV2Params params {};
    params.registeredWorkspace = reinterpret_cast<void *>(uintptr_t {0x1000});
    params.dstLocal = reinterpret_cast<const int32_t *>(uintptr_t {0x2000});
    params.bs = 8;
    params.h = 3584;
    params.topK = 16;
    params.nvS = 128;
    params.aivCoreNum = 32;
    params.activeOutputOffset = activeOutputOffset;
    params.stream = reinterpret_cast<aclrtStream>(uintptr_t {0x3000});
    return params;
}

TileXRMoonEp::CombineV2LaunchContext ValidContext()
{
    TileXRMoonEp::CombineV2LaunchContext context {};
    context.devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x4000});
    context.layout.scratchOffset[0] = 4096;
    context.layout.scratchOffset[1] = 8192;
    context.magic = 1;
    return context;
}

void TestConfiguresDynamicUb()
{
    Reset();
    uint64_t activeOutputOffset = 0;
    const TileXRMoonEp::CombineV2Params params =
        ValidParams(&activeOutputOffset);
    const TileXRMoonEp::CombineV2LaunchContext context = ValidContext();

    CheckStatus(TileXRMoonEp::TileXRMoonEpLaunchCombineV2Kernel(params, context),
        TileXR::TILEXR_SUCCESS, "configured launch");
    Check(registrationCalls == 1 && launchCalls == 1,
        "configured launch did not reach Runtime");
    Check(capturedBlockDim == params.aivCoreNum,
        "configured launch did not use the requested block dimension");
    Check(capturedCfg.schemMode == 1,
        "configured launch did not preserve batch scheduling mode");
    Check(capturedCfg.localMemorySize == static_cast<uint32_t>(kA5UbBytes),
        "configured launch did not pass the device UB size");
    Check(activeOutputOffset == context.layout.scratchOffset[1],
        "configured launch did not publish the active epoch");
}

void TestRejectsUnavailableDevice()
{
    Reset();
    getDeviceReturn = 1;
    uint64_t activeOutputOffset = 0;
    CheckStatus(TileXRMoonEp::TileXRMoonEpLaunchCombineV2Kernel(
        ValidParams(&activeOutputOffset), ValidContext()),
        TileXR::TILEXR_ERROR_INTERNAL, "device query failure");
    Check(registrationCalls == 0 && launchCalls == 0,
        "device query failure still launched the kernel");
}

void TestRejectsInvalidDynamicUb()
{
    Reset();
    getDeviceInfoReturn = 1;
    uint64_t activeOutputOffset = 0;
    CheckStatus(TileXRMoonEp::TileXRMoonEpLaunchCombineV2Kernel(
        ValidParams(&activeOutputOffset), ValidContext()),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "UB query failure");

    Reset();
    dynamicUbBytes = kMinUbBytes - 1;
    CheckStatus(TileXRMoonEp::TileXRMoonEpLaunchCombineV2Kernel(
        ValidParams(&activeOutputOffset), ValidContext()),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL, "insufficient dynamic UB");
    Check(registrationCalls == 0 && launchCalls == 0,
        "invalid dynamic UB still launched the kernel");
}

} // namespace

extern "C" {
extern const unsigned char TileXRMoonEpCombineV2KernelBinaryData[] = {0};
extern const std::size_t TileXRMoonEpCombineV2KernelBinarySize = 1;

aclError aclrtGetDevice(int32_t *deviceId)
{
    if (getDeviceReturn == ACL_SUCCESS && deviceId != nullptr) {
        *deviceId = 0;
    }
    return getDeviceReturn;
}

aclError aclrtGetDeviceInfo(uint32_t, aclrtDevAttr attr, int64_t *value)
{
    if (attr != ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE) {
        return 1;
    }
    if (getDeviceInfoReturn == ACL_SUCCESS && value != nullptr) {
        *value = dynamicUbBytes;
    }
    return getDeviceInfoReturn;
}

rtError_t rtKernelLaunchWithFlagV2(const void *, uint32_t blockDim,
    rtArgsEx_t *,
    void *, rtStream_t, uint32_t, const rtTaskCfgInfo_t *cfgInfo)
{
    ++launchCalls;
    capturedBlockDim = blockDim;
    if (cfgInfo != nullptr) {
        capturedCfg = *cfgInfo;
    }
    return launchReturn;
}
} // extern "C"

namespace TileXRMoonEp {

int EnsureMoonEpKernelRegistered(KernelRegistrationState &,
    const unsigned char *, std::size_t, const void *, const char *)
{
    ++registrationCalls;
    return registrationReturn;
}

} // namespace TileXRMoonEp

int main()
{
    TestConfiguresDynamicUb();
    TestRejectsUnavailableDevice();
    TestRejectsInvalidDynamicUb();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
