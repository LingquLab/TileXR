#include "moonep_kernel_registration.h"

#include <iostream>

#include "runtime/kernel.h"
#include "tilexr_types.h"

namespace TileXRMoonEp {

int EnsureMoonEpKernelRegistered(KernelRegistrationState &state,
    const unsigned char *binaryData, std::size_t binarySize,
    const void *signature, const char *kernelName)
{
    std::lock_guard<std::mutex> guard(state.mutex);
    if (state.registered) {
        return TileXR::TILEXR_SUCCESS;
    }
    if (binaryData == nullptr || binarySize == 0 ||
        signature == nullptr || kernelName == nullptr) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    rtDevBinary_t binary {};
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.version = 0;
    binary.data = binaryData;
    binary.length = static_cast<uint64_t>(binarySize);

    void *binaryHandle = nullptr;
    rtError_t ret = rtDevBinaryRegister(&binary, &binaryHandle);
    if (ret != RT_ERROR_NONE) {
        std::cerr << "TileXR MoonEP rtDevBinaryRegister failed, ret=" << ret
                  << ", kernel=" << kernelName
                  << ", binarySize=" << binarySize << std::endl;
        return TileXR::TILEXR_ERROR_MKIRT;
    }

    ret = rtFunctionRegister(binaryHandle, signature, kernelName, kernelName, 0);
    if (ret != RT_ERROR_NONE) {
        std::cerr << "TileXR MoonEP rtFunctionRegister failed, ret=" << ret
                  << ", kernel=" << kernelName << std::endl;
        (void)rtDevBinaryUnRegister(binaryHandle);
        return TileXR::TILEXR_ERROR_MKIRT;
    }

    state.binaryHandle = binaryHandle;
    state.registered = true;
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXRMoonEp
