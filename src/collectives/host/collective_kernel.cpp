/*
 * Copyright (c) 2024-2026 TileXR Project
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "collective_kernel.h"

#include "collective_utils.h"
#include "runtime/kernel.h"

namespace TileXRCollectives {
namespace Host {
namespace {

const void *g_collectiveKernelStub = nullptr;

bool AreCollectiveKernelsRegistered()
{
    return g_collectiveKernelStub != nullptr;
}

} // namespace

int LaunchCollectiveKernel(TileXRCommPtr comm, TileXR::TileXRType type, const HostLaunchContext &context,
                           void *sendBuf, void *recvBuf, int64_t kernelCount,
                           TileXR::TileXRDataType dataType, uint32_t blockDim,
                           aclrtStream stream)
{
    if ((type != TileXR::TileXRType::ALL_GATHER && type != TileXR::TileXRType::ALL2ALL) ||
        comm == nullptr || context.hostArgs == nullptr || context.devArgs == nullptr || sendBuf == nullptr || recvBuf == nullptr ||
        kernelCount <= 0 || blockDim == 0 || !IsSupportedDataType(dataType)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    if (!AreCollectiveKernelsRegistered()) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    int64_t magic = 0;
    const int magicRet = TileXRCommNextMagic(comm, &magic);
    if (magicRet != TileXR::TILEXR_SUCCESS) {
        return magicRet;
    }

    AscendCCLKernelArgs args {};
    args.input = sendBuf;
    args.output = recvBuf;
    args.commArgsPtr = context.devArgs;
    args.count = kernelCount;
    args.magic = magic;
    args.op = static_cast<int>(type);

    rtArgsEx_t argsInfo {};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);
    const rtError_t ret = rtKernelLaunchWithFlagV2(g_collectiveKernelStub, blockDim, &argsInfo, nullptr,
        static_cast<rtStream_t>(stream), RT_KERNEL_DEFAULT, nullptr);
    return ret == RT_ERROR_NONE ? TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_MKIRT;
}

} // namespace Host
} // namespace TileXRCollectives
