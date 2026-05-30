/*
 * Copyright (c) 2024-2026 TileXR Project
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "collective_launcher.h"

#include "tilexr_types.h"

namespace TileXRCollectives {
namespace Host {

int PrepareHostLaunchContext(TileXRCommPtr comm, HostLaunchContext &context)
{
    context = HostLaunchContext {};
    int ret = TileXRGetCommArgsHost(comm, context.hostArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (context.hostArgs == nullptr) {
        context = HostLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    ret = TileXRGetCommArgsDev(comm, context.devArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        context = HostLaunchContext {};
        return ret;
    }
    if (context.devArgs == nullptr) {
        context = HostLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace Host
} // namespace TileXRCollectives
