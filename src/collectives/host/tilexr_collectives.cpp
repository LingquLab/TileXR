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
#include "tilexr_collectives.h"

namespace {

void TouchHostLaunchContext(TileXRCommPtr comm)
{
    if (comm == nullptr) {
        return;
    }

    TileXRCollectives::Host::HostLaunchContext context;
    (void)TileXRCollectives::Host::PrepareHostLaunchContext(comm, context);
}

} // namespace

int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,
                    TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                    aclrtStream stream)
{
    (void)sendBuf;
    (void)recvBuf;
    (void)sendCount;
    (void)dataType;
    (void)stream;
    TouchHostLaunchContext(comm);
    return TileXR::TILEXR_ERROR_INTERNAL;
}

int TileXRAllToAll(void *sendBuf, void *recvBuf, int64_t sendCount,
                   TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                   aclrtStream stream)
{
    (void)sendBuf;
    (void)recvBuf;
    (void)sendCount;
    (void)dataType;
    (void)stream;
    TouchHostLaunchContext(comm);
    return TileXR::TILEXR_ERROR_INTERNAL;
}
