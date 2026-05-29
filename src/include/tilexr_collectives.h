/*
 * Copyright (c) 2024-2026 TileXR Project
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEXR_COLLECTIVES_H
#define TILEXR_COLLECTIVES_H

#include <cstdint>
#include "acl/acl_base.h"
#include "tilexr_api.h"
#include "tilexr_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int TileXRAllGather(void *sendBuf, void *recvBuf, int64_t sendCount,
                    TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                    aclrtStream stream);
int TileXRAllToAll(void *sendBuf, void *recvBuf, int64_t sendCount,
                   TileXR::TileXRDataType dataType, TileXRCommPtr comm,
                   aclrtStream stream);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // TILEXR_COLLECTIVES_H
