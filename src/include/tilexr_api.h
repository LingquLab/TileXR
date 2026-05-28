/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEXR_API_H
#define TILEXR_API_H

#include <cstddef>
#include <string>
#include "comm_args.h"
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void *TileXRCommPtr;
typedef uint32_t TileXRUDMAMemHandle;
#define TILEXRUNIQUE_ID_BYTES 128
typedef struct { char internal[TILEXRUNIQUE_ID_BYTES]; } TileXRUniqueId;

int TileXRGetUniqueId(TileXRUniqueId *uniqueId, int commDomain);

int TileXRCommInitRankLocal(int rankSize, int rank, TileXRCommPtr *comm);

int TileXRCommInitRank(TileXRUniqueId commId, int rankSize, int rank, TileXRCommPtr *comm);

int TileXRCommInitRankWithCustDomainSize(int commDomain, int bufferSize, int rankSize, int rank, TileXRCommPtr *comm);

int TileXRCommInitRankWithDomain(int commDomain, int rankSize, int rank, TileXRCommPtr *comm);

int TileXRGetCommArgsDev(TileXRCommPtr comm, GM_ADDR &commArgsPtr);

int TileXRGetCommArgsHost(TileXRCommPtr comm, TileXR::CommArgs *&commArgsPtr);

int TileXRUDMARegister(TileXRCommPtr comm, GM_ADDR localPtr, size_t bytes, TileXRUDMAMemHandle *handle);

int TileXRUDMAUnregister(TileXRCommPtr comm, TileXRUDMAMemHandle handle);

int TileXRGetUDMARegistryDev(TileXRCommPtr comm, GM_ADDR &registryPtr);

void TileXRPrintDFX2Log(TileXRCommPtr comm);

int TileXRCommInit(int rank, int rankSize, TileXRCommPtr *comms);

int TileXRCommInitAll(uint32_t ndev, int32_t* devices, TileXRCommPtr *comms);

int TileXRCommInitThread(int rank, int rankSize, const char *uid, TileXRCommPtr *comms);

int TileXRCommDestroy(TileXRCommPtr comm);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // TILEXR_API_H
