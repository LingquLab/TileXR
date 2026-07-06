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

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include "comm_args.h"
#include "tilexr_udma_reg.h"
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
typedef uint8_t *GM_ADDR;
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void *TileXRCommPtr;
typedef void *TileXRDirectCcuPreparedTasksPtr;
typedef uint32_t TileXRUDMAMemHandle;
#define TILEXRUNIQUE_ID_BYTES 128
#define TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES 2048
#define TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_WINDOW_MISSION 0U
#define TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_WINDOW_FULL_REPOSITORY 1U
#define TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_DATA_LEN_INSTRUCTION_BYTES 0U
#define TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_DATA_LEN_DESCRIPTOR_BYTES 1U
#define TILEXR_DIRECT_CCU_REPOSITORY_MEMORY_ALLOC_ACL 0U
#define TILEXR_DIRECT_CCU_REPOSITORY_MEMORY_ALLOC_ACL_MODULE3 1U
#define TILEXR_DIRECT_CCU_REPOSITORY_MEMORY_ALLOC_RT_HBM 2U
#define TILEXR_DIRECT_CCU_INSTALL_ORDER_REPOSITORY_FIRST 0U
#define TILEXR_DIRECT_CCU_INSTALL_ORDER_LOWER_LAYER_FIRST 1U
#define TILEXR_DIRECT_CCU_MEMORY_COPY_REMOTE_TO_LOCAL 0U
#define TILEXR_DIRECT_CCU_MEMORY_COPY_LOCAL_TO_REMOTE 1U
#define TILEXR_DIRECT_CCU_SQE_ARGS_LEN 13U
typedef struct { char internal[TILEXRUNIQUE_ID_BYTES]; } TileXRUniqueId;

typedef struct TileXRDirectCcuPrepareOptions {
    uint32_t syncResourceCount;
    /* 0 is a valid explicit no-SQE-load mode; use TILEXR_DIRECT_CCU_SQE_ARGS_LEN for full SQE args. */
    uint32_t sqeArgCount;
    uint32_t syncInstructionCount;
    uint32_t bindingsPerSyncResource;
    uint16_t missionStartId;
    uint16_t instructionStartId;
    uint16_t missionInstructionStartId;
    uint16_t xnStartId;
    uint16_t gsaStartId;
    uint16_t remoteXnStartId;
    uint16_t remoteXnCount;
    uint16_t ckeStartId;
    uint16_t channelStartId;
    uint16_t localWaitCkeStartId;
    uint16_t localWaitCkeCount;
    uint16_t remoteNotifyCkeStartId;
    uint16_t remoteNotifyCkeCount;
    uint32_t repositoryInstallWindow;
    uint32_t repositoryInstallDataLenMode;
    uint32_t repositoryMemoryAllocMode;
    uint32_t installOrder;
    uint32_t deviceId;
    uint32_t rank;
    const char *provider;
} TileXRDirectCcuPrepareOptions;

typedef struct TileXRDirectCcuMemoryCopyPrepareOptions {
    TileXRDirectCcuPrepareOptions prepare;
    uint64_t localSourceAddr;
    uint64_t localDestinationAddr;
    uint64_t bytes;
    uint32_t peerRank;
    uint32_t direction;
} TileXRDirectCcuMemoryCopyPrepareOptions;

typedef struct TileXRDirectCcuPrepareReport {
    bool pipelineBuilt;
    bool installAttempted;
    bool installSucceeded;
    bool submitReady;
    uint32_t requiredInstallSurfaceCount;
    uint32_t publicVerifiedInstallSurfaceCount;
    uint32_t missingInstallSurfaceCount;
    uint32_t taskCount;
    uint32_t submitTaskCount;
    char message[TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES];
} TileXRDirectCcuPrepareReport;

typedef struct TileXRDirectCcuSubmitReport {
    bool submitted;
    uint32_t taskCount;
    uint32_t submittedTaskCount;
    char message[TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES];
} TileXRDirectCcuSubmitReport;

typedef struct TileXRDirectCcuTaskInfo {
    uint8_t dieId;
    uint8_t missionId;
    uint16_t timeout;
    uint16_t instStartId;
    uint16_t instCnt;
    uint32_t key;
    uint32_t argSize;
    uint64_t args[TILEXR_DIRECT_CCU_SQE_ARGS_LEN];
} TileXRDirectCcuTaskInfo;

#define TILEXR_DIRECT_CCU_INSTRUCTION_WORDS 4U
typedef struct TileXRDirectCcuInstructionWords {
    uint64_t words[TILEXR_DIRECT_CCU_INSTRUCTION_WORDS];
} TileXRDirectCcuInstructionWords;

typedef struct TileXRDirectCcuInstructionReadbackReport {
    bool readbackAttempted;
    uint32_t readInstructionCount;
    uint32_t opcode;
    int driverRet;
    int opRet;
    char message[TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES];
} TileXRDirectCcuInstructionReadbackReport;

int TileXRGetUniqueId(TileXRUniqueId *uniqueId, int commDomain);

int TileXRCommInitRankLocal(int rankSize, int rank, TileXRCommPtr *comm);

int TileXRCommInitRank(TileXRUniqueId commId, int rankSize, int rank, TileXRCommPtr *comm);

int TileXRCommInitRankWithCustDomainSize(int commDomain, int bufferSize, int rankSize, int rank, TileXRCommPtr *comm);

int TileXRCommInitRankWithDomain(int commDomain, int rankSize, int rank, TileXRCommPtr *comm);

int TileXRCommInitRankDirectCcuWithDomain(int commDomain, int rankSize, int rank, TileXRCommPtr *comm);

#ifdef __cplusplus
int TileXRGetCommArgsDev(TileXRCommPtr comm, GM_ADDR &commArgsPtr);

int TileXRGetCommArgsHost(TileXRCommPtr comm, TileXR::CommArgs *&commArgsPtr);
#endif

int TileXRCommNextMagic(TileXRCommPtr comm, int64_t *magic);

int TileXRUDMARegister(TileXRCommPtr comm, GM_ADDR localPtr, size_t bytes, TileXRUDMAMemHandle *handle);

int TileXRUDMAUnregister(TileXRCommPtr comm, TileXRUDMAMemHandle handle);

#ifdef __cplusplus
int TileXRGetUDMARegistryDev(TileXRCommPtr comm, GM_ADDR &registryPtr);

int TileXRGetUDMARegistryHost(TileXRCommPtr comm, const TileXR::TileXRUDMARegistry **registry);
#endif

int TileXRSDMAAvailable(TileXRCommPtr comm, bool *available);

int TileXRGetSDMAWorkspaceDev(TileXRCommPtr comm, GM_ADDR *workspace);

int TileXRCommPrepareDirectCcu(
    TileXRCommPtr comm,
    const TileXRDirectCcuPrepareOptions *options,
    TileXRDirectCcuPreparedTasksPtr *prepared,
    TileXRDirectCcuPrepareReport *report);

int TileXRCommPrepareDirectCcuMemoryCopy(
    TileXRCommPtr comm,
    const TileXRDirectCcuMemoryCopyPrepareOptions *options,
    TileXRDirectCcuPreparedTasksPtr *prepared,
    TileXRDirectCcuPrepareReport *report);

int TileXRDirectCcuGetPreparedTask(
    TileXRDirectCcuPreparedTasksPtr prepared,
    uint32_t index,
    TileXRDirectCcuTaskInfo *task);

int TileXRDirectCcuSubmitPrepared(
    TileXRDirectCcuPreparedTasksPtr prepared,
    void *stream,
    TileXRDirectCcuSubmitReport *report);

int TileXRDirectCcuSubmitPreparedTask(
    TileXRDirectCcuPreparedTasksPtr prepared,
    uint32_t index,
    void *stream,
    TileXRDirectCcuSubmitReport *report);

int TileXRCommReadDirectCcuInstructions(
    TileXRCommPtr comm,
    uint8_t dieId,
    uint16_t instructionStartId,
    uint32_t instructionCount,
    TileXRDirectCcuInstructionWords *instructions,
    TileXRDirectCcuInstructionReadbackReport *report);

int TileXRDirectCcuDestroyPrepared(TileXRDirectCcuPreparedTasksPtr prepared);

void TileXRPrintDFX2Log(TileXRCommPtr comm);

int TileXRCommInit(int rank, int rankSize, TileXRCommPtr *comms);

int TileXRCommInitAll(uint32_t ndev, int32_t* devices, TileXRCommPtr *comms);

int TileXRCommInitThread(int rank, int rankSize, const char *uid, TileXRCommPtr *comms);

int TileXRCommDestroy(TileXRCommPtr comm);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // TILEXR_API_H
