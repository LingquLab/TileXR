/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tilexr_api.h"
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <acl/acl_rt.h>

#include "tilexr_comm.h"
#include "tilexr_log.h"
#include "tools/socket/tilexr_sock_exchange.h"

using namespace std;
using namespace TileXR;

namespace {

constexpr const char* TILEXR_DIRECT_CCU_PUBLIC_PROVIDER = "tilexr-public-direct-ccu";
constexpr const char* TILEXR_DIRECT_CCU_BARRIER_MODE_ENV = "TILEXR_CCU_DIRECT_BARRIER_MODE";
constexpr uint32_t TILEXR_DIRECT_CCU_MEMORY_COPY_INSTRUCTION_COUNT = 7U;

struct TileXRDirectCcuPreparedTasks {
    ~TileXRDirectCcuPreparedTasks()
    {
        (void)TileXRCcuReleaseDirectInstallAttemptResources(attempt);
    }

    TileXRCcuDirectInstallAttempt attempt;
};

void CopyDirectCcuMessage(const std::string& message, char* output)
{
    if (output == nullptr) {
        return;
    }
    std::memset(output, 0, TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES);
    if (message.empty()) {
        return;
    }
    std::strncpy(output, message.c_str(), TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES - 1);
}

void FillPublicPrepareReport(
    const TileXRCcuDirectInstallReport& source,
    TileXRDirectCcuPrepareReport* report)
{
    if (report == nullptr) {
        return;
    }
    *report = TileXRDirectCcuPrepareReport {};
    report->pipelineBuilt = source.pipelineBuilt;
    report->installAttempted = source.installAttempted;
    report->installSucceeded = source.installSucceeded;
    report->submitReady = source.submitReady;
    report->requiredInstallSurfaceCount = source.requiredInstallSurfaceCount;
    report->publicVerifiedInstallSurfaceCount = source.publicVerifiedInstallSurfaceCount;
    report->missingInstallSurfaceCount = source.missingInstallSurfaceCount;
    report->taskCount = source.taskCount;
    report->submitTaskCount = source.submitTaskCount;
    CopyDirectCcuMessage(source.message, report->message);
}

void FillPublicSubmitReport(
    const TileXRCcuDirectSubmitReport& source,
    TileXRDirectCcuSubmitReport* report)
{
    if (report == nullptr) {
        return;
    }
    *report = TileXRDirectCcuSubmitReport {};
    report->submitted = source.submitted;
    report->taskCount = source.taskCount;
    report->submittedTaskCount = source.submittedTaskCount;
    CopyDirectCcuMessage(source.message, report->message);
}

void FillPublicInstructionReadbackReport(
    uint32_t readInstructionCount,
    const TileXRCcuDriverAdapterReport& source,
    TileXRDirectCcuInstructionReadbackReport* report)
{
    if (report == nullptr) {
        return;
    }
    *report = TileXRDirectCcuInstructionReadbackReport {};
    report->readbackAttempted = true;
    report->readInstructionCount = readInstructionCount;
    report->opcode = source.opcode;
    report->driverRet = source.driverRet;
    report->opRet = source.opRet;
    CopyDirectCcuMessage(source.message, report->message);
}

TileXRCcuRepositoryInstallWindow RepositoryInstallWindowFromPublic(uint32_t value)
{
    return value == TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_WINDOW_FULL_REPOSITORY ?
        TileXRCcuRepositoryInstallWindow::FullRepository :
        TileXRCcuRepositoryInstallWindow::Mission;
}

TileXRCcuRepositoryInstallDataLenMode RepositoryInstallDataLenModeFromPublic(uint32_t value)
{
    return value == TILEXR_DIRECT_CCU_REPOSITORY_INSTALL_DATA_LEN_DESCRIPTOR_BYTES ?
        TileXRCcuRepositoryInstallDataLenMode::DescriptorBytes :
        TileXRCcuRepositoryInstallDataLenMode::InstructionBytes;
}

TileXRCcuRepositoryMemoryAllocMode RepositoryMemoryAllocModeFromPublic(uint32_t value)
{
    if (value == TILEXR_DIRECT_CCU_REPOSITORY_MEMORY_ALLOC_RT_HBM) {
        return TileXRCcuRepositoryMemoryAllocMode::RtHbm;
    }
    return value == TILEXR_DIRECT_CCU_REPOSITORY_MEMORY_ALLOC_ACL_MODULE3 ?
        TileXRCcuRepositoryMemoryAllocMode::AclModule3 :
        TileXRCcuRepositoryMemoryAllocMode::Acl;
}

TileXRCcuInstallOrder InstallOrderFromPublic(uint32_t value)
{
    return value == TILEXR_DIRECT_CCU_INSTALL_ORDER_LOWER_LAYER_FIRST ?
        TileXRCcuInstallOrder::InstallLowerLayerFirst :
        TileXRCcuInstallOrder::RepositoryFirst;
}

TileXRCcuDirectInstallOptions MakeDirectCcuOptions(const TileXRDirectCcuPrepareOptions& publicOptions)
{
    TileXRCcuDirectInstallOptions options;
    options.syncResourceCount = publicOptions.syncResourceCount;
    options.sqeArgCount = publicOptions.sqeArgCount;
    options.syncInstructionCount = publicOptions.syncInstructionCount;
    options.bindingsPerSyncResource = publicOptions.bindingsPerSyncResource;
    options.missionStartId = publicOptions.missionStartId;
    options.instructionStartId = publicOptions.instructionStartId;
    options.missionInstructionStartId = publicOptions.missionInstructionStartId;
    options.xnStartId = publicOptions.xnStartId;
    options.gsaStartId = publicOptions.gsaStartId;
    options.remoteXnStartId = publicOptions.remoteXnStartId;
    options.remoteXnCount = publicOptions.remoteXnCount;
    options.ckeStartId = publicOptions.ckeStartId;
    options.channelStartId = publicOptions.channelStartId;
    options.localWaitCkeStartId = publicOptions.localWaitCkeStartId;
    options.localWaitCkeCount = publicOptions.localWaitCkeCount;
    options.remoteNotifyCkeStartId = publicOptions.remoteNotifyCkeStartId;
    options.remoteNotifyCkeCount = publicOptions.remoteNotifyCkeCount;
    options.repositoryInstallOptions.window =
        RepositoryInstallWindowFromPublic(publicOptions.repositoryInstallWindow);
    options.repositoryInstallOptions.dataLenMode =
        RepositoryInstallDataLenModeFromPublic(publicOptions.repositoryInstallDataLenMode);
    options.repositoryMemoryAllocMode =
        RepositoryMemoryAllocModeFromPublic(publicOptions.repositoryMemoryAllocMode);
    options.installOrder = InstallOrderFromPublic(publicOptions.installOrder);
    options.deviceId = publicOptions.deviceId;
    options.rank = publicOptions.rank;
    const char* barrierMode = std::getenv(TILEXR_DIRECT_CCU_BARRIER_MODE_ENV);
    if (barrierMode != nullptr && std::strcmp(barrierMode, "sync_cke") == 0) {
        options.barrierMode = TileXRCcuBarrierMode::SyncCke;
    } else if (barrierMode != nullptr && std::strcmp(barrierMode, "sync_cke_set_wait") == 0) {
        options.barrierMode = TileXRCcuBarrierMode::SyncCkeSetWait;
    } else if (barrierMode != nullptr && std::strcmp(barrierMode, "sync_cke_post_only") == 0) {
        options.barrierMode = TileXRCcuBarrierMode::SyncCkePostOnly;
    } else if (barrierMode != nullptr && std::strcmp(barrierMode, "local_cke") == 0) {
        options.barrierMode = TileXRCcuBarrierMode::LocalCke;
    } else if (barrierMode != nullptr && std::strcmp(barrierMode, "local_cke_post_only") == 0) {
        options.barrierMode = TileXRCcuBarrierMode::LocalCkePostOnly;
    } else if (barrierMode != nullptr && std::strcmp(barrierMode, "sync_xn_post_only") == 0) {
        options.barrierMode = TileXRCcuBarrierMode::SyncXnPostOnly;
    } else if (barrierMode != nullptr && std::strcmp(barrierMode, "sync_xn_load_post_only") == 0) {
        options.barrierMode = TileXRCcuBarrierMode::SyncXnLoadPostOnly;
    }
    options.provider = (publicOptions.provider == nullptr || publicOptions.provider[0] == '\0') ?
        TILEXR_DIRECT_CCU_PUBLIC_PROVIDER :
        publicOptions.provider;
    return options;
}

TileXRDirectCcuPreparedTasks* PreparedHandle(TileXRDirectCcuPreparedTasksPtr prepared)
{
    return static_cast<TileXRDirectCcuPreparedTasks*>(prepared);
}

TileXRCcuMemoryCopyDirection MemoryCopyDirectionFromPublic(uint32_t direction)
{
    return direction == TILEXR_DIRECT_CCU_MEMORY_COPY_LOCAL_TO_REMOTE ?
        TileXRCcuMemoryCopyDirection::LocalToRemote :
        TileXRCcuMemoryCopyDirection::RemoteToLocal;
}

} // namespace

int TileXRCommInitRankLocal(int rankSize, int rank, TileXRCommPtr *comm)
{
    TILEXR_LOG(INFO) << "using tilexr c++ api! rank" << rank;
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm ptr is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    *comm = nullptr;
    unique_ptr<TileXRComm> c(new (std::nothrow) TileXRComm(rank, rankSize));
    if (c == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    int ret = c->Init();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr init failed! ret: " << ret;
        return ret;
    }
    *comm = c.release();
    return TILEXR_SUCCESS;
}

int TileXRGetUniqueId(TileXRUniqueId *uniqueId, int commDomain)
{
    if (uniqueId == nullptr) {
        TILEXR_LOG(ERROR) << "uniqueId is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    int res = BootstrapGetUniqueId(reinterpret_cast<struct TileXRBootstrapHandle *>(uniqueId), commDomain);
    if (res != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr BootstrapGetUniqueId failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRCommInitRank(TileXRUniqueId commId, int rankSize, int rank, TileXRCommPtr *comm)
{
    TILEXR_LOG(INFO) << "using tilexr c++ api! rank" << rank;
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm ptr is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    *comm = nullptr;
    unique_ptr<TileXRComm> c(new (std::nothrow) TileXRComm(rank, rankSize, commId));
    if (c == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    int ret = c->Init();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr init failed! ret: " << ret;
        return ret;
    }
    *comm = c.release();
    return TILEXR_SUCCESS;
}

int TileXRCommInitRankWithCustDomainSize(int commDomain, int bufferSize, int rankSize, int rank, TileXRCommPtr *comm)
{
    TILEXR_LOG(INFO) << "using tilexr c++ api! rank" << rank;
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm ptr is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    *comm = nullptr;

    constexpr int minBufferSize = TILEXR_COMM_BUFFER_SIZE;
    if (bufferSize < minBufferSize) {
        TILEXR_LOG(ERROR) << "tilexr comm buffer size " << bufferSize << " MBytes should not be less than " <<
            minBufferSize << " MBytes!";
        return TILEXR_ERROR_INTERNAL;
    }

    unique_ptr<TileXRComm> c(new (std::nothrow) TileXRComm(rank, rankSize, commDomain, bufferSize));
    if (c == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    int ret = c->Init();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr init failed! ret: " << ret;
        return ret;
    }
    *comm = c.release();
    return TILEXR_SUCCESS;
}

int TileXRCommInitRankWithDomain(int commDomain, int rankSize, int rank, TileXRCommPtr *comm)
{
    constexpr int minBufferSize = TILEXR_COMM_BUFFER_SIZE;
    return TileXRCommInitRankWithCustDomainSize(commDomain, minBufferSize, rankSize, rank, comm);
}

int TileXRCommInitRankDirectCcuWithDomain(int commDomain, int rankSize, int rank, TileXRCommPtr *comm)
{
    TILEXR_LOG(INFO) << "using tilexr direct CCU only api! rank" << rank;
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr direct CCU only comm ptr is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    *comm = nullptr;
    unique_ptr<TileXRComm> c(new (std::nothrow) TileXRComm(rank, rankSize, commDomain, TILEXR_COMM_BUFFER_SIZE));
    if (c == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed for direct CCU only init. rank : "
                          << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    int ret = c->InitDirectCcuOnly();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr direct CCU only init failed! ret: " << ret;
        return ret;
    }
    *comm = c.release();
    return TILEXR_SUCCESS;
}

int TileXRGetCommArgsDev(TileXRCommPtr comm, GM_ADDR &commArgsPtr)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    auto* tilexr = static_cast<TileXRComm *>(comm);
    commArgsPtr = tilexr->GetCommArgsPtr();
    return TILEXR_SUCCESS;
}

int TileXRGetCommArgsHost(TileXRCommPtr comm, TileXR::CommArgs *&commArgsPtr)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    commArgsPtr = c->GetCommArgs();
    return TILEXR_SUCCESS;
}

int TileXRCommNextMagic(TileXRCommPtr comm, int64_t *magic)
{
    if (comm == nullptr || magic == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRCommNextMagic invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    *magic = c->NextMagic();
    return TILEXR_SUCCESS;
}

int TileXRUDMARegister(TileXRCommPtr comm, GM_ADDR localPtr, size_t bytes, TileXRUDMAMemHandle *handle)
{
    if (comm == nullptr || localPtr == nullptr || handle == nullptr || bytes == 0) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    return c->RegisterUDMAMemory(localPtr, bytes, handle);
}

int TileXRUDMAUnregister(TileXRCommPtr comm, TileXRUDMAMemHandle handle)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRUDMAUnregister invalid comm";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    return c->UnregisterUDMAMemory(handle);
}

int TileXRGetUDMARegistryDev(TileXRCommPtr comm, GM_ADDR &registryPtr)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRGetUDMARegistryDev invalid comm";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    registryPtr = c->GetUDMARegistryPtr();
    return TILEXR_SUCCESS;
}

int TileXRGetUDMARegistryHost(TileXRCommPtr comm, const TileXR::TileXRUDMARegistry **registry)
{
    if (comm == nullptr || registry == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRGetUDMARegistryHost invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    *registry = c->GetUDMARegistryHost();
    return *registry == nullptr ? TILEXR_ERROR_NOT_INITIALIZED : TILEXR_SUCCESS;
}

int TileXRSDMAAvailable(TileXRCommPtr comm, bool *available)
{
    if (comm == nullptr || available == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRSDMAAvailable invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    *available = c->IsSDMAAvailable();
    return TILEXR_SUCCESS;
}

int TileXRGetSDMAWorkspaceDev(TileXRCommPtr comm, GM_ADDR *workspace)
{
    if (comm == nullptr || workspace == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRGetSDMAWorkspaceDev invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* c = static_cast<TileXRComm *>(comm);
    *workspace = c->GetSDMAWorkspacePtr();
    return TILEXR_SUCCESS;
}

int TileXRCommPrepareDirectCcu(
    TileXRCommPtr comm,
    const TileXRDirectCcuPrepareOptions *options,
    TileXRDirectCcuPreparedTasksPtr *prepared,
    TileXRDirectCcuPrepareReport *report)
{
    if (prepared != nullptr) {
        *prepared = nullptr;
    }
    if (report != nullptr) {
        *report = TileXRDirectCcuPrepareReport {};
    }
    if (comm == nullptr || options == nullptr || prepared == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRCommPrepareDirectCcu invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    std::unique_ptr<TileXRDirectCcuPreparedTasks> handle(new (std::nothrow) TileXRDirectCcuPreparedTasks);
    if (handle == nullptr) {
        return TILEXR_ERROR_INTERNAL;
    }

    auto* tilexr = static_cast<TileXRComm*>(comm);
    TileXRCcuDirectInstallReport internalReport;
    TileXRCcuDirectInstallOptions internalOptions = MakeDirectCcuOptions(*options);
    const int ret = tilexr->PrepareDirectCcuInstallAttempt(internalOptions, &handle->attempt, &internalReport);
    FillPublicPrepareReport(internalReport, report);
    const bool installDiagnosticReady =
        internalReport.pipelineBuilt &&
        internalReport.installAttempted &&
        internalReport.installSucceeded &&
        !internalReport.submitReady;
    if (ret != TILEXR_SUCCESS && !installDiagnosticReady) {
        return ret;
    }

    *prepared = handle.release();
    return TILEXR_SUCCESS;
}

int TileXRCommPrepareDirectCcuMemoryCopy(
    TileXRCommPtr comm,
    const TileXRDirectCcuMemoryCopyPrepareOptions *options,
    TileXRDirectCcuPreparedTasksPtr *prepared,
    TileXRDirectCcuPrepareReport *report)
{
    if (prepared != nullptr) {
        *prepared = nullptr;
    }
    if (report != nullptr) {
        *report = TileXRDirectCcuPrepareReport {};
    }
    if (comm == nullptr || options == nullptr || prepared == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRCommPrepareDirectCcuMemoryCopy invalid input";
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    std::unique_ptr<TileXRDirectCcuPreparedTasks> handle(new (std::nothrow) TileXRDirectCcuPreparedTasks);
    if (handle == nullptr) {
        return TILEXR_ERROR_INTERNAL;
    }

    TileXRCcuDirectInstallOptions internalOptions = MakeDirectCcuOptions(options->prepare);
    internalOptions.sqeArgCount = 0;
    internalOptions.syncResourceCount = 1;
    internalOptions.syncInstructionCount = std::max<uint32_t>(
        internalOptions.syncInstructionCount,
        TILEXR_DIRECT_CCU_MEMORY_COPY_INSTRUCTION_COUNT);
    internalOptions.bindingsPerSyncResource =
        internalOptions.bindingsPerSyncResource == 0 ? 1 : internalOptions.bindingsPerSyncResource;
    const TileXRCcuMemoryCopyDirection direction = MemoryCopyDirectionFromPublic(options->direction);

    auto* tilexr = static_cast<TileXRComm*>(comm);
    TileXRCcuDirectInstallReport internalReport;
    const int ret = tilexr->PrepareDirectCcuMemoryCopyInstallAttempt(
        internalOptions,
        options->localSourceAddr,
        options->localDestinationAddr,
        options->bytes,
        options->peerRank,
        direction,
        &handle->attempt,
        &internalReport);
    FillPublicPrepareReport(internalReport, report);
    const bool installDiagnosticReady =
        internalReport.pipelineBuilt &&
        internalReport.installAttempted &&
        internalReport.installSucceeded &&
        !internalReport.submitReady;
    if (ret != TILEXR_SUCCESS && !installDiagnosticReady) {
        return ret;
    }

    *prepared = handle.release();
    return TILEXR_SUCCESS;
}

int TileXRDirectCcuGetPreparedTask(
    TileXRDirectCcuPreparedTasksPtr prepared,
    uint32_t index,
    TileXRDirectCcuTaskInfo *task)
{
    if (prepared == nullptr || task == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const auto* handle = PreparedHandle(prepared);
    if (index >= handle->attempt.submitTasks.size()) {
        return TILEXR_ERROR_NOT_FOUND;
    }

    const TileXRCcuTask& source = handle->attempt.submitTasks[index];
    *task = TileXRDirectCcuTaskInfo {};
    task->dieId = source.dieId;
    task->missionId = source.missionId;
    task->timeout = source.timeout;
    task->instStartId = source.instStartId;
    task->instCnt = source.instCnt;
    task->key = source.key;
    task->argSize = source.argSize;
    std::memcpy(task->args, source.args, sizeof(task->args));
    return TILEXR_SUCCESS;
}

int TileXRDirectCcuSubmitPrepared(
    TileXRDirectCcuPreparedTasksPtr prepared,
    void *stream,
    TileXRDirectCcuSubmitReport *report)
{
    if (report != nullptr) {
        *report = TileXRDirectCcuSubmitReport {};
    }
    if (prepared == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* handle = PreparedHandle(prepared);
    TileXRCcuDirectSubmitReport internalReport;
    const int ret = TileXRCcuSubmitPreparedTasks(
        handle->attempt.submitTasks,
        stream,
        nullptr,
        nullptr,
        &internalReport);
    FillPublicSubmitReport(internalReport, report);
    return ret;
}

int TileXRDirectCcuSubmitPreparedTask(
    TileXRDirectCcuPreparedTasksPtr prepared,
    uint32_t index,
    void *stream,
    TileXRDirectCcuSubmitReport *report)
{
    if (report != nullptr) {
        *report = TileXRDirectCcuSubmitReport {};
    }
    if (prepared == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* handle = PreparedHandle(prepared);
    if (index >= handle->attempt.submitTasks.size()) {
        if (report != nullptr) {
            report->taskCount = static_cast<uint32_t>(handle->attempt.submitTasks.size());
            CopyDirectCcuMessage("selected direct CCU submit task is missing", report->message);
        }
        return TILEXR_ERROR_NOT_FOUND;
    }

    std::vector<TileXRCcuTask> selectedTasks(1U);
    selectedTasks[0] = handle->attempt.submitTasks[index];
    TileXRCcuDirectSubmitReport internalReport;
    const int ret = TileXRCcuSubmitPreparedTasks(
        selectedTasks,
        stream,
        nullptr,
        nullptr,
        &internalReport);
    FillPublicSubmitReport(internalReport, report);
    return ret;
}

int TileXRCommReadDirectCcuInstructions(
    TileXRCommPtr comm,
    uint8_t dieId,
    uint16_t instructionStartId,
    uint32_t instructionCount,
    TileXRDirectCcuInstructionWords *instructions,
    TileXRDirectCcuInstructionReadbackReport *report)
{
    if (report != nullptr) {
        *report = TileXRDirectCcuInstructionReadbackReport {};
    }
    if (comm == nullptr || instructions == nullptr || instructionCount == 0 ||
        instructionCount > TILEXR_CCU_MAX_DATA_ARRAY_SIZE) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto* tilexr = static_cast<TileXRComm*>(comm);
    TileXRCcuDriverAdapterReport internalReport;
    const uint32_t instructionBytes =
        instructionCount * static_cast<uint32_t>(sizeof(TileXRDirectCcuInstructionWords));
    const int ret = tilexr->ReadDirectCcuInstructionsForDebug(
        dieId,
        instructionStartId,
        instructions,
        instructionCount,
        instructionBytes,
        &internalReport);
    FillPublicInstructionReadbackReport(instructionCount, internalReport, report);
    return ret;
}

#if defined(TILEXR_CCU_TESTING)
extern "C" TileXRDirectCcuPreparedTasksPtr TileXRDirectCcuCreatePreparedForTest(
    const TileXRDirectCcuTaskInfo* tasks,
    uint32_t taskCount)
{
    if (tasks == nullptr || taskCount == 0) {
        return nullptr;
    }
    std::unique_ptr<TileXRDirectCcuPreparedTasks> handle(new (std::nothrow) TileXRDirectCcuPreparedTasks);
    if (handle == nullptr) {
        return nullptr;
    }
    handle->attempt.submitTasks.reserve(taskCount);
    for (uint32_t i = 0; i < taskCount; ++i) {
        TileXRCcuTask task {};
        task.dieId = tasks[i].dieId;
        task.missionId = tasks[i].missionId;
        task.timeout = tasks[i].timeout;
        task.instStartId = tasks[i].instStartId;
        task.instCnt = tasks[i].instCnt;
        task.key = tasks[i].key;
        task.argSize = tasks[i].argSize;
        std::memcpy(task.args, tasks[i].args, sizeof(task.args));
        handle->attempt.submitTasks.push_back(task);
    }
    return handle.release();
}
#endif

int TileXRDirectCcuDestroyPrepared(TileXRDirectCcuPreparedTasksPtr prepared)
{
    if (prepared == nullptr) {
        return TILEXR_INVALID_VALUE;
    }
    delete PreparedHandle(prepared);
    return TILEXR_SUCCESS;
}

void TileXRPrintDFX2Log(TileXRCommPtr comm)
{
    if (comm == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comm is nullptr!";
        return;
    }
    auto* tilexr = static_cast<TileXRComm *>(comm);
    TILEXR_LOG(INFO) << tilexr->PrintDFX();
}

int TileXRCommInit(int rank, int rankSize, TileXRCommPtr *comms)
{
    if (comms == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comms is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    *comms = nullptr;
    *comms = new (std::nothrow) TileXRComm(rank, rankSize);
    if (*comms == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRCommInitAll(uint32_t ndev, int32_t *devices, TileXRCommPtr *comms)
{
    if (comms == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comms is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    if (devices == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr devices is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    for (uint32_t i = 0; i < ndev; ++i) {
        comms[i] = nullptr;
    }
    static int commDomain = 0;
    commDomain++;
    vector<unique_ptr<TileXRComm>> commHolders;
    commHolders.reserve(ndev);
    for (uint32_t i = 0; i < ndev; ++i) {
        unique_ptr<TileXRComm> commHolder(new (std::nothrow) TileXRComm(i, ndev, commDomain, TILEXR_COMM_BUFFER_SIZE));
        if (commHolder == nullptr) {
            TILEXR_LOG(ERROR) << "TileXRComm create failed. dev : " << i << ", ndev : " << ndev;
            return TILEXR_ERROR_INTERNAL;
        }
        commHolders.emplace_back(std::move(commHolder));
    }
    static atomic<int> uid;
    uid++;
    vector<unique_ptr<thread>> threads;
    atomic<int> error(TILEXR_SUCCESS);
    for (uint32_t r = 0; r < ndev; r++) {
        threads.emplace_back(make_unique<thread>(
            [&](int rank) {
                aclrtSetDevice(devices[rank]);
                auto* c = commHolders[rank].get();
                int ret = c->InitThread("uid" + to_string(uid));
                if (ret != TILEXR_SUCCESS) {
                    error.store(ret);
                }
            },
            r));
    }
    for (auto &t : threads) {
        t->join();
    }
    threads.clear();
    int ret = error.load();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXRCommInitAll init failed! ret: " << ret;
        return ret;
    }
    for (uint32_t i = 0; i < ndev; ++i) {
        comms[i] = commHolders[i].release();
    }
    return TILEXR_SUCCESS;
}

int TileXRCommInitThread(int rank, int rankSize, const char *uid, TileXRCommPtr *comms)
{
    if (uid == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr uid is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    if (comms == nullptr) {
        TILEXR_LOG(ERROR) << "tilexr comms is nullptr!";
        return TILEXR_ERROR_INTERNAL;
    }
    *comms = nullptr;
    if (rank >= rankSize) {
        TILEXR_LOG(ERROR) << "tilexr rank : " << rank << " rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    unique_ptr<TileXRComm> c(new (std::nothrow) TileXRComm(rank, rankSize));
    if (c == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRComm create failed. rank : " << rank << ", rankSize : " << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    int ret = c->InitThread(string(uid));
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "tilexr init thread failed! ret: " << ret;
        return ret;
    }
    *comms = c.release();
    return TILEXR_SUCCESS;
}

int TileXRCommDestroy(TileXRCommPtr comm)
{
    if (comm == nullptr) {
        return TILEXR_INVALID_VALUE;
    }
    auto *c = static_cast<TileXRComm *>(comm);

    delete c;
    return TILEXR_SUCCESS;
}
