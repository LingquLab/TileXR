/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEXR_COMM_H
#define TILEXR_COMM_H

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include "../include/tilexr_sdma_types.h"
#include "../include/tilexr_udma_reg.h"
#include "../include/tilexr_types.h"
#include "../include/tilexr_api.h"
#include "../include/comm_args.h"
#include "ccu/tilexr_ccu_direct_orchestrator.h"
#include "ccu/tilexr_ccu_direct_runtime.h"
#include "ccu/tilexr_ccu_lower_layer_plan_builder.h"

namespace TileXR {
constexpr int IPC_NAME_SIZE = 65;

class TileXRSockExchange;
class TileXRUDMATransport;
class TileXRSDMATransport;
class TileXRComm {
public:
    TileXRComm(int rank, int rankSize);
    TileXRComm(int rank, int rankSize, int commDomain, int bufferSize);
    TileXRComm(int rank, int rankSize, TileXRUniqueId commId);
    ~TileXRComm();
    TileXRComm(const TileXRComm &) = delete;
    TileXRComm &operator=(const TileXRComm &) = delete;
    int Init();
    int InitDirectCcuOnly();
    int InitThread(const std::string &uid = "default");
    int GetRank() const;
    int GetRankSize() const;
    int GetCommSize() const;
    const PhysicalInfo &GetPhysicalInfo() const;
    GM_ADDR GetCommArgsPtr();
    CommArgs* GetCommArgs();
    int64_t NextMagic();
    int RegisterUDMAMemory(GM_ADDR localPtr, size_t bytes, TileXRUDMAMemHandle *handle);
    int UnregisterUDMAMemory(TileXRUDMAMemHandle handle);
    GM_ADDR GetUDMARegistryPtr() const;
    const TileXRUDMARegistry* GetUDMARegistryHost() const;
    int RefreshDirectCcuBasicInfo(uint8_t dieId = 0);
    bool HasDirectCcuBasicInfo() const;
    int GetDirectCcuBasicInfoStatus() const;
    const TileXRCcuBasicInfo *GetDirectCcuBasicInfo() const;
    const TileXRCcuDriverAdapterReport &GetDirectCcuBasicInfoReport() const;
    int ConfigureDirectCcuLowerLayerTemplate(const TileXRCcuLowerLayerTransportSnapshot &templateSnapshot);
    int ConfigureDirectCcuVerifiedEndpointRoutes(
        const std::vector<TileXRCcuLowerLayerTransportRoute> &verifiedRoutes);
    int ConfigureDirectCcuLocalVerifiedEndpointRoute(const TileXRCcuLowerLayerTransportRoute &route);
    int ConfigureDirectCcuLowerLayerTemplateFromAllocation(
        const TileXRCcuResourceAllocation &allocation,
        const std::vector<TileXRCcuRemoteCcuBufferInfo> &remoteCcuBuffers);
    int PrepareDirectCcuLowerLayerTemplateFromAllocation(const TileXRCcuResourceAllocation &allocation);
    int RefreshDirectCcuLowerLayerPlan();
    bool HasDirectCcuLowerLayerPlan() const;
    int GetDirectCcuLowerLayerPlanStatus() const;
    const TileXRCcuLowerLayerPlanBuilderReport &GetDirectCcuLowerLayerPlanReport() const;
    const TileXRCcuLowerLayerInstallPlan *GetDirectCcuLowerLayerPlan() const;
    int ReadDirectCcuInstructionsForDebug(
        uint8_t dieId,
        uint16_t instructionStartId,
        void *instructions,
        uint32_t instructionCount,
        uint32_t instructionBytes,
        TileXRCcuDriverAdapterReport *report);
    int PrepareDirectCcuInstallAttempt(
        const TileXRCcuDirectInstallOptions &options,
        TileXRCcuDirectInstallAttempt *attempt,
        TileXRCcuDirectInstallReport *report);
    int PrepareDirectCcuMemoryCopyInstallAttempt(
        const TileXRCcuDirectInstallOptions &options,
        uint64_t localSourceAddr,
        uint64_t localDestinationAddr,
        uint64_t bytes,
        uint32_t peerRank,
        TileXRCcuMemoryCopyDirection direction,
        TileXRCcuDirectInstallAttempt *attempt,
        TileXRCcuDirectInstallReport *report);
    bool IsSDMAAvailable() const;
    GM_ADDR GetSDMAWorkspacePtr() const;
    SDMAInitStatus GetSDMAInitStatus() const;
    std::string PrintDFX();
    friend class Lccl;
    friend class Lcoc;
    friend class LcclTest;

private:
    int SetMemoryName(std::string &name);
    int SetIpcPidSdid(std::string &name, const uint32_t *pids, const int64_t *sdids) const;
    int OpenIpcMem(const char names[TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE]);
    int GetDev();
    int GetDevThread(const std::string &uid = "");
    int EnablePeerAccess();
    int InitCommMem();
    int InitCommon();
    void CloseIpcMem();
    void FreePeerMem(GM_ADDR &mem) const;
    int InitMem();
    int GetSidId(int64_t sdids[TILEXR_MAX_RANK_SIZE], int rankSize);
    int GetPid(uint32_t *pids);
    int GetName(std::string &name, char names[TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE]) const;
    int SyncCommArgs();
    int InitDumpAddr();
    int InitUDMA();
    int InitDirectCcuRuntime();
    int InitSDMA();
    int UpdateCommArgsDev();
    void FreeUDMARegistry();
    void ResetDirectCcuBasicInfo();
    void ResetDirectCcuLowerLayerPlan();
    int FillDirectCcuLowerLayerPlanFromAllocation(
        const TileXRCcuResourceAllocation &allocation,
        TileXRCcuLowerLayerInstallPlan *plan,
        TileXRCcuLowerLayerPlanBuilderReport *report);
    int ExchangeDirectCcuRemoteNotifyCke(
        const TileXRCcuResourceAllocation &allocation,
        std::vector<TileXRCcuRemoteCcuBufferInfo> *remoteCcuBuffers,
        TileXRCcuLowerLayerPlanBuilderReport *report);
    static int PrepareDirectCcuLowerLayerPlanCallback(
        const TileXRCcuResourceAllocation &allocation,
        TileXRCcuLowerLayerInstallPlan *plan,
        TileXRCcuLowerLayerPlanBuilderReport *report,
        void *userData);
    static int DirectCcuAllGatherCallback(
        const void *sendBuf,
        size_t sendBytes,
        void *recvBuf,
        void *userData);
    int DirectCcuThreadAllGather(const void *sendBuf, size_t sendBytes, void *recvBuf);
    void ResetSDMAState();

private:
    int rank_ = 0;  // global rank id
    int rankSize_ = 0;  // global rank size
    int commSize_ = 0;  // local TileXRComm size
    int localRank_ = -1;
    uint32_t localRankSize_ = 0;
    int devId_ = 0;
    std::atomic<int64_t> magic_ {1};
    bool inited_ = false;
    bool ipcMemInited_ = false;
    std::string uid_ = {};
    std::vector<int> devList_ = {};
    int commDomain_ = {};
    int bufferSize_ = TILEXR_COMM_BUFFER_SIZE;

    // shared ping pong buff，这个地址就是一开始申请在HBM上的，所以host上可以取到，但不能直接修改。
    GM_ADDR peerMem_[TILEXR_MAX_RANK_SIZE] = {};
    PhysicalInfo physicalInfo_ = {};
    CommArgs commArgs_ = {};    // host侧
    GM_ADDR commArgsPtr_ = nullptr; // device侧
    TileXRUniqueId commId_ = {};
    TileXRSockExchange *socketExchange_ = nullptr;
    bool isEnableMsprofOp_ = false;
    GM_ADDR udmaInfoDev_ = nullptr;
    GM_ADDR udmaRegistryDev_ = nullptr;
    GM_ADDR udmaRegisteredPtr_ = nullptr;
    TileXRUDMARegistry udmaRegistry_ = {};
    std::unique_ptr<TileXRUDMATransport> udmaTransport_;
    std::unique_ptr<TileXRCcuDirectRuntime> ccuDirectRuntime_;
    bool directCcuBasicInfoValid_ = false;
    int directCcuBasicInfoStatus_ = TILEXR_ERROR_NOT_FOUND;
    TileXRCcuBasicInfo directCcuBasicInfo_ = {};
    TileXRCcuDriverAdapterReport directCcuBasicInfoReport_ = {};
    bool directCcuLowerLayerTemplateConfigured_ = false;
    bool directCcuLowerLayerPlanValid_ = false;
    int directCcuLowerLayerPlanStatus_ = TILEXR_ERROR_NOT_FOUND;
    TileXRCcuLowerLayerTransportSnapshot directCcuLowerLayerTemplate_ = {};
    TileXRCcuLowerLayerTransportSnapshot directCcuLowerLayerSnapshot_ = {};
    TileXRCcuLowerLayerInstallPlan directCcuLowerLayerPlan_ = {};
    TileXRCcuLowerLayerPlanBuilderReport directCcuLowerLayerPlanReport_ = {};
    std::vector<TileXRCcuLowerLayerTransportRoute> directCcuVerifiedEndpointRoutes_ = {};
    TileXRCcuLowerLayerTransportRoute directCcuLocalVerifiedEndpointRoute_ = {};
    bool directCcuLocalVerifiedEndpointRouteValid_ = false;
    uint64_t directCcuThreadAllGatherRound_ = 0;
    GM_ADDR sdmaWorkspaceDev_ = nullptr;
    SDMAInitStatus sdmaInitStatus_ = SDMAInitStatus::DISABLED_BY_ENV;
    std::unique_ptr<TileXRSDMATransport> sdmaTransport_;
};
} // TileXR

#endif // TILEXR_COMM_H
