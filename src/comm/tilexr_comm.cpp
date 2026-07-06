/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tilexr_comm.h"
#include "tilexr_internal.h"
#include "ccu/tilexr_ccu_memory_program.h"
#include "ccu/tilexr_ccu_repository.h"
#include "sdma/tilexr_sdma_transport.h"
#include "udma/tilexr_udma_transport.h"

#include <acl/acl_rt.h>
#include <chrono>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <map>
#include <set>
#include <thread>
#include <sstream>
#include <iomanip>
#include <cstring>

#include "tilexr_log.h"
#include "tools/socket/tilexr_sock_exchange.h"

#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rts/rts_device.h"
#include "runtime/rt_ffts.h"

enum TopologyType : int {
    TOPOLOGY_HCCS = 0,
    TOPOLOGY_PIX,
    TOPOLOGY_PIB,
    TOPOLOGY_PHB,
    TOPOLOGY_SYS,
    TOPOLOGY_SIO,
    TOPOLOGY_HCCS_SW
};

using namespace std;
using namespace chrono;

namespace TileXR {
constexpr int HCCL_IPC_PID_ARRAY_SIZE = 1; // 固定每次只传一个PID数据
constexpr int TILEXR_INIT_TIMEOUT = 600;
constexpr uint32_t TILEXR_CCU_DIRECT_MEMORY_COPY_INSTRUCTION_COUNT = 7U;
static map<string, GM_ADDR [TILEXR_MAX_RANK_SIZE]> g_localPeerMemMap;
static map<string, int[TILEXR_MAX_RANK_SIZE]> g_devList;
struct TileXRThreadAllGatherState {
    std::vector<uint8_t> data[TILEXR_MAX_RANK_SIZE];
    uint64_t arrivals = 0;
    uint64_t departures = 0;
    size_t bytes = 0;
};
static map<string, TileXRThreadAllGatherState> g_directCcuAllGatherStates;
static std::mutex g_mtx;
static std::mutex g_udmaMtx;
static bool g_udmaUnavailable = false;
static std::mutex g_ccuDirectRuntimeMtx;
static bool g_ccuDirectRuntimeUnavailable = false;
static std::string g_ccuDirectRuntimeUnavailableMessage;
static std::mutex g_sdmaMtx;
static bool g_sdmaUnavailable = false;

uint8_t SelectDirectCcuInstallDieId()
{
    const char *text = std::getenv("TILEXR_CCU_DIRECT_INSTALL_DIE_ID");
    if (text == nullptr || text[0] == '\0') {
        return 0;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 0);
    if (end == text || *end != '\0' || parsed > 1UL) {
        return 0;
    }
    return static_cast<uint8_t>(parsed);
}

uint32_t SelectDirectCcuPeerLocalXnOffset(size_t peerLocalIndex, uint32_t syncIndex, size_t peerRouteCount)
{
    if (peerRouteCount == 0) {
        return 0;
    }
    return static_cast<uint32_t>(peerLocalIndex) +
        static_cast<uint32_t>(syncIndex / peerRouteCount) * static_cast<uint32_t>(peerRouteCount);
}

uint32_t SelectDirectCcuChannelBoundRemoteXnOffset(size_t peerLocalIndex, uint32_t syncIndex, size_t peerRouteCount)
{
    return SelectDirectCcuPeerLocalXnOffset(peerLocalIndex, syncIndex, peerRouteCount);
}

uint16_t DirectCcuRemoteXnProofSpan(uint16_t syncRouteCount)
{
    if (syncRouteCount == 0) {
        return 0;
    }
    return syncRouteCount;
}

uint16_t SelectDirectCcuChannelBoundRemoteXnId(
    uint16_t remoteXnStartId,
    size_t peerLocalIndex,
    uint32_t syncIndex,
    size_t peerRouteCount)
{
    return static_cast<uint16_t>(
        static_cast<uint32_t>(remoteXnStartId) +
        SelectDirectCcuChannelBoundRemoteXnOffset(peerLocalIndex, syncIndex, peerRouteCount));
}

uint16_t SelectDirectCcuRemoteNotifyCkeId(uint16_t remoteNotifyCkeStartId, size_t routeIndex)
{
    return static_cast<uint16_t>(static_cast<uint32_t>(remoteNotifyCkeStartId) + routeIndex);
}

std::string ProcessDirectCcuRuntimeUnavailableMessage()
{
    lock_guard<mutex> lock(g_ccuDirectRuntimeMtx);
    if (!g_ccuDirectRuntimeUnavailable) {
        return {};
    }
    return g_ccuDirectRuntimeUnavailableMessage.empty() ?
        "direct CCU runtime unavailable after process-level init failure" :
        "direct CCU runtime unavailable after process-level init failure: " +
            g_ccuDirectRuntimeUnavailableMessage;
}

struct DirectCcuMemoryCopyEndpoint {
    uint64_t sourceAddr = 0;
    uint64_t sourceToken = 0;
    uint64_t destinationAddr = 0;
    uint64_t destinationToken = 0;
    uint64_t bytes = 0;
    uint32_t rank = 0;
    uint32_t valid = 0;
};

int QueryDirectCcuProcessMemoryToken(uint64_t addr, uint64_t bytes, uint64_t *packedToken)
{
    if (addr == 0 || bytes == 0 || packedToken == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *packedToken = 0;
    rtMemUbTokenInfo info {};
    info.va = addr;
    info.size = bytes;
    const rtError_t ret = rtUbDevQueryInfo(QUERY_PROCESS_TOKEN, &info);
    if (ret != RT_ERROR_NONE) {
        return TILEXR_ERROR_MKIRT;
    }
    constexpr uint32_t tokenIdRightShift = 8U;
    const uint32_t tokenId = info.tokenId >> tokenIdRightShift;
    *packedToken = TileXRCcuPackMemoryToken(tokenId, info.tokenValue, true);
    return *packedToken == 0 ? TILEXR_ERROR_NOT_FOUND : TILEXR_SUCCESS;
}

int BuildDirectCcuLocalMemoryCopyEndpoint(
    uint32_t rank,
    uint64_t sourceAddr,
    uint64_t destinationAddr,
    uint64_t bytes,
    DirectCcuMemoryCopyEndpoint *endpoint)
{
    if (endpoint == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *endpoint = DirectCcuMemoryCopyEndpoint {};
    endpoint->rank = rank;
    endpoint->bytes = bytes;
    endpoint->sourceAddr = sourceAddr;
    endpoint->destinationAddr = destinationAddr;
    int ret = QueryDirectCcuProcessMemoryToken(sourceAddr, bytes, &endpoint->sourceToken);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    ret = QueryDirectCcuProcessMemoryToken(destinationAddr, bytes, &endpoint->destinationToken);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    endpoint->valid = 1;
    return TILEXR_SUCCESS;
}


// 如果是互联的链路，返回false； 对910B2C那些不互联的链路，返回true
bool SkipUnusedChannel910B2C(int curRank, int peerRank, ChipName chipName)
{
    if (chipName == ChipName::CHIP_910B2C) {
        constexpr int rankSizePerNode = 8;
        // 双节点16P中不用的链路: 不在同一个节点 且rank在节点内序号不同； 在调用时将跳过
        if ((curRank / rankSizePerNode != peerRank / rankSizePerNode)
            && (std::abs(curRank - peerRank) != rankSizePerNode)) {
            return true;
        }
    }
    return false;
}

int TileXRComm::InitDumpAddr()
{
    constexpr uint32_t dumpCoreCnt = 75;
    constexpr uint32_t dumpSizePerCore = 1 * 1024 * 1024;
    constexpr uint32_t dumpWorkspaceSize = dumpCoreCnt * dumpSizePerCore;
    GM_ADDR dumpAddr = nullptr;
    int ret = 0;
    ret = aclrtMalloc(reinterpret_cast<void **>(&dumpAddr), dumpWorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMalloc err " << __LINE__ << " " << ret;
        return TILEXR_ERROR_INTERNAL;
    }
    ret = aclrtMemset(dumpAddr, dumpWorkspaceSize, 0, dumpWorkspaceSize);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMemset err " << __LINE__ << " " << ret;
        aclrtFree(dumpAddr);
        return TILEXR_ERROR_INTERNAL;
    }
 
    GM_ADDR memory = static_cast<GM_ADDR>(std::malloc(dumpWorkspaceSize));
    if (!memory) {
        TILEXR_LOG(ERROR) << "std::malloc err " << __LINE__;
        aclrtFree(dumpAddr);
        return TILEXR_ERROR_INTERNAL;
    }
    // Zero out allocated memory
    for (size_t i = 0; i < dumpWorkspaceSize; ++i) {
        ((uint8_t*)memory)[i] = 0;
    }
    // 遍历每个block进行初始化
        for (uint32_t i = 0; i < dumpCoreCnt; ++i) {
        // 计算当前block的起始地址
        GM_ADDR blockStart = memory + i * dumpSizePerCore;
        GM_ADDR deviceBlockStart = dumpAddr + i * dumpSizePerCore;
        
        // 初始化BlockInfo
        LcclDumpBlockInfo* blockInfo = reinterpret_cast<LcclDumpBlockInfo*>(blockStart);
        blockInfo->len = dumpSizePerCore;
        blockInfo->core = i;
        blockInfo->blockNum = 0;
        blockInfo->dumpOffset = dumpSizePerCore - sizeof(LcclDumpBlockInfo);
        blockInfo->magic = 0; // 示例魔法值
        blockInfo->dumpAddr = reinterpret_cast<uint64_t>(deviceBlockStart + sizeof(LcclDumpBlockInfo));
    }
 
    ret = aclrtMemcpy(dumpAddr, dumpWorkspaceSize, memory, dumpWorkspaceSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMemcpy err " << __LINE__ << " " << ret;
        std::free(memory);
        aclrtFree(dumpAddr);
        return TILEXR_ERROR_INTERNAL;
    }
    std::free(memory);
 
    commArgs_.dumpAddr = dumpAddr;
    return TILEXR_SUCCESS;
}

int TileXRComm::InitUDMA()
{
    if (rankSize_ <= 1) {
        TILEXR_LOG(INFO) << "InitUDMA skipped for single-rank communicator";
        return TILEXR_SUCCESS;
    }

    {
        lock_guard<mutex> lock(g_udmaMtx);
        if (g_udmaUnavailable) {
            TILEXR_LOG(INFO) << "InitUDMA skipped after previous UDMA init failure";
            return TILEXR_SUCCESS;
        }
    }

    udmaTransport_.reset(new (nothrow) TileXRUDMATransport());
    if (udmaTransport_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXRUDMATransport allocation failed, UDMA disabled";
        return TILEXR_SUCCESS;
    }
    TileXRUDMATransportOptions options {};
    options.rank = rank_;
    options.rankSize = rankSize_;
    options.devId = devId_;
    options.exchange = socketExchange_;
    int ret = udmaTransport_->Init(options);
    if (ret != TILEXR_SUCCESS || !udmaTransport_->IsAvailable()) {
        TILEXR_LOG(WARN) << "TileXR UDMA init failed: " << ret << ", UDMA disabled";
        lock_guard<mutex> lock(g_udmaMtx);
        g_udmaUnavailable = true;
        udmaTransport_.reset();
        return TILEXR_SUCCESS;
    }

    udmaInfoDev_ = udmaTransport_->GetUDMAInfoDev();
    commArgs_.udmaInfoPtr = udmaInfoDev_;
    commArgs_.extraFlag |= ExtraFlag::UDMA;

    TILEXR_LOG(INFO) << "InitUDMA success, rank " << rank_ << "/" << rankSize_;
    return TILEXR_SUCCESS;
}

int TileXRComm::InitDirectCcuRuntime()
{
    if (rankSize_ <= 1) {
        TILEXR_LOG(INFO) << "direct CCU runtime skipped for single-rank communicator";
        return TILEXR_SUCCESS;
    }

    lock_guard<mutex> lock(g_ccuDirectRuntimeMtx);
    if (g_ccuDirectRuntimeUnavailable) {
        TILEXR_LOG(INFO) << "direct CCU runtime skipped after previous init failure";
        return TILEXR_SUCCESS;
    }

    ccuDirectRuntime_.reset(new (nothrow) TileXRCcuDirectRuntime());
    if (ccuDirectRuntime_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXRCcuDirectRuntime allocation failed, direct CCU disabled";
        return TILEXR_SUCCESS;
    }

    TileXRCcuDirectRuntimeOptions options {};
    options.rank = rank_;
    options.rankSize = rankSize_;
    options.devId = devId_;
    options.allGather = &TileXRComm::DirectCcuAllGatherCallback;
    options.allGatherUserData = this;
    TileXRCcuDirectRuntimeReport runtimeReport;
    const int ret = ccuDirectRuntime_->Init(options, &runtimeReport);
    if (ret != TILEXR_SUCCESS || !ccuDirectRuntime_->IsAvailable()) {
        TILEXR_LOG(WARN) << "TileXR direct CCU runtime init failed: " << ret
                         << ", logicDevId " << runtimeReport.logicDevId
                         << ", devicePhyId " << runtimeReport.devicePhyId
                         << ", hdcType " << runtimeReport.hdcType
                         << ", raInitialized " << (runtimeReport.raInitialized ? 1 : 0)
                         << ", ccuTlvInitialized " << (runtimeReport.ccuTlvInitialized ? 1 : 0)
                         << ", " << runtimeReport.message << ", direct CCU disabled";
        g_ccuDirectRuntimeUnavailable = true;
        g_ccuDirectRuntimeUnavailableMessage = runtimeReport.message;
        ResetDirectCcuBasicInfo();
        ccuDirectRuntime_.reset();
        return TILEXR_SUCCESS;
    }

    const int ccuInfoRet = RefreshDirectCcuBasicInfo(0);
    if (ccuInfoRet != TILEXR_SUCCESS && ccuInfoRet != TILEXR_ERROR_NOT_FOUND) {
        TILEXR_LOG(WARN) << "direct CCU basic info refresh failed after runtime init: " << ccuInfoRet
                         << ", " << directCcuBasicInfoReport_.message;
    }

        TILEXR_LOG(INFO) << "InitDirectCcuRuntime success, rank " << rank_ << "/" << rankSize_
                     << " logicDevId " << runtimeReport.logicDevId
                     << " devicePhyId " << runtimeReport.devicePhyId
                     << " hdcType " << runtimeReport.hdcType
                     << " raInitialized " << (runtimeReport.raInitialized ? 1 : 0)
                     << " ccuTlvInitialized " << (runtimeReport.ccuTlvInitialized ? 1 : 0);
    return TILEXR_SUCCESS;
}

int TileXRComm::InitSDMA()
{
    {
        lock_guard<mutex> lock(g_sdmaMtx);
        if (g_sdmaUnavailable) {
            TILEXR_LOG(INFO) << "InitSDMA skipped after previous SDMA init failure";
            sdmaInitStatus_ = SDMAInitStatus::PTO_UNAVAILABLE;
            return TILEXR_SUCCESS;
        }
    }

    sdmaTransport_.reset(new (nothrow) TileXRSDMATransport());
    if (sdmaTransport_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXRSDMATransport allocation failed, SDMA disabled";
        sdmaInitStatus_ = SDMAInitStatus::INIT_FAILED;
        return TILEXR_SUCCESS;
    }

    TileXRSDMATransportOptions options {};
    options.devId = devId_;
    int ret = sdmaTransport_->Init(options);
    sdmaInitStatus_ = sdmaTransport_->GetLastStatus();
    if (ret != TILEXR_SUCCESS || !sdmaTransport_->IsAvailable()) {
        if (sdmaInitStatus_ != SDMAInitStatus::DISABLED_BY_ENV) {
            TILEXR_LOG(WARN) << "TileXR SDMA init unavailable, status " << static_cast<int>(sdmaInitStatus_);
            lock_guard<mutex> lock(g_sdmaMtx);
            g_sdmaUnavailable = true;
        }
        sdmaTransport_.reset();
        sdmaWorkspaceDev_ = nullptr;
        commArgs_.sdmaWorkspacePtr = nullptr;
        return TILEXR_SUCCESS;
    }

    sdmaWorkspaceDev_ = sdmaTransport_->GetWorkspaceDev();
    if (sdmaWorkspaceDev_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXR SDMA workspace is null, SDMA disabled";
        sdmaInitStatus_ = SDMAInitStatus::NULL_WORKSPACE;
        commArgs_.extraFlag &= ~ExtraFlag::SDMA;
        commArgs_.sdmaWorkspacePtr = nullptr;
        sdmaWorkspaceDev_ = nullptr;
        sdmaTransport_.reset();
        return TILEXR_SUCCESS;
    }

    commArgs_.sdmaWorkspacePtr = sdmaWorkspaceDev_;
    commArgs_.extraFlag |= ExtraFlag::SDMA;
    sdmaInitStatus_ = SDMAInitStatus::INITIALIZED;
    TILEXR_LOG(INFO) << "InitSDMA success, workspace " << static_cast<void*>(sdmaWorkspaceDev_);
    return TILEXR_SUCCESS;
}

void TileXRComm::ResetSDMAState()
{
    commArgs_.extraFlag &= ~ExtraFlag::SDMA;
    commArgs_.sdmaWorkspacePtr = nullptr;
    sdmaWorkspaceDev_ = nullptr;
    sdmaInitStatus_ = SDMAInitStatus::DISABLED_BY_ENV;
    if (sdmaTransport_ != nullptr) {
        sdmaTransport_->Shutdown();
        sdmaTransport_.reset();
    }
}

bool TileXRComm::IsSDMAAvailable() const
{
    return (commArgs_.extraFlag & ExtraFlag::SDMA) != 0 && commArgs_.sdmaWorkspacePtr != nullptr;
}

GM_ADDR TileXRComm::GetSDMAWorkspacePtr() const
{
    return sdmaWorkspaceDev_;
}

SDMAInitStatus TileXRComm::GetSDMAInitStatus() const
{
    return sdmaInitStatus_;
}

int TileXRComm::SyncCommArgs()
{
    commArgs_.rank = rank_;
    commArgs_.localRank = localRank_;
    commArgs_.rankSize = rankSize_;
    commArgs_.localRankSize = localRankSize_;
    for (int i = 0; i < rankSize_; ++i) {
        commArgs_.peerMems[i] = peerMem_[i];    // 这里不会越界，之前有逻辑校验过越界了
    }

    if (isEnableMsprofOp_) {
        if (InitDumpAddr() != TILEXR_SUCCESS) {
            return TILEXR_ERROR_INTERNAL;
        }

        uint64_t fftsVal = 0;
        uint32_t fftsLen = 0;
        int error = rtGetC2cCtrlAddr(&fftsVal, &fftsLen);
        if (error != RT_ERROR_NONE) {
            TILEXR_LOG(ERROR) << "rtGetC2cCtrlAddr err:" << error;
            return TILEXR_ERROR_MKIRT;
        }
        commArgs_.fftsVal = fftsVal;
    }

    int ret = 0;
    ret = aclrtMalloc(reinterpret_cast<void **>(&commArgsPtr_), sizeof(commArgs_), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMalloc err " << __LINE__ << " " << ret;
        return TILEXR_ERROR_INTERNAL;
    }
    ret = aclrtMemcpy(commArgsPtr_, sizeof(commArgs_), &commArgs_, sizeof(commArgs_), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMemcpy err " << __LINE__ << " " << ret;
        FreePeerMem(commArgsPtr_);
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRComm::UpdateCommArgsDev()
{
    if (commArgsPtr_ == nullptr) {
        return TILEXR_SUCCESS;
    }
    int ret = aclrtMemcpy(commArgsPtr_, sizeof(commArgs_), &commArgs_, sizeof(commArgs_), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMemcpy update comm args err " << ret;
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

void TileXRComm::FreeUDMARegistry()
{
    if (udmaRegistryDev_ != nullptr) {
        aclError ret = aclrtFree(udmaRegistryDev_);
        if (ret != ACL_SUCCESS) {
            TILEXR_LOG(WARN) << "Free UDMA registry failed: " << ret;
        }
        udmaRegistryDev_ = nullptr;
    }
    commArgs_.udmaRegistryPtr = nullptr;
    udmaRegistry_ = TileXRUDMARegistry {};
}

void TileXRComm::ResetDirectCcuBasicInfo()
{
    directCcuBasicInfoValid_ = false;
    directCcuBasicInfoStatus_ = TILEXR_ERROR_NOT_FOUND;
    directCcuBasicInfo_ = TileXRCcuBasicInfo {};
    directCcuBasicInfoReport_ = TileXRCcuDriverAdapterReport {};
}

void TileXRComm::ResetDirectCcuLowerLayerPlan()
{
    directCcuLowerLayerPlanValid_ = false;
    directCcuLowerLayerPlanStatus_ = TILEXR_ERROR_NOT_FOUND;
    directCcuLowerLayerSnapshot_ = TileXRCcuLowerLayerTransportSnapshot {};
    directCcuLowerLayerPlan_ = TileXRCcuLowerLayerInstallPlan {};
    directCcuLowerLayerPlanReport_ = TileXRCcuLowerLayerPlanBuilderReport {};
}

int TileXRComm::RegisterUDMAMemory(GM_ADDR localPtr, size_t bytes, TileXRUDMAMemHandle *handle)
{
    if (!inited_) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister requires initialized communicator";
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    if (localPtr == nullptr || bytes == 0 || handle == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!((commArgs_.extraFlag & ExtraFlag::UDMA) != 0 && commArgs_.udmaInfoPtr != nullptr)) {
        TILEXR_LOG(WARN) << "TileXRUDMARegister called while UDMA is unavailable";
        return TILEXR_ERROR_NOT_FOUND;
    }
    if (!uid_.empty()) {
        TILEXR_LOG(WARN) << "TileXRUDMARegister is not supported in InitThread mode";
        return TILEXR_ERROR_INTERNAL;
    }

    TileXRUDMARegionDesc localRegion {};
    localRegion.base = localPtr;
    localRegion.bytes = bytes;

    if (udmaTransport_ == nullptr || !udmaTransport_->IsAvailable()) {
        TILEXR_LOG(ERROR) << "TileXR UDMA transport is unavailable";
        return TILEXR_ERROR_NOT_FOUND;
    }
    int ret = udmaTransport_->RegisterMemory(localPtr, bytes);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR UDMA memory registration failed: " << ret;
        return TILEXR_ERROR_INTERNAL;
    }

    if (socketExchange_ == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister requires live socket exchange";
        udmaTransport_->UnregisterMemory(localPtr);
        return TILEXR_ERROR_INTERNAL;
    }
    std::vector<TileXRUDMARegionDesc> allRegions(rankSize_);
    ret = socketExchange_->AllGather(&localRegion, 1, allRegions.data());
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister allgather failed: " << ret;
        udmaTransport_->UnregisterMemory(localPtr);
        return ret;
    }

    TileXRUDMARegistry nextRegistry {};
    nextRegistry.rankSize = static_cast<uint32_t>(rankSize_);
    nextRegistry.regionCount = 1;
    for (int i = 0; i < rankSize_; ++i) {
        if (allRegions[i].base == nullptr || allRegions[i].bytes == 0) {
            TILEXR_LOG(ERROR) << "TileXRUDMARegister received invalid region from rank " << i;
            udmaTransport_->UnregisterMemory(localPtr);
            return TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        nextRegistry.regions[i] = allRegions[i];
    }

    GM_ADDR nextRegistryDev = nullptr;
    ret = aclrtMalloc(reinterpret_cast<void **>(&nextRegistryDev), sizeof(nextRegistry), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMalloc UDMA registry failed: " << ret;
        udmaTransport_->UnregisterMemory(localPtr);
        return TILEXR_ERROR_INTERNAL;
    }
    ret = aclrtMemcpy(nextRegistryDev, sizeof(nextRegistry), &nextRegistry, sizeof(nextRegistry), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMemcpy UDMA registry failed: " << ret;
        aclrtFree(nextRegistryDev);
        udmaTransport_->UnregisterMemory(localPtr);
        return TILEXR_ERROR_INTERNAL;
    }

    if (udmaRegisteredPtr_ != nullptr) {
        udmaTransport_->UnregisterMemory(udmaRegisteredPtr_);
        udmaRegisteredPtr_ = nullptr;
    }
    FreeUDMARegistry();
    udmaRegistry_ = nextRegistry;
    udmaRegistryDev_ = nextRegistryDev;
    udmaRegisteredPtr_ = localPtr;
    commArgs_.udmaRegistryPtr = udmaRegistryDev_;
    *handle = 0;
    ret = UpdateCommArgsDev();
    if (ret != TILEXR_SUCCESS) {
        udmaTransport_->UnregisterMemory(localPtr);
        udmaRegisteredPtr_ = nullptr;
        FreeUDMARegistry();
        return ret;
    }

    return TILEXR_SUCCESS;
}

int TileXRComm::UnregisterUDMAMemory(TileXRUDMAMemHandle handle)
{
    if (handle != 0) {
        return TILEXR_ERROR_NOT_FOUND;
    }
    if (udmaRegisteredPtr_ != nullptr && udmaTransport_ != nullptr) {
        int ret = udmaTransport_->UnregisterMemory(udmaRegisteredPtr_);
        if (ret != TILEXR_SUCCESS) {
            TILEXR_LOG(WARN) << "TileXR UDMA memory unregistration failed: " << ret;
        }
        udmaRegisteredPtr_ = nullptr;
    }
    FreeUDMARegistry();
    return UpdateCommArgsDev();
}

GM_ADDR TileXRComm::GetUDMARegistryPtr() const
{
    return udmaRegistryDev_;
}

const TileXRUDMARegistry* TileXRComm::GetUDMARegistryHost() const
{
    return UDMARegistryValid(&udmaRegistry_, rankSize_) ? &udmaRegistry_ : nullptr;
}

int TileXRComm::RefreshDirectCcuBasicInfo(uint8_t dieId)
{
    ResetDirectCcuBasicInfo();
    if (ccuDirectRuntime_ == nullptr || !ccuDirectRuntime_->IsAvailable()) {
        directCcuBasicInfoReport_.message = "direct CCU runtime is unavailable for basic info";
        directCcuBasicInfoStatus_ = TILEXR_ERROR_NOT_FOUND;
        return directCcuBasicInfoStatus_;
    }

    TileXRCcuBasicInfo basicInfo;
    TileXRCcuDriverAdapterReport report;
    const int ret = ccuDirectRuntime_->QueryBasicInfo(dieId, &basicInfo, &report);
    directCcuBasicInfoReport_ = report;
    if (ret != TILEXR_SUCCESS) {
        directCcuBasicInfoStatus_ = ret;
        return directCcuBasicInfoStatus_;
    }

    directCcuBasicInfo_ = basicInfo;
    directCcuBasicInfoReport_.message = "direct CCU basic info cached";
    directCcuBasicInfoValid_ = true;
    directCcuBasicInfoStatus_ = TILEXR_SUCCESS;
    return TILEXR_SUCCESS;
}

bool TileXRComm::HasDirectCcuBasicInfo() const
{
    return directCcuBasicInfoValid_;
}

int TileXRComm::GetDirectCcuBasicInfoStatus() const
{
    return directCcuBasicInfoStatus_;
}

const TileXRCcuBasicInfo *TileXRComm::GetDirectCcuBasicInfo() const
{
    return directCcuBasicInfoValid_ ? &directCcuBasicInfo_ : nullptr;
}

const TileXRCcuDriverAdapterReport &TileXRComm::GetDirectCcuBasicInfoReport() const
{
    return directCcuBasicInfoReport_;
}

int TileXRComm::ConfigureDirectCcuLowerLayerTemplate(
    const TileXRCcuLowerLayerTransportSnapshot &templateSnapshot)
{
    directCcuLowerLayerTemplate_ = templateSnapshot;
    directCcuLowerLayerTemplateConfigured_ = true;
    return RefreshDirectCcuLowerLayerPlan();
}

int TileXRComm::ConfigureDirectCcuVerifiedEndpointRoutes(
    const std::vector<TileXRCcuLowerLayerTransportRoute> &verifiedRoutes)
{
    TileXRCcuLowerLayerTransportSnapshot validationSnapshot;
    validationSnapshot.routes = verifiedRoutes;
    TileXRCcuLowerLayerPlanBuilderReport report;
    int ret = TileXRCcuOverlayVerifiedEndpointRoutes(verifiedRoutes, &validationSnapshot, &report);
    directCcuLowerLayerPlanReport_ = report;
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanStatus_ = ret;
        return ret;
    }

    directCcuVerifiedEndpointRoutes_ = verifiedRoutes;
    if (directCcuLowerLayerTemplateConfigured_) {
        return RefreshDirectCcuLowerLayerPlan();
    }
    return TILEXR_SUCCESS;
}

int TileXRComm::ConfigureDirectCcuLocalVerifiedEndpointRoute(
    const TileXRCcuLowerLayerTransportRoute &route)
{
    TileXRCcuLowerLayerTransportSnapshot validationSnapshot;
    validationSnapshot.routes.push_back(route);
    TileXRCcuLowerLayerPlanBuilderReport report;
    std::vector<TileXRCcuLowerLayerTransportRoute> routes {route};
    int ret = TileXRCcuOverlayVerifiedEndpointRoutes(routes, &validationSnapshot, &report);
    directCcuLowerLayerPlanReport_ = report;
    if (ret != TILEXR_SUCCESS) {
        directCcuLocalVerifiedEndpointRoute_ = TileXRCcuLowerLayerTransportRoute {};
        directCcuLocalVerifiedEndpointRouteValid_ = false;
        directCcuLowerLayerPlanStatus_ = ret;
        return ret;
    }

    directCcuLocalVerifiedEndpointRoute_ = route;
    directCcuLocalVerifiedEndpointRouteValid_ = true;
    if (ccuDirectRuntime_ != nullptr && ccuDirectRuntime_->IsAvailable()) {
        return ccuDirectRuntime_->ConfigureLocalVerifiedEndpointRoute(route);
    }
    return TILEXR_SUCCESS;
}

int TileXRComm::ConfigureDirectCcuLowerLayerTemplateFromAllocation(
    const TileXRCcuResourceAllocation &allocation,
    const std::vector<TileXRCcuRemoteCcuBufferInfo> &remoteCcuBuffers)
{
    ResetDirectCcuLowerLayerPlan();
    if (!directCcuBasicInfoValid_) {
        directCcuLowerLayerPlanReport_.message =
            "direct CCU basic info is unavailable for lower-layer transport template";
        directCcuLowerLayerPlanStatus_ = TILEXR_ERROR_NOT_FOUND;
        return directCcuLowerLayerPlanStatus_;
    }

    TileXRCcuLowerLayerTransportSnapshot templateSnapshot;
    TileXRCcuLowerLayerPlanBuilderReport report;
    int ret = TileXRCcuBuildLowerLayerTransportTemplate(
        directCcuBasicInfo_,
        allocation,
        remoteCcuBuffers,
        &templateSnapshot,
        &report);
    directCcuLowerLayerPlanReport_ = report;
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }
    ret = TileXRCcuOverlayVerifiedEndpointRoutes(
        directCcuVerifiedEndpointRoutes_,
        &templateSnapshot,
        &directCcuLowerLayerPlanReport_);
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }

    directCcuLowerLayerTemplate_ = templateSnapshot;
    directCcuLowerLayerTemplateConfigured_ = true;
    return RefreshDirectCcuLowerLayerPlan();
}

int TileXRComm::PrepareDirectCcuLowerLayerTemplateFromAllocation(
    const TileXRCcuResourceAllocation &allocation)
{
    ResetDirectCcuLowerLayerPlan();
    if (!directCcuBasicInfoValid_) {
        directCcuLowerLayerPlanReport_.message =
            "direct CCU basic info is unavailable for lower-layer transport template";
        directCcuLowerLayerPlanStatus_ = TILEXR_ERROR_NOT_FOUND;
        return directCcuLowerLayerPlanStatus_;
    }
    if (ccuDirectRuntime_ == nullptr || !ccuDirectRuntime_->IsAvailable()) {
        directCcuLowerLayerPlanReport_.message =
            "direct CCU runtime is unavailable for resource window registration";
        directCcuLowerLayerPlanStatus_ = TILEXR_ERROR_NOT_FOUND;
        return directCcuLowerLayerPlanStatus_;
    }

    int ret = ccuDirectRuntime_->RegisterCcuResourceRmaBuffer(directCcuBasicInfo_.resourceAddr);
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanReport_.message = "failed to register direct CCU resource window";
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }

    TileXRCcuLocalResourceWindowInfo localCcuResourceWindow;
    ret = ccuDirectRuntime_->ExportLocalCcuRmaBuffer(&localCcuResourceWindow);
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanReport_.message = "failed to export direct CCU local resource window token";
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }
    if (directCcuLocalVerifiedEndpointRouteValid_) {
        ret = ccuDirectRuntime_->ConfigureLocalVerifiedEndpointRoute(directCcuLocalVerifiedEndpointRoute_);
        if (ret != TILEXR_SUCCESS) {
            directCcuLowerLayerPlanReport_.message = "failed to configure direct CCU local verified endpoint route";
            directCcuLowerLayerPlanStatus_ = ret;
            return directCcuLowerLayerPlanStatus_;
        }
    } else {
        TileXRCcuDirectRuntimeReport endpointRouteReport;
        ret = ccuDirectRuntime_->RefreshLocalVerifiedEndpointRoute(&endpointRouteReport);
        if (ret != TILEXR_SUCCESS && ret != TILEXR_ERROR_NOT_FOUND) {
            TILEXR_LOG(WARN) << "direct CCU local endpoint route collection failed closed: "
                             << ret << ", " << endpointRouteReport.message;
        }
    }

    std::vector<TileXRCcuRemoteCcuBufferInfo> remoteCcuBuffers;
    ret = ccuDirectRuntime_->ExportRemoteCcuRmaBuffers(&remoteCcuBuffers);
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanReport_.message = "failed to export direct CCU peer resource window tokens";
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }
    ret = ExchangeDirectCcuRemoteNotifyCke(allocation, &remoteCcuBuffers, &directCcuLowerLayerPlanReport_);
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }

    TileXRCcuLowerLayerTransportSnapshot templateSnapshot;
    TileXRCcuLowerLayerPlanBuilderReport report;
    ret = TileXRCcuBuildLowerLayerTransportTemplate(
        directCcuBasicInfo_,
        allocation,
        remoteCcuBuffers,
        &templateSnapshot,
        &report);
    directCcuLowerLayerPlanReport_ = report;
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }
    templateSnapshot.msidToken.dieId = directCcuBasicInfo_.dieId;
    templateSnapshot.msidToken.msId = directCcuBasicInfo_.msId;
    templateSnapshot.msidToken.tokenId = localCcuResourceWindow.tokenId;
    templateSnapshot.msidToken.tokenValue = localCcuResourceWindow.tokenValue;
    templateSnapshot.msidToken.valid = true;
    ret = TileXRCcuOverlayVerifiedEndpointRoutes(
        directCcuVerifiedEndpointRoutes_,
        &templateSnapshot,
        &directCcuLowerLayerPlanReport_);
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }

    directCcuLowerLayerTemplate_ = templateSnapshot;
    directCcuLowerLayerTemplateConfigured_ = true;
    return RefreshDirectCcuLowerLayerPlan();
}

int TileXRComm::FillDirectCcuLowerLayerPlanFromAllocation(
    const TileXRCcuResourceAllocation &allocation,
    TileXRCcuLowerLayerInstallPlan *plan,
    TileXRCcuLowerLayerPlanBuilderReport *report)
{
    if (plan == nullptr || report == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    int ret = PrepareDirectCcuLowerLayerTemplateFromAllocation(allocation);
    if (ret != TILEXR_SUCCESS) {
        *report = directCcuLowerLayerPlanReport_;
        return ret;
    }
    if (!directCcuLowerLayerPlanValid_) {
        *report = directCcuLowerLayerPlanReport_;
        return TILEXR_ERROR_NOT_FOUND;
    }
    *plan = directCcuLowerLayerPlan_;
    *report = directCcuLowerLayerPlanReport_;
    return TILEXR_SUCCESS;
}

int TileXRComm::ExchangeDirectCcuRemoteNotifyCke(
    const TileXRCcuResourceAllocation &allocation,
    std::vector<TileXRCcuRemoteCcuBufferInfo> *remoteCcuBuffers,
    TileXRCcuLowerLayerPlanBuilderReport *report)
{
    if (remoteCcuBuffers == nullptr) {
        if (report != nullptr) {
            report->message = "missing direct CCU remote notify CKE exchange inputs";
        }
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (rankSize_ <= 1 || rank_ < 0 || rank_ >= rankSize_) {
        if (report != nullptr) {
            report->message = "invalid direct CCU peer XN/CKE exchange shape";
        }
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const size_t peerRouteCount = static_cast<size_t>(rankSize_ - 1);
    const size_t syncRouteCount = allocation.remoteXn.num;
    if (allocation.localXn.num == 0 ||
        allocation.localWaitCke.num == 0 ||
        allocation.remoteNotifyCke.num == 0 ||
        allocation.remoteXn.num < static_cast<uint16_t>(rankSize_ - 1) ||
        allocation.localWaitCke.num < allocation.remoteXn.num ||
        allocation.remoteNotifyCke.num < allocation.remoteXn.num ||
        allocation.channels.num == 0 ||
        remoteCcuBuffers->size() != peerRouteCount) {
        if (report != nullptr) {
            report->message = "invalid direct CCU peer XN/CKE exchange shape";
        }
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    struct PeerResourceExchange {
        uint16_t localXnStartId;
        uint16_t localXnCount;
        uint16_t remoteXnStartId;
        uint16_t remoteXnCount;
        uint16_t localWaitCkeStartId;
        uint16_t localWaitCkeCount;
        uint16_t remoteNotifyCkeStartId;
        uint16_t remoteNotifyCkeCount;
        uint16_t channelStartId;
        uint16_t channelCount;
    };
    PeerResourceExchange local {
        allocation.localXn.startId,
        allocation.localXn.num,
        allocation.remoteXn.startId,
        DirectCcuRemoteXnProofSpan(allocation.remoteXn.num),
        allocation.localWaitCke.startId,
        allocation.localWaitCke.num,
        allocation.remoteNotifyCke.startId,
        allocation.remoteNotifyCke.num,
        allocation.channels.startId,
        allocation.channels.num,
    };
    std::vector<PeerResourceExchange> all(rankSize_);
    const int ret = DirectCcuAllGatherCallback(&local, sizeof(local), all.data(), this);
    if (ret != TILEXR_SUCCESS) {
        if (report != nullptr) {
            report->message = "failed to exchange direct CCU peer XN/CKE resources";
        }
        return ret;
    }

    std::vector<int> peerRanks;
    peerRanks.reserve(peerRouteCount);
    for (int peer = 0; peer < rankSize_; ++peer) {
        if (peer != rank_) {
            peerRanks.push_back(peer);
        }
    }
    if (peerRanks.size() != peerRouteCount) {
        if (report != nullptr) {
            report->message = "invalid direct CCU peer XN/CKE exchange shape";
        }
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    std::vector<TileXRCcuRemoteCcuBufferInfo> peerCcuBuffers = *remoteCcuBuffers;
    remoteCcuBuffers->assign(syncRouteCount, TileXRCcuRemoteCcuBufferInfo{});

    size_t routeIndex = 0;
    for (uint32_t syncIndex = 0; syncIndex < allocation.remoteXn.num; ++syncIndex) {
        const size_t peerBufferIndex = syncIndex % peerRouteCount;
        const int peer = peerRanks[peerBufferIndex];
        const PeerResourceExchange &peerResources = all[peer];
        const size_t peerLocalIndex = static_cast<size_t>(rank_ < peer ? rank_ : rank_ - 1);
        const uint32_t peerLocalXnOffset =
            SelectDirectCcuPeerLocalXnOffset(peerLocalIndex, syncIndex, peerRouteCount);
        const uint32_t selectedRemoteXnOffset =
            SelectDirectCcuChannelBoundRemoteXnOffset(peerLocalIndex, syncIndex, peerRouteCount);
        const uint32_t peerLocalWaitCkeOffset = routeIndex;
        if (peerResources.localXnCount == 0 ||
            peerResources.remoteXnCount == 0 ||
            peerResources.localWaitCkeCount == 0 ||
            peerResources.channelCount == 0 ||
            peerLocalXnOffset >= peerResources.localXnCount ||
            selectedRemoteXnOffset >= peerResources.remoteXnCount ||
            peerLocalIndex >= peerResources.channelCount ||
            peerLocalWaitCkeOffset >= peerResources.localWaitCkeCount) {
            if (report != nullptr) {
                report->message = "peer direct CCU local XN/CKE resources are incomplete";
            }
            return TILEXR_ERROR_NOT_FOUND;
        }
        uint16_t channelBoundRemoteXnId = SelectDirectCcuChannelBoundRemoteXnId(
            peerResources.remoteXnStartId,
            peerLocalIndex,
            syncIndex,
            peerRouteCount);
        const uint16_t peerLocalXnId =
            static_cast<uint16_t>(static_cast<uint32_t>(peerResources.localXnStartId) + peerLocalXnOffset);
        uint16_t remoteNotifyCke =
            static_cast<uint16_t>(static_cast<uint32_t>(peerResources.localWaitCkeStartId) +
                peerLocalWaitCkeOffset);
        (*remoteCcuBuffers)[routeIndex] = peerCcuBuffers[peerBufferIndex];
        (*remoteCcuBuffers)[routeIndex].remoteXnId = channelBoundRemoteXnId;
        (*remoteCcuBuffers)[routeIndex].remoteNotifyCke = remoteNotifyCke;
        const bool peerLocalXnOwnerVerified =
            static_cast<uint32_t>(peerLocalXnId) >= peerResources.localXnStartId &&
            static_cast<uint32_t>(peerLocalXnId) <
                static_cast<uint32_t>(peerResources.localXnStartId) + peerResources.localXnCount;
        const bool notifyCkeOwnerVerified =
            static_cast<uint32_t>(remoteNotifyCke) >= peerResources.localWaitCkeStartId &&
            static_cast<uint32_t>(remoteNotifyCke) <
                static_cast<uint32_t>(peerResources.localWaitCkeStartId) + peerResources.localWaitCkeCount;
        const bool localChannelOwnerVerified =
            allocation.channels.num != 0 &&
            peerLocalXnOwnerVerified &&
            static_cast<uint32_t>(channelBoundRemoteXnId) >= peerResources.remoteXnStartId &&
            static_cast<uint32_t>(channelBoundRemoteXnId) <
                static_cast<uint32_t>(peerResources.remoteXnStartId) + peerResources.remoteXnCount &&
            routeIndex < allocation.channels.num &&
            peerResources.channelStartId != 0 &&
            peerLocalIndex < peerResources.channelCount;
        const bool transportResourceExchangeVerified =
            notifyCkeOwnerVerified &&
            allocation.localWaitCke.num != 0 &&
            routeIndex < allocation.localWaitCke.num &&
            peerLocalWaitCkeOffset < peerResources.localWaitCkeCount;
        (*remoteCcuBuffers)[routeIndex].channelResourceOwnerVerified = localChannelOwnerVerified;
        (*remoteCcuBuffers)[routeIndex].transportResourceExchangeVerified = transportResourceExchangeVerified;
        ++routeIndex;
    }
    return TILEXR_SUCCESS;
}

int TileXRComm::DirectCcuAllGatherCallback(
    const void *sendBuf,
    size_t sendBytes,
    void *recvBuf,
    void *userData)
{
    auto *comm = static_cast<TileXRComm *>(userData);
    if (comm == nullptr || sendBuf == nullptr || recvBuf == nullptr || sendBytes == 0) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (comm->socketExchange_ == nullptr) {
        return comm->DirectCcuThreadAllGather(sendBuf, sendBytes, recvBuf);
    }
    return comm->socketExchange_->AllGather(
        static_cast<const uint8_t *>(sendBuf),
        sendBytes,
        static_cast<uint8_t *>(recvBuf));
}

int TileXRComm::DirectCcuThreadAllGather(const void *sendBuf, size_t sendBytes, void *recvBuf)
{
    if (sendBuf == nullptr || recvBuf == nullptr || sendBytes == 0 || rank_ < 0 ||
        rank_ >= rankSize_ || rankSize_ <= 0 || uid_.empty()) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    const uint64_t round = directCcuThreadAllGatherRound_++;
    const std::string key = uid_ + ":ccu:" + std::to_string(round);
    auto start = high_resolution_clock::now();
    for (;;) {
        {
            lock_guard<mutex> lock(g_mtx);
            auto &state = g_directCcuAllGatherStates[key];
            if (state.bytes == 0) {
                state.bytes = sendBytes;
            } else if (state.bytes != sendBytes) {
                g_directCcuAllGatherStates.erase(key);
                return TILEXR_ERROR_PARA_CHECK_FAIL;
            }
            if (state.data[rank_].empty()) {
                state.data[rank_].resize(sendBytes);
                std::memcpy(state.data[rank_].data(), sendBuf, sendBytes);
                ++state.arrivals;
            }
            if (state.arrivals == static_cast<uint64_t>(rankSize_)) {
                auto *output = static_cast<uint8_t *>(recvBuf);
                for (int i = 0; i < rankSize_; ++i) {
                    std::memcpy(output + static_cast<size_t>(i) * sendBytes, state.data[i].data(), sendBytes);
                }
                ++state.departures;
                if (state.departures == static_cast<uint64_t>(rankSize_)) {
                    g_directCcuAllGatherStates.erase(key);
                }
                return TILEXR_SUCCESS;
            }
        }
        const std::string processUnavailableMessage = ProcessDirectCcuRuntimeUnavailableMessage();
        if (!processUnavailableMessage.empty()) {
            lock_guard<mutex> lock(g_mtx);
            g_directCcuAllGatherStates.erase(key);
            TILEXR_LOG(ERROR) << "direct CCU thread allgather abort rank " << rank_ << "/" << rankSize_
                              << " uid " << uid_ << " round " << round << ", "
                              << processUnavailableMessage;
            return TILEXR_ERROR_NOT_FOUND;
        }
        this_thread::sleep_for(1ms);
        auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - start);
        if (elapsed.count() > TILEXR_INIT_TIMEOUT) {
            lock_guard<mutex> lock(g_mtx);
            g_directCcuAllGatherStates.erase(key);
            TILEXR_LOG(ERROR) << "direct CCU thread allgather timeout rank " << rank_ << "/" << rankSize_
                              << " uid " << uid_ << " round " << round;
            return TILEXR_ERROR_TIMEOUT;
        }
    }
}

int TileXRComm::PrepareDirectCcuLowerLayerPlanCallback(
    const TileXRCcuResourceAllocation &allocation,
    TileXRCcuLowerLayerInstallPlan *plan,
    TileXRCcuLowerLayerPlanBuilderReport *report,
    void *userData)
{
    auto *comm = static_cast<TileXRComm *>(userData);
    if (comm == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return comm->FillDirectCcuLowerLayerPlanFromAllocation(allocation, plan, report);
}

int TileXRComm::PrepareDirectCcuInstallAttempt(
    const TileXRCcuDirectInstallOptions &options,
    TileXRCcuDirectInstallAttempt *attempt,
    TileXRCcuDirectInstallReport *report)
{
    if (!inited_) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport{};
            report->message = "TileXRComm is not initialized for direct CCU install attempt";
        }
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    const std::string processUnavailableMessage = ProcessDirectCcuRuntimeUnavailableMessage();
    if (!processUnavailableMessage.empty()) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport{};
            report->message = processUnavailableMessage;
        }
        return TILEXR_ERROR_NOT_FOUND;
    }
    const uint8_t installDieId = SelectDirectCcuInstallDieId();
    if (!directCcuBasicInfoValid_ || directCcuBasicInfo_.dieId != installDieId) {
        const int ret = RefreshDirectCcuBasicInfo(installDieId);
        if (ret != TILEXR_SUCCESS) {
            if (report != nullptr) {
                *report = TileXRCcuDirectInstallReport{};
                report->message = directCcuBasicInfoReport_.message;
            }
            return ret;
        }
    }
    if (ccuDirectRuntime_ == nullptr || !ccuDirectRuntime_->IsAvailable()) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport{};
            report->message = "direct CCU runtime is unavailable for install attempt";
        }
        return TILEXR_ERROR_NOT_FOUND;
    }

    TileXRCcuDriverAdapter adapter;
    TileXRCcuDriverAdapterReport adapterReport;
    int ret = ccuDirectRuntime_->CreateDriverAdapter(&adapter, &adapterReport);
    if (ret != TILEXR_SUCCESS) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport{};
            report->message = adapterReport.message;
        }
        return ret;
    }

    TileXRCcuDirectInstallOptions next = options;
    next.basicInfo = &directCcuBasicInfo_;
    next.offlineOnly = false;
    next.driverAdapter = &adapter;
    next.repositoryMemoryOps = TileXRCcuMakeRepositoryDeviceMemoryOps(next.repositoryMemoryAllocMode);
    next.repositoryMemoryUserData = nullptr;
    next.lowerLayerPlan = nullptr;
    next.prepareLowerLayerPlan = &TileXRComm::PrepareDirectCcuLowerLayerPlanCallback;
    next.lowerLayerPlanUserData = this;
    if (next.provider.empty()) {
        next.provider = "tilexr-comm-direct-ccu";
    }

    return TileXRCcuRunDirectInstallAttempt(next, attempt, report);
}

int TileXRComm::PrepareDirectCcuMemoryCopyInstallAttempt(
    const TileXRCcuDirectInstallOptions &options,
    uint64_t localSourceAddr,
    uint64_t localDestinationAddr,
    uint64_t bytes,
    uint32_t peerRank,
    TileXRCcuMemoryCopyDirection direction,
    TileXRCcuDirectInstallAttempt *attempt,
    TileXRCcuDirectInstallReport *report)
{
    if (!inited_) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport {};
            report->message = "TileXRComm is not initialized for direct CCU memory copy install attempt";
        }
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    if (localSourceAddr == 0 || localDestinationAddr == 0 || bytes == 0 ||
        peerRank >= static_cast<uint32_t>(rankSize_) || peerRank == static_cast<uint32_t>(rank_)) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport {};
            report->message = "invalid direct CCU memory copy endpoint";
        }
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const std::string processUnavailableMessage = ProcessDirectCcuRuntimeUnavailableMessage();
    if (!processUnavailableMessage.empty()) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport {};
            report->message = processUnavailableMessage;
        }
        return TILEXR_ERROR_NOT_FOUND;
    }
    const uint8_t installDieId = SelectDirectCcuInstallDieId();
    if (!directCcuBasicInfoValid_ || directCcuBasicInfo_.dieId != installDieId) {
        const int ret = RefreshDirectCcuBasicInfo(installDieId);
        if (ret != TILEXR_SUCCESS) {
            if (report != nullptr) {
                *report = TileXRCcuDirectInstallReport {};
                report->message = directCcuBasicInfoReport_.message;
            }
            return ret;
        }
    }
    if (ccuDirectRuntime_ == nullptr || !ccuDirectRuntime_->IsAvailable()) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport {};
            report->message = "direct CCU runtime is unavailable for memory copy install attempt";
        }
        return TILEXR_ERROR_NOT_FOUND;
    }

    DirectCcuMemoryCopyEndpoint localEndpoint;
    int ret = BuildDirectCcuLocalMemoryCopyEndpoint(
        static_cast<uint32_t>(rank_),
        localSourceAddr,
        localDestinationAddr,
        bytes,
        &localEndpoint);
    if (ret != TILEXR_SUCCESS) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport {};
            report->message = "failed to query direct CCU memory copy local buffer token";
        }
        return ret;
    }

    std::vector<DirectCcuMemoryCopyEndpoint> allEndpoints(static_cast<size_t>(rankSize_));
    ret = DirectCcuAllGatherCallback(
        &localEndpoint,
        sizeof(localEndpoint),
        allEndpoints.data(),
        this);
    if (ret != TILEXR_SUCCESS) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport {};
            report->message = "failed to exchange direct CCU memory copy peer endpoints";
        }
        return ret;
    }
    const DirectCcuMemoryCopyEndpoint &peerEndpoint = allEndpoints[peerRank];
    if (peerEndpoint.valid == 0 || peerEndpoint.bytes != bytes) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport {};
            report->message = "invalid direct CCU memory copy peer endpoint";
        }
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    TileXRCcuDirectMemoryCopySpec memoryCopy;
    memoryCopy.direction = direction;
    memoryCopy.lengthBytes = bytes;
    if (direction == TileXRCcuMemoryCopyDirection::RemoteToLocal) {
        memoryCopy.localAddr = localEndpoint.destinationAddr;
        memoryCopy.localToken = localEndpoint.destinationToken;
        memoryCopy.remoteAddr = peerEndpoint.sourceAddr;
        memoryCopy.remoteToken = peerEndpoint.sourceToken;
    } else {
        memoryCopy.localAddr = localEndpoint.sourceAddr;
        memoryCopy.localToken = localEndpoint.sourceToken;
        memoryCopy.remoteAddr = peerEndpoint.destinationAddr;
        memoryCopy.remoteToken = peerEndpoint.destinationToken;
    }

    TileXRCcuDriverAdapter adapter;
    TileXRCcuDriverAdapterReport adapterReport;
    ret = ccuDirectRuntime_->CreateDriverAdapter(&adapter, &adapterReport);
    if (ret != TILEXR_SUCCESS) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport {};
            report->message = adapterReport.message;
        }
        return ret;
    }

    TileXRCcuDirectInstallOptions next = options;
    next.basicInfo = &directCcuBasicInfo_;
    next.offlineOnly = false;
    next.driverAdapter = &adapter;
    next.repositoryMemoryOps = TileXRCcuMakeRepositoryDeviceMemoryOps(next.repositoryMemoryAllocMode);
    next.repositoryMemoryUserData = nullptr;
    next.lowerLayerPlan = nullptr;
    next.prepareLowerLayerPlan = &TileXRComm::PrepareDirectCcuLowerLayerPlanCallback;
    next.lowerLayerPlanUserData = this;
    next.sqeArgCount = 0;
    next.syncResourceCount = 1;
    next.syncInstructionCount = std::max<uint32_t>(
        next.syncInstructionCount,
        TILEXR_CCU_DIRECT_MEMORY_COPY_INSTRUCTION_COUNT);
    next.bindingsPerSyncResource = next.bindingsPerSyncResource == 0 ? 1 : next.bindingsPerSyncResource;
    if (next.provider.empty()) {
        next.provider = "tilexr-comm-direct-ccu-memory-copy";
    }

    return TileXRCcuRunDirectMemoryCopyInstallAttempt(next, memoryCopy, attempt, report);
}

int TileXRComm::RefreshDirectCcuLowerLayerPlan()
{
    ResetDirectCcuLowerLayerPlan();
    if (!directCcuLowerLayerTemplateConfigured_) {
        directCcuLowerLayerPlanReport_.message = "direct CCU lower-layer template is not configured";
        directCcuLowerLayerPlanStatus_ = TILEXR_ERROR_NOT_FOUND;
        return directCcuLowerLayerPlanStatus_;
    }
    if (ccuDirectRuntime_ == nullptr || !ccuDirectRuntime_->IsAvailable()) {
        directCcuLowerLayerPlanReport_.message = "direct CCU runtime is unavailable for lower-layer planning";
        directCcuLowerLayerPlanStatus_ = TILEXR_ERROR_NOT_FOUND;
        return directCcuLowerLayerPlanStatus_;
    }

    TileXRCcuLowerLayerTransportSnapshot snapshot;
    int ret = ccuDirectRuntime_->ExportLowerLayerTransportSnapshot(directCcuLowerLayerTemplate_, &snapshot);
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanReport_.message = "failed to export direct CCU lower-layer transport snapshot";
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }
    ret = TileXRCcuOverlayVerifiedEndpointRoutes(
        directCcuVerifiedEndpointRoutes_,
        &snapshot,
        &directCcuLowerLayerPlanReport_);
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }

    TileXRCcuLowerLayerInstallPlan plan;
    TileXRCcuLowerLayerPlanBuilderReport report;
    ret = TileXRCcuBuildLowerLayerInstallPlanFromTransportSnapshot(snapshot, &plan, &report);
    directCcuLowerLayerPlanReport_ = report;
    if (ret != TILEXR_SUCCESS) {
        directCcuLowerLayerPlanStatus_ = ret;
        return directCcuLowerLayerPlanStatus_;
    }

    directCcuLowerLayerSnapshot_ = snapshot;
    directCcuLowerLayerPlan_ = plan;
    directCcuLowerLayerPlanReport_.message = "direct CCU lower-layer install plan cached";
    directCcuLowerLayerPlanValid_ = true;
    directCcuLowerLayerPlanStatus_ = TILEXR_SUCCESS;
    return TILEXR_SUCCESS;
}

bool TileXRComm::HasDirectCcuLowerLayerPlan() const
{
    return directCcuLowerLayerPlanValid_;
}

int TileXRComm::GetDirectCcuLowerLayerPlanStatus() const
{
    return directCcuLowerLayerPlanStatus_;
}

const TileXRCcuLowerLayerPlanBuilderReport &TileXRComm::GetDirectCcuLowerLayerPlanReport() const
{
    return directCcuLowerLayerPlanReport_;
}

const TileXRCcuLowerLayerInstallPlan *TileXRComm::GetDirectCcuLowerLayerPlan() const
{
    return directCcuLowerLayerPlanValid_ ? &directCcuLowerLayerPlan_ : nullptr;
}

int TileXRComm::ReadDirectCcuInstructionsForDebug(
    uint8_t dieId,
    uint16_t instructionStartId,
    void *instructions,
    uint32_t instructionCount,
    uint32_t instructionBytes,
    TileXRCcuDriverAdapterReport *report)
{
    if (report != nullptr) {
        *report = TileXRCcuDriverAdapterReport{};
    }
    if (!inited_) {
        if (report != nullptr) {
            report->message = "TileXRComm is not initialized for direct CCU instruction readback";
        }
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    if (ccuDirectRuntime_ == nullptr || !ccuDirectRuntime_->IsAvailable()) {
        if (report != nullptr) {
            report->message = "direct CCU runtime is unavailable for instruction readback";
        }
        return TILEXR_ERROR_NOT_FOUND;
    }

    TileXRCcuDriverAdapter adapter;
    int ret = ccuDirectRuntime_->CreateDriverAdapter(&adapter, report);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    return adapter.ReadInstructions(dieId, instructionStartId, instructions, instructionCount, instructionBytes, report);
}

int TileXRComm::InitCommon()
{
    // enable peer device
    if (EnablePeerAccess() != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "EnablePeerAccess failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    const char *lcclDeterministic = std::getenv("LCCL_DETERMINISTIC");
    if (lcclDeterministic && (string(lcclDeterministic) == "1" || string(lcclDeterministic) == "true")) {
        commArgs_.extraFlag |= ExtraFlag::DETERMINISTIC;
    }
    if (GetChipName() == ChipName::CHIP_910B2C) {
        commArgs_.extraFlag |= ExtraFlag::TOPO_910B2C;
    }
    if (GetChipName() >= ChipName::CHIP_910_9391) {
        commArgs_.extraFlag |= ExtraFlag::TOPO_910_93;
    }
    if (GetChipName() > ChipName::CHIP_910_9362) {
        commArgs_.extraFlag |= ExtraFlag::TOPO_910A5;
    }
    if (GetChipName() == ChipName::CHIP_910A5) {
        commArgs_.extraFlag |= ExtraFlag::PERF_CYCLE_A5;
    }
    constexpr uint32_t AI_CORE_NUM_20 = 20;
    if (GetCoreNum(GetChipName()) > AI_CORE_NUM_20) {
        commArgs_.extraFlag |= ExtraFlag::IS_GREATER_THAN_40_AIV;
    }

    localRank_ = rank_ % localRankSize_;
    return TILEXR_SUCCESS;
}

void TileXRComm::CloseIpcMem()
{
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_ || peerMem_[i] == nullptr) {
            continue;
        }
        int ret = rtIpcCloseMemory(static_cast<void *>(peerMem_[i]));
        if (ret != RT_ERROR_NONE) {
            TILEXR_LOG(WARN) << "Close ipc[" << i << "] memory failed! ret: " << ret;
        }
        peerMem_[i] = nullptr;
    }
}

void TileXRComm::FreePeerMem(GM_ADDR &mem) const
{
    if (mem != nullptr) {
        aclError aclRet = aclrtFree(mem);
        if (aclRet != ACL_SUCCESS) {
            TILEXR_LOG(ERROR) << "Free share memory failed! ret: " << aclRet;
        }
    }
    mem = nullptr;
}

int TileXRComm::Init()
{
    if (inited_) {
        return TILEXR_SUCCESS;
    }
    if (rank_ < 0 || rank_ >= rankSize_ || rankSize_ <= 0 || rankSize_ > TILEXR_MAX_RANK_SIZE) {
        TILEXR_LOG(ERROR) << "The rank is invalid! rank:" << rank_ << " rankSize:" << rankSize_;
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (TileXRSockExchange::CheckValid(commId_)) {
        socketExchange_ = new (nothrow) TileXRSockExchange(rank_, rankSize_, commId_);
    } else {
        socketExchange_ = new (nothrow) TileXRSockExchange(rank_, rankSize_, commDomain_);
    }
    if (socketExchange_ == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRSockExchange create failed. rank : " << rank_ << " rankSize:" << rankSize_;
        return TILEXR_ERROR_INTERNAL;
    }
    int ret = GetDev();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "init context failed! ret: " << ret;
        return ret;
    }

    TILEXR_LOG(INFO) << "rank " << rank_ << "/" << rankSize_ << " running devId:" << devId_;

    if (InitCommon() != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "init common failed!";
        return TILEXR_ERROR_INTERNAL;
    }

    TILEXR_LOG(DEBUG) << "Prepare to InitCommMem localRankSize_ -> " << localRankSize_ << ", localRank_ -> " << localRank_;
    if (InitCommMem() != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "InitCommMem failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    TILEXR_LOG(DEBUG) << "InitCommMem " << rank_ << "/" << rankSize_ << ", localRank_ : " << localRank_ <<
            ", localRankSize_ : " << localRankSize_ << " success";

    // 新增：初始化 UDMA
    ret = InitUDMA();
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    ret = InitDirectCcuRuntime();
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    ret = InitSDMA();
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }

    // set comm args in device.
    ret = SyncCommArgs();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "SyncCommArgs failed! ret: " << ret;
        return ret;
    }
    TILEXR_LOG(INFO) << "TileXRCommInit " << rank_ << "/" << rankSize_ << " success. extraFlag:" << commArgs_.extraFlag <<
        " commArgs_.localRank : " << commArgs_.localRank << " commArgs_.localRankSize : " << commArgs_.localRankSize;
    inited_ = true;
    return TILEXR_SUCCESS;
}

int TileXRComm::InitDirectCcuOnly()
{
    if (inited_) {
        return TILEXR_SUCCESS;
    }
    if (rank_ < 0 || rank_ >= rankSize_ || rankSize_ <= 0 || rankSize_ > TILEXR_MAX_RANK_SIZE) {
        TILEXR_LOG(ERROR) << "The rank is invalid for direct CCU only init! rank:" << rank_
                          << " rankSize:" << rankSize_;
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (TileXRSockExchange::CheckValid(commId_)) {
        socketExchange_ = new (nothrow) TileXRSockExchange(rank_, rankSize_, commId_);
    } else {
        socketExchange_ = new (nothrow) TileXRSockExchange(rank_, rankSize_, commDomain_);
    }
    if (socketExchange_ == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRSockExchange create failed for direct CCU only init. rank : "
                          << rank_ << " rankSize:" << rankSize_;
        return TILEXR_ERROR_INTERNAL;
    }

    int ret = GetDev();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "direct CCU only init context failed! ret: " << ret;
        return ret;
    }
    TILEXR_LOG(INFO) << "direct CCU only init rank " << rank_ << "/" << rankSize_
                     << " running devId:" << devId_;

    if (InitCommon() != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "direct CCU only init common failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    ret = InitDirectCcuRuntime();
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    inited_ = true;
    return TILEXR_SUCCESS;
}

int TileXRComm::InitThread(const std::string &uid)
{
    if (inited_) {
        return TILEXR_SUCCESS;
    }
    if (rank_ < 0 || rank_ >= rankSize_ || rankSize_ <= 0 || rankSize_ > TILEXR_MAX_RANK_SIZE) {
        TILEXR_LOG(ERROR) << "The rank is invalid! rank:" << rank_ << "rankSize:" << rankSize_;
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (GetDevThread(uid) != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "get devs failed.";
        return TILEXR_ERROR_INTERNAL;
    }
    TILEXR_LOG(INFO) << "rank " << rank_ << "/" << rankSize_ << " running devId:" << devId_ << "uid: " << uid;

    if (InitCommon() != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "init common failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    {
        lock_guard<mutex> lock(g_mtx);
        if (g_localPeerMemMap.find(uid) == g_localPeerMemMap.end()) {
            for (int i = 0; i < rankSize_; ++i) {
                g_localPeerMemMap[uid][i] = nullptr;
            }
        }
        uid_ = uid;
    }
    int ret = InitMem();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "InitMem failed! ret: " << ret;
        return ret;
    }
    g_localPeerMemMap[uid][rank_] = peerMem_[rank_];

    auto start = high_resolution_clock::now();
    for (int i = 0; i < rankSize_; ++i) {
        while (g_localPeerMemMap[uid][i] == nullptr) { // check other threads
            this_thread::sleep_for(1ms);
            auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - start);
            if (elapsed.count() > TILEXR_INIT_TIMEOUT) {
                TILEXR_LOG(ERROR) << "Lccl Init timeout!";
                FreePeerMem(g_localPeerMemMap[uid][rank_]);
                return TILEXR_ERROR_TIMEOUT;
            }
        }
        peerMem_[i] = g_localPeerMemMap[uid][i];
    }
    localRank_ = rank_;
    localRankSize_ = rankSize_;

    // 注意：InitThread 为单进程多线程模式，不支持 UDMA（需要 socketExchange_ 进行跨进程协调）
    // UDMA 主要用于跨进程/跨节点通信，线程模式使用进程内共享内存即可
    TILEXR_LOG(DEBUG) << "Thread mode: UDMA initialization skipped (single-process multi-thread scenario)";

    ret = InitDirectCcuRuntime();
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    ret = InitSDMA();
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }
    ret = SyncCommArgs();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "SyncCommArgs failed! ret: " << ret;
        return ret;
    }
    TILEXR_LOG(INFO) << "Lccl init multi thread " << rank_ << "/" << rankSize_ << " success, uid:" << uid;
    inited_ = true;
    return TILEXR_SUCCESS;
}

/**
 * @brief 函数内部会有检测，是否需要进行 aclrtDeviceEnablePeerAccess，如果芯片为310P且是HCCS链路，则不调用此函数。
 *
 *
 */
int TileXRComm::EnablePeerAccess()
{
    physicalInfo_.chipName = GetChipName();
    for (auto &dev : devList_) {
        if (devId_ == dev) {
            continue;
        }
        // 处理910B2C 16卡通信的特例
        if (SkipUnusedChannel910B2C(dev, devId_, GetChipName())) {
            continue;
        }

        int64_t value = 0;
        if (rtGetPairDevicesInfo(devId_, dev, 0, &value) != RT_ERROR_NONE) {
            TILEXR_LOG(WARN) << devId_ << " & " << dev << " pair devices info failed to get";
        } else {
            TILEXR_LOG(DEBUG) << devId_ << " <-----> " << dev << ", halGetPairDevicesInfo: *value = " << value;
        }

        // 如果310P未来通信域要支持两卡四芯的话，这里需要做更改。并且现在默认服务器上机器只有一个链路种类。
        if (value == TOPOLOGY_HCCS || value == TOPOLOGY_SIO || value == TOPOLOGY_HCCS_SW ||
            GetChipName() == ChipName::CHIP_910B2C) {
            physicalInfo_.physicalLink = PhysicalLink::HCCS;
            commArgs_.extraFlag &= ~(ExtraFlag::TOPO_PCIE);
        } else if (physicalInfo_.physicalLink == PhysicalLink::RESERVED) {
            physicalInfo_.physicalLink = PhysicalLink::PCIE;
            commArgs_.extraFlag |= ExtraFlag::TOPO_PCIE;
        }

        physicalInfo_.coreNum = GetCoreNum(physicalInfo_.chipName);

        // value里的0实际上对应驱动枚举类的 TOPOLOGY_HCCS
        if (physicalInfo_.chipName == ChipName::CHIP_310P3 && value == 0) {
            TILEXR_LOG(WARN) << "warn aclrtDeviceEnablePeerAccess is skipped! peerDeviceId = " << dev;
            continue;
        }

        aclError ret = aclrtDeviceEnablePeerAccess(dev, 0);
        if (ret != ACL_SUCCESS) {
            TILEXR_LOG(ERROR) << "err aclrtDeviceEnablePeerAccess failed peerDeviceId = " << dev << " ,rank = " << rank_
                           << ", value = " << value << ", flags = " << 0 << "," << __LINE__ << ": " << ret;
            return TILEXR_ERROR_INTERNAL;
        }
    }
    TILEXR_LOG(DEBUG) << "EnablePeerAccess succeed" << rank_;
    return TILEXR_SUCCESS;
}

int TileXRComm::GetDev()
{
    // 这里这个nodeNum可以理解为Y轴长度，手动控制的话将这个拦截修改即可。
    int nodeNum = socketExchange_->GetNodeNum();
    if (nodeNum <= 0 || nodeNum > rankSize_) {
        TILEXR_LOG(ERROR) << "error! node num : " << nodeNum << " rank size: " << rankSize_;
        return TILEXR_ERROR_INTERNAL;
    }
    localRankSize_ = rankSize_ < 0 ? 0 : rankSize_ / nodeNum;
    localRank_ = rank_ % localRankSize_;
    TILEXR_LOG(DEBUG) << "GetDev : localRankSize_ : " << localRankSize_ << " localRank_: " << localRank_
                    << "  rank :" << rank_ << "   rankSize :" << rankSize_;
    devList_.resize(rankSize_);
    // get current id and broadcast
    aclError aclRet = aclrtGetDevice(&devId_);
    if (aclRet != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtGetDevice error! ret: " << aclRet;
        return TILEXR_ERROR_INTERNAL;
    }
    // get other rank dev id, put into devList_
    int ret = socketExchange_->AllGather(&devId_, 1, devList_.data());
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXRSockExchange AllGather error! ret: " << ret;
        return TILEXR_ERROR_INTERNAL;
    }
    std::string devIdStr = "";
    for (int i = 0; i < rankSize_; ++i) {
        devIdStr += (i == 0 ? "" : ", ");
        devIdStr += to_string(devList_[i]);
    }
    TILEXR_LOG(DEBUG) << "rank " << rank_ << " devId: " << devId_ << ", otherDevList : " << devIdStr;
    TILEXR_LOG(INFO) << "AllGather: Get other rank dev id success";
    return TILEXR_SUCCESS;
}

int TileXRComm::GetDevThread(const std::string &uid)
{
    devList_.resize(rankSize_);
    // get current id and broadcast
    aclError aclRet = aclrtGetDevice(&devId_);
    if (aclRet != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtGetDevice error! ret: " << aclRet;
        return TILEXR_ERROR_INTERNAL;
    }
    {
        std::lock_guard<std::mutex> lock(g_mtx);
        if (g_devList.find(uid) == g_devList.end()) {
            for (int i = 0; i < rankSize_; ++i) {
                g_devList[uid][i] = 0;
            }
        }
    }
    g_devList[uid][rank_] = devId_ + 1; // 0 is invalid
    auto start = high_resolution_clock::now();
    for (int i = 0; i < rankSize_; ++i) {
        while (g_devList[uid][i] == 0) { // check other threads
            this_thread::sleep_for(1ms);
            auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - start);
            if (elapsed.count() > TILEXR_INIT_TIMEOUT) {
                TILEXR_LOG(ERROR) << "Lccl Init timeout!";
                return TILEXR_ERROR_TIMEOUT;
            }
        }
        devList_.at(i) = g_devList[uid][i] - 1;
    }
    return TILEXR_SUCCESS;
}

int TileXRComm::InitMem()
{
    // 申请并初始化IpcBuff
    constexpr int32_t bufferSizeUint = 1024 * 1024;
    int tilexrBuffSize = bufferSize_ * bufferSizeUint + TILEXR_FLAG_BUFF_BYTES;

    TILEXR_LOG(DEBUG) << "tilexr buffer size " << tilexrBuffSize;
    aclError ret = aclrtMalloc(
        reinterpret_cast<void **>(&peerMem_[rank_]), tilexrBuffSize,
        (GetChipName() == ChipName::CHIP_310P3) ? ACL_MEM_MALLOC_HUGE_FIRST_P2P : ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "allocate device mem error " << __FILE__ << ":" << __LINE__ << " " << ret;
        return TILEXR_ERROR_INTERNAL;
    }
    TILEXR_LOG(DEBUG) << "peerMem[rank" << rank_ << "], allocate finished.";
    aclrtMemset(peerMem_[rank_], tilexrBuffSize, 0, tilexrBuffSize);
    return TILEXR_SUCCESS;
}

int TileXRComm::GetPid(uint32_t *pids)
{
    if (rtDeviceGetBareTgid(&pids[rank_]) != RT_ERROR_NONE) {  // 获取docker外的进程id，bare指docker�?        TILEXR_LOG(ERROR) << "DeviceGetBareTgid err " << __LINE__;
        return TILEXR_ERROR_INTERNAL;
    }
    int ret = socketExchange_->AllGather(&pids[rank_], 1, pids);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXRSockExchange AllGather error! ret: " << ret;
        return ret;
    }
    for (int i = 0; i < rankSize_; ++i) {
        TILEXR_LOG(DEBUG) << "rank : " << rank_ << ", otherRank : " << i << " pid[" << i << "]: " << pids[i];
    }
    TILEXR_LOG(DEBUG) << "AllGather: Get other rank pid";
    return TILEXR_SUCCESS;
}

int TileXRComm::GetSidId(int64_t sdids[TILEXR_MAX_RANK_SIZE], int rankSize)
{
    if (rank_ >= rankSize) {
        TILEXR_LOG(ERROR) << "TileXRComm::GetSidId err rank_ >= rankSize " << rank_ << ">=" << rankSize;
        return TILEXR_ERROR_INTERNAL;
    }
    if ((physicalInfo_.chipName >= ChipName::CHIP_910_9391) && (physicalInfo_.chipName < ChipName::RESERVED)) {
        const int rtModuleTypeSystem = 0;
        const int infoTypeSdid = 26;
        if (rtGetDeviceInfo(devList_[rank_], rtModuleTypeSystem, infoTypeSdid, &sdids[rank_]) != RT_ERROR_NONE) {
            TILEXR_LOG(ERROR) << "DeviceGetDeviceInfo err " << __LINE__;
            return TILEXR_ERROR_INTERNAL;
        }
        TILEXR_LOG(DEBUG) << "rank " << rank_ << " dev id: " << devList_[rank_]
                       << " rtGetDeviceInfo sdid: " << sdids[rank_];

        int ret = socketExchange_->AllGather(&sdids[rank_], 1, sdids);
        if (ret != TILEXR_SUCCESS) {
            TILEXR_LOG(ERROR) << "TileXRSockExchange AllGather error! ret: " << ret;
            return ret;
        }
        for (int i = 0; i < rankSize_; ++i) {
            TILEXR_LOG(DEBUG) << "rank " << i << " sdid: " << sdids[i];
        }
        TILEXR_LOG(DEBUG) << "AllGather: Get other rank sdid";
    }
    return TILEXR_SUCCESS;
}

int TileXRComm::GetName(string &name, char names[TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE]) const
{
    int ret = socketExchange_->AllGather<char>(name.c_str(), IPC_NAME_SIZE, names[0]);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXRSockExchange AllGather error! ret: " << ret;
        return TILEXR_ERROR_INTERNAL;
    }
    for (int i = 0; i < rankSize_; ++i) {
        TILEXR_LOG(DEBUG) << "rank " << i << " mem name: " << names[i];
    }
    TILEXR_LOG(DEBUG) << "AllGather: Get other rank mem name";
    return TILEXR_SUCCESS;
}

int TileXRComm::InitCommMem()
{
    int ret = InitMem();
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "InitMem error! ret: " << ret;
        return ret;
    }

    // 获取所有进程的pid
    uint32_t pids[TILEXR_MAX_RANK_SIZE] = {0};
    ret = GetPid(pids);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "GetPid error! ret: " << ret;
        return ret;
    }

    // 获取所有进程的sdid
    int64_t sdids[TILEXR_MAX_RANK_SIZE] = {0};
    ret = GetSidId(sdids, rankSize_);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "GetSidId error! ret: " << ret;
        return ret;
    }

    // 获取所有进程的mem name
    string name;
    if (SetMemoryName(name) != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "SetMemoryName err ";
        return TILEXR_ERROR_INTERNAL;
    }

    if (SetIpcPidSdid(name, pids, sdids) != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "SetIpcPidSdid failed!";
        return TILEXR_ERROR_INTERNAL;
    }

    TILEXR_LOG(DEBUG) << "rank " << rank_ << " mem name: " << name << " name len: " << name.size();
    char names[TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE];
    name.resize(IPC_NAME_SIZE);
    ret = GetName(name, names);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "GetName error! ret: " << ret;
        return ret;
    }

    if (OpenIpcMem(names) != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "rank: " << rank_ << " OpenIpcMem failed!";
        return TILEXR_ERROR_INTERNAL;
    }
    return TILEXR_SUCCESS;
}

int TileXRComm::OpenIpcMem(const char names[TILEXR_MAX_RANK_SIZE][IPC_NAME_SIZE])
{
    static mutex mut;
    lock_guard<mutex> lock(mut);
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_) {
            continue;
        }
        // 处理910B2C 16卡通信的特例
        if (SkipUnusedChannel910B2C(rank_, i, GetChipName())) {
            continue;
        }
        int ret = rtIpcOpenMemory(reinterpret_cast<void **>(&peerMem_[i]), names[i]);
        if (ret != RT_ERROR_NONE) {
            CloseIpcMem();
            TILEXR_LOG(ERROR) << "rank : " << rank_ << " localRank : " << localRank_ << " peerMem: " << i <<
                " devId : " << devId_ << " peerDevId : " << (i < static_cast<int>(devList_.size()) ? devList_[i] : -1) <<
                " localRankSize : " << localRankSize_ << " ipcNameLen : " << std::strlen(names[i]) <<
                " ipcNamePrefix : " << std::string(names[i], std::min<size_t>(std::strlen(names[i]), 16U)) <<
                " IpcOpenMemory err " << ret;
            return TILEXR_ERROR_INTERNAL;
        }
    }
    ipcMemInited_ = true;
    return TILEXR_SUCCESS;
}

int TileXRComm::SetMemoryName(string &name)
{
    char nameModified[IPC_NAME_SIZE] = {};
    int memRank = rank_;
    constexpr int32_t bufferSizeUint = 1024 * 1024;
    int tilexrBuffSize = bufferSize_ * bufferSizeUint + TILEXR_FLAG_BUFF_BYTES;
    if (rtIpcSetMemoryName(peerMem_[memRank], tilexrBuffSize, nameModified, IPC_NAME_SIZE) != RT_ERROR_NONE) {
        return TILEXR_ERROR_INTERNAL;
    }
    name = nameModified;
    return TILEXR_SUCCESS;
}

int TileXRComm::SetIpcPidSdid(string &name, const uint32_t *pids, const int64_t *sdids) const
{
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_) {
            continue;
        }

        if (UseLegacyIpcPid(physicalInfo_.chipName)) {
            // 910B
            int32_t pidInt32 = pids[i];
            int rtRet = rtSetIpcMemPid(name.c_str(), &pidInt32, HCCL_IPC_PID_ARRAY_SIZE);
            if (rtRet != RT_ERROR_NONE) {
                TILEXR_LOG(ERROR) << "err " << rtRet;
                return TILEXR_ERROR_INTERNAL;
            }
        } else {
            // 910A3
            int32_t pidInt32 = pids[i];
            int rtRet = rtSetIpcMemorySuperPodPid(name.c_str(), sdids[i], &pidInt32, HCCL_IPC_PID_ARRAY_SIZE);
            if (rtRet != RT_ERROR_NONE) {
                TILEXR_LOG(ERROR) << "err " << rtRet;
                return TILEXR_ERROR_INTERNAL;
            }
        }
    }
    return TILEXR_SUCCESS;
}

TileXRComm::~TileXRComm()
{
    {
        lock_guard<mutex> lock(g_mtx);
        if (g_localPeerMemMap.find(uid_) != g_localPeerMemMap.end()) {
            g_localPeerMemMap.erase(uid_);
        }
    }
    if (ipcMemInited_) {
        CloseIpcMem();
        ipcMemInited_ = false;
    }
    if (socketExchange_) {
        delete socketExchange_;
        socketExchange_ = nullptr;
    }
    FreePeerMem(commArgs_.dumpAddr);
    FreeUDMARegistry();
    FreePeerMem(peerMem_[rank_]);
    FreePeerMem(commArgsPtr_);

    if (udmaTransport_ != nullptr) {
        udmaTransport_->Shutdown();
        udmaTransport_.reset();
    }
    udmaRegisteredPtr_ = nullptr;
    udmaInfoDev_ = nullptr;
    ResetDirectCcuBasicInfo();
    ResetSDMAState();
}

TileXRComm::TileXRComm(int rank, int rankSize) : rank_(rank), rankSize_(rankSize)
{
}

TileXRComm::TileXRComm(int rank, int rankSize, int commDomain, int bufferSize)
    : rank_(rank), rankSize_(rankSize), commDomain_(commDomain), bufferSize_(bufferSize)
{
}

TileXRComm::TileXRComm(int rank, int rankSize, TileXRUniqueId commId)
    : rank_(rank), rankSize_(rankSize), commId_(commId)
{
}

int TileXRComm::GetRank() const
{
    return rank_;
}

int TileXRComm::GetRankSize() const
{
    return rankSize_;
}

int TileXRComm::GetCommSize() const
{
    return commSize_;
}

const PhysicalInfo &TileXRComm::GetPhysicalInfo() const
{
    return physicalInfo_;
}

GM_ADDR TileXRComm::GetCommArgsPtr()
{
    return commArgsPtr_;
}

CommArgs* TileXRComm::GetCommArgs()
{
    return &commArgs_;
}

int64_t TileXRComm::NextMagic()
{
    return magic_.fetch_add(1);
}

std::string TileXRComm::PrintDFX()
{
    if (commArgsPtr_ == nullptr) {
        return "no comm args";
    }

    int ret = aclrtMemcpy(&commArgs_, sizeof(commArgs_), commArgsPtr_, sizeof(commArgs_),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMemcpy err " << __LINE__ << " " << ret;
        return "acl mem copy error";
    }
    stringstream ss;
    // 输出CommArgs基本属性
    ss << "CommArgs {"
       << "\n  rank: " << commArgs_.rank
       << "\n  localRank: " << commArgs_.localRank
       << "\n  rankSize: " << commArgs_.rankSize
       << "\n  localRankSize: " << commArgs_.localRankSize
       << "\n  extraFlag:  0x" << std::hex << std::setfill('0') << commArgs_.extraFlag;

    // 输出peerMems数组内容
    ss << "\n  peerMems: [";
    for (int i = 0; i < TILEXR_MAX_RANK_SIZE; ++i) {
        if (commArgs_.peerMems[i] == nullptr) {
            continue;
        }
        if (i > 0) {
            ss << ", ";
        }
        ss << "{id: " << static_cast<void *>(commArgs_.peerMems[i]) << "}";
    }
    ss << "]";

    // magic数组内容
    ss << "\n  magics: [";
    for (int i = 0; i < rankSize_; ++i) {
        ss << std::dec << commArgs_.magics[i] << ",";
    }
    ss << "] \n";

    // 输出dfx数组内容
    ss << "\n  dfx: [";
    const int dfxGroupCount = 5;
    for (int i = 0; i < DFX_COUNT; ++i) {
        if (i % dfxGroupCount == 0) {
            ss << "\n    " << std::dec << setw(dfxGroupCount) << i << ": ";
        }
        ss << "0x"<< std::hex << commArgs_.dfx[i] << ", ";
    }
    ss << "\n    ]";

    ss << "\n  sdma: {"
       << " enabled: " << (((commArgs_.extraFlag & ExtraFlag::SDMA) != 0) ? "true" : "false")
       << ", status: " << static_cast<int>(sdmaInitStatus_)
       << ", workspace: " << static_cast<void*>(commArgs_.sdmaWorkspacePtr)
       << " }";

    ss << "\n}";
    return ss.str();
}

}  // TileXR
