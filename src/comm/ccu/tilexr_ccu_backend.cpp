/*
 * Copyright (c) 2026 TileXR Project
 */
#include "ccu/tilexr_ccu_backend.h"

#include "ccu/tilexr_ccu_direct_orchestrator.h"
#include "ccu/tilexr_ccu_direct_runtime.h"
#include "ccu/tilexr_ccu_memory_program.h"
#include "ccu/tilexr_ccu_repository.h"
#include "tilexr_log.h"
#include "tools/socket/tilexr_sock_exchange.h"

#include <acl/acl_rt.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include "runtime/dev.h"
#include "runtime/mem.h"
#include "runtime/rts/rts_device.h"

using namespace std;
using namespace chrono;

namespace TileXR {

constexpr int TILEXR_INIT_TIMEOUT = 600;
constexpr uint32_t TILEXR_CCU_DIRECT_MEMORY_COPY_INSTRUCTION_COUNT = 7U;

struct TileXRThreadAllGatherState {
    std::vector<uint8_t> data[TILEXR_MAX_RANK_SIZE];
    uint64_t arrivals = 0;
    uint64_t departures = 0;
    size_t bytes = 0;
};
static map<string, TileXRThreadAllGatherState> g_directCcuAllGatherStates;
static std::mutex g_mtx;
static std::mutex g_ccuDirectRuntimeMtx;
static bool g_ccuDirectRuntimeUnavailable = false;
static std::string g_ccuDirectRuntimeUnavailableMessage;

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

class TileXRCcuBackend::Impl {
public:
    int Init(const TileXRCcuBackendOptions &options);
    void Shutdown();
    bool Available() const;

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

private:
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
    static int DirectCcuAllGatherCallback(const void *sendBuf, size_t sendBytes, void *recvBuf, void *userData);
    int DirectCcuThreadAllGather(const void *sendBuf, size_t sendBytes, void *recvBuf);

    TileXRCcuBackendOptions options_ = {};
    int rank_ = 0;
    int rankSize_ = 0;
    int devId_ = 0;
    std::string uid_ = {};
    TileXRSockExchange *socketExchange_ = nullptr;
    bool initialized_ = false;
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
};

void TileXRCcuBackend::Impl::Shutdown()
{
    initialized_ = false;
    ResetDirectCcuBasicInfo();
    ResetDirectCcuLowerLayerPlan();
    directCcuLowerLayerTemplateConfigured_ = false;
    directCcuLowerLayerTemplate_ = TileXRCcuLowerLayerTransportSnapshot {};
    directCcuVerifiedEndpointRoutes_.clear();
    directCcuLocalVerifiedEndpointRoute_ = TileXRCcuLowerLayerTransportRoute {};
    directCcuLocalVerifiedEndpointRouteValid_ = false;
    directCcuThreadAllGatherRound_ = 0;
    if (ccuDirectRuntime_ != nullptr) {
        ccuDirectRuntime_->Shutdown();
        ccuDirectRuntime_.reset();
    }
    options_ = TileXRCcuBackendOptions {};
    socketExchange_ = nullptr;
}

bool TileXRCcuBackend::Impl::Available() const
{
    return initialized_ && ccuDirectRuntime_ != nullptr && ccuDirectRuntime_->IsAvailable();
}

int TileXRCcuBackend::Impl::Init(const TileXRCcuBackendOptions &options)
{
    Shutdown();
    options_ = options;
    rank_ = options.rank;
    rankSize_ = options.rankSize;
    devId_ = options.devId;
    uid_ = options.uid;
    socketExchange_ = options.exchange;
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

    TileXRCcuDirectRuntimeOptions runtimeOptions {};
    runtimeOptions.rank = rank_;
    runtimeOptions.rankSize = rankSize_;
    runtimeOptions.devId = devId_;
    runtimeOptions.allGather = &TileXRCcuBackend::Impl::DirectCcuAllGatherCallback;
    runtimeOptions.allGatherUserData = this;
    TileXRCcuDirectRuntimeReport runtimeReport;
    const int ret = ccuDirectRuntime_->Init(runtimeOptions, &runtimeReport);
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
    initialized_ = true;
    return TILEXR_SUCCESS;
}

void TileXRCcuBackend::Impl::ResetDirectCcuBasicInfo()
{
    directCcuBasicInfoValid_ = false;
    directCcuBasicInfoStatus_ = TILEXR_ERROR_NOT_FOUND;
    directCcuBasicInfo_ = TileXRCcuBasicInfo {};
    directCcuBasicInfoReport_ = TileXRCcuDriverAdapterReport {};
}

void TileXRCcuBackend::Impl::ResetDirectCcuLowerLayerPlan()
{
    directCcuLowerLayerPlanValid_ = false;
    directCcuLowerLayerPlanStatus_ = TILEXR_ERROR_NOT_FOUND;
    directCcuLowerLayerSnapshot_ = TileXRCcuLowerLayerTransportSnapshot {};
    directCcuLowerLayerPlan_ = TileXRCcuLowerLayerInstallPlan {};
    directCcuLowerLayerPlanReport_ = TileXRCcuLowerLayerPlanBuilderReport {};
}

int TileXRCcuBackend::Impl::RefreshDirectCcuBasicInfo(uint8_t dieId)
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

bool TileXRCcuBackend::Impl::HasDirectCcuBasicInfo() const
{
    return directCcuBasicInfoValid_;
}

int TileXRCcuBackend::Impl::GetDirectCcuBasicInfoStatus() const
{
    return directCcuBasicInfoStatus_;
}

const TileXRCcuBasicInfo *TileXRCcuBackend::Impl::GetDirectCcuBasicInfo() const
{
    return directCcuBasicInfoValid_ ? &directCcuBasicInfo_ : nullptr;
}

const TileXRCcuDriverAdapterReport &TileXRCcuBackend::Impl::GetDirectCcuBasicInfoReport() const
{
    return directCcuBasicInfoReport_;
}

int TileXRCcuBackend::Impl::ConfigureDirectCcuLowerLayerTemplate(
    const TileXRCcuLowerLayerTransportSnapshot &templateSnapshot)
{
    directCcuLowerLayerTemplate_ = templateSnapshot;
    directCcuLowerLayerTemplateConfigured_ = true;
    return RefreshDirectCcuLowerLayerPlan();
}

int TileXRCcuBackend::Impl::ConfigureDirectCcuVerifiedEndpointRoutes(
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

int TileXRCcuBackend::Impl::ConfigureDirectCcuLocalVerifiedEndpointRoute(
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

int TileXRCcuBackend::Impl::ConfigureDirectCcuLowerLayerTemplateFromAllocation(
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

int TileXRCcuBackend::Impl::PrepareDirectCcuLowerLayerTemplateFromAllocation(
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

int TileXRCcuBackend::Impl::FillDirectCcuLowerLayerPlanFromAllocation(
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

int TileXRCcuBackend::Impl::ExchangeDirectCcuRemoteNotifyCke(
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

int TileXRCcuBackend::Impl::DirectCcuAllGatherCallback(
    const void *sendBuf,
    size_t sendBytes,
    void *recvBuf,
    void *userData)
{
    auto *backend = static_cast<TileXRCcuBackend::Impl *>(userData);
    if (backend == nullptr || sendBuf == nullptr || recvBuf == nullptr || sendBytes == 0) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (backend->socketExchange_ == nullptr) {
        return backend->DirectCcuThreadAllGather(sendBuf, sendBytes, recvBuf);
    }
    return backend->socketExchange_->AllGather(
        static_cast<const uint8_t *>(sendBuf),
        sendBytes,
        static_cast<uint8_t *>(recvBuf));
}

int TileXRCcuBackend::Impl::DirectCcuThreadAllGather(const void *sendBuf, size_t sendBytes, void *recvBuf)
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

int TileXRCcuBackend::Impl::PrepareDirectCcuLowerLayerPlanCallback(
    const TileXRCcuResourceAllocation &allocation,
    TileXRCcuLowerLayerInstallPlan *plan,
    TileXRCcuLowerLayerPlanBuilderReport *report,
    void *userData)
{
    auto *backend = static_cast<TileXRCcuBackend::Impl *>(userData);
    if (backend == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return backend->FillDirectCcuLowerLayerPlanFromAllocation(allocation, plan, report);
}

int TileXRCcuBackend::Impl::PrepareDirectCcuInstallAttempt(
    const TileXRCcuDirectInstallOptions &options,
    TileXRCcuDirectInstallAttempt *attempt,
    TileXRCcuDirectInstallReport *report)
{
    if (!initialized_) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport{};
            report->message = "TileXRCcuBackend is not initialized for direct CCU install attempt";
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
    next.prepareLowerLayerPlan = &TileXRCcuBackend::Impl::PrepareDirectCcuLowerLayerPlanCallback;
    next.lowerLayerPlanUserData = this;
    if (next.provider.empty()) {
        next.provider = "tilexr-comm-direct-ccu";
    }

    return TileXRCcuRunDirectInstallAttempt(next, attempt, report);
}

int TileXRCcuBackend::Impl::PrepareDirectCcuMemoryCopyInstallAttempt(
    const TileXRCcuDirectInstallOptions &options,
    uint64_t localSourceAddr,
    uint64_t localDestinationAddr,
    uint64_t bytes,
    uint32_t peerRank,
    TileXRCcuMemoryCopyDirection direction,
    TileXRCcuDirectInstallAttempt *attempt,
    TileXRCcuDirectInstallReport *report)
{
    if (!initialized_) {
        if (report != nullptr) {
            *report = TileXRCcuDirectInstallReport {};
            report->message = "TileXRCcuBackend is not initialized for direct CCU memory copy install attempt";
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
    next.prepareLowerLayerPlan = &TileXRCcuBackend::Impl::PrepareDirectCcuLowerLayerPlanCallback;
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

int TileXRCcuBackend::Impl::RefreshDirectCcuLowerLayerPlan()
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

bool TileXRCcuBackend::Impl::HasDirectCcuLowerLayerPlan() const
{
    return directCcuLowerLayerPlanValid_;
}

int TileXRCcuBackend::Impl::GetDirectCcuLowerLayerPlanStatus() const
{
    return directCcuLowerLayerPlanStatus_;
}

const TileXRCcuLowerLayerPlanBuilderReport &TileXRCcuBackend::Impl::GetDirectCcuLowerLayerPlanReport() const
{
    return directCcuLowerLayerPlanReport_;
}

const TileXRCcuLowerLayerInstallPlan *TileXRCcuBackend::Impl::GetDirectCcuLowerLayerPlan() const
{
    return directCcuLowerLayerPlanValid_ ? &directCcuLowerLayerPlan_ : nullptr;
}

int TileXRCcuBackend::Impl::ReadDirectCcuInstructionsForDebug(
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
    if (!initialized_) {
        if (report != nullptr) {
            report->message = "TileXRCcuBackend is not initialized for direct CCU instruction readback";
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
TileXRCcuBackend::TileXRCcuBackend() : impl_(new (std::nothrow) Impl())
{
}

TileXRCcuBackend::~TileXRCcuBackend()
{
    Shutdown();
}

int TileXRCcuBackend::Init(const TileXRCcuBackendOptions &options)
{
    if (impl_ == nullptr) {
        impl_.reset(new (std::nothrow) Impl());
        if (impl_ == nullptr) {
            return TILEXR_ERROR_INTERNAL;
        }
    }
    return impl_->Init(options);
}

void TileXRCcuBackend::Shutdown()
{
    if (impl_ != nullptr) {
        impl_->Shutdown();
    }
}

bool TileXRCcuBackend::Available() const
{
    return impl_ != nullptr && impl_->Available();
}

bool TileXRCcuBackend::Supports(const TileXRCcuCollectiveRequest &request) const
{
    (void)request;
    return false;
}

int TileXRCcuBackend::PrepareCollective(const TileXRCcuCollectiveRequest &request, TileXRCcuCollectivePlan *plan)
{
    if (plan == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *plan = TileXRCcuCollectivePlan {};
    if (!Available()) {
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    (void)request;
    return TILEXR_ERROR_NOT_SUPPORT;
}

int TileXRCcuBackend::SubmitCollective(const TileXRCcuCollectivePlan &plan, aclrtStream)
{
    if (!Available()) {
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    return plan.ready ? TILEXR_ERROR_NOT_SUPPORT : TILEXR_ERROR_PARA_CHECK_FAIL;
}

#ifdef TILEXR_CCU_TESTING
bool TileXRCcuBackend::RuntimeInitializedForTest() const
{
    return Available();
}
#endif

} // namespace TileXR
