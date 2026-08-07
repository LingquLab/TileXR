/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_context.h"

#include <acl/acl_rt.h>

#include <algorithm>
#include <array>
#include <new>
#include <string>
#include <vector>

#include "tilexr_api.h"
#include "tilexr_log.h"
#include "tools/socket/tilexr_sock_exchange.h"
#include "udma/tilexr_udma_transport.h"

namespace TileXR {

TileXRUDMAContext::TileXRUDMAContext() = default;

TileXRUDMAContext::~TileXRUDMAContext()
{
    Shutdown();
}

int TileXRUDMAContext::Init(const TileXRUDMAContextOptions& options)
{
    options_ = options;
    if (options_.rankSize <= 1) {
        TILEXR_LOG(INFO) << "InitUDMA skipped for single-rank communicator";
        return TILEXR_SUCCESS;
    }

    UDMAQpConfig qpConfig;
    const int configRet = LoadAndAgreeQpConfig(qpConfig);
    if (configRet != TILEXR_SUCCESS) {
        TILEXR_LOG(WARN) << "TileXR UDMA QP configuration unavailable: " << configRet
                         << ", UDMA disabled";
        return TILEXR_SUCCESS;
    }

    transport_.reset(new (std::nothrow) TileXRUDMATransport());
    const int transportAllocationStatus = AgreeStatus(
        transport_ == nullptr ? TILEXR_ERROR_INTERNAL : TILEXR_SUCCESS);
    if (transportAllocationStatus != TILEXR_SUCCESS) {
        transport_.reset();
        TILEXR_LOG(WARN) << "TileXRUDMATransport allocation failed on at least one rank, UDMA disabled";
        return TILEXR_SUCCESS;
    }

    TileXRUDMATransportOptions transportOptions {};
    transportOptions.rank = options_.rank;
    transportOptions.rankSize = options_.rankSize;
    transportOptions.localRankSize = options_.localRankSize;
    transportOptions.devId = options_.devId;
    transportOptions.nonPinRegistration = options_.nonPinRegistration;
    transportOptions.exchange = options_.exchange;
    transportOptions.qpConfig = qpConfig;
    int ret = transport_->Init(transportOptions);
    if (ret != TILEXR_SUCCESS || !transport_->IsAvailable()) {
        TILEXR_LOG(WARN) << "TileXR UDMA init failed: " << ret << ", UDMA disabled";
        transport_.reset();
        return TILEXR_SUCCESS;
    }

    udmaInfoDev_ = transport_->GetUDMAInfoDev();
    if (udmaInfoDev_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXR UDMA info is null, UDMA disabled";
        transport_->Shutdown();
        transport_.reset();
        return TILEXR_SUCCESS;
    }

    lifecycle_ = Lifecycle::TransportReady;
    ret = ApplyCommArgsState(GetCommArgsState());
    if (ret != TILEXR_SUCCESS) {
        Shutdown();
        return ret;
    }

    TILEXR_LOG(INFO) << "InitUDMA success, rank " << options_.rank << "/" << options_.rankSize;
    return TILEXR_SUCCESS;
}

void TileXRUDMAContext::Shutdown()
{
    TileXRUDMACommArgsState disabledState {};
    const int publishRet = ApplyCommArgsState(disabledState);
    if (publishRet != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR UDMA shutdown failed to clear comm args: " << publishRet;
    }

    int memoryCleanupRet = TILEXR_SUCCESS;
    if (transport_ != nullptr) {
        memoryCleanupRet = transport_->CleanupAllMemory();
    }
    const int registryCleanupRet = CleanupAllRegistries();
    if (memoryCleanupRet != TILEXR_SUCCESS || registryCleanupRet != TILEXR_SUCCESS ||
        (transport_ != nullptr && transport_->HasMemoryCleanupPending()) ||
        udmaRegistryDev_ != nullptr || !retiredRegistryDevs_.empty()) {
        TILEXR_LOG(ERROR) << "TileXR UDMA shutdown retains memory cleanup state"
                          << ", transport ret " << memoryCleanupRet
                          << ", registry ret " << registryCleanupRet
                          << ", registry ptr " << reinterpret_cast<uintptr_t>(udmaRegistryDev_)
                          << ", retired registries " << retiredRegistryDevs_.size();
    }

    if (transport_ != nullptr) {
        transport_->Shutdown();
        transport_.reset();
    }

    retiredRegistryDevs_.clear();
    udmaRegistryDev_ = nullptr;
    registry_ = TileXRUDMARegistry {};
    registeredPtr_ = nullptr;
    registeredBytes_ = 0;
    udmaInfoDev_ = nullptr;
    lifecycle_ = Lifecycle::Unavailable;
}

bool TileXRUDMAContext::IsAvailable() const
{
    return lifecycle_ != Lifecycle::Unavailable && udmaInfoDev_ != nullptr &&
        transport_ != nullptr && transport_->IsAvailable();
}

TileXRUDMACommArgsState TileXRUDMAContext::GetCommArgsState() const
{
    TileXRUDMACommArgsState state {};
    state.available = IsAvailable();
    state.infoDev = state.available ? udmaInfoDev_ : nullptr;
    state.registryDev = state.available && lifecycle_ == Lifecycle::MemoryReady ? udmaRegistryDev_ : nullptr;
    return state;
}

int TileXRUDMAContext::RegisterMemory(GM_ADDR localPtr, size_t bytes, TileXRUDMAMemHandle* handle)
{
    if (localPtr == nullptr || bytes == 0 || handle == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!IsAvailable()) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister called while UDMA is unavailable";
        return TILEXR_ERROR_NOT_FOUND;
    }
    if (options_.threadMode) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister is not supported in InitThread mode";
        return TILEXR_ERROR_INTERNAL;
    }
    if (options_.exchange == nullptr) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister requires live socket exchange";
        return TILEXR_ERROR_INTERNAL;
    }
    if (lifecycle_ == Lifecycle::CleanupPending) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister requires pending UDMA memory cleanup first";
        return TILEXR_ERROR_INTERNAL;
    }

    const TileXRUDMACommArgsState previousState = GetCommArgsState();
    int ret = transport_->PrepareMemory(localPtr, bytes);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR UDMA memory registration failed: " << ret;
        const int localCleanupStatus = transport_->HasMemoryCleanupPending()
            ? TILEXR_ERROR_INTERNAL : TILEXR_SUCCESS;
        const int cleanupStatus = AgreeStatus(localCleanupStatus);
        if (cleanupStatus != TILEXR_SUCCESS) {
            EnterCleanupPending("candidate prepare rollback did not complete on every rank");
        }
        return ret;
    }

    TileXRUDMARegionDesc localRegion {};
    localRegion.base = localPtr;
    localRegion.bytes = bytes;
    std::vector<TileXRUDMARegionDesc> allRegions(options_.rankSize);
    ret = options_.exchange->AllGather(&localRegion, 1, allRegions.data());
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister allgather failed: " << ret;
        const int cleanupRet = transport_->AbortPreparedMemory();
        if (cleanupRet != TILEXR_SUCCESS || transport_->HasMemoryCleanupPending()) {
            EnterCleanupPending("candidate rollback failed after region exchange");
        }
        return ret;
    }

    TileXRUDMARegistry nextRegistry {};
    GM_ADDR nextRegistryDev = nullptr;
    int candidateStatus = TILEXR_SUCCESS;
    if (options_.rankSize <= 0 || options_.rankSize > static_cast<int>(TILEXR_MAX_RANK_SIZE)) {
        candidateStatus = TILEXR_ERROR_PARA_CHECK_FAIL;
    } else {
        nextRegistry.rankSize = static_cast<uint32_t>(options_.rankSize);
        nextRegistry.regionCount = 1;
        for (int i = 0; i < options_.rankSize; ++i) {
            if (allRegions[i].base == nullptr || allRegions[i].bytes == 0) {
                TILEXR_LOG(ERROR) << "TileXRUDMARegister received invalid region from rank " << i;
                candidateStatus = TILEXR_ERROR_PARA_CHECK_FAIL;
                break;
            }
            nextRegistry.regions[i] = allRegions[i];
        }
    }

    if (candidateStatus == TILEXR_SUCCESS) {
        const aclError allocRet = aclrtMalloc(
            reinterpret_cast<void**>(&nextRegistryDev), sizeof(nextRegistry), ACL_MEM_MALLOC_HUGE_FIRST);
        if (allocRet != ACL_SUCCESS) {
            TILEXR_LOG(ERROR) << "aclrtMalloc UDMA registry failed: " << allocRet;
            candidateStatus = TILEXR_ERROR_INTERNAL;
        }
    }

    if (candidateStatus == TILEXR_SUCCESS) {
        const aclError copyRet = aclrtMemcpy(nextRegistryDev, sizeof(nextRegistry),
            &nextRegistry, sizeof(nextRegistry), ACL_MEMCPY_HOST_TO_DEVICE);
        if (copyRet != ACL_SUCCESS) {
            TILEXR_LOG(ERROR) << "aclrtMemcpy UDMA registry failed: " << copyRet;
            candidateStatus = TILEXR_ERROR_INTERNAL;
        }
    }

    const int agreedCandidateStatus = AgreeStatus(candidateStatus);
    if (agreedCandidateStatus != TILEXR_SUCCESS) {
        int cleanupRet = transport_->AbortPreparedMemory();
        const int registryRet = FreeDeviceRegistry(nextRegistryDev);
        if (registryRet != TILEXR_SUCCESS) {
            RetainRegistry(nextRegistryDev);
        }
        if (cleanupRet == TILEXR_SUCCESS && registryRet != TILEXR_SUCCESS) {
            cleanupRet = registryRet;
        }
        const int agreedCleanupStatus = AgreeStatus(cleanupRet);
        if (agreedCleanupStatus != TILEXR_SUCCESS || transport_->HasMemoryCleanupPending() ||
            !retiredRegistryDevs_.empty()) {
            EnterCleanupPending("candidate registry rollback did not complete on every rank");
        }
        return agreedCandidateStatus;
    }

    TileXRUDMACommArgsState nextState {};
    nextState.available = true;
    nextState.infoDev = transport_->GetPreparedUDMAInfoDev();
    nextState.registryDev = nextRegistryDev;
    const int localPublishStatus = nextState.infoDev == nullptr
        ? TILEXR_ERROR_NOT_INITIALIZED : ApplyCommArgsState(nextState);
    const int publishStatus = AgreeStatus(localPublishStatus);
    if (publishStatus != TILEXR_SUCCESS) {
        const int localRestoreStatus = ApplyCommArgsState(previousState);
        const int restoreStatus = AgreeStatus(localRestoreStatus);
        if (restoreStatus != TILEXR_SUCCESS) {
            RetainRegistry(nextRegistryDev);
            EnterCleanupPending("candidate publication could not restore the previous comm args");
            return publishStatus;
        }

        int cleanupRet = transport_->AbortPreparedMemory();
        const int registryRet = FreeDeviceRegistry(nextRegistryDev);
        if (registryRet != TILEXR_SUCCESS) {
            RetainRegistry(nextRegistryDev);
        }
        if (cleanupRet == TILEXR_SUCCESS && registryRet != TILEXR_SUCCESS) {
            cleanupRet = registryRet;
        }
        const int agreedCleanupStatus = AgreeStatus(cleanupRet);
        if (agreedCleanupStatus != TILEXR_SUCCESS || transport_->HasMemoryCleanupPending() ||
            !retiredRegistryDevs_.empty()) {
            EnterCleanupPending("candidate rollback failed after comm args restoration");
        }
        return publishStatus;
    }

    const GM_ADDR nextInfoDev = nextState.infoDev;
    const int localCommitStatus = transport_->CommitPreparedMemory();
    const int commitStatus = AgreeStatus(localCommitStatus);
    if (commitStatus != TILEXR_SUCCESS) {
        RetainRegistry(nextRegistryDev);
        EnterCleanupPending("candidate publication could not commit transport ownership");
        return commitStatus;
    }

    GM_ADDR previousRegistryDev = udmaRegistryDev_;
    udmaRegistryDev_ = nextRegistryDev;
    nextRegistryDev = nullptr;
    RetainRegistry(previousRegistryDev);
    udmaInfoDev_ = nextInfoDev;
    registry_ = nextRegistry;
    registeredPtr_ = localPtr;
    registeredBytes_ = bytes;
    lifecycle_ = Lifecycle::MemoryReady;
    *handle = 0;

    int cleanupRet = transport_->CleanupRetiredMemory();
    const int registryCleanupRet = CleanupRetiredRegistries();
    if (cleanupRet == TILEXR_SUCCESS && registryCleanupRet != TILEXR_SUCCESS) {
        cleanupRet = registryCleanupRet;
    }
    const int agreedCleanupStatus = AgreeStatus(cleanupRet);
    if (agreedCleanupStatus != TILEXR_SUCCESS || transport_->HasMemoryCleanupPending() ||
        !retiredRegistryDevs_.empty()) {
        EnterCleanupPending("replaced registration cleanup did not complete on every rank");
        return agreedCleanupStatus == TILEXR_SUCCESS ? TILEXR_ERROR_INTERNAL : agreedCleanupStatus;
    }
    return TILEXR_SUCCESS;
}

int TileXRUDMAContext::UnregisterMemory(TileXRUDMAMemHandle handle)
{
    if (handle != 0) {
        return TILEXR_ERROR_NOT_FOUND;
    }

    if (!IsAvailable() || transport_ == nullptr) {
        return TILEXR_ERROR_NOT_FOUND;
    }

    TileXRUDMACommArgsState hiddenState {};
    hiddenState.available = true;
    hiddenState.infoDev = transport_->GetBaseUDMAInfoDev();
    hiddenState.registryDev = nullptr;
    int ret = hiddenState.infoDev == nullptr
        ? TILEXR_ERROR_NOT_INITIALIZED : ApplyCommArgsState(hiddenState);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXRUDMAUnregister failed to clear comm args: " << ret;
        return ret;
    }
    udmaInfoDev_ = hiddenState.infoDev;
    lifecycle_ = Lifecycle::CleanupPending;

    ret = transport_->CleanupAllMemory();
    const int registryRet = CleanupAllRegistries();
    if (ret == TILEXR_SUCCESS && registryRet != TILEXR_SUCCESS) {
        ret = registryRet;
    }
    if (ret == TILEXR_SUCCESS && (transport_->HasMemoryCleanupPending() ||
        udmaRegistryDev_ != nullptr || !retiredRegistryDevs_.empty())) {
        ret = TILEXR_ERROR_INTERNAL;
    }
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR UDMA memory unregistration failed: " << ret;
        return ret;
    }

    registeredPtr_ = nullptr;
    registeredBytes_ = 0;
    registry_ = TileXRUDMARegistry {};
    lifecycle_ = Lifecycle::TransportReady;
    return TILEXR_SUCCESS;
}

GM_ADDR TileXRUDMAContext::GetRegistryDev() const
{
    return lifecycle_ == Lifecycle::MemoryReady ? udmaRegistryDev_ : nullptr;
}

const TileXRUDMARegistry* TileXRUDMAContext::GetRegistryHost() const
{
    return lifecycle_ == Lifecycle::MemoryReady && UDMARegistryValid(&registry_, options_.rankSize)
        ? &registry_ : nullptr;
}

uint32_t TileXRUDMAContext::GetQpCount() const
{
    return IsAvailable() ? transport_->GetQpCount() : 0U;
}

int TileXRUDMAContext::ApplyCommArgsState(const TileXRUDMACommArgsState& state) const
{
    if (options_.updateCommArgs == nullptr) {
        return TILEXR_SUCCESS;
    }
    return options_.updateCommArgs(state, options_.updateCommArgsUserData);
}

int TileXRUDMAContext::LoadAndAgreeQpConfig(UDMAQpConfig& config) const
{
    if (options_.exchange == nullptr || options_.rankSize <= 0) {
        return TILEXR_ERROR_INTERNAL;
    }

    std::string parseError;
    const UDMAQpConfigParseStatus parseStatus = LoadUDMAQpConfigFromEnv(config, &parseError);
    if (parseStatus != UDMAQpConfigParseStatus::SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR UDMA QP route specification is invalid: " << parseError;
    }
    const UDMAQpConfigWireDescriptor localDescriptor =
        BuildUDMAQpConfigWireDescriptor(config, parseStatus);
    std::vector<UDMAQpConfigWireDescriptor> allDescriptors(options_.rankSize);
    const int exchangeRet = options_.exchange->AllGather(&localDescriptor, 1, allDescriptors.data());
    if (exchangeRet != TILEXR_SUCCESS) {
        return exchangeRet;
    }

    std::string descriptorError;
    for (int rank = 0; rank < options_.rankSize; ++rank) {
        if (!ValidateUDMAQpConfigWireDescriptor(allDescriptors[rank], &descriptorError)) {
            TILEXR_LOG(ERROR) << "TileXR UDMA QP descriptor from rank " << rank
                              << " is invalid: " << descriptorError;
            return TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        if (allDescriptors[rank].parseStatus !=
            static_cast<uint32_t>(UDMAQpConfigParseStatus::SUCCESS)) {
            TILEXR_LOG(ERROR) << "TileXR UDMA QP configuration failed on rank " << rank;
            return TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        if (!UDMAQpConfigWireDescriptorsEqual(allDescriptors[0], allDescriptors[rank])) {
            TILEXR_LOG(ERROR) << "TileXR UDMA QP configuration mismatch at rank " << rank;
            return TILEXR_ERROR_PARA_CHECK_FAIL;
        }
    }
    if (!UDMAQpConfigFromWireDescriptor(allDescriptors[0], config, &descriptorError)) {
        TILEXR_LOG(ERROR) << "TileXR UDMA agreed QP descriptor is unusable: " << descriptorError;
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return TILEXR_SUCCESS;
}

int TileXRUDMAContext::AgreeStatus(int localStatus) const
{
    if (options_.exchange == nullptr || options_.rankSize <= 0 ||
        options_.rankSize > TILEXR_MAX_RANK_SIZE) {
        return TILEXR_ERROR_INTERNAL;
    }
    int32_t local = static_cast<int32_t>(localStatus);
    std::array<int32_t, TILEXR_MAX_RANK_SIZE> allStatus {};
    allStatus.fill(TILEXR_ERROR_INTERNAL);
    const int exchangeRet = options_.exchange->AllGather(&local, 1, allStatus.data());
    if (exchangeRet != TILEXR_SUCCESS) {
        return exchangeRet;
    }
    for (int rank = 0; rank < options_.rankSize; ++rank) {
        if (allStatus[rank] != TILEXR_SUCCESS) {
            return static_cast<int>(allStatus[rank]);
        }
    }
    return TILEXR_SUCCESS;
}

int TileXRUDMAContext::FreeDeviceRegistry(GM_ADDR& registryDev) const
{
    if (registryDev == nullptr) {
        return TILEXR_SUCCESS;
    }
    const aclError ret = aclrtFree(registryDev);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "Free UDMA registry failed: " << ret
                          << ", ptr " << reinterpret_cast<uintptr_t>(registryDev);
        return TILEXR_ERROR_INTERNAL;
    }
    registryDev = nullptr;
    return TILEXR_SUCCESS;
}

int TileXRUDMAContext::CleanupRetiredRegistries()
{
    int firstError = TILEXR_SUCCESS;
    for (auto it = retiredRegistryDevs_.begin(); it != retiredRegistryDevs_.end();) {
        const int ret = FreeDeviceRegistry(*it);
        if (firstError == TILEXR_SUCCESS && ret != TILEXR_SUCCESS) {
            firstError = ret;
        }
        if (*it == nullptr) {
            it = retiredRegistryDevs_.erase(it);
        } else {
            ++it;
        }
    }
    return firstError;
}

int TileXRUDMAContext::CleanupAllRegistries()
{
    int firstError = FreeDeviceRegistry(udmaRegistryDev_);
    const int retiredRet = CleanupRetiredRegistries();
    if (firstError == TILEXR_SUCCESS && retiredRet != TILEXR_SUCCESS) {
        firstError = retiredRet;
    }
    return firstError;
}

void TileXRUDMAContext::RetainRegistry(GM_ADDR& registryDev)
{
    if (registryDev == nullptr) {
        return;
    }
    if (std::find(retiredRegistryDevs_.begin(), retiredRegistryDevs_.end(), registryDev) ==
        retiredRegistryDevs_.end()) {
        retiredRegistryDevs_.push_back(registryDev);
    }
    registryDev = nullptr;
}

void TileXRUDMAContext::EnterCleanupPending(const char* reason)
{
    lifecycle_ = Lifecycle::CleanupPending;
    if (transport_ != nullptr) {
        const GM_ADDR ownedPtr = transport_->GetRegisteredMemoryPtr();
        if (ownedPtr != nullptr) {
            registeredPtr_ = ownedPtr;
            registeredBytes_ = transport_->GetRegisteredMemoryBytes();
        }
    }

    TILEXR_LOG(ERROR) << "TileXR UDMA entered cleanup-pending state: " << reason;
    TileXRUDMACommArgsState hiddenState {};
    hiddenState.available = IsAvailable();
    hiddenState.infoDev = hiddenState.available ? udmaInfoDev_ : nullptr;
    hiddenState.registryDev = nullptr;
    const int ret = ApplyCommArgsState(hiddenState);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR UDMA failed to hide registry in cleanup-pending state: " << ret;
    }
}

} // namespace TileXR
