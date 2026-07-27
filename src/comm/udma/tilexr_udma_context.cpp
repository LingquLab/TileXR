/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_context.h"

#include <acl/acl_rt.h>

#include <algorithm>
#include <mutex>
#include <new>
#include <vector>

#include "tilexr_api.h"
#include "tilexr_log.h"
#include "tools/socket/tilexr_sock_exchange.h"
#include "udma/tilexr_udma_transport.h"

namespace TileXR {

namespace {

std::mutex g_udmaMtx;
bool g_udmaUnavailable = false;

} // namespace

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

    {
        std::lock_guard<std::mutex> lock(g_udmaMtx);
        if (g_udmaUnavailable) {
            TILEXR_LOG(INFO) << "InitUDMA skipped after previous UDMA init failure";
            return TILEXR_SUCCESS;
        }
    }

    transport_.reset(new (std::nothrow) TileXRUDMATransport());
    if (transport_ == nullptr) {
        TILEXR_LOG(WARN) << "TileXRUDMATransport allocation failed, UDMA disabled";
        return TILEXR_SUCCESS;
    }

    TileXRUDMATransportOptions transportOptions {};
    transportOptions.rank = options_.rank;
    transportOptions.rankSize = options_.rankSize;
    transportOptions.devId = options_.devId;
    transportOptions.exchange = options_.exchange;
    int ret = transport_->Init(transportOptions);
    if (ret != TILEXR_SUCCESS || !transport_->IsAvailable()) {
        TILEXR_LOG(WARN) << "TileXR UDMA init failed: " << ret << ", UDMA disabled";
        {
            std::lock_guard<std::mutex> lock(g_udmaMtx);
            g_udmaUnavailable = true;
        }
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

    available_ = true;
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
    registeredPtr_ = nullptr;
    registeredBytes_ = 0;
    FreeRegistry();

    if (transport_ != nullptr) {
        transport_->Shutdown();
        transport_.reset();
    }
    udmaInfoDev_ = nullptr;
    available_ = false;
}

bool TileXRUDMAContext::IsAvailable() const
{
    return available_ && udmaInfoDev_ != nullptr && transport_ != nullptr && transport_->IsAvailable();
}

TileXRUDMACommArgsState TileXRUDMAContext::GetCommArgsState() const
{
    TileXRUDMACommArgsState state {};
    state.available = IsAvailable();
    state.infoDev = state.available ? udmaInfoDev_ : nullptr;
    state.registryDev = state.available ? udmaRegistryDev_ : nullptr;
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

    const GM_ADDR previousRegisteredPtr = registeredPtr_;
    const size_t previousRegisteredBytes = registeredBytes_;
    const GM_ADDR previousRegistryDev = udmaRegistryDev_;
    const TileXRUDMARegistry previousRegistry = registry_;

    int ret = transport_->RegisterMemory(localPtr, bytes);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR UDMA memory registration failed: " << ret;
        if (previousRegisteredPtr != nullptr) {
            (void)RestoreTransportRegistration(previousRegisteredPtr, previousRegisteredBytes);
        }
        return TILEXR_ERROR_INTERNAL;
    }

    TileXRUDMARegionDesc localRegion {};
    localRegion.base = localPtr;
    localRegion.bytes = bytes;
    std::vector<TileXRUDMARegionDesc> allRegions(options_.rankSize);
    ret = options_.exchange->AllGather(&localRegion, 1, allRegions.data());
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXRUDMARegister allgather failed: " << ret;
        (void)RollbackTransportRegistration(localPtr, previousRegisteredPtr, previousRegisteredBytes);
        return ret;
    }

    TileXRUDMARegistry nextRegistry {};
    nextRegistry.rankSize = static_cast<uint32_t>(options_.rankSize);
    nextRegistry.regionCount = 1;
    for (int i = 0; i < options_.rankSize; ++i) {
        if (allRegions[i].base == nullptr || allRegions[i].bytes == 0) {
            TILEXR_LOG(ERROR) << "TileXRUDMARegister received invalid region from rank " << i;
            (void)RollbackTransportRegistration(localPtr, previousRegisteredPtr, previousRegisteredBytes);
            return TILEXR_ERROR_PARA_CHECK_FAIL;
        }
        nextRegistry.regions[i] = allRegions[i];
    }

    GM_ADDR nextRegistryDev = nullptr;
    ret = aclrtMalloc(reinterpret_cast<void**>(&nextRegistryDev), sizeof(nextRegistry), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMalloc UDMA registry failed: " << ret;
        (void)RollbackTransportRegistration(localPtr, previousRegisteredPtr, previousRegisteredBytes);
        return TILEXR_ERROR_INTERNAL;
    }

    ret = aclrtMemcpy(
        nextRegistryDev, sizeof(nextRegistry), &nextRegistry, sizeof(nextRegistry), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "aclrtMemcpy UDMA registry failed: " << ret;
        FreeDeviceRegistry(nextRegistryDev);
        (void)RollbackTransportRegistration(localPtr, previousRegisteredPtr, previousRegisteredBytes);
        return TILEXR_ERROR_INTERNAL;
    }

    TileXRUDMACommArgsState nextState {};
    nextState.available = true;
    nextState.infoDev = udmaInfoDev_;
    nextState.registryDev = nextRegistryDev;
    ret = ApplyCommArgsState(nextState);
    if (ret != TILEXR_SUCCESS) {
        FreeDeviceRegistry(nextRegistryDev);
        (void)RollbackTransportRegistration(localPtr, previousRegisteredPtr, previousRegisteredBytes);
        return ret;
    }

    udmaRegistryDev_ = nextRegistryDev;
    registry_ = nextRegistry;
    registeredPtr_ = localPtr;
    registeredBytes_ = bytes;
    *handle = 0;

    GM_ADDR oldRegistryDev = previousRegistryDev;
    FreeDeviceRegistry(oldRegistryDev);
    (void)previousRegistry;
    return TILEXR_SUCCESS;
}

int TileXRUDMAContext::UnregisterMemory(TileXRUDMAMemHandle handle)
{
    if (handle != 0) {
        return TILEXR_ERROR_NOT_FOUND;
    }

    TileXRUDMACommArgsState nextState {};
    nextState.available = IsAvailable();
    nextState.infoDev = nextState.available ? udmaInfoDev_ : nullptr;
    nextState.registryDev = nullptr;
    int ret = ApplyCommArgsState(nextState);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXRUDMAUnregister failed to clear comm args: " << ret;
        return ret;
    }

    if (registeredPtr_ != nullptr && transport_ != nullptr) {
        ret = transport_->UnregisterMemory(registeredPtr_);
        if (ret != TILEXR_SUCCESS) {
            TILEXR_LOG(ERROR) << "TileXR UDMA memory unregistration failed: " << ret;
        }
    }
    registeredPtr_ = nullptr;
    registeredBytes_ = 0;
    FreeRegistry();
    return TILEXR_SUCCESS;
}

GM_ADDR TileXRUDMAContext::GetRegistryDev() const
{
    return udmaRegistryDev_;
}

const TileXRUDMARegistry* TileXRUDMAContext::GetRegistryHost() const
{
    return UDMARegistryValid(&registry_, options_.rankSize) ? &registry_ : nullptr;
}

int TileXRUDMAContext::ApplyCommArgsState(const TileXRUDMACommArgsState& state) const
{
    if (options_.updateCommArgs == nullptr) {
        return TILEXR_SUCCESS;
    }
    return options_.updateCommArgs(state, options_.updateCommArgsUserData);
}

int TileXRUDMAContext::RollbackTransportRegistration(
    GM_ADDR localPtr, GM_ADDR previousRegisteredPtr, size_t previousRegisteredBytes) const
{
    if (previousRegisteredPtr != nullptr) {
        return RestoreTransportRegistration(previousRegisteredPtr, previousRegisteredBytes);
    }
    if (transport_ == nullptr || localPtr == nullptr) {
        TILEXR_LOG(ERROR) << "TileXR UDMA rollback registration called with invalid state";
        return TILEXR_ERROR_NOT_FOUND;
    }
    const int ret = transport_->UnregisterMemory(localPtr);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR UDMA failed to roll back local registration: " << ret;
    }
    return ret;
}

int TileXRUDMAContext::RestoreTransportRegistration(GM_ADDR localPtr, size_t bytes) const
{
    if (transport_ == nullptr || localPtr == nullptr || bytes == 0) {
        TILEXR_LOG(ERROR) << "TileXR UDMA restore registration called with invalid state";
        return TILEXR_ERROR_NOT_FOUND;
    }
    const int ret = transport_->RegisterMemory(localPtr, bytes);
    if (ret != TILEXR_SUCCESS) {
        TILEXR_LOG(ERROR) << "TileXR UDMA failed to restore previous registration: " << ret;
    }
    return ret;
}

void TileXRUDMAContext::FreeRegistry()
{
    FreeDeviceRegistry(udmaRegistryDev_);
    registry_ = TileXRUDMARegistry {};
}

void TileXRUDMAContext::FreeDeviceRegistry(GM_ADDR& registryDev) const
{
    if (registryDev == nullptr) {
        return;
    }
    aclError ret = aclrtFree(registryDev);
    if (ret != ACL_SUCCESS) {
        TILEXR_LOG(ERROR) << "Free UDMA registry failed: " << ret;
    }
    registryDev = nullptr;
}

} // namespace TileXR
