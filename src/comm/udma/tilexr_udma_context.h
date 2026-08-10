/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_CONTEXT_H
#define TILEXR_UDMA_CONTEXT_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tilexr_api.h"
#include "tilexr_types.h"
#include "tilexr_udma_reg.h"
#include "udma/tilexr_udma_config.h"

namespace TileXR {

class TileXRSockExchange;
class TileXRUDMATransport;

struct TileXRUDMACommArgsState {
    bool available = false;
    bool sharedQp = false;
    GM_ADDR infoDev = nullptr;
    GM_ADDR registryDev = nullptr;
};

using TileXRUDMACommArgsUpdateFn = int (*)(const TileXRUDMACommArgsState& state, void* userData);

struct TileXRUDMAContextOptions {
    int rank = 0;
    int rankSize = 0;
    int localRankSize = 0;
    int devId = 0;
    bool sharedQpDomain = false;
    bool threadMode = false;
    bool nonPinRegistration = false;
    TileXRSockExchange* exchange = nullptr;
    TileXRUDMACommArgsUpdateFn updateCommArgs = nullptr;
    void* updateCommArgsUserData = nullptr;
};

class TileXRUDMAContext {
public:
    TileXRUDMAContext();
    ~TileXRUDMAContext();
    TileXRUDMAContext(const TileXRUDMAContext&) = delete;
    TileXRUDMAContext& operator=(const TileXRUDMAContext&) = delete;

    int Init(const TileXRUDMAContextOptions& options);
    void Shutdown();

    bool IsAvailable() const;
    TileXRUDMACommArgsState GetCommArgsState() const;

    int RegisterMemory(GM_ADDR localPtr, size_t bytes, TileXRUDMAMemHandle* handle);
    int UnregisterMemory(TileXRUDMAMemHandle handle);

    GM_ADDR GetRegistryDev() const;
    const TileXRUDMARegistry* GetRegistryHost() const;
    uint32_t GetQpCount() const;

private:
    enum class Lifecycle {
        Unavailable,
        TransportReady,
        MemoryReady,
        CleanupPending,
    };

    int ApplyCommArgsState(const TileXRUDMACommArgsState& state) const;
    int LoadAndAgreeQpConfig(UDMAQpConfig& config) const;
    int AgreeStatus(int localStatus) const;
    int FreeDeviceRegistry(GM_ADDR& registryDev) const;
    int CleanupRetiredRegistries();
    int CleanupAllRegistries();
    void RetainRegistry(GM_ADDR& registryDev);
    void EnterCleanupPending(const char* reason);

    TileXRUDMAContextOptions options_ {};
    Lifecycle lifecycle_ = Lifecycle::Unavailable;
    GM_ADDR udmaInfoDev_ = nullptr;
    GM_ADDR udmaRegistryDev_ = nullptr;
    std::vector<GM_ADDR> retiredRegistryDevs_;
    GM_ADDR registeredPtr_ = nullptr;
    size_t registeredBytes_ = 0;
    TileXRUDMARegistry registry_ {};
    std::unique_ptr<TileXRUDMATransport> transport_;
};

} // namespace TileXR

#endif // TILEXR_UDMA_CONTEXT_H
