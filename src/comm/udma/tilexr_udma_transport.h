/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_TRANSPORT_H
#define TILEXR_UDMA_TRANSPORT_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "comm_args.h"
#include "tilexr_api.h"
#include "tilexr_types.h"
#include "tilexr_udma_types.h"
#include "udma/tilexr_hccp_defs.h"
#include "udma/tilexr_hccp_loader.h"
#include "udma/tilexr_udma_layout.h"

namespace TileXR {

class TileXRSockExchange;

struct TileXRUDMATransportOptions {
    int rank = 0;
    int rankSize = 0;
    int devId = 0;
    TileXRSockExchange* exchange = nullptr;
};

class TileXRUDMATransport {
public:
    TileXRUDMATransport();
    ~TileXRUDMATransport();
    TileXRUDMATransport(const TileXRUDMATransport&) = delete;
    TileXRUDMATransport& operator=(const TileXRUDMATransport&) = delete;

    int Init(const TileXRUDMATransportOptions& options);
    int RegisterMemory(GM_ADDR localPtr, size_t bytes);
    int UnregisterMemory(GM_ADDR localPtr);
    void Shutdown();

    bool IsAvailable() const;
    GM_ADDR GetUDMAInfoDev() const;

private:
    struct PerEidState;

    int OpenDevice();
    int BuildRoutes();
    int CreateContexts();
    int CreateQueues();
    int ImportQueues();
    int RefreshUDMAInfo();
    int RegisterMemoryOnContexts(GM_ADDR localPtr, size_t bytes);
    int ExchangeAndImportMemory();
    int AllocDeviceScalar(void** ptr, size_t bytes) const;
    void FreeDeviceScalar(void*& ptr) const;
    void CleanupQueues();
    void CleanupMemory();
    void CleanupContexts();
    uint32_t FallbackLocalEid() const;

    TileXRHccpLoader loader_;
    TileXRUDMATransportOptions options_ {};
    bool available_ = false;
    bool tsdOpened_ = false;
    bool raInitialized_ = false;
    pid_t subPid_ = 0;
    uint32_t logicDevId_ = 0;
    uint32_t deviceIdOffset_ = 0;
    uint32_t eidCount_ = 0;
    uint32_t qpNum_ = 1;
    std::map<uint32_t, void*> ctxHandleByEid_;
    std::map<uint32_t, void*> tokenHandleByEid_;
    std::map<int, uint32_t> peerLocalEid_;
    std::map<int, uint32_t> peerRemoteEid_;
    std::map<uint32_t, PerEidState> states_;
    std::map<uint32_t, HccpEid> localEidByEid_;
    MemoryRegionMap registeredMem_;
    std::vector<void*> remoteMemHandles_;
    std::map<uint32_t, UDMAMemInfo> localMemInfoByEid_;
    GM_ADDR udmaInfoDev_ = nullptr;
    GM_ADDR eidTableDev_ = nullptr;
    uint32_t udmaInfoSize_ = 0;
    GM_ADDR registeredPtr_ = nullptr;
};

} // namespace TileXR

#endif // TILEXR_UDMA_TRANSPORT_H
