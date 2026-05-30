/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_TRANSPORT_H
#define TILEXR_SDMA_TRANSPORT_H

#include <memory>

#include "comm_args.h"
#include "tilexr_sdma_types.h"

namespace TileXR {

struct TileXRSDMATransportOptions {
    int devId = 0;
};

class TileXRSDMATransport {
public:
    TileXRSDMATransport();
    ~TileXRSDMATransport();
    TileXRSDMATransport(const TileXRSDMATransport&) = delete;
    TileXRSDMATransport& operator=(const TileXRSDMATransport&) = delete;

    int Init(const TileXRSDMATransportOptions& options);
    void Shutdown();

    bool IsAvailable() const;
    GM_ADDR GetWorkspaceDev() const;
    SDMAInitStatus GetLastStatus() const;

private:
    struct Impl;

    static bool EnvEnabled();

    TileXRSDMATransportOptions options_ {};
    bool available_ = false;
    GM_ADDR workspaceDev_ = nullptr;
    SDMAInitStatus lastStatus_ = SDMAInitStatus::DISABLED_BY_ENV;
    std::unique_ptr<Impl> impl_;
};

} // namespace TileXR

#endif // TILEXR_SDMA_TRANSPORT_H
