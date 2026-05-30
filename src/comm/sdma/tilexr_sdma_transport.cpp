/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "sdma/tilexr_sdma_transport.h"

#include <cstdlib>
#include <string>

#include "tilexr_log.h"
#include "tilexr_types.h"

namespace TileXR {

TileXRSDMATransport::~TileXRSDMATransport()
{
    Shutdown();
}

bool TileXRSDMATransport::EnvEnabled()
{
    const char* value = std::getenv("TILEXR_ENABLE_SDMA");
    if (value == nullptr) {
        return false;
    }
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

int TileXRSDMATransport::Init(const TileXRSDMATransportOptions& options)
{
    options_ = options;
    available_ = false;
    workspaceDev_ = nullptr;

    if (!EnvEnabled()) {
        lastStatus_ = SDMAInitStatus::DISABLED_BY_ENV;
        TILEXR_LOG(INFO) << "TileXR SDMA disabled; set TILEXR_ENABLE_SDMA=1 to enable";
        return TILEXR_SUCCESS;
    }

    lastStatus_ = SDMAInitStatus::PTO_UNAVAILABLE;
    TILEXR_LOG(WARN) << "TileXR SDMA PTO support is not compiled in yet";
    return TILEXR_SUCCESS;
}

void TileXRSDMATransport::Shutdown()
{
    available_ = false;
    workspaceDev_ = nullptr;
}

bool TileXRSDMATransport::IsAvailable() const
{
    return available_;
}

GM_ADDR TileXRSDMATransport::GetWorkspaceDev() const
{
    return workspaceDev_;
}

SDMAInitStatus TileXRSDMATransport::GetLastStatus() const
{
    return lastStatus_;
}

} // namespace TileXR
