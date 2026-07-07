/*
 * Copyright (c) 2026 TileXR Project
 */
#include "ccu/tilexr_ccu_backend.h"

namespace TileXR {

TileXRCcuBackend::TileXRCcuBackend() = default;

TileXRCcuBackend::~TileXRCcuBackend()
{
    Shutdown();
}

int TileXRCcuBackend::Init(const TileXRCcuBackendOptions &options)
{
    options_ = options;
    initialized_ = true;
    return TILEXR_SUCCESS;
}

void TileXRCcuBackend::Shutdown()
{
    initialized_ = false;
}

bool TileXRCcuBackend::Available() const
{
    return initialized_;
}

bool TileXRCcuBackend::Supports(const TileXRCcuCollectiveRequest &request) const
{
    return initialized_ && request.type == TileXRType::ALL_GATHER;
}

int TileXRCcuBackend::PrepareCollective(const TileXRCcuCollectiveRequest &request, TileXRCcuCollectivePlan *plan)
{
    if (plan == nullptr) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!initialized_) {
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    if (!Supports(request)) {
        return TILEXR_ERROR_NOT_SUPPORT;
    }
    *plan = TileXRCcuCollectivePlan {};
    plan->ready = true;
    return TILEXR_SUCCESS;
}

int TileXRCcuBackend::SubmitCollective(const TileXRCcuCollectivePlan &plan, aclrtStream)
{
    if (!initialized_) {
        return TILEXR_ERROR_NOT_INITIALIZED;
    }
    return plan.ready ? TILEXR_SUCCESS : TILEXR_ERROR_PARA_CHECK_FAIL;
}

} // namespace TileXR
