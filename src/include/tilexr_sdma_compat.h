/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_COMPAT_H
#define TILEXR_SDMA_COMPAT_H

#include "comm_args.h"
#include "tilexr_sdma_types.h"

// Preserve installed generated config when available, while allowing callers to
// override TILEXR_HAVE_PTO_SDMA before including this header.
#if defined(__has_include)
#if __has_include("tilexr_sdma_config.h")
#include "tilexr_sdma_config.h"
#endif
#endif

#ifndef TILEXR_HAVE_PTO_SDMA
#define TILEXR_HAVE_PTO_SDMA 0
#endif

#if TILEXR_ASCENDC_AICORE_COMPILE && defined(TILEXR_HAVE_PTO_SDMA) && TILEXR_HAVE_PTO_SDMA

#ifndef PTO_COMM_NOT_SUPPORTED
#define PTO_COMM_NOT_SUPPORTED 1
#endif

#include "pto/npu/comm/async/sdma/sdma_async_intrin.hpp"

namespace TileXR {
namespace detail {

using SDMAScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, TILEXR_SDMA_SCRATCH_BYTES>;

__aicore__ inline bool TileXRSDMABuildSession(
    SDMAScratchTile& scratch,
    __gm__ uint8_t* workspace,
    pto::comm::sdma::SdmaSession& session,
    uint32_t channelGroupIdx)
{
    pto::TASSIGN(scratch, 0);
    pto::comm::sdma::SdmaBaseConfig config {
        TILEXR_SDMA_DEFAULT_BLOCK_BYTES,
        TILEXR_SDMA_DEFAULT_COMM_BLOCK_OFFSET,
        TILEXR_SDMA_DEFAULT_QUEUE_NUM
    };
    return pto::comm::sdma::BuildSdmaSession(scratch, workspace, session, 0, config, channelGroupIdx);
}

__aicore__ inline uint64_t TileXRSDMAPostPut(
    __gm__ uint8_t* dst,
    __gm__ uint8_t* src,
    uint64_t bytes,
    const pto::comm::sdma::SdmaSession& session)
{
    return pto::comm::sdma::__sdma_put_async(dst, src, bytes, session.execCtx);
}

__aicore__ inline bool TileXRSDMAWait(uint64_t eventHandle, const pto::comm::sdma::SdmaSession& session)
{
    return pto::comm::sdma::detail::SdmaWaitEvent(eventHandle, session);
}

} // namespace detail
} // namespace TileXR

#endif

#endif // TILEXR_SDMA_COMPAT_H
