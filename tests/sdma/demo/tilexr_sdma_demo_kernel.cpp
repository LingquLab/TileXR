/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

// Demo-local CANN 9.1.0 workaround: this SDMA-only kernel does not exercise
// PTO async prefetch, whose unused backend can be pulled in before SDMA helper
// symbols are visible.
#ifndef PTO_NPU_TPREFETCH_ASYNC_HPP
#define PTO_NPU_TPREFETCH_ASYNC_HPP
#endif

#include "kernel_operator.h"
#include "tilexr_sdma.h"

extern "C" __global__ __aicore__ void tilexr_sdma_copy_kernel(
    GM_ADDR commArgsGM,
    GM_ADDR dstGM,
    GM_ADDR srcGM,
    GM_ADDR debugGM,
    uint32_t bytes)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto dst = reinterpret_cast<__gm__ uint8_t*>(dstGM);
    auto src = reinterpret_cast<__gm__ uint8_t*>(srcGM);
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM);

    if ASCEND_IS_AIV {
        if (debug != nullptr) {
            debug[0] = TileXR::TILEXR_SDMA_DEMO_MAGIC;
            debug[1] = static_cast<int32_t>(AscendC::GetBlockIdx());
            debug[2] = static_cast<int32_t>(bytes);
            debug[3] = TileXR::SDMAEnabled(args) ? 1 : 0;
            debug[4] = 0;
            debug[5] = 0;
        }
        if (AscendC::GetBlockIdx() != 0) {
            return;
        }
        uint64_t event = TileXR::SDMACopyNbi(args, dst, src, static_cast<uint64_t>(bytes), 0);
        if (debug != nullptr) {
            debug[4] = event == 0 ? 0 : 1;
        }
        bool waitOk = TileXR::SDMAWait(args, event, 0);
        if (debug != nullptr) {
            debug[5] = waitOk ? 1 : 0;
        }
    }
}

extern "C" void launch_tilexr_sdma_copy(
    uint32_t blockDim,
    void* stream,
    GM_ADDR commArgs,
    GM_ADDR dst,
    GM_ADDR src,
    GM_ADDR debug,
    uint32_t bytes)
{
    tilexr_sdma_copy_kernel<<<blockDim, nullptr, stream>>>(commArgs, dst, src, debug, bytes);
}
