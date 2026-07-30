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
#include "tilexr_sdma_benchmark_types.h"

__aicore__ inline uint64_t BenchmarkSubmit(
    __gm__ TileXR::CommArgs* args,
    __gm__ uint8_t* destination,
    __gm__ uint8_t* source,
    uint32_t bytes,
    uint32_t channel,
    uint32_t batchCopies)
{
    return TileXR::SDMACopyStridedNbi(
        args, destination, source, static_cast<uint64_t>(bytes), batchCopies,
        static_cast<uint64_t>(bytes), static_cast<uint64_t>(bytes), channel);
}

extern "C" __global__ __aicore__ void tilexr_sdma_copy_kernel(
    GM_ADDR commArgsGM,
    GM_ADDR dstGM,
    GM_ADDR srcGM,
    GM_ADDR debugGM,
    uint32_t bytes,
    uint32_t firstChannel,
    uint32_t iterations)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    const uint32_t block = static_cast<uint32_t>(AscendC::GetBlockIdx());
    const uint32_t channel = firstChannel + block;
    auto dst = reinterpret_cast<__gm__ uint8_t*>(dstGM) + static_cast<uint64_t>(block) * bytes;
    auto src = reinterpret_cast<__gm__ uint8_t*>(srcGM) + static_cast<uint64_t>(block) * bytes;
    auto debug = reinterpret_cast<__gm__ int32_t*>(debugGM) + block * 16U;

    if ASCEND_IS_AIV {
        if (debug != nullptr) {
            debug[0] = TileXR::TILEXR_SDMA_DEMO_MAGIC;
            debug[1] = static_cast<int32_t>(block);
            debug[2] = static_cast<int32_t>(bytes);
            debug[3] = TileXR::SDMAEnabled(args) ? 1 : 0;
            debug[4] = 0;
            debug[5] = 0;
            debug[6] = static_cast<int32_t>(channel);
            debug[7] = 0;
            debug[8] = 0;
            debug[9] = static_cast<int32_t>(iterations);
        }
        uint64_t event = TileXR::SDMACopyNbi(
            args, dst, src, static_cast<uint64_t>(bytes), channel);
        uint64_t busyEvent = 0U;
#if TILEXR_SDMA_A5_AICORE_COMPILE
        busyEvent = TileXR::SDMACopyNbi(
            args, dst, src, static_cast<uint64_t>(bytes), channel);
#endif
        if (debug != nullptr) {
            debug[4] = event == 0 ? 0 : 1;
#if TILEXR_SDMA_A5_AICORE_COMPILE
            debug[7] = static_cast<int32_t>(event & 0xFFFFFFFFULL);
            debug[8] = busyEvent == 0 ? 1 : 0;
#else
            debug[7] = event == 0 ? 0 : 1;
            debug[8] = -1;
#endif
        }
        bool waitOk = TileXR::SDMAWait(args, event, channel);
#if TILEXR_SDMA_A5_AICORE_COMPILE
        auto workspace = reinterpret_cast<__gm__ TileXR::detail::A5SdmaWorkspace*>(
            args->sdmaWorkspacePtr);
        auto channelState = &workspace->channels[channel];
        if (debug != nullptr) {
            debug[10] = TileXR::detail::A5SdmaWorkspaceValid(workspace) ? 1 : 0;
            debug[11] = channelState->sqBase != 0U ? 1 : 0;
            debug[12] = channelState->rtsqAddress != 0U ? 1 : 0;
            debug[13] = TileXR::detail::A5SdmaQueueStateValid(
                channelState->tail, channelState->depth) ? 1 : 0;
            debug[14] = channelState->streamId <= 0xFFFFU ? 1 : 0;
            debug[15] = waitOk ? 1 : 0;
        }
#endif
        for (uint32_t iteration = 1U; iteration < iterations && waitOk; ++iteration) {
            const uint64_t nextEvent = TileXR::SDMACopyNbi(
                args, dst, src, static_cast<uint64_t>(bytes), channel);
            if (debug != nullptr) {
#if TILEXR_SDMA_A5_AICORE_COMPILE
                debug[7] = static_cast<int32_t>(nextEvent & 0xFFFFFFFFULL);
#else
                debug[7] = nextEvent == 0 ? debug[7] : static_cast<int32_t>(iteration + 1U);
#endif
            }
            waitOk = nextEvent != 0U && TileXR::SDMAWait(args, nextEvent, channel);
            event = nextEvent;
        }
        if (debug != nullptr) {
            debug[5] = waitOk ? 1 : 0;
#if TILEXR_SDMA_A5_AICORE_COMPILE
            pipe_barrier(PIPE_ALL);
            TileXR::detail::A5SdmaCleanCacheLine(
                reinterpret_cast<__gm__ uint8_t*>(debug));
#endif
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
    uint32_t bytes,
    uint32_t firstChannel,
    uint32_t iterations)
{
    tilexr_sdma_copy_kernel<<<blockDim, nullptr, stream>>>(
        commArgs, dst, src, debug, bytes, firstChannel, iterations);
}

extern "C" __global__ __aicore__ void tilexr_sdma_benchmark_kernel(
    GM_ADDR commArgsGM,
    GM_ADDR dstGM,
    GM_ADDR srcGM,
    GM_ADDR samplesGM,
    uint32_t bytes,
    uint32_t channel,
    uint32_t warmupIterations,
    uint32_t measuredIterations,
    uint32_t sampleCount,
    uint32_t phaseProfile,
    uint32_t batchCopies,
    uint32_t workingSetSlots)
{
    if ASCEND_IS_AIV {
        auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
        auto dst = reinterpret_cast<__gm__ uint8_t*>(dstGM);
        auto src = reinterpret_cast<__gm__ uint8_t*>(srcGM);
        auto samples = reinterpret_cast<__gm__ TileXR::test::SdmaBenchmarkSample*>(samplesGM);
        const uint64_t batchBytes = static_cast<uint64_t>(bytes) * batchCopies;
        uint64_t operationIndex = 0U;

        uint32_t status = TileXR::SDMAEnabled(args) && workingSetSlots != 0U
            ? TileXR::test::TILEXR_SDMA_BENCHMARK_OK
            : TileXR::test::TILEXR_SDMA_BENCHMARK_DISABLED;
        for (uint32_t iteration = 0U;
             iteration < warmupIterations && status == TileXR::test::TILEXR_SDMA_BENCHMARK_OK;
             ++iteration) {
            const uint64_t offset = (operationIndex++ % workingSetSlots) * batchBytes;
            const uint64_t event = BenchmarkSubmit(
                args, dst + offset, src + offset, bytes, channel, batchCopies);
            if (event == 0U) {
                status = TileXR::test::TILEXR_SDMA_BENCHMARK_WARMUP_SUBMIT_FAILED;
            } else if (!TileXR::SDMAWait(args, event, channel)) {
                status = TileXR::test::TILEXR_SDMA_BENCHMARK_WARMUP_WAIT_FAILED;
            }
        }

        pipe_barrier(PIPE_ALL);
        for (uint32_t sampleIndex = 0U; sampleIndex < sampleCount; ++sampleIndex) {
            uint32_t completed = 0U;
            uint64_t startCycle = 0U;
            uint64_t endCycle = 0U;
            uint64_t submitCycles = 0U;
            uint64_t waitCycles = 0U;
            uint64_t completionCycles = 0U;
            uint64_t releaseCycles = 0U;
            if (status == TileXR::test::TILEXR_SDMA_BENCHMARK_OK) {
                startCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
                if (phaseProfile != 0U) {
                    for (uint32_t iteration = 0U; iteration < measuredIterations; ++iteration) {
                        const uint64_t phaseStart =
                            static_cast<uint64_t>(AscendC::GetSystemCycle());
                        const uint64_t offset =
                            (operationIndex++ % workingSetSlots) * batchBytes;
                        const uint64_t event = BenchmarkSubmit(
                            args, dst + offset, src + offset, bytes, channel, batchCopies);
                        const uint64_t submitEnd =
                            static_cast<uint64_t>(AscendC::GetSystemCycle());
                        submitCycles += submitEnd - phaseStart;
                        if (event == 0U) {
                            status = TileXR::test::TILEXR_SDMA_BENCHMARK_SUBMIT_FAILED;
                            break;
                        }
#if TILEXR_SDMA_A5_AICORE_COMPILE
                        uint32_t eventChannel = 0U;
                        uint32_t eventGeneration = 0U;
                        if (!TileXR::detail::A5SdmaDecodeEvent(
                                event, eventChannel, eventGeneration)) {
                            status = TileXR::test::TILEXR_SDMA_BENCHMARK_WAIT_FAILED;
                            break;
                        }
                        auto workspace = reinterpret_cast<
                            __gm__ TileXR::detail::A5SdmaWorkspace*>(args->sdmaWorkspacePtr);
                        auto completion = reinterpret_cast<
                            __gm__ TileXR::detail::A5SdmaCompletionLine*>(
                                workspace->channels[eventChannel].completionRecordAddress);
                        while (TileXR::detail::A5SdmaReadCompletion(completion) !=
                               eventGeneration) {
                        }
                        const uint64_t completionEnd =
                            static_cast<uint64_t>(AscendC::GetSystemCycle());
#else
                        const uint64_t completionEnd = submitEnd;
#endif
                        const bool waitOk = TileXR::SDMAWait(args, event, channel);
                        const uint64_t waitEnd =
                            static_cast<uint64_t>(AscendC::GetSystemCycle());
                        waitCycles += waitEnd - submitEnd;
                        completionCycles += completionEnd - submitEnd;
                        releaseCycles += waitEnd - completionEnd;
                        if (!waitOk) {
                            status = TileXR::test::TILEXR_SDMA_BENCHMARK_WAIT_FAILED;
                            break;
                        }
                        completed += batchCopies;
                    }
                } else {
                    for (uint32_t iteration = 0U; iteration < measuredIterations; ++iteration) {
                        const uint64_t offset =
                            (operationIndex++ % workingSetSlots) * batchBytes;
                        const uint64_t event = BenchmarkSubmit(
                            args, dst + offset, src + offset, bytes, channel, batchCopies);
                        if (event == 0U) {
                            status = TileXR::test::TILEXR_SDMA_BENCHMARK_SUBMIT_FAILED;
                            break;
                        }
                        if (!TileXR::SDMAWait(args, event, channel)) {
                            status = TileXR::test::TILEXR_SDMA_BENCHMARK_WAIT_FAILED;
                            break;
                        }
                        completed += batchCopies;
                    }
                }
                pipe_barrier(PIPE_ALL);
                endCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
            }
            samples[sampleIndex].cycles = endCycle - startCycle;
            samples[sampleIndex].completed = completed;
            samples[sampleIndex].status = status;
            samples[sampleIndex].submitCycles = submitCycles;
            samples[sampleIndex].waitCycles = waitCycles;
            samples[sampleIndex].completionCycles = completionCycles;
            samples[sampleIndex].releaseCycles = releaseCycles;
        }

#if TILEXR_SDMA_A5_AICORE_COMPILE
        pipe_barrier(PIPE_ALL);
        for (uint32_t sampleIndex = 0U; sampleIndex < sampleCount; ++sampleIndex) {
            TileXR::detail::A5SdmaCleanCacheLine(
                reinterpret_cast<__gm__ uint8_t*>(samples + sampleIndex));
        }
        pipe_barrier(PIPE_ALL);
        dsb(DSB_DDR);
#endif
    }
}

extern "C" void launch_tilexr_sdma_benchmark(
    void* stream,
    GM_ADDR commArgs,
    GM_ADDR dst,
    GM_ADDR src,
    GM_ADDR samples,
    uint32_t bytes,
    uint32_t channel,
    uint32_t warmupIterations,
    uint32_t measuredIterations,
    uint32_t sampleCount,
    uint32_t phaseProfile,
    uint32_t batchCopies,
    uint32_t workingSetSlots)
{
    tilexr_sdma_benchmark_kernel<<<1, nullptr, stream>>>(
        commArgs, dst, src, samples, bytes, channel,
        warmupIterations, measuredIterations, sampleCount, phaseProfile,
        batchCopies, workingSetSlots);
}
