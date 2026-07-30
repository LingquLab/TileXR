/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_SDMA_BENCHMARK_TYPES_H
#define TILEXR_SDMA_BENCHMARK_TYPES_H

#include <cstdint>

namespace TileXR {
namespace test {

constexpr uint32_t TILEXR_SDMA_BENCHMARK_OK = 0U;
constexpr uint32_t TILEXR_SDMA_BENCHMARK_DISABLED = 1U;
constexpr uint32_t TILEXR_SDMA_BENCHMARK_WARMUP_SUBMIT_FAILED = 2U;
constexpr uint32_t TILEXR_SDMA_BENCHMARK_WARMUP_WAIT_FAILED = 3U;
constexpr uint32_t TILEXR_SDMA_BENCHMARK_SUBMIT_FAILED = 4U;
constexpr uint32_t TILEXR_SDMA_BENCHMARK_WAIT_FAILED = 5U;

struct alignas(64) SdmaBenchmarkSample {
    uint64_t cycles;
    uint32_t completed;
    uint32_t status;
    uint64_t submitCycles;
    uint64_t waitCycles;
    uint64_t completionCycles;
    uint64_t releaseCycles;
    uint64_t reserved[2];
};

static_assert(sizeof(SdmaBenchmarkSample) == 64U,
              "SDMA benchmark sample must occupy one cache line");

} // namespace test
} // namespace TileXR

#endif // TILEXR_SDMA_BENCHMARK_TYPES_H
