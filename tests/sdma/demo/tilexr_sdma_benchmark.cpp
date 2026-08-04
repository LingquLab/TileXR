/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "tilexr_api.h"
#include "tilexr_sdma_a5_types.h"
#include "tilexr_sdma_benchmark_types.h"
#include "tilexr_types.h"

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
    uint32_t workingSetSlots);

namespace {
constexpr uint32_t kDefaultBytes = 4096U;
constexpr uint32_t kDefaultWarmup = 20U;
constexpr uint32_t kDefaultIterations = 100U;
constexpr uint32_t kDefaultSamples = 10U;
constexpr uint32_t kAlignmentBytes = 64U;
constexpr uint32_t kMaxSamples = 100U;
constexpr uint32_t kMaxBatchCopies = 32U;
constexpr size_t kMinWorkingSetBytes = 64U * 1024U * 1024U;
constexpr int kDeviceId = 0;
constexpr double kAscend950CyclesPerUs = 1000.0;

bool CheckAcl(const std::string& label, aclError ret)
{
    if (ret == ACL_SUCCESS) {
        return true;
    }
    std::cerr << "ERROR: " << label << " failed with " << ret << std::endl;
    return false;
}

bool CheckTileXR(const std::string& label, int ret)
{
    if (ret == TileXR::TILEXR_SUCCESS) {
        return true;
    }
    std::cerr << "ERROR: " << label << " failed with " << ret << std::endl;
    return false;
}

bool ParseUint32(const char* text, uint32_t* value)
{
    if (text == nullptr || value == nullptr || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    *value = static_cast<uint32_t>(parsed);
    return true;
}

void FillPattern(std::vector<uint8_t>& data)
{
    for (size_t index = 0U; index < data.size(); ++index) {
        data[index] = static_cast<uint8_t>((index * 37U + 11U) & 0xFFU);
    }
}

void Cleanup(uint8_t* src,
             uint8_t* dst,
             TileXR::test::SdmaBenchmarkSample* samples,
             TileXRCommPtr comm,
             aclrtStream stream,
             bool deviceSet,
             bool aclInitialized)
{
    if (src != nullptr) {
        (void)aclrtFree(src);
    }
    if (dst != nullptr) {
        (void)aclrtFree(dst);
    }
    if (samples != nullptr) {
        (void)aclrtFree(samples);
    }
    if (comm != nullptr) {
        (void)TileXRCommDestroy(comm);
    }
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        (void)aclrtResetDevice(kDeviceId);
    }
    if (aclInitialized) {
        (void)aclFinalize();
    }
}

const char* StatusName(uint32_t status)
{
    switch (status) {
        case TileXR::test::TILEXR_SDMA_BENCHMARK_OK:
            return "ok";
        case TileXR::test::TILEXR_SDMA_BENCHMARK_DISABLED:
            return "sdma_disabled";
        case TileXR::test::TILEXR_SDMA_BENCHMARK_WARMUP_SUBMIT_FAILED:
            return "warmup_submit_failed";
        case TileXR::test::TILEXR_SDMA_BENCHMARK_WARMUP_WAIT_FAILED:
            return "warmup_wait_failed";
        case TileXR::test::TILEXR_SDMA_BENCHMARK_SUBMIT_FAILED:
            return "submit_failed";
        case TileXR::test::TILEXR_SDMA_BENCHMARK_WAIT_FAILED:
            return "wait_failed";
        default:
            return "unknown";
    }
}
} // namespace

int main(int argc, char** argv)
{
    uint32_t bytes = kDefaultBytes;
    uint32_t warmupIterations = kDefaultWarmup;
    uint32_t measuredIterations = kDefaultIterations;
    uint32_t sampleCount = kDefaultSamples;
    uint32_t channel = 0U;
    uint32_t phaseProfile = 0U;
    uint32_t batchCopies = 1U;
    if (argc > 8 ||
        (argc >= 2 && !ParseUint32(argv[1], &bytes)) ||
        (argc >= 3 && !ParseUint32(argv[2], &warmupIterations)) ||
        (argc >= 4 && !ParseUint32(argv[3], &measuredIterations)) ||
        (argc >= 5 && !ParseUint32(argv[4], &sampleCount)) ||
        (argc >= 6 && !ParseUint32(argv[5], &channel)) ||
        (argc >= 7 && !ParseUint32(argv[6], &phaseProfile)) ||
        (argc == 8 && !ParseUint32(argv[7], &batchCopies)) ||
        bytes == 0U || (bytes % kAlignmentBytes) != 0U || measuredIterations == 0U ||
        sampleCount == 0U || sampleCount > kMaxSamples ||
        channel >= TileXR::detail::TILEXR_SDMA_A5_CHANNEL_COUNT || phaseProfile > 1U ||
        batchCopies == 0U || batchCopies > kMaxBatchCopies ||
        measuredIterations > std::numeric_limits<uint32_t>::max() / batchCopies ||
        static_cast<uint64_t>(bytes) * batchCopies > std::numeric_limits<size_t>::max()) {
        std::cerr << "ERROR: usage: tilexr_sdma_benchmark "
                     "[aligned-bytes [warmup [iterations [samples [channel "
                     "[phase-profile [batch-copies]]]]]]]"
                  << std::endl;
        return 1;
    }

    (void)setenv("TILEXR_ENABLE_SDMA", "1", 1);
    bool aclInitialized = false;
    bool deviceSet = false;
    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    uint8_t* src = nullptr;
    uint8_t* dst = nullptr;
    TileXR::test::SdmaBenchmarkSample* deviceSamples = nullptr;

    if (!CheckAcl("aclInit", aclInit(nullptr))) {
        return 1;
    }
    aclInitialized = true;
    if (!CheckAcl("aclrtSetDevice", aclrtSetDevice(kDeviceId))) {
        Cleanup(src, dst, deviceSamples, comm, stream, deviceSet, aclInitialized);
        return 1;
    }
    deviceSet = true;
    const char* socName = aclrtGetSocName();
    if (socName == nullptr || std::strstr(socName, "Ascend950") == nullptr) {
        std::cerr << "ERROR: device-cycle conversion is only defined here for Ascend950, got "
                  << (socName == nullptr ? "<null>" : socName) << std::endl;
        Cleanup(src, dst, deviceSamples, comm, stream, deviceSet, aclInitialized);
        return 1;
    }
    if (!CheckAcl("aclrtCreateStream", aclrtCreateStream(&stream)) ||
        !CheckTileXR("TileXRCommInitRankLocal", TileXRCommInitRankLocal(1, 0, &comm))) {
        Cleanup(src, dst, deviceSamples, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    bool sdmaAvailable = false;
    GM_ADDR sdmaWorkspace = nullptr;
    GM_ADDR commArgsDev = nullptr;
    if (!CheckTileXR("TileXRSDMAAvailable", TileXRSDMAAvailable(comm, &sdmaAvailable)) ||
        !CheckTileXR("TileXRGetSDMAWorkspaceDev", TileXRGetSDMAWorkspaceDev(comm, &sdmaWorkspace)) ||
        !CheckTileXR("TileXRGetCommArgsDev", TileXRGetCommArgsDev(comm, commArgsDev)) ||
        !sdmaAvailable || sdmaWorkspace == nullptr || commArgsDev == nullptr) {
        std::cerr << "ERROR: TileXR direct SDMA is unavailable" << std::endl;
        Cleanup(src, dst, deviceSamples, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    const size_t transferBytes = static_cast<size_t>(bytes) * batchCopies;
    const size_t workingSetSlots = std::max<size_t>(1U, kMinWorkingSetBytes / transferBytes);
    const size_t allocationBytes = transferBytes * workingSetSlots;
    const size_t sampleBytes = static_cast<size_t>(sampleCount) *
        sizeof(TileXR::test::SdmaBenchmarkSample);
    if (!CheckAcl("aclrtMalloc src", aclrtMalloc(
            reinterpret_cast<void**>(&src), allocationBytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl("aclrtMalloc dst", aclrtMalloc(
            reinterpret_cast<void**>(&dst), allocationBytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl("aclrtMalloc samples", aclrtMalloc(
            reinterpret_cast<void**>(&deviceSamples), sampleBytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
        Cleanup(src, dst, deviceSamples, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    std::vector<uint8_t> hostSrc(allocationBytes);
    std::vector<uint8_t> hostDst(allocationBytes, 0U);
    std::vector<TileXR::test::SdmaBenchmarkSample> hostSamples(sampleCount);
    FillPattern(hostSrc);
    if (!CheckAcl("aclrtMemcpy H2D src", aclrtMemcpy(
            src, allocationBytes, hostSrc.data(), hostSrc.size(), ACL_MEMCPY_HOST_TO_DEVICE)) ||
        !CheckAcl("aclrtMemcpy H2D dst", aclrtMemcpy(
            dst, allocationBytes, hostDst.data(), hostDst.size(), ACL_MEMCPY_HOST_TO_DEVICE)) ||
        !CheckAcl("aclrtMemcpy H2D samples", aclrtMemcpy(
            deviceSamples, sampleBytes, hostSamples.data(), sampleBytes, ACL_MEMCPY_HOST_TO_DEVICE))) {
        Cleanup(src, dst, deviceSamples, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    launch_tilexr_sdma_benchmark(
        stream, commArgsDev, reinterpret_cast<GM_ADDR>(dst), reinterpret_cast<GM_ADDR>(src),
        reinterpret_cast<GM_ADDR>(deviceSamples), bytes, channel,
        warmupIterations, measuredIterations, sampleCount, phaseProfile, batchCopies,
        static_cast<uint32_t>(workingSetSlots));
    if (!CheckAcl("aclrtSynchronizeStream", aclrtSynchronizeStream(stream)) ||
        !CheckAcl("aclrtMemcpy D2H dst", aclrtMemcpy(
            hostDst.data(), hostDst.size(), dst, allocationBytes, ACL_MEMCPY_DEVICE_TO_HOST)) ||
        !CheckAcl("aclrtMemcpy D2H samples", aclrtMemcpy(
            hostSamples.data(), sampleBytes, deviceSamples, sampleBytes, ACL_MEMCPY_DEVICE_TO_HOST))) {
        Cleanup(src, dst, deviceSamples, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    const uint64_t submittedBatches = static_cast<uint64_t>(warmupIterations) +
        static_cast<uint64_t>(measuredIterations) * sampleCount;
    const size_t touchedSlots = static_cast<size_t>(std::min<uint64_t>(
        static_cast<uint64_t>(workingSetSlots), submittedBatches));
    const size_t touchedBytes = touchedSlots * transferBytes;
    bool valid = std::equal(hostDst.begin(), hostDst.begin() + touchedBytes, hostSrc.begin()) &&
        std::all_of(hostDst.begin() + touchedBytes, hostDst.end(),
                    [](uint8_t value) { return value == 0U; });
    std::vector<double> cyclesPerCopy;
    cyclesPerCopy.reserve(sampleCount);
    const uint32_t expectedCopies = measuredIterations * batchCopies;
    for (uint32_t index = 0U; index < sampleCount; ++index) {
        const auto& sample = hostSamples[index];
        if (sample.status != TileXR::test::TILEXR_SDMA_BENCHMARK_OK ||
            sample.completed != expectedCopies || sample.cycles == 0U) {
            std::cerr << "ERROR: sample " << index << " status=" << StatusName(sample.status)
                      << " completed=" << sample.completed << "/" << expectedCopies
                      << " cycles=" << sample.cycles << std::endl;
            valid = false;
            continue;
        }
        cyclesPerCopy.push_back(
            static_cast<double>(sample.cycles) / static_cast<double>(expectedCopies));
    }
    if (!valid || cyclesPerCopy.size() != sampleCount) {
        Cleanup(src, dst, deviceSamples, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    const double meanCycles = std::accumulate(cyclesPerCopy.begin(), cyclesPerCopy.end(), 0.0) /
        static_cast<double>(cyclesPerCopy.size());
    double squaredDifferenceSum = 0.0;
    for (double cycles : cyclesPerCopy) {
        const double difference = cycles - meanCycles;
        squaredDifferenceSum += difference * difference;
    }
    const double stddevCycles = std::sqrt(
        squaredDifferenceSum / static_cast<double>(cyclesPerCopy.size()));
    const auto minmaxCycles = std::minmax_element(cyclesPerCopy.begin(), cyclesPerCopy.end());
    const double meanUs = meanCycles / kAscend950CyclesPerUs;
    const double stddevUs = stddevCycles / kAscend950CyclesPerUs;
    const double minUs = *minmaxCycles.first / kAscend950CyclesPerUs;
    const double maxUs = *minmaxCycles.second / kAscend950CyclesPerUs;
    const double bandwidthGBps = static_cast<double>(bytes) / meanUs / 1000.0;
    const double bandwidthGiBps = static_cast<double>(bytes) / meanUs * 1000000.0 /
        (1024.0 * 1024.0 * 1024.0);
    double meanSubmitUs = 0.0;
    double meanWaitUs = 0.0;
    double meanCompletionUs = 0.0;
    double meanReleaseUs = 0.0;
    if (phaseProfile != 0U) {
        for (const auto& sample : hostSamples) {
            meanSubmitUs += static_cast<double>(sample.submitCycles);
            meanWaitUs += static_cast<double>(sample.waitCycles);
            meanCompletionUs += static_cast<double>(sample.completionCycles);
            meanReleaseUs += static_cast<double>(sample.releaseCycles);
        }
        const double phaseDivisor = static_cast<double>(sampleCount) *
            static_cast<double>(expectedCopies) * kAscend950CyclesPerUs;
        meanSubmitUs /= phaseDivisor;
        meanWaitUs /= phaseDivisor;
        meanCompletionUs /= phaseDivisor;
        meanReleaseUs /= phaseDivisor;
    }

    std::cout << std::fixed << std::setprecision(6);
    for (uint32_t index = 0U; index < sampleCount; ++index) {
        std::cout << "SAMPLE index=" << index
                  << " total_cycles=" << hostSamples[index].cycles
                  << " cycles_per_copy=" << cyclesPerCopy[index]
                  << " latency_us=" << cyclesPerCopy[index] / kAscend950CyclesPerUs << std::endl;
    }
    std::cout << "RESULT soc=" << socName
              << " bytes=" << bytes
              << " channel=" << channel
              << " warmup=" << warmupIterations
              << " batches=" << measuredIterations
              << " batch_copies=" << batchCopies
              << " copies=" << expectedCopies
              << " allocation_bytes=" << allocationBytes
              << " working_set_slots=" << workingSetSlots
              << " verified_slots=" << touchedSlots
              << " samples=" << sampleCount
              << " mean_cycles=" << meanCycles
              << " mean_us=" << meanUs
              << " stddev_us=" << stddevUs
              << " min_us=" << minUs
              << " max_us=" << maxUs
              << " bandwidth_GBps=" << bandwidthGBps
              << " bandwidth_GiBps=" << bandwidthGiBps << std::endl;
    if (phaseProfile != 0U) {
        std::cout << "PHASE submit_us=" << meanSubmitUs
                  << " wait_us=" << meanWaitUs
                  << " completion_us=" << meanCompletionUs
                  << " release_us=" << meanReleaseUs
                  << " combined_us=" << (meanSubmitUs + meanWaitUs) << std::endl;
    }
    std::cout << "PASS direct SDMA benchmark data verification" << std::endl;

    Cleanup(src, dst, deviceSamples, comm, stream, deviceSet, aclInitialized);
    return 0;
}
