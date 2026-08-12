/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "runtime/kernel.h"
#include "tilexr_api.h"
#include "tilexr_types.h"

extern "C" {
extern const unsigned char TileXRUDMAProfileProbeKernelBinaryData[];
extern const std::size_t TileXRUDMAProfileProbeKernelBinarySize;
}

namespace {

constexpr uintptr_t kKernelSignatureValue = UINT64_C(0x5458505200001000);
constexpr const char* kKernelName = "tilexr_udma_profile_probe_kernel";
constexpr uint64_t kProbeMagic = UINT64_C(0x5458505250524f42);
constexpr size_t kAlignment = 2U * 1024U * 1024U;
constexpr uint64_t kMaxTransferBytes = 16U * 1024U * 1024U;
constexpr uint32_t kBatchCount = 4U;
constexpr size_t kProfileRegionBytes =
    static_cast<size_t>(kMaxTransferBytes) * kBatchCount;
constexpr size_t kLegacyTransferBytes = 256U * 1024U;
constexpr size_t kLegacyRegionBytes = 2U * kLegacyTransferBytes;
constexpr size_t kLegacyDestinationOffset = kLegacyTransferBytes;
constexpr uint32_t kProfileRegionCount = 4U;
constexpr uint32_t kMinimumQpCount = 3U;
constexpr uint32_t kStatusWords = 11U;
constexpr uint32_t kModeProfileTransfer = 0U;
constexpr uint32_t kModeProfileConsume = 1U;
constexpr uint32_t kModeLegacyGet = 2U;
constexpr uint32_t kLegacyPatternRegion = 7U;

constexpr uint64_t kTransferSizes[] = {
    48U * 1024U,
    256U * 1024U,
    1U * 1024U * 1024U,
    2U * 1024U * 1024U,
    4U * 1024U * 1024U,
    8U * 1024U * 1024U,
    16U * 1024U * 1024U,
};

struct DeviceRegion {
    void* allocation = nullptr;
    GM_ADDR base = nullptr;
    size_t bytes = 0;
};

struct KernelRegistration {
    void* binaryHandle = nullptr;
    bool registered = false;
};

int GetEnvInt(const char* name, int defaultValue)
{
    const char* value = std::getenv(name);
    return value == nullptr ? defaultValue : std::atoi(value);
}

uint64_t GetEnvUint64(const char* name, uint64_t defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return defaultValue;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    return end != value && end != nullptr && end[0] == '\0' ?
        static_cast<uint64_t>(parsed) : defaultValue;
}

int GetRank()
{
    const char* names[] = {
        "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "RANK"};
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr) {
            return std::atoi(value);
        }
    }
    return 0;
}

int GetRankSize()
{
    const char* names[] = {
        "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "RANK_SIZE"};
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr) {
            return std::atoi(value);
        }
    }
    return 1;
}

int GetLocalRank()
{
    const char* names[] = {
        "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK",
        "LOCAL_RANK"};
    for (const char* name : names) {
        const char* value = std::getenv(name);
        if (value != nullptr) {
            return std::atoi(value);
        }
    }
    return 0;
}

void Log(int rank, const std::string& message)
{
    std::cout << "[rank " << rank << "] " << message << std::endl;
}

bool CheckAcl(int rank, const char* step, aclError ret)
{
    if (ret == ACL_SUCCESS) {
        return true;
    }
    std::cerr << "[rank " << rank << "] ERROR: " << step
              << " failed, ret=" << ret << std::endl;
    return false;
}

bool CheckTileXR(int rank, const char* step, int ret)
{
    if (ret == TileXR::TILEXR_SUCCESS) {
        return true;
    }
    std::cerr << "[rank " << rank << "] ERROR: " << step
              << " failed, ret=" << ret << std::endl;
    return false;
}

bool AllocateAlignedRegion(int rank, size_t bytes, DeviceRegion& region)
{
    if (bytes == 0 || bytes > std::numeric_limits<size_t>::max() - (kAlignment - 1U)) {
        return false;
    }
    const size_t allocationBytes = bytes + kAlignment - 1U;
    if (!CheckAcl(rank, "aclrtMalloc aligned region",
            aclrtMalloc(&region.allocation, allocationBytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
        return false;
    }
    const uintptr_t raw = reinterpret_cast<uintptr_t>(region.allocation);
    const uintptr_t aligned = (raw + kAlignment - 1U) & ~(static_cast<uintptr_t>(kAlignment) - 1U);
    region.base = reinterpret_cast<GM_ADDR>(aligned);
    region.bytes = bytes;
    if ((aligned % kAlignment) != 0U || aligned < raw ||
        aligned - raw > allocationBytes - bytes) {
        std::cerr << "[rank " << rank << "] ERROR: failed to derive a 2 MiB-aligned device region"
                  << std::endl;
        return false;
    }
    return true;
}

void FreeRegion(DeviceRegion& region)
{
    if (region.allocation != nullptr) {
        (void)aclrtFree(region.allocation);
    }
    region = {};
}

uint8_t PatternByte(int rank, uint32_t region, uint64_t offset)
{
    const uint64_t mixed = offset * UINT64_C(1315423911) +
        static_cast<uint64_t>(rank + 1) * UINT64_C(2654435761) +
        static_cast<uint64_t>(region + 1U) * UINT64_C(2246822519);
    return static_cast<uint8_t>((mixed ^ (mixed >> 17U) ^ (mixed >> 31U)) & 0xFFU);
}

bool InitializePattern(int rank, const DeviceRegion& region, uint32_t patternRegion,
    size_t patternBytes, size_t offset = 0U)
{
    if (offset > region.bytes || patternBytes > region.bytes - offset) {
        return false;
    }
    std::vector<uint8_t> host(patternBytes);
    for (size_t index = 0; index < patternBytes; ++index) {
        host[index] = PatternByte(rank, patternRegion, index);
    }
    return CheckAcl(rank, "aclrtMemcpy pattern H2D",
        aclrtMemcpy(region.base + offset, region.bytes - offset,
            host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE));
}

bool ValidatePattern(int rank, const char* label, GM_ADDR device, size_t bytes,
    int sourceRank, uint32_t sourceRegion, uint64_t sourceOffset = 0U)
{
    std::vector<uint8_t> host(bytes);
    if (!CheckAcl(rank, "aclrtMemcpy result D2H",
            aclrtMemcpy(host.data(), host.size(), device, bytes, ACL_MEMCPY_DEVICE_TO_HOST))) {
        return false;
    }
    for (size_t index = 0; index < bytes; ++index) {
        const uint8_t expected = PatternByte(
            sourceRank, sourceRegion, sourceOffset + index);
        if (host[index] != expected) {
            std::cerr << "[rank " << rank << "] ERROR: " << label
                      << " byte mismatch at offset=" << index
                      << " got=" << static_cast<uint32_t>(host[index])
                      << " expected=" << static_cast<uint32_t>(expected) << std::endl;
            return false;
        }
    }
    return true;
}

const void* KernelSignature()
{
    return reinterpret_cast<const void*>(kKernelSignatureValue);
}

bool RegisterKernel(int rank, KernelRegistration& registration)
{
    if (registration.registered) {
        return true;
    }
    rtDevBinary_t binary {};
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.version = 0;
    binary.data = TileXRUDMAProfileProbeKernelBinaryData;
    binary.length = static_cast<uint64_t>(TileXRUDMAProfileProbeKernelBinarySize);
    rtError_t ret = rtDevBinaryRegister(&binary, &registration.binaryHandle);
    if (ret != RT_ERROR_NONE) {
        std::cerr << "[rank " << rank << "] ERROR: rtDevBinaryRegister failed, ret="
                  << ret << std::endl;
        return false;
    }
    ret = rtFunctionRegister(registration.binaryHandle, KernelSignature(),
        kKernelName, kKernelName, 0);
    if (ret != RT_ERROR_NONE) {
        std::cerr << "[rank " << rank << "] ERROR: rtFunctionRegister failed, ret="
                  << ret << std::endl;
        (void)rtDevBinaryUnRegister(registration.binaryHandle);
        registration.binaryHandle = nullptr;
        return false;
    }
    registration.registered = true;
    return true;
}

void UnregisterKernel(KernelRegistration& registration)
{
    if (registration.binaryHandle != nullptr) {
        (void)rtDevBinaryUnRegister(registration.binaryHandle);
    }
    registration = {};
}

bool LaunchKernel(int rank, KernelRegistration& registration, aclrtStream stream,
    GM_ADDR commArgs, GM_ADDR profileInfo, GM_ADDR profileRegistry,
    GM_ADDR legacy, GM_ADDR consumer, GM_ADDR status, int peer,
    uint32_t qpIdx, uint32_t localRegion, uint32_t remoteRegion,
    uint64_t transferBytes, uint32_t batchCount, uint32_t mode)
{
    if (!RegisterKernel(rank, registration)) {
        return false;
    }
    struct KernelArgs {
        GM_ADDR commArgs;
        GM_ADDR profileInfo;
        GM_ADDR profileRegistry;
        GM_ADDR legacy;
        GM_ADDR consumer;
        GM_ADDR status;
        int32_t peer;
        uint32_t qpIdx;
        uint32_t localRegion;
        uint32_t remoteRegion;
        uint64_t transferBytes;
        uint32_t batchCount;
        uint32_t mode;
    } args {
        commArgs, profileInfo, profileRegistry, legacy, consumer, status,
        peer, qpIdx, localRegion, remoteRegion, transferBytes, batchCount, mode
    };
    rtArgsEx_t argsInfo {};
    argsInfo.args = &args;
    argsInfo.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfgInfo {};
    cfgInfo.schemMode = 1;
    const rtError_t ret = rtKernelLaunchWithFlagV2(KernelSignature(), 1U,
        &argsInfo, nullptr, static_cast<rtStream_t>(stream), 0U, &cfgInfo);
    if (ret != RT_ERROR_NONE) {
        std::cerr << "[rank " << rank << "] ERROR: rtKernelLaunchWithFlagV2 failed, ret="
                  << ret << std::endl;
        return false;
    }
    return true;
}

bool ReadAndValidateStatus(int rank, GM_ADDR statusDev, uint32_t mode,
    uint32_t qpIdx, uint32_t remoteRegion, uint64_t batchBytes)
{
    uint64_t status[kStatusWords] = {};
    if (!CheckAcl(rank, "aclrtMemcpy status D2H",
            aclrtMemcpy(status, sizeof(status), statusDev, sizeof(status),
                ACL_MEMCPY_DEVICE_TO_HOST))) {
        return false;
    }
    const bool common = status[0] == kProbeMagic &&
        status[1] == TileXR::TILEXR_SUCCESS &&
        status[4] == TileXR::TILEXR_SUCCESS &&
        status[9] == batchBytes && status[10] == mode;
    const bool profile = mode == kModeLegacyGet ||
        (status[2] != 0U && status[3] == TileXR::TILEXR_SUCCESS &&
         status[5] == qpIdx && status[6] == 0U && status[7] == remoteRegion);
    if (!common || !profile) {
        std::cerr << "[rank " << rank << "] ERROR: kernel status mismatch"
                  << " magic=" << std::hex << status[0] << std::dec
                  << " post=" << status[1] << " frontier=" << status[2]
                  << " flush=" << status[3] << " quiet=" << status[4]
                  << " qp=" << status[5] << " localRegion=" << status[6]
                  << " remoteRegion=" << status[7] << " batch=" << status[8]
                  << " bytes=" << status[9] << " mode=" << status[10] << std::endl;
        return false;
    }
    return true;
}

bool MeasureProfileCase(int rank, int device, int peer,
    KernelRegistration& registration, aclrtStream stream, GM_ADDR commArgs,
    const TileXR::TileXRUDMAProfileView& view, const DeviceRegion& staging,
    const DeviceRegion& consumer, const DeviceRegion& status,
    uint32_t qpIdx, uint32_t remoteRegion, uint64_t transferBytes,
    uint32_t mode, uint32_t warmupIterations, uint32_t timedIterations,
    uint64_t sourcePatternOffset = 0U)
{
    const uint64_t batchBytes = transferBytes * kBatchCount;
    if (batchBytes > staging.bytes || batchBytes > consumer.bytes) {
        return false;
    }
    if (!CheckAcl(rank, "aclrtMemset staging",
            aclrtMemset(staging.base, staging.bytes, 0xA5, batchBytes)) ||
        !CheckAcl(rank, "aclrtMemset consumer",
            aclrtMemset(consumer.base, consumer.bytes, 0x5A, batchBytes))) {
        return false;
    }

    for (uint32_t iteration = 0; iteration < warmupIterations; ++iteration) {
        if (!LaunchKernel(rank, registration, stream, commArgs, view.infoDev,
                view.registryDev, nullptr, consumer.base, status.base, peer,
                qpIdx, 0U, remoteRegion, transferBytes, kBatchCount, mode)) {
            return false;
        }
    }
    if (!CheckAcl(rank, "aclrtSynchronizeStream warmup", aclrtSynchronizeStream(stream))) {
        return false;
    }

    for (uint32_t iteration = 0; iteration < timedIterations; ++iteration) {
        aclrtEvent start = nullptr;
        aclrtEvent stop = nullptr;
        if (!CheckAcl(rank, "aclrtCreateEvent start", aclrtCreateEvent(&start)) ||
            !CheckAcl(rank, "aclrtCreateEvent stop", aclrtCreateEvent(&stop))) {
            if (start != nullptr) {
                (void)aclrtDestroyEvent(start);
            }
            if (stop != nullptr) {
                (void)aclrtDestroyEvent(stop);
            }
            return false;
        }
        bool ok = CheckAcl(rank, "aclrtRecordEvent start", aclrtRecordEvent(start, stream));
        ok = ok && LaunchKernel(rank, registration, stream, commArgs, view.infoDev,
            view.registryDev, nullptr, consumer.base, status.base, peer,
            qpIdx, 0U, remoteRegion, transferBytes, kBatchCount, mode);
        ok = ok && CheckAcl(rank, "aclrtRecordEvent stop", aclrtRecordEvent(stop, stream));
        ok = ok && CheckAcl(rank, "aclrtSynchronizeEvent stop", aclrtSynchronizeEvent(stop));
        float elapsedMs = 0.0F;
        ok = ok && CheckAcl(rank, "aclrtEventElapsedTime",
            aclrtEventElapsedTime(&elapsedMs, start, stop));
        (void)aclrtDestroyEvent(start);
        (void)aclrtDestroyEvent(stop);
        if (!ok || elapsedMs <= 0.0F) {
            return false;
        }
        const double udmaGBps = static_cast<double>(batchBytes) /
            (static_cast<double>(elapsedMs) * 1.0e6);
        const uint64_t stagingIoBytes = mode == kModeProfileConsume ? 2U * batchBytes : 0U;
        std::cout << "TILEXR_UDMA_PROFILE_PROBE_JSON {"
                  << "\"record\":\"timing\","
                  << "\"rank\":" << rank << ","
                  << "\"device\":" << device << ","
                  << "\"mode\":\""
                  << (mode == kModeProfileConsume ? "transfer_consume" : "transfer_only") << "\","
                  << "\"qp\":" << qpIdx << ","
                  << "\"remote_region\":" << remoteRegion << ","
                  << "\"wqe_bytes\":" << transferBytes << ","
                  << "\"batch_count\":" << kBatchCount << ","
                  << "\"batch_bytes\":" << batchBytes << ","
                  << "\"staging_io_bytes\":" << stagingIoBytes << ","
                  << "\"iteration\":" << iteration << ","
                  << "\"elapsed_ms\":" << std::fixed << std::setprecision(6) << elapsedMs << ","
                  << "\"udma_gbps\":" << std::setprecision(6) << udmaGBps
                  << "}" << std::defaultfloat << std::endl;
    }

    if (!ReadAndValidateStatus(rank, status.base, mode, qpIdx, remoteRegion, batchBytes) ||
        !ValidatePattern(rank, "staging", staging.base, batchBytes,
            peer, remoteRegion, sourcePatternOffset)) {
        return false;
    }
    if (mode == kModeProfileConsume &&
        !ValidatePattern(rank, "consumer", consumer.base, batchBytes,
            peer, remoteRegion, sourcePatternOffset)) {
        return false;
    }
    std::cout << "TILEXR_UDMA_PROFILE_PROBE_JSON {"
              << "\"record\":\"correctness\",\"rank\":" << rank
              << ",\"mode\":\""
              << (mode == kModeProfileConsume ? "transfer_consume" : "transfer_only")
              << "\",\"qp\":" << qpIdx
              << ",\"remote_region\":" << remoteRegion
              << ",\"wqe_bytes\":" << transferBytes
              << ",\"batch_bytes\":" << batchBytes
              << ",\"byte_exact\":true}" << std::endl;
    return true;
}

bool RunLegacyGet(int rank, int peer, const char* phase,
    KernelRegistration& registration, aclrtStream stream, GM_ADDR commArgs,
    const DeviceRegion& legacy, const DeviceRegion& consumer,
    const DeviceRegion& status)
{
    GM_ADDR destination = legacy.base + kLegacyDestinationOffset;
    if (!CheckAcl(rank, "aclrtMemset legacy destination",
            aclrtMemset(destination, legacy.bytes - kLegacyDestinationOffset,
                0xCC, kLegacyTransferBytes)) ||
        !LaunchKernel(rank, registration, stream, commArgs, nullptr, nullptr,
            destination, consumer.base, status.base, peer, 0U, 0U, 0U,
            kLegacyTransferBytes, 1U, kModeLegacyGet) ||
        !CheckAcl(rank, "aclrtSynchronizeStream legacy", aclrtSynchronizeStream(stream)) ||
        !ReadAndValidateStatus(rank, status.base, kModeLegacyGet,
            0U, 0U, kLegacyTransferBytes) ||
        !ValidatePattern(rank, "legacy destination", destination,
            kLegacyTransferBytes, peer, kLegacyPatternRegion)) {
        return false;
    }
    std::cout << "TILEXR_UDMA_PROFILE_PROBE_JSON {"
              << "\"record\":\"lifecycle\",\"rank\":" << rank
              << ",\"phase\":\"" << phase
              << "\",\"legacy_get_byte_exact\":true}" << std::endl;
    return true;
}

void Cleanup(int rank, int device, TileXRCommPtr comm, aclrtStream stream,
    KernelRegistration& registration, std::vector<DeviceRegion>& profileRegions,
    DeviceRegion& legacy, DeviceRegion& consumer, DeviceRegion& status)
{
    UnregisterKernel(registration);
    if (comm != nullptr) {
        (void)TileXRCommDestroy(comm);
    }
    for (DeviceRegion& region : profileRegions) {
        FreeRegion(region);
    }
    FreeRegion(legacy);
    FreeRegion(consumer);
    FreeRegion(status);
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    (void)aclrtResetDevice(device);
    (void)aclFinalize();
    Log(rank, "cleanup complete");
}

} // namespace

int main()
{
    const int rank = GetRank();
    const int rankSize = GetRankSize();
    const int localRank = GetLocalRank();
    const int device = GetEnvInt("TILEXR_PROFILE_PROBE_DEVICE_BASE", 0) + localRank;
    const uint32_t warmupIterations = static_cast<uint32_t>(
        std::max(0, GetEnvInt("TILEXR_PROFILE_PROBE_WARMUP", 2)));
    const uint32_t timedIterations = static_cast<uint32_t>(
        std::max(1, GetEnvInt("TILEXR_PROFILE_PROBE_ITERATIONS", 5)));
    const bool registrationOnly =
        GetEnvInt("TILEXR_PROFILE_PROBE_REGISTRATION_ONLY", 0) != 0;
    const uint64_t sourceRegistrationOffset = GetEnvUint64(
        "TILEXR_PROFILE_PROBE_SOURCE_REG_OFFSET", 0U);
    const uint64_t sourceRegistrationBytes = GetEnvUint64(
        "TILEXR_PROFILE_PROBE_SOURCE_REG_BYTES", kProfileRegionBytes);
    const uint64_t sourceViewOffset = GetEnvUint64(
        "TILEXR_PROFILE_PROBE_SOURCE_VIEW_OFFSET", 0U);
    const uint64_t defaultSourceViewBytes =
        sourceViewOffset <= sourceRegistrationBytes ?
            sourceRegistrationBytes - sourceViewOffset : 0U;
    const uint64_t sourceViewBytes = GetEnvUint64(
        "TILEXR_PROFILE_PROBE_SOURCE_VIEW_BYTES", defaultSourceViewBytes);
    if (rankSize != 2 || rank < 0 || rank >= rankSize || device < 0) {
        std::cerr << "ERROR: profile probe requires exactly two ranks and valid device ids" << std::endl;
        return 2;
    }
    const int peer = 1 - rank;

    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    KernelRegistration kernelRegistration;
    std::vector<DeviceRegion> profileRegions(kProfileRegionCount);
    DeviceRegion legacy;
    DeviceRegion consumer;
    DeviceRegion status;
    TileXRUDMAMemHandle legacyHandle = 0U;
    TileXRUDMAProfileHandle profileHandle = 0U;
    bool legacyRegistered = false;
    bool profileRegistered = false;
    bool initialized = false;

    if (!CheckAcl(rank, "aclInit", aclInit(nullptr)) ||
        !CheckAcl(rank, "aclrtSetDevice", aclrtSetDevice(device)) ||
        !CheckAcl(rank, "aclrtCreateStream", aclrtCreateStream(&stream))) {
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 1;
    }
    initialized = true;
    (void)initialized;

    if (!CheckTileXR(rank, "TileXRCommInitRankLocal",
            TileXRCommInitRankLocal(rankSize, rank, &comm))) {
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 1;
    }
    uint32_t qpCount = 0U;
    GM_ADDR commArgs = nullptr;
    if (!CheckTileXR(rank, "TileXRUDMAGetQpCount", TileXRUDMAGetQpCount(comm, &qpCount)) ||
        qpCount < kMinimumQpCount ||
        !CheckTileXR(rank, "TileXRGetCommArgsDev", TileXRGetCommArgsDev(comm, commArgs))) {
        std::cerr << "[rank " << rank << "] ERROR: probe needs at least three UDMA QPs, got "
                  << qpCount << std::endl;
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 1;
    }

    bool ok = true;
    for (DeviceRegion& region : profileRegions) {
        ok = AllocateAlignedRegion(rank, kProfileRegionBytes, region) && ok;
    }
    ok = AllocateAlignedRegion(rank, kLegacyRegionBytes, legacy) && ok;
    ok = AllocateAlignedRegion(rank, kProfileRegionBytes, consumer) && ok;
    ok = AllocateAlignedRegion(rank, kAlignment, status) && ok;
    if (!ok) {
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 1;
    }
    if (sourceRegistrationBytes == 0U || sourceViewBytes == 0U ||
        sourceRegistrationOffset > kProfileRegionBytes ||
        sourceRegistrationBytes > kProfileRegionBytes - sourceRegistrationOffset ||
        sourceViewOffset > sourceRegistrationBytes ||
        sourceViewBytes > sourceRegistrationBytes - sourceViewOffset) {
        std::cerr << "[rank " << rank
                  << "] ERROR: source registration range exceeds the aligned backing allocation"
                  << std::endl;
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 2;
    }
    for (uint32_t region = 1U; region < kProfileRegionCount; ++region) {
        ok = InitializePattern(rank, profileRegions[region], region,
            profileRegions[region].bytes) && ok;
    }
    ok = InitializePattern(rank, legacy, kLegacyPatternRegion,
        kLegacyTransferBytes) && ok;
    if (!ok) {
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 1;
    }

    if (!CheckTileXR(rank, "TileXRUDMARegister legacy",
            TileXRUDMARegister(comm, legacy.base, legacy.bytes, &legacyHandle))) {
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 1;
    }
    legacyRegistered = true;
    TileXR::CommArgs* commArgsHost = nullptr;
    if (!CheckTileXR(rank, "TileXRGetCommArgsHost",
            TileXRGetCommArgsHost(comm, commArgsHost)) || commArgsHost == nullptr) {
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 1;
    }
    const GM_ADDR legacyInfoBeforeProfile = commArgsHost->udmaInfoPtr;
    const GM_ADDR legacyRegistryBeforeProfile = commArgsHost->udmaRegistryPtr;

    TileXR::TileXRUDMAProfileDesc desc {};
    desc.regionCount = kProfileRegionCount;
    desc.qpBindingCount = qpCount;
    for (uint32_t region = 0U; region < desc.regionCount; ++region) {
        desc.regions[region].base = profileRegions[region].base;
        desc.regions[region].bytes = profileRegions[region].bytes;
    }
    for (uint32_t region = 1U; region < desc.regionCount; ++region) {
        desc.regions[region].registrationBase =
            profileRegions[region].base + sourceRegistrationOffset;
        desc.regions[region].registrationBytes = sourceRegistrationBytes;
        desc.regions[region].base =
            desc.regions[region].registrationBase + sourceViewOffset;
        desc.regions[region].bytes = sourceViewBytes;
    }
    for (uint32_t qp = 0U; qp < qpCount; ++qp) {
        desc.qpBindings[qp].localRegion = 0U;
        desc.qpBindings[qp].remoteRegion = qp % 3U + 1U;
    }
    if (!CheckTileXR(rank, "TileXRUDMAProfileRegister",
            TileXRUDMAProfileRegister(comm, &desc, &profileHandle))) {
        (void)TileXRUDMAUnregister(comm, legacyHandle);
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 1;
    }
    profileRegistered = true;
    TileXR::TileXRUDMAProfileView view {};
    if (!CheckTileXR(rank, "TileXRUDMAProfileQuery",
            TileXRUDMAProfileQuery(comm, profileHandle, &view)) ||
        view.infoDev == nullptr || view.registryDev == nullptr ||
        view.registryHost == nullptr || view.qpCount != qpCount ||
        commArgsHost->udmaInfoPtr != legacyInfoBeforeProfile ||
        commArgsHost->udmaRegistryPtr != legacyRegistryBeforeProfile) {
        std::cerr << "[rank " << rank << "] ERROR: profile registration mutated legacy CommArgs"
                  << std::endl;
        (void)TileXRUDMAProfileUnregister(comm, profileHandle);
        (void)TileXRUDMAUnregister(comm, legacyHandle);
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return 1;
    }

    if (registrationOnly) {
        std::cout << "TILEXR_UDMA_PROFILE_PROBE_JSON {"
                  << "\"record\":\"registration\",\"rank\":" << rank
                  << ",\"source_offset\":" << sourceRegistrationOffset
                  << ",\"source_bytes\":" << sourceRegistrationBytes
                  << ",\"view_offset\":" << sourceViewOffset
                  << ",\"view_bytes\":" << sourceViewBytes
                  << ",\"source_base_mod_2m\":"
                  << (reinterpret_cast<uintptr_t>(
                          desc.regions[1].registrationBase) % kAlignment)
                  << ",\"passed\":true}" << std::endl;
        const bool profileUnregistered = CheckTileXR(rank,
            "TileXRUDMAProfileUnregister registration-only",
            TileXRUDMAProfileUnregister(comm, profileHandle));
        profileRegistered = !profileUnregistered;
        const bool legacyUnregistered = CheckTileXR(rank,
            "TileXRUDMAUnregister registration-only",
            TileXRUDMAUnregister(comm, legacyHandle));
        legacyRegistered = !legacyUnregistered;
        Cleanup(rank, device, comm, stream, kernelRegistration,
            profileRegions, legacy, consumer, status);
        return profileUnregistered && legacyUnregistered ? 0 : 1;
    }

    for (uint32_t qp = 0U; qp < qpCount && ok; ++qp) {
        const uint32_t remoteRegion = desc.qpBindings[qp].remoteRegion;
        for (uint64_t transferBytes : kTransferSizes) {
            if (transferBytes > desc.regions[remoteRegion].bytes / kBatchCount) {
                continue;
            }
            ok = MeasureProfileCase(rank, device, peer, kernelRegistration,
                stream, commArgs, view, profileRegions[0], consumer, status,
                qp, remoteRegion, transferBytes, kModeProfileTransfer,
                warmupIterations, timedIterations,
                sourceRegistrationOffset + sourceViewOffset) && ok;
            ok = MeasureProfileCase(rank, device, peer, kernelRegistration,
                stream, commArgs, view, profileRegions[0], consumer, status,
                qp, remoteRegion, transferBytes, kModeProfileConsume,
                warmupIterations, timedIterations,
                sourceRegistrationOffset + sourceViewOffset) && ok;
        }
    }
    ok = RunLegacyGet(rank, peer, "profile_and_legacy_coexist",
        kernelRegistration, stream, commArgs, legacy, consumer, status) && ok;

    if (ok) {
        ok = CheckTileXR(rank, "TileXRUDMAProfileUnregister",
            TileXRUDMAProfileUnregister(comm, profileHandle));
        profileRegistered = !ok;
    }
    ok = ok && RunLegacyGet(rank, peer, "legacy_after_profile_unregister",
        kernelRegistration, stream, commArgs, legacy, consumer, status);

    if (ok) {
        profileHandle = 0U;
        ok = CheckTileXR(rank, "TileXRUDMAProfileRegister second",
            TileXRUDMAProfileRegister(comm, &desc, &profileHandle));
        profileRegistered = ok;
    }
    if (ok) {
        view = {};
        ok = CheckTileXR(rank, "TileXRUDMAProfileQuery second",
            TileXRUDMAProfileQuery(comm, profileHandle, &view));
    }
    if (ok) {
        ok = CheckTileXR(rank, "TileXRUDMAUnregister legacy",
            TileXRUDMAUnregister(comm, legacyHandle));
        legacyRegistered = !ok;
    }
    if (ok) {
        const uint32_t remoteRegion = desc.qpBindings[0].remoteRegion;
        ok = MeasureProfileCase(rank, device, peer, kernelRegistration,
            stream, commArgs, view, profileRegions[0], consumer, status,
            0U, remoteRegion, 2U * 1024U * 1024U, kModeProfileTransfer,
            0U, 1U, sourceRegistrationOffset + sourceViewOffset);
        if (ok) {
            std::cout << "TILEXR_UDMA_PROFILE_PROBE_JSON {"
                      << "\"record\":\"lifecycle\",\"rank\":" << rank
                      << ",\"phase\":\"profile_after_legacy_unregister\","
                      << "\"profile_get_byte_exact\":true}" << std::endl;
        }
    }

    if (profileRegistered) {
        const bool unregistered = CheckTileXR(rank, "TileXRUDMAProfileUnregister final",
            TileXRUDMAProfileUnregister(comm, profileHandle));
        profileRegistered = !unregistered;
        ok = unregistered && ok;
    }
    if (legacyRegistered) {
        const bool unregistered = CheckTileXR(rank, "TileXRUDMAUnregister final",
            TileXRUDMAUnregister(comm, legacyHandle));
        legacyRegistered = !unregistered;
        ok = unregistered && ok;
    }

    if (ok) {
        std::cout << "TILEXR_UDMA_PROFILE_PROBE_JSON {"
                  << "\"record\":\"summary\",\"rank\":" << rank
                  << ",\"device\":" << device
                  << ",\"qp_count\":" << qpCount
                  << ",\"profile_regions\":" << kProfileRegionCount
                  << ",\"region_alignment_bytes\":" << kAlignment
                  << ",\"max_wqe_bytes\":" << kMaxTransferBytes
                  << ",\"max_batch_bytes\":" << kProfileRegionBytes
                  << ",\"passed\":true}" << std::endl;
        std::cout << "[rank " << rank << "] TileXR UDMA profile probe success" << std::endl;
    }
    Cleanup(rank, device, comm, stream, kernelRegistration,
        profileRegions, legacy, consumer, status);
    return ok ? 0 : 1;
}
