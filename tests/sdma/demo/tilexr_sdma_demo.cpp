/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "tilexr_api.h"
#include "tilexr_sdma_types.h"
#include "tilexr_types.h"

extern "C" void launch_tilexr_sdma_copy(
    uint32_t blockDim,
    void* stream,
    GM_ADDR commArgs,
    GM_ADDR dst,
    GM_ADDR src,
    GM_ADDR debug,
    uint32_t bytes);

namespace {
constexpr uint32_t kDefaultBytes = 4096;
constexpr uint32_t kAlignmentBytes = 64;
constexpr size_t kDebugWords = 6;
constexpr int kDeviceId = 0;

bool CheckAcl(const std::string& label, aclError ret)
{
    std::cout << label << " ret=" << ret << std::endl;
    if (ret != ACL_SUCCESS) {
        std::cerr << "ERROR: " << label << " failed with " << ret << std::endl;
        return false;
    }
    return true;
}

bool CheckTileXR(const std::string& label, int ret)
{
    std::cout << label << " ret=" << ret << std::endl;
    if (ret != TileXR::TILEXR_SUCCESS) {
        std::cerr << "ERROR: " << label << " failed with " << ret << std::endl;
        return false;
    }
    return true;
}

bool ParseBytes(const char* text, uint32_t* bytes)
{
    if (text == nullptr || bytes == nullptr || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 ||
        value > std::numeric_limits<uint32_t>::max() || (value % kAlignmentBytes) != 0) {
        return false;
    }
    *bytes = static_cast<uint32_t>(value);
    return true;
}

void FillPattern(std::vector<uint8_t>& data)
{
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 37U + 11U) & 0xFFU);
    }
}

void Cleanup(
    uint8_t* src,
    uint8_t* dst,
    int32_t* debug,
    TileXRCommPtr comm,
    aclrtStream stream,
    bool deviceSet,
    bool aclInitialized)
{
    if (src != nullptr) {
        (void)CheckAcl("aclrtFree src", aclrtFree(src));
    }
    if (dst != nullptr) {
        (void)CheckAcl("aclrtFree dst", aclrtFree(dst));
    }
    if (debug != nullptr) {
        (void)CheckAcl("aclrtFree debug", aclrtFree(debug));
    }
    if (comm != nullptr) {
        (void)CheckTileXR("TileXRCommDestroy", TileXRCommDestroy(comm));
    }
    if (stream != nullptr) {
        (void)CheckAcl("aclrtDestroyStream", aclrtDestroyStream(stream));
    }
    if (deviceSet) {
        (void)CheckAcl("aclrtResetDevice", aclrtResetDevice(kDeviceId));
    }
    if (aclInitialized) {
        (void)CheckAcl("aclFinalize", aclFinalize());
    }
}

bool CopyHostToDevice(void* dst, size_t dstSize, const void* src, size_t srcSize, const std::string& name)
{
    return CheckAcl("aclrtMemcpy H2D " + name,
        aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_HOST_TO_DEVICE));
}

bool CopyDeviceToHost(void* dst, size_t dstSize, const void* src, size_t srcSize, const std::string& name)
{
    return CheckAcl("aclrtMemcpy D2H " + name,
        aclrtMemcpy(dst, dstSize, src, srcSize, ACL_MEMCPY_DEVICE_TO_HOST));
}

bool VerifyDebug(const std::vector<int32_t>& debug, uint32_t bytes)
{
    bool ok = true;
    if (debug[0] != TileXR::TILEXR_SDMA_DEMO_MAGIC) {
        std::cerr << "ERROR: debug magic mismatch: got " << debug[0] << std::endl;
        ok = false;
    }
    if (debug[2] != static_cast<int32_t>(bytes)) {
        std::cerr << "ERROR: debug bytes mismatch: got " << debug[2] << std::endl;
        ok = false;
    }
    if (debug[3] != 1) {
        std::cerr << "ERROR: SDMA was not enabled in kernel debug word" << std::endl;
        ok = false;
    }
    if (debug[4] != 1) {
        std::cerr << "ERROR: SDMA event was not posted" << std::endl;
        ok = false;
    }
    if (debug[5] != 1) {
        std::cerr << "ERROR: SDMA wait did not report success" << std::endl;
        ok = false;
    }
    return ok;
}
} // namespace

int main(int argc, char** argv)
{
    uint32_t bytes = kDefaultBytes;
    if (argc > 2 || (argc == 2 && !ParseBytes(argv[1], &bytes))) {
        std::cerr << "ERROR: bytes must be a non-zero 64-byte aligned unsigned integer" << std::endl;
        return 1;
    }

    (void)setenv("TILEXR_ENABLE_SDMA", "1", 1);

    bool aclInitialized = false;
    bool deviceSet = false;
    TileXRCommPtr comm = nullptr;
    aclrtStream stream = nullptr;
    uint8_t* src = nullptr;
    uint8_t* dst = nullptr;
    int32_t* debug = nullptr;

    if (!CheckAcl("aclInit", aclInit(nullptr))) {
        return 1;
    }
    aclInitialized = true;
    if (!CheckAcl("aclrtSetDevice(0)", aclrtSetDevice(kDeviceId))) {
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 1;
    }
    deviceSet = true;
    if (!CheckAcl("aclrtCreateStream", aclrtCreateStream(&stream))) {
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    if (!CheckTileXR("TileXRCommInitRankLocal", TileXRCommInitRankLocal(1, 0, &comm))) {
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    bool sdmaAvailable = false;
    GM_ADDR sdmaWorkspace = nullptr;
    if (!CheckTileXR("TileXRSDMAAvailable", TileXRSDMAAvailable(comm, &sdmaAvailable)) ||
        !CheckTileXR("TileXRGetSDMAWorkspaceDev", TileXRGetSDMAWorkspaceDev(comm, &sdmaWorkspace))) {
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 1;
    }
    std::cout << "SDMA available=" << (sdmaAvailable ? "true" : "false") << std::endl;
    std::cout << "SDMA workspace=" << static_cast<void*>(sdmaWorkspace) << std::endl;
    if (!sdmaAvailable || sdmaWorkspace == nullptr) {
        std::cerr << "ERROR: TileXR SDMA is unavailable" << std::endl;
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 4;
    }

    GM_ADDR commArgsDev = nullptr;
    if (!CheckTileXR("TileXRGetCommArgsDev", TileXRGetCommArgsDev(comm, commArgsDev)) || commArgsDev == nullptr) {
        std::cerr << "ERROR: failed to get TileXR CommArgs device pointer" << std::endl;
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    if (!CheckAcl("aclrtMalloc src",
            aclrtMalloc(reinterpret_cast<void**>(&src), bytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl("aclrtMalloc dst",
            aclrtMalloc(reinterpret_cast<void**>(&dst), bytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
        !CheckAcl("aclrtMalloc debug",
            aclrtMalloc(reinterpret_cast<void**>(&debug), kDebugWords * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST))) {
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    std::vector<uint8_t> hostSrc(bytes);
    std::vector<uint8_t> hostDst(bytes, 0);
    std::vector<int32_t> hostDebug(kDebugWords, 0);
    FillPattern(hostSrc);

    if (!CopyHostToDevice(src, bytes, hostSrc.data(), hostSrc.size(), "src") ||
        !CopyHostToDevice(dst, bytes, hostDst.data(), hostDst.size(), "dst") ||
        !CopyHostToDevice(debug, kDebugWords * sizeof(int32_t), hostDebug.data(),
            hostDebug.size() * sizeof(int32_t), "debug")) {
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    launch_tilexr_sdma_copy(
        1, stream, commArgsDev, reinterpret_cast<GM_ADDR>(dst), reinterpret_cast<GM_ADDR>(src),
        reinterpret_cast<GM_ADDR>(debug), bytes);
    if (!CheckAcl("aclrtSynchronizeStream", aclrtSynchronizeStream(stream))) {
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    if (!CopyDeviceToHost(hostDst.data(), hostDst.size(), dst, bytes, "dst") ||
        !CopyDeviceToHost(hostDebug.data(), hostDebug.size() * sizeof(int32_t), debug,
            kDebugWords * sizeof(int32_t), "debug")) {
        Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
        return 1;
    }

    std::cout << "debug words:";
    for (size_t i = 0; i < hostDebug.size(); ++i) {
        std::cout << " d" << i << "=" << hostDebug[i];
    }
    std::cout << std::endl;

    const bool dataOk = (hostDst == hostSrc);
    if (!dataOk) {
        auto mismatch = std::mismatch(hostDst.begin(), hostDst.end(), hostSrc.begin());
        const size_t index = static_cast<size_t>(mismatch.first - hostDst.begin());
        std::cerr << "ERROR: dst/src mismatch at byte " << index
                  << " dst=" << static_cast<int>(*mismatch.first)
                  << " src=" << static_cast<int>(*mismatch.second) << std::endl;
    }
    const bool debugOk = VerifyDebug(hostDebug, bytes);

    Cleanup(src, dst, debug, comm, stream, deviceSet, aclInitialized);
    if (!dataOk || !debugOk) {
        std::cerr << "ERROR: TileXR SDMA demo verification failed" << std::endl;
        return 1;
    }

    std::cout << "PASS TileXR SDMA copied " << bytes << " bytes correctly" << std::endl;
    return 0;
}
