#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include "acl/acl.h"
#include "tilexr_api.h"
#include "tilexr_moonep_combine_v2.h"
#include "tilexr_types.h"

namespace {

constexpr int kRankCount = 8;
constexpr int kDeviceCount = 8;
constexpr int kCommDomain = 41;
constexpr int kInvocationCount = 2;
constexpr int64_t kBs = 8;
constexpr int64_t kH = 3584;
constexpr int64_t kTopK = 16;
constexpr int64_t kSlots = 128;
constexpr uint32_t kAivCoreNum = 16;
constexpr uint32_t kQpCount = 32;

[[noreturn]] void Abort(int rank, const std::string &step, int status)
{
    std::cerr << "[rank " << rank << "] " << step
              << " failed, status=" << status << std::endl;
    MPI_Abort(MPI_COMM_WORLD, status == 0 ? 1 : status);
    std::abort();
}

uint16_t SourceValue(int sourceRank, int sourceSlot, int column)
{
    const uint32_t payload = static_cast<uint32_t>(
        sourceRank * 131 + sourceSlot * 17 + column) & 0x03ffU;
    return static_cast<uint16_t>(0x3c00U + payload);
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (world != kRankCount) {
        Abort(rank, "MPI world size", world);
    }

    const int device = rank % kDeviceCount;
    if (aclInit(nullptr) != ACL_SUCCESS) {
        Abort(rank, "aclInit", 1);
    }
    if (aclrtSetDevice(device) != ACL_SUCCESS) {
        Abort(rank, "aclrtSetDevice", 1);
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        Abort(rank, "aclrtCreateStream", 1);
    }

    TileXRCommPtr comm = nullptr;
    int ret = TileXRCommInitRankWithSharedQpDomain(
        kCommDomain, world, rank, &comm);
    if (ret != TileXR::TILEXR_SUCCESS) {
        Abort(rank, "TileXRCommInitRankWithSharedQpDomain", ret);
    }

    TileXR::CommArgs *commArgs = nullptr;
    uint32_t qpCount = 0;
    ret = TileXRGetCommArgsHost(comm, commArgs);
    if (ret != TileXR::TILEXR_SUCCESS || commArgs == nullptr ||
        (commArgs->extraFlag & TileXR::ExtraFlag::UDMA_SHARED_QP) == 0U) {
        Abort(rank, "shared-QP CommArgs validation", ret);
    }
    ret = TileXRUDMAGetQpCount(comm, &qpCount);
    if (ret != TileXR::TILEXR_SUCCESS || qpCount != kQpCount) {
        Abort(rank, "TileXRUDMAGetQpCount", ret);
    }

    uint64_t workspaceBytes = 0;
    uint64_t profileOffset = 0;
    uint64_t outputOffsets[2] = {};
    ret = TileXRMoonEpCombineGetWorkspaceSizeV2(
        kBs, kH, kTopK, kSlots, TILEXR_MOONEP_DTYPE_BFLOAT16,
        &workspaceBytes, &profileOffset, &outputOffsets[0], &outputOffsets[1]);
    if (ret != TILEXR_MOONEP_SUCCESS || workspaceBytes == 0) {
        Abort(rank, "TileXRMoonEpCombineGetWorkspaceSizeV2", ret);
    }

    void *workspace = nullptr;
    int32_t *dst = nullptr;
    const size_t rowElements = static_cast<size_t>(kH);
    const size_t sourceElements = static_cast<size_t>(kSlots) * rowElements;
    const size_t sourceBytes = sourceElements * sizeof(uint16_t);
    const size_t dstBytes = static_cast<size_t>(kSlots) * sizeof(int32_t);
    if (aclrtMalloc(&workspace, static_cast<size_t>(workspaceBytes),
            ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void **>(&dst), dstBytes,
            ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        Abort(rank, "aclrtMalloc", 1);
    }
    if (aclrtMemset(workspace, static_cast<size_t>(workspaceBytes), 0,
            static_cast<size_t>(workspaceBytes)) != ACL_SUCCESS) {
        Abort(rank, "aclrtMemset workspace", 1);
    }

    const int slotsPerRank = static_cast<int>(kSlots) / world;
    std::vector<uint16_t> source(sourceElements);
    std::vector<int32_t> destinations(static_cast<size_t>(kSlots));
    for (int slot = 0; slot < kSlots; ++slot) {
        const int targetRank = slot % world;
        const int targetSlot = rank * slotsPerRank + slot / world;
        destinations[static_cast<size_t>(slot)] =
            targetRank * static_cast<int>(kSlots) + targetSlot;
        for (int column = 0; column < kH; ++column) {
            source[static_cast<size_t>(slot) * rowElements +
                static_cast<size_t>(column)] = SourceValue(rank, slot, column);
        }
    }
    if (aclrtMemcpy(workspace, static_cast<size_t>(workspaceBytes),
            source.data(), sourceBytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(dst, dstBytes, destinations.data(), dstBytes,
            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        Abort(rank, "input H2D copy", 1);
    }

    TileXRUDMAMemHandle handle = 0;
    ret = TileXRUDMARegister(comm, static_cast<GM_ADDR>(workspace),
        static_cast<size_t>(workspaceBytes), &handle);
    if (ret != TileXR::TILEXR_SUCCESS) {
        Abort(rank, "TileXRUDMARegister", ret);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    uint64_t activeOutputOffset = 0;
    uint64_t previousOutputOffset = 0;
    std::vector<uint16_t> output(sourceElements);
    int localOk = 1;
    for (int invocation = 0; invocation < kInvocationCount; ++invocation) {
        ret = TileXRMoonEpCombineV2(workspace, dst, comm, kBs, kH, kTopK,
            kSlots, kAivCoreNum, &activeOutputOffset,
            TILEXR_MOONEP_DTYPE_BFLOAT16, stream);
        if (ret != TILEXR_MOONEP_SUCCESS) {
            Abort(rank, "TileXRMoonEpCombineV2", ret);
        }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            Abort(rank, "aclrtSynchronizeStream", 1);
        }
        if ((activeOutputOffset != outputOffsets[0] &&
                activeOutputOffset != outputOffsets[1]) ||
            (invocation != 0 && activeOutputOffset == previousOutputOffset)) {
            Abort(rank, "active output offset", 1);
        }
        previousOutputOffset = activeOutputOffset;

        const void *outputDevice = static_cast<const uint8_t *>(workspace) +
            activeOutputOffset;
        if (aclrtMemcpy(output.data(), sourceBytes, outputDevice, sourceBytes,
                ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            Abort(rank, "output D2H copy", 1);
        }

        for (int targetSlot = 0; targetSlot < kSlots && localOk != 0;
            ++targetSlot) {
            const int sourceRank = targetSlot / slotsPerRank;
            const int sourceSlot =
                (targetSlot % slotsPerRank) * world + rank;
            for (int column = 0; column < kH; ++column) {
                const size_t index =
                    static_cast<size_t>(targetSlot) * rowElements +
                    static_cast<size_t>(column);
                const uint16_t expected =
                    SourceValue(sourceRank, sourceSlot, column);
                if (output[index] != expected) {
                    std::cerr << "[rank " << rank
                              << "] output mismatch invocation=" << invocation
                              << " target_slot=" << targetSlot
                              << " source_rank=" << sourceRank
                              << " source_slot=" << sourceSlot
                              << " column=" << column
                              << " got=" << output[index]
                              << " expected=" << expected << std::endl;
                    localOk = 0;
                    break;
                }
            }
        }
    }

    int globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    const int unregisterRet = TileXRUDMAUnregister(comm, handle);
    MPI_Barrier(MPI_COMM_WORLD);

    const int destroyRet = TileXRCommDestroy(comm);
    const aclError freeDstRet = aclrtFree(dst);
    const aclError freeWorkspaceRet = aclrtFree(workspace);
    const aclError destroyStreamRet = aclrtDestroyStream(stream);
    const aclError resetRet = aclrtResetDevice(device);
    const aclError finalizeRet = aclFinalize();
    const int cleanupOk = unregisterRet == TileXR::TILEXR_SUCCESS &&
        destroyRet == TileXR::TILEXR_SUCCESS && freeDstRet == ACL_SUCCESS &&
        freeWorkspaceRet == ACL_SUCCESS && destroyStreamRet == ACL_SUCCESS &&
        resetRet == ACL_SUCCESS && finalizeRet == ACL_SUCCESS;
    int globalCleanupOk = 0;
    MPI_Allreduce(&cleanupOk, &globalCleanupOk, 1, MPI_INT, MPI_MIN,
        MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "MoonEP Combine V2 hardware probe "
                  << (globalOk != 0 && globalCleanupOk != 0 ? "passed" : "failed")
                  << " logical_ranks=" << world
                  << " physical_devices=" << kDeviceCount
                  << " invocations=" << kInvocationCount
                  << " qp_count=" << qpCount
                  << " workspace_bytes=" << workspaceBytes
                  << " profile_offset=" << profileOffset
                  << " active_output_offset=" << activeOutputOffset
                  << std::endl;
    }
    MPI_Finalize();
    return globalOk != 0 && globalCleanupOk != 0 ? 0 : 1;
}
