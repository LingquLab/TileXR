#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#ifdef TILEXR_SOURCE_ROOT
const char *kSourceRoot = TILEXR_SOURCE_ROOT;
#else
const char *kSourceRoot = ".";
#endif

std::string JoinPath(const std::string &base, const std::string &path)
{
    if (base.empty() || base[base.size() - 1] == '/') {
        return base + path;
    }
    return base + "/" + path;
}

bool ReadFile(const std::string &relativePath, std::string *contents)
{
    const std::string fullPath = JoinPath(kSourceRoot, relativePath);
    std::ifstream stream(fullPath.c_str());
    if (!stream.is_open()) {
        std::cerr << "missing file: " << relativePath << std::endl;
        ++g_failures;
        return false;
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();
    *contents = buffer.str();
    return true;
}

void CheckContains(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) == std::string::npos) {
        std::cerr << label << " missing: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string &label, const std::string &contents, const std::string &needle)
{
    if (contents.find(needle) != std::string::npos) {
        std::cerr << label << " contains forbidden string: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckOrdered(const std::string &label, const std::string &contents, const std::string &first,
    const std::string &second)
{
    const std::string::size_type firstPos = contents.find(first);
    const std::string::size_type secondPos = contents.find(second);
    if (firstPos == std::string::npos || secondPos == std::string::npos || firstPos >= secondPos) {
        std::cerr << label << " expected order: " << first << " before " << second << std::endl;
        ++g_failures;
    }
}

std::string SliceBetween(const std::string &contents, const std::string &begin, const std::string &end)
{
    const std::string::size_type beginPos = contents.find(begin);
    if (beginPos == std::string::npos) {
        return std::string();
    }

    const std::string::size_type endPos = contents.find(end, beginPos + begin.size());
    if (endPos == std::string::npos) {
        return contents.substr(beginPos);
    }
    return contents.substr(beginPos, endPos - beginPos);
}

void TestKernelUsesTileXRPeerMemory()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "extern \"C\" __global__ __aicore__ void tilexr_ep_dispatch_kernel");
    CheckContains(path, contents, "launch_tilexr_ep_dispatch_kernel");
    CheckContains(path, contents, "CommArgs");
    CheckContains(path, contents, "peerMems");
    CheckContains(path, contents, "GlobalTensor<GM_ADDR> peerMems");
    CheckContains(path, contents, "peerMems.GetValue(peer)");
    CheckContains(path, contents, "IPC_DATA_OFFSET");
    CheckContains(path, contents, "SyncCollectives");
    CheckContains(path, contents, "DataCopyPad");
    CheckContains(path, contents, "kEpStepWindowCleared");
    CheckContains(path, contents, "kEpStepDispatchReady");
    CheckContains(path, contents, "kEpStepDispatchDrained");
    CheckContains(path, contents, "LoadInt32FromGm");
    CheckContains(path, contents, "LoadAssistTupleFromGm");
    CheckContains(path, contents, "StoreWindowHeader");
    CheckContains(path, contents, "StoreSlotHeader");
    CheckContains(path, contents, "StoreAssistTuple");
    CheckContains(path, contents, "TileXREpFlushDispatchSlotHeaders");
    CheckContains(path, contents, "sourceWindow = TileXREpWindowBase");
    CheckNotContains(path, contents, "expertIds[");
    CheckNotContains(path, contents, "assistBase[item]");
    CheckNotContains(path, contents, "args->peerMems[peer]");
    CheckNotContains(path, contents, "slot->count");
    CheckNotContains(path, contents, "assist[index]");
}

void TestCrossNodeDispatchUsesUDMARegistry()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "tilexr_udma.h");
    CheckContains(path, contents, "TileXR::UDMARegistryEnabled(args)");
    CheckContains(path, contents, "TileXREpUsesUdmaWindow");
    CheckContains(path, contents, "TileXR::UDMAPutNbi<uint8_t>");
    CheckContains(path, contents, "TileXR::UDMAQuiet(args, dstRank)");
    CheckContains(path, contents, "TileXREpFlushDispatchSlotHeaders");
    CheckContains(path, contents, "sourceWindow = TileXREpWindowBase");
}

void TestCrossNodeDispatchPullsRemoteSlots()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "tilexr_ep_dispatch_cross_node_kernel");
    CheckContains(path, contents, "launch_tilexr_ep_dispatch_cross_node_kernel");
    CheckContains(path, contents, "TileXREpPullUdmaSlots");
    CheckContains(path, contents, "TileXR::UDMAGetNbi<uint8_t>");
    CheckContains(path, contents, "TileXREpNotifyUdmaReady");
    CheckContains(path, contents, "TileXREpWaitUdmaReady");
    CheckContains(path, contents, "TileXREpNotifyAllUdmaReady");
    CheckContains(path, contents, "TileXREpWaitAllUdmaReady");
    CheckContains(path, contents, "TileXR::UDMAPutSignalNbi<uint8_t>");
}

void TestCrossNodeDispatchSeparatesLocalAndRemotePeers()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXREpIsSameNodePeer");
    CheckContains(path, contents, "TileXREpUsesUdmaPeer");
    CheckContains(path, contents, "if (localRankSize <= 1)");
    CheckContains(path, contents, "TileXREpPublishLocalUdmaSlot");
    CheckContains(path, contents, "TileXREpDispatchWriteWindow");
    CheckContains(path, contents, "args->localRankSize");
    CheckContains(path, contents, "if (localRankSize > 1)");
    CheckContains(path, contents, "dstRank != rank && TileXREpIsSameNodePeer(rank, dstRank, localRankSize)");
    CheckContains(path, contents, "srcRank != rank &&");
    CheckContains(path, contents, "TileXREpIsSameNodePeer(rank, srcRank, localRankSize)");
    CheckContains(path, contents, "sameNodeSource ? rank : srcRank");
    CheckContains(path, contents, "!TileXREpIsSameNodePeer(rank, peer, localRankSize)");
}

void TestHostDispatchSplitsCrossNodeKernel()
{
    const std::string path = "src/ep/host/ep_kernel_launch.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "launch_tilexr_ep_dispatch_cross_node_kernel");
    CheckContains(path, contents, "TileXREpUsesCrossNodeKernel");
}

void TestDispatchHelpersLiveInDispatchHelperFile()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_helpers.h";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXREpRouteToDstRank");
    CheckContains(path, contents, "TileXREpCopyRoutePayload");
    CheckContains(path, contents, "TileXREpStoreDispatchSlotHeader");
    CheckContains(path, contents, "TileXREpStoreAssistTuple");
    CheckContains(path, contents, "localWindow + PayloadOffset(dstRank, slotBytes)");
}

void TestCombineHelpersLiveInCombineHelperFile()
{
    const std::string path = "src/ep/kernels/tilexr_ep_combine_helpers.h";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXREpDrainSourceWindow");
    CheckContains(path, contents, "TileXREpWaitDispatchSlotReady");
    CheckContains(path, contents, "slotMagic");
    CheckContains(path, contents, "TileXREpLoadAssistTuple");
    CheckContains(path, contents, "TileXREpGetCombineTokenId");
    CheckContains(path, contents, "TileXREpGetCombineTopKId");
    CheckContains(path, contents, "sourceWindow + SlotOffset(slotRank, slotBytes)");
}

void TestCombineKernelUsesTileXRPeerMemory()
{
    const std::string path = "src/ep/kernels/tilexr_ep_combine_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "extern \"C\" __global__ __aicore__ void tilexr_ep_combine_kernel");
    CheckContains(path, contents, "launch_tilexr_ep_combine_kernel");
    CheckContains(path, contents, "kEpStepCombineWindowCleared");
    CheckContains(path, contents, "kEpStepCombineReady");
    CheckContains(path, contents, "ScatterCombineRows");
    CheckContains(path, contents, "DrainCombineRows");
    CheckContains(path, contents, "AccumulateRowFp16");
    CheckContains(path, contents, "TileXREpCombineHalfBitsToFloat");
    CheckContains(path, contents, "uint16_t u;");
    CheckContains(path, contents, "float16_t h;");
    CheckContains(path, contents, "return static_cast<float>(value.h);");
    CheckContains(path, contents, "auto src = reinterpret_cast<__gm__ uint16_t *>(srcGM)");
    CheckContains(path, contents, "TileXREpCombineHalfBitsToFloat(ld_dev(src + elem, 0))");
    CheckNotContains(path, contents, "const int32_t exponent");
    CheckNotContains(path, contents, "const int32_t mantissa");
    CheckContains(path, contents, "tilexr_ep_combine_cross_node_kernel");
    CheckContains(path, contents, "ScatterCombineRowsDataAsFlag");
    CheckContains(path, contents, "TileXREpDrainCrossNodeCombineRows");
    CheckContains(path, contents, "TileXREpClearDataAsFlagFlags");
    CheckContains(path, contents, "TileXREpFinalizeDataAsFlagFlags");
    CheckContains(path, contents, "TileXREpDataAsFlagRoundIndex");
    CheckContains(path, contents, "TileXREpDataAsFlagReadyTag");
    CheckContains(path, contents, "TileXREpCombineDataAsFlagRecvIfReady");
    CheckContains(path, contents, "TileXREpStoreCombineDataAsFlagStatusValue");
    CheckContains(path, contents, "CombineDataAsFlagSendRoundOffset");
    CheckContains(path, contents, "CombineDataAsFlagRemoteRecvRoundOffset");
    CheckContains(path, contents, "tilexr_ep_combine_udma_wqe_simt.h");
    CheckContains(path, contents, "TileXREpPostCombineDataAsFlagWqes");
    CheckContains(path, contents, "recvDone[TileXR::TILEXR_MAX_RANK_SIZE]");
    CheckContains(path, contents, "remainingPeers");
    CheckContains(path, contents, "madeProgress");
    CheckContains(path, contents, "while (remainingPeers > 0");
    CheckContains(path, contents, "TileXR::TILEXR_UDMA_MAX_RETRY_TIMES");
    CheckNotContains(path, contents, "tilexr_ep_combine_cross_node_drain_kernel");
    CheckNotContains(path, contents, "launch_tilexr_ep_combine_cross_node_drain_kernel");
    CheckNotContains(path, contents, "TileXREpNotifyRemoteUdmaReadySeparate(args");
    CheckNotContains(path, contents, "TileXREpWaitRemoteUdmaReady(sendWindow");
    CheckNotContains(path, contents, "DataAsFlagSend(");
    CheckNotContains(path, contents, "TileXR::UDMAPutNbi<uint8_t>(args, peer");
    CheckNotContains(path, contents, "tilexr_ep_dispatch_kernel");
    CheckNotContains(path, contents, "kCombineDafDebugStopStage");
    CheckNotContains(path, contents, "TileXREpWaitFirstDataAsFlagReadyWord");

    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "TileXREpLaunchCombineKernel");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "launch_tilexr_ep_combine_kernel");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch,
            "launch_tilexr_ep_combine_cross_node_kernel");
        CheckNotContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch,
            "launch_tilexr_ep_combine_cross_node_drain_kernel");
        CheckNotContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch,
            "aclrtSynchronizeStream(params.stream)");
        CheckNotContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch,
            "ACL_MEMCPY_DEVICE_TO_HOST");
    }
}

void TestKernelCommonHasCombineHelpers()
{
    const std::string path = "src/ep/kernels/tilexr_ep_kernel_common.h";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "UDMASecondOperationOffset");
    CheckContains(path, contents, "TileXREpNotifyRemoteUdmaReadySeparate");
    CheckContains(path, contents, "TileXREpWaitRemoteUdmaReady");
    CheckContains(path, contents, "TileXREpStoreStatusValue");
    CheckContains(path, contents, "TileXREpFlushUdmaSourceWindow");
    CheckContains(path, contents, "IsValidShape");
    CheckContains(path, contents, "CombineDataAsFlagRecvWindowOffset");
    CheckContains(path, contents, "CombineDataAsFlagSendWindowOffset");
    CheckContains(path, contents, "CombineDataAsFlagSendRoundOffset");
    CheckContains(path, contents, "CombineDataAsFlagRemoteRecvWindowOffset");
    CheckContains(path, contents, "CombineDataAsFlagRemoteRecvRoundOffset");
    CheckContains(path, contents, "CombineDataAsFlagStatusOffset");
    CheckContains(path, contents, "TileXREpDataAsFlagReadyTag");
    CheckContains(path, contents, "TileXREpDataAsFlagRoundIndex");
    CheckContains(path, contents, "TileXREpReadyValue");

    std::string epWindow;
    if (ReadFile("src/ep/common/ep_window.h", &epWindow)) {
        CheckContains("src/ep/common/ep_window.h", epWindow, "kEpStepCombineDataAsFlagReady");
    }

    std::string combine;
    if (ReadFile("src/ep/kernels/tilexr_ep_combine_kernel.cpp", &combine)) {
        CheckContains("src/ep/kernels/tilexr_ep_combine_kernel.cpp", combine, "kEpStatusRemoteReadyTimeout");
    }
}

void TestCombineDataAsFlagHelperFile()
{
    const std::string path = "src/ep/kernels/tilexr_ep_combine_data_as_flag.h";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "tilexr_data_as_flag.h");
    CheckContains(path, contents, "TileXREpCopyGmToDataAsFlagPayload");
    CheckContains(path, contents, "TileXREpStoreDataAsFlagSlotHeader");
    CheckContains(path, contents, "TileXREpWriteDataAsFlagAssist");
    CheckContains(path, contents, "TileXREpClearDataAsFlagFlags");
    CheckContains(path, contents, "TileXREpFinalizeDataAsFlagFlags");
    CheckNotContains(path, contents, "TileXREpWaitFirstDataAsFlagReadyWord");
    CheckContains(path, contents, "TileXREpCombineDataAsFlagCheckExpected");
    CheckContains(path, contents, "TileXREpCombineDataAsFlagRecvIfReady");
    CheckContains(path, contents, "kCombineDafRecvNotReady");
    CheckContains(path, contents, "kCombineDafRecvDone");
    CheckContains(path, contents, "kCombineDafRecvError");
    CheckContains(path, contents, "TileXREpRevalidateDataAsFlagAfterRecv");
    CheckContains(path, contents, "TileXREpInvalidateDataAsFlagPayload");
    CheckContains(path, contents, "TileXREpCopyDataAsFlagPayloadTailBypass");
    CheckContains(path, contents, "TileXREpInvalidateLocalCacheLines");
    CheckContains(path, contents,
        "static_assert(TileXR::DATA_AS_FLAG_ALIGN_BYTES % sizeof(uint64_t) == 0U");
    CheckContains(path, contents,
        "static_assert(kCombineDafPayloadCacheSafeBytes % static_cast<int64_t>(sizeof(uint64_t)) == 0");
    CheckContains(path, contents,
        "tail bypass from silently dropping sub-word tails");
    const std::string flagsMatchBody = SliceBetween(contents,
        "TileXREpDataAsFlagFlagsMatchExpected",
        "TileXREpCombineDataAsFlagCheckExpected");
    CheckNotContains("TileXREpDataAsFlagFlagsMatchExpected", flagsMatchBody,
        "TileXREpInvalidateLocalCacheLines(flag");
    CheckContains(path, contents, "ld_dev(flagWords + idx, 0)");
    CheckContains(path, contents, "ld_dev(src + idx, 0)");
    CheckContains(path, contents, "st_dev(value, dst + idx, 0)");
    CheckNotContains(path, contents, "LoadUint64FromGm(flagAddr");
    const std::string invalidateBody = SliceBetween(contents,
        "TileXREpInvalidateDataAsFlagPayload",
        "TileXREpCopyDataAsFlagPayloadToGm");
    CheckContains("TileXREpInvalidateDataAsFlagPayload", invalidateBody,
        "kCombineDafPayloadCacheSafeBytes");
    CheckNotContains("TileXREpInvalidateDataAsFlagPayload", invalidateBody,
        "DATA_AS_FLAG_BLOCK_BYTES");
    CheckNotContains(path, contents, "TileXREpCombineDataAsFlagCheckExpectedAndRecvWithRetry");
    const std::string recvIfReadyBody = SliceBetween(contents,
        "TileXREpCombineDataAsFlagRecvIfReady",
        "} // namespace");
    CheckOrdered("TileXREpCombineDataAsFlagRecvIfReady", recvIfReadyBody,
        "if (!TileXREpCombineDataAsFlagCheckExpected", "TileXREpCopyDataAsFlagPayloadToGm");
    CheckOrdered("TileXREpCombineDataAsFlagRecvIfReady", recvIfReadyBody,
        "TileXREpCopyDataAsFlagPayloadToGm", "TileXREpRevalidateDataAsFlagAfterRecv");
    CheckOrdered("TileXREpCombineDataAsFlagRecvIfReady", recvIfReadyBody,
        "TileXREpRevalidateDataAsFlagAfterRecv", "TileXREpClearDataAsFlagFlags");
    CheckContains(path, contents, "DATA_AS_FLAG_PAYLOAD_BYTES");
    CheckContains(path, contents, "DATA_AS_FLAG_FLAG_OFFSET_BYTES");
    CheckContains(path, contents, "DATA_AS_FLAG_FLAG_BYTES");
    CheckContains(path, contents, "TileXR::UDMACleanCacheLines");
    CheckContains(path, contents, "expectedReadyTag");
    CheckNotContains(path, contents, "DATA_AS_FLAG_READY_VALUE");
    CheckNotContains(path, contents, "DataAsFlagCheck(");
    CheckNotContains(path, contents, "DataAsFlagCheckAndRecv(");
    CheckNotContains(path, contents, "TileXR::UDMAPutNbi");
    CheckNotContains(path, contents, "TileXREpNotifyRemoteUdmaReadySeparate");
}

void TestCombineSimtWqeHelperFile()
{
    const std::string path = "src/ep/kernels/tilexr_ep_combine_udma_wqe_simt.h";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "simt_api/asc_simt.h");
    CheckContains(path, contents, "struct CombineDafWqeTask");
    CheckContains(path, contents, "struct CombineDafDoorbellTask");
    CheckContains(path, contents, "TILEXR_COMBINE_DAF_WQE_MODE_API");
    CheckContains(path, contents, "TILEXR_COMBINE_DAF_WQE_MODE_SCALAR_RAW");
    CheckContains(path, contents, "TILEXR_COMBINE_DAF_WQE_MODE_SIMT_RAW");
    CheckContains(path, contents, "#define TILEXR_COMBINE_DAF_WQE_MODE_SIMT_RAW 1");
    CheckContains(path, contents, "TileXR::UDMAPutNbi<uint8_t>");
    CheckContains(path, contents, "__simt_vf__");
    CheckContains(path, contents, "TileXREpCombineSimtWriteWqeVf");
    CheckContains(path, contents, "asc_vf_call");
    CheckContains(path, contents, "threadIdx.x");
    CheckContains(path, contents, "blockDim.x");
    CheckContains(path, contents, "asc_syncthreads");
    CheckContains(path, contents, "TileXREpCombineSimtWriteWqeRaw");
    CheckContains(path, contents, "TileXR::UDMACleanCacheLines");
    CheckContains(path, contents, "static_cast<uint64_t>(tasks[idx].wqeSize)");
    CheckNotContains(path, contents, "dcci(");
    CheckContains(path, contents, "remoteAddr");
    CheckContains(path, contents, "wqeAddr");
    CheckContains(path, contents, "rmtEidL");
    CheckContains(path, contents, "TileXR::UDMAPollCQWhenSQOverflow");
    CheckContains(path, contents, "wq->tailAddr != 0");
    CheckContains(path, contents, "scq->dbAddr != 0");
    CheckContains(path, contents, "st_dev(");
    CheckContains(path, contents, "TileXR::UDMAQuiet");
    CheckContains(path, contents, "roundIndex");
    CheckContains(path, contents, "CombineDataAsFlagRemoteRecvRoundOffset");
    CheckNotContains(path, contents, "AscendC::Simt::VF_CALL");
    CheckNotContains(path, contents, "AscendC::Simt::Dim3");
    CheckNotContains(path, contents, "Simt::ThreadBarrier");
    CheckNotContains(path, contents, "HcclAiRMAWQ");
    CheckNotContains(path, contents, "aclshmem");
    CheckNotContains(path, contents, "AscendC::printf");
    CheckNotContains(path, contents, "TileXREpCommitCombineDataAsFlagScalarRawTaskByTask");
    CheckNotContains(path, contents, "TileXREpCombineScalarWriteWqeRaw");
    CheckNotContains(path, contents, "constexpr bool kCombineDafUseApiWqe");
    CheckNotContains(path, contents, "constexpr bool kCombineDafUseSimtWqe");
}

void TestCombineUdmaReservationPlansBeforeCommit()
{
    const std::string path = "src/ep/kernels/tilexr_ep_combine_udma_wqe_simt.h";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXREpPlanCombineDataAsFlagPeerWqeReservations");
    CheckContains(path, contents, "TileXREpCommitCombineDataAsFlagPeerWqeReservations");

    const std::string planBody = SliceBetween(contents,
        "TileXREpPlanCombineDataAsFlagPeerWqeReservations",
        "TileXREpCommitCombineDataAsFlagPeerWqeReservations");
    const std::string commitBody = SliceBetween(contents,
        "TileXREpCommitCombineDataAsFlagPeerWqeReservations",
        "__simt_vf__");
    const std::string postBody = SliceBetween(contents,
        "TileXREpPostCombineDataAsFlagWqes",
        "} // namespace");
    CheckNotContains("TileXREpPlanCombineDataAsFlagPeerWqeReservations", planBody, "st_dev(");
    CheckNotContains("TileXREpReserveCombineDataAsFlagPeerWqes", SliceBetween(contents,
        "TileXREpReserveCombineDataAsFlagPeerWqes",
        "__simt_callee__"), "TileXREpCommitCombineDataAsFlagPeerWqeReservations");
    CheckContains("TileXREpCommitCombineDataAsFlagPeerWqeReservations", commitBody,
        "reinterpret_cast<__gm__ uint32_t *>(doorbells[idx].dbAddr)");
    CheckContains("TileXREpCommitCombineDataAsFlagPeerWqeReservations", commitBody,
        "reinterpret_cast<__gm__ uint32_t *>(doorbells[idx].headAddr)");
    CheckContains("TileXREpCommitCombineDataAsFlagPeerWqeReservations", commitBody,
        "reinterpret_cast<__gm__ uint32_t *>(doorbells[idx].wqeCntAddr)");
    CheckOrdered("TileXREpPostCombineDataAsFlagWqes", postBody, "TileXREpReserveCombineDataAsFlagPeerWqes",
        "TileXR::UDMACleanCacheLines");
    CheckOrdered("TileXREpPostCombineDataAsFlagWqes", postBody, "TileXR::UDMACleanCacheLines",
        "TileXREpCommitCombineDataAsFlagPeerWqeReservations(doorbells");
}

void TestCrossNodeCombineAbortAndVisibilityGuards()
{
    const std::string path = "src/ep/kernels/tilexr_ep_combine_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXREpAbortCrossNodeCombineLocal");
    CheckContains(path, contents, "TileXREpAbortCrossNodeCombineLocal(sync");
    CheckContains(path, contents, "kEpStepCombineRelayReady");
    CheckContains(path, contents, "TileXREpFlushCrossNodeSameNodeIpcSlots");
    CheckContains(path, contents, "TileXREpInvalidateLocalCacheLines(sourceWindow + SlotOffset(expertRank, slotBytes), slotBytes)");
}

void TestCombineWqeModeBuildControls()
{
    std::string epCMake;
    if (ReadFile("src/ep/CMakeLists.txt", &epCMake)) {
        CheckContains("src/ep/CMakeLists.txt", epCMake, "TILEXR_COMBINE_DAF_WQE_MODE");
        CheckContains("src/ep/CMakeLists.txt", epCMake,
            "set(TILEXR_COMBINE_DAF_WQE_MODE \"simt_raw\"");
        CheckContains("src/ep/CMakeLists.txt", epCMake, "api scalar_raw simt_raw");
        CheckContains("src/ep/CMakeLists.txt", epCMake, "TILEXR_COMBINE_DAF_WQE_MODE_API");
        CheckContains("src/ep/CMakeLists.txt", epCMake, "TILEXR_COMBINE_DAF_WQE_MODE_SCALAR_RAW");
        CheckContains("src/ep/CMakeLists.txt", epCMake, "TILEXR_COMBINE_DAF_WQE_MODE_SIMT_RAW");
    }

    std::string buildScript;
    if (ReadFile("tests/ep/build.sh", &buildScript)) {
        CheckContains("tests/ep/build.sh", buildScript, "TILEXR_COMBINE_DAF_WQE_MODE");
        CheckContains("tests/ep/build.sh", buildScript,
            "COMBINE_DAF_WQE_MODE=\"${TILEXR_COMBINE_DAF_WQE_MODE:-simt_raw}\"");
        CheckContains("tests/ep/build.sh", buildScript, "api|scalar_raw|simt_raw");
        CheckContains("tests/ep/build.sh", buildScript, "-DTILEXR_COMBINE_DAF_WQE_MODE=");
    }
}

void TestDispatchDemoRunsCombine()
{
    const std::string path = "tests/ep/demo/tilexr_ep_dispatch_demo.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TileXRMoeEpCombine");
    CheckContains(path, contents, "ValidateCombineOutputs");
    CheckContains(path, contents, "combine validation");
    CheckNotContains(path, contents, "DumpSlotHeader");
    CheckNotContains(path, contents, "TILEXR_EP_DEMO_DUMP_WINDOW");
    CheckNotContains(path, contents, "TILEXR_EP_DEMO_DUMP_COMBINE_WINDOW");
}

void TestKernelForwardsActiveMask()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "xActiveMask");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "xActiveMaskGM");
    }

    std::string dispatchHelpers;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_helpers.h", &dispatchHelpers)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_helpers.h", dispatchHelpers, "TileXREpIsTokenActive");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_helpers.h", dispatchHelpers, "xActiveMaskGM == nullptr");
    }
}

void TestKernelForwardsExpertTokenNumsType()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "expertTokenNumsType");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "expertTokenNumsType");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpClearExpertTokenNums");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpFinalizeExpertTokenNums");
    }

    std::string combineHelpers;
    if (ReadFile("src/ep/kernels/tilexr_ep_combine_helpers.h", &combineHelpers)) {
        CheckContains("src/ep/kernels/tilexr_ep_combine_helpers.h", combineHelpers,
            "TileXREpIncrementExpertTokenNum");
        CheckContains("src/ep/kernels/tilexr_ep_combine_helpers.h", combineHelpers,
            "running += expertTokenNumsOut[localExpert]");
    }
}

void TestKernelForwardsTpRecvCountsOut()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "tpRecvCountsOut");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "tpWorldSize");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "tpRankId");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "tpRecvCountsOutGM");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpAppendTpGroupRows");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpTpGroupStartRank");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "tpWorldSize");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "tpRankId");
    }

    const std::string demoPath = "tests/ep/demo/tilexr_ep_dispatch_demo.cpp";
    std::string demo;
    if (ReadFile(demoPath, &demo)) {
        CheckContains(demoPath, demo, "TILEXR_EP_DEMO_TP_RECV_COUNTS");
        CheckContains(demoPath, demo, "TILEXR_EP_DEMO_TP_WORLD_SIZE");
        CheckContains(demoPath, demo, "TILEXR_EP_DEMO_TP_RANK_ID");
        CheckContains(demoPath, demo, "BuildExpectedTpRoutes");
        CheckContains(demoPath, demo, "ValidateTpRecvCounts");
    }
}

void TestDispatchDemoExercisesV2OptionalInputs()
{
    const std::string path = "tests/ep/demo/tilexr_ep_dispatch_demo.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckContains(path, contents, "TILEXR_EP_DEMO_ACTIVE_MASK");
    CheckContains(path, contents, "TILEXR_EP_DEMO_EXPERT_TOKEN_NUMS_TYPE");
    CheckContains(path, contents, "xActiveMaskDev");
    CheckContains(path, contents, "expertTokenNumsType");
}

void TestKernelForwardsSharedExpertConfig()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "sharedExpertNum");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "sharedExpertRankNum");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpRouteToDstRank");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "sharedExpertRankNum");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpLocalExpertCount");
    }

    std::string demo;
    if (ReadFile("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", &demo)) {
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "TILEXR_EP_DEMO_SHARED_EXPERT_NUM");
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "TILEXR_EP_DEMO_SHARED_EXPERT_RANK_NUM");
    }
}

void TestKernelForwardsStaticQuantConfig()
{
    std::string hostLaunch;
    if (ReadFile("src/ep/host/ep_kernel_launch.cpp", &hostLaunch)) {
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "scales");
        CheckContains("src/ep/host/ep_kernel_launch.cpp", hostLaunch, "quantMode");
    }

    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "scalesGM");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "quantMode");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpCopyStaticQuantRoutePayload");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "TileXREpClampInt8");
    }

    std::string demo;
    if (ReadFile("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", &demo)) {
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "TILEXR_EP_DEMO_QUANT_MODE");
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "TILEXR_EP_DEMO_STATIC_QUANT_SCALE");
    }
}

void TestKernelForwardsPerTokenDynamicQuantConfig()
{
    std::string kernel;
    if (ReadFile("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", &kernel)) {
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "kEpQuantModePerTokenDynamic");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel,
            "TileXREpCopyPerTokenDynamicQuantRoutePayload");
        CheckContains("src/ep/kernels/tilexr_ep_dispatch_kernel.cpp", kernel, "dynamicScalesOutGM");
    }

    std::string demo;
    if (ReadFile("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", &demo)) {
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "usePerTokenDynamicQuant");
        CheckContains("tests/ep/demo/tilexr_ep_dispatch_demo.cpp", demo, "DynamicScaleForXValue");
    }
}

void TestClearLocalWindowDoesNotPreclearSlotHeaders()
{
    const std::string path = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
    std::string contents;
    if (!ReadFile(path, &contents)) {
        return;
    }

    CheckNotContains(path, contents,
        "for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {\n"
        "        StoreSlotHeader(localWindow + SlotOffset(srcRank, slotBytes), 0, srcRank, 0, 0, tBuf);\n"
        "    }");
}

void TestNoForbiddenDependencies()
{
    const std::vector<std::string> paths = {
        "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp",
        "src/ep/kernels/tilexr_ep_combine_kernel.cpp",
        "src/ep/kernels/tilexr_ep_kernel_common.h",
        "src/ep/kernels/tilexr_ep_dispatch_helpers.h",
        "src/ep/kernels/tilexr_ep_combine_helpers.h",
        "src/ep/host/ep_kernel_launch.cpp",
        "src/ep/CMakeLists.txt",
    };
    const std::vector<std::string> forbidden = {
        "src/mc2",
        "3rdparty/ops-transformer",
        "GetHcclContext",
        "TileXRUDMARegister",
        "shmem",
    };

    for (std::vector<std::string>::const_iterator path = paths.begin(); path != paths.end(); ++path) {
        std::string contents;
        if (!ReadFile(*path, &contents)) {
            continue;
        }
        for (std::vector<std::string>::const_iterator needle = forbidden.begin(); needle != forbidden.end(); ++needle) {
            CheckNotContains(*path, contents, *needle);
        }
    }
}

} // namespace

int main()
{
    TestKernelUsesTileXRPeerMemory();
    TestCrossNodeDispatchUsesUDMARegistry();
    TestCrossNodeDispatchPullsRemoteSlots();
    TestCrossNodeDispatchSeparatesLocalAndRemotePeers();
    TestHostDispatchSplitsCrossNodeKernel();
    TestDispatchHelpersLiveInDispatchHelperFile();
    TestCombineHelpersLiveInCombineHelperFile();
    TestCombineKernelUsesTileXRPeerMemory();
    TestKernelCommonHasCombineHelpers();
    TestCombineDataAsFlagHelperFile();
    TestCombineSimtWqeHelperFile();
    TestCombineUdmaReservationPlansBeforeCommit();
    TestCrossNodeCombineAbortAndVisibilityGuards();
    TestCombineWqeModeBuildControls();
    TestDispatchDemoRunsCombine();
    TestKernelForwardsActiveMask();
    TestKernelForwardsExpertTokenNumsType();
    TestKernelForwardsTpRecvCountsOut();
    TestDispatchDemoExercisesV2OptionalInputs();
    TestKernelForwardsSharedExpertConfig();
    TestKernelForwardsStaticQuantConfig();
    TestKernelForwardsPerTokenDynamicQuantConfig();
    TestClearLocalWindowDoesNotPreclearSlotHeaders();
    TestNoForbiddenDependencies();
    if (g_failures != 0) {
        std::cerr << g_failures << " TileXR EP kernel source checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR EP kernel source checks passed" << std::endl;
    return 0;
}
