#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TILEXR_SOURCE_ROOT
#error TILEXR_SOURCE_ROOT must be defined
#endif

namespace {

std::string Read(const std::string &path)
{
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

std::string Lower(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

bool Require(const std::string &text, const std::string &needle,
    const char *message)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

bool Reject(const std::string &text, const std::string &needle,
    const char *message)
{
    if (text.find(needle) != std::string::npos) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

bool RequireBefore(const std::string &text, const std::string &first,
    const std::string &second, const char *message)
{
    const std::size_t firstPos = text.find(first);
    const std::size_t secondPos = text.find(second);
    if (firstPos == std::string::npos || secondPos == std::string::npos ||
        firstPos >= secondPos) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

std::size_t CountOccurrences(const std::string &text,
    const std::string &needle)
{
    std::size_t count = 0U;
    for (std::size_t position = text.find(needle);
        position != std::string::npos;
        position = text.find(needle, position + needle.size())) {
        ++count;
    }
    return count;
}

std::string Section(const std::string &text, const std::string &begin,
    const std::string &end)
{
    const std::size_t beginPos = text.find(begin);
    if (beginPos == std::string::npos) {
        return std::string();
    }
    const std::size_t endPos = text.find(end, beginPos + begin.size());
    return text.substr(beginPos, endPos == std::string::npos ?
        std::string::npos : endPos - beginPos);
}

} // namespace

int main()
{
    const std::string root = TILEXR_SOURCE_ROOT;
    const std::string rootCmake = Read(root + "/CMakeLists.txt");
    const std::string moonEpCmake = Read(root + "/src/moonep/CMakeLists.txt");
    const std::string v1Cmake = Read(root + "/src/moonep/combine/CMakeLists.txt");
    const std::string v2Cmake = Read(root + "/src/moonep/combine_v2/CMakeLists.txt");
    const std::string kernelEntry = Read(root +
        "/src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.cpp");
    const std::string kernelImpl = Read(root +
        "/src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h");
    const std::string profileHeader = Read(root +
        "/src/moonep/combine_v2/common/combine_v2_profile.h");
    const std::string hardwareProbe = Read(root +
        "/tests/moonep_combine_v2/demo/tilexr_moonep_combine_v2_hardware_probe.cpp");
    const std::string traceTool = Read(root +
        "/tools/moonep/combine_v2_trace.py");
    const std::string clusterLauncher = Read(root +
        "/tools/moonep/run_combine_v2_perf_cluster.sh");
    const std::string multihostLauncher = Read(root +
        "/tools/moonep/run_combine_v2_perf_multihost.sh");
    const std::string host = Read(root +
        "/src/moonep/combine_v2/host/combine_v2_host.cpp");
    const std::string hostHeader = Read(root +
        "/src/moonep/combine_v2/host/combine_v2_host.h");
    const std::string launch = Read(root +
        "/src/moonep/combine_v2/host/combine_v2_launch.cpp");
    const std::string publicHeader = Read(root +
        "/src/include/tilexr_moonep_combine_v2.h");
    const std::string registration = Read(root +
        "/src/moonep/common/moonep_kernel_registration.h");
    const std::string peerPrefill = Section(kernelImpl,
        "inline void MoonEpCombineV2PrefillPeerWqesVf(",
        "inline void MoonEpCombineV2PrefillFullmeshWqesVf(");
    const std::string fullmeshPrefill = Section(kernelImpl,
        "inline void MoonEpCombineV2PrefillFullmeshWqesVf(",
        "inline void MoonEpCombineV2BuildPayloadWqesVf(");
    const std::string directBuilder = Section(kernelImpl,
        "inline void MoonEpCombineV2BuildPayloadWqesVf(",
        "inline void MoonEpCombineV2BuildFullmeshPayloadWqesVf(");
    const std::string fullmeshBuilder = Section(kernelImpl,
        "inline void MoonEpCombineV2BuildFullmeshPayloadWqesVf(",
        "inline void MoonEpCombineV2BuildFullSyncWqesVf(");
    const std::string fullSyncBuilder = Section(kernelImpl,
        "inline void MoonEpCombineV2BuildFullSyncWqesVf(",
        "#endif");
    const std::string vectorSelection = Section(kernelImpl,
        "__aicore__ inline uint32_t MoonEpCombineV2::SelectPeerIndices(",
        "__aicore__ inline uint64_t MoonEpCombineV2::LoadToken(");
    const std::string submitPair = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::SubmitPair(",
        "__aicore__ inline bool MoonEpCombineV2::SendRemoteStep(");
    const std::string fullmeshSubmit = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::SubmitFullmeshBatch(",
        "__aicore__ inline bool MoonEpCombineV2::SendFullmeshStep(");
    const std::string fullmeshSend = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::SendFullmeshStep(",
        "__aicore__ inline void MoonEpCombineV2::CopySelfRowsIn(");
    const std::string fullmeshWait = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::WaitFullmeshCq(",
        "__aicore__ inline bool MoonEpCombineV2::WaitStepGrant(");
    const std::string selfStep = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::SendSelfStep(",
        "__aicore__ inline bool MoonEpCombineV2::WaitInboundDone(");
    const std::string selfGrant = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::SubmitSelfGrant(",
        "__aicore__ inline bool MoonEpCombineV2::SendSelfStep(");
    const std::string inboundDone = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::WaitInboundDone()",
        "__aicore__ inline void MoonEpCombineV2::InitReduceBuffers()");
    const std::string localGrant = Section(kernelImpl,
        "__aicore__ inline void MoonEpCombineV2::PublishLocalGrant(",
        "__aicore__ inline void MoonEpCombineV2::CopyIssueToSq(");
    const std::string process = Section(kernelImpl,
        "__aicore__ inline void MoonEpCombineV2::Process()",
        "} // namespace");
    const std::string fullSyncPublisher = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::PublishFullSyncBatch(",
        "__aicore__ inline bool MoonEpCombineV2::WaitFullSyncCq(");
    const std::string fullSyncWait = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::WaitFullSyncCq(",
        "__aicore__ inline bool MoonEpCombineV2::FullSyncSignalMatches(");
    const std::string barrierServer = Section(hardwareProbe,
        "bool BarrierServer(", "bool BarrierClient(");
    const std::string timedBatch = Section(hardwareProbe,
        "CheckAcl(rank, \"aclrtRecordEvent batch start\"",
        "if (options.profile) {");

    bool ok = true;
    ok &= Require(rootCmake, "add_subdirectory(tests/moonep_combine_v2)",
        "Combine V2 tests are not wired at the root");
    ok &= Require(moonEpCmake, "add_subdirectory(combine_v2)",
        "Combine V2 source directory is not configured");
    ok &= Require(moonEpCmake, "tilexr-moonep-combine-v2",
        "Combine V2 is not linked into the MoonEP aggregate library");
    ok &= Reject(v1Cmake, "combine_v2",
        "Combine V1 build definition contains V2 changes");

    ok &= Require(timedBatch, "for (int iteration = 0; iteration < options.iterations; ++iteration)",
        "Combine V2 timed batch does not contain the iteration loop");
    ok &= Require(timedBatch, "LaunchCombine(",
        "Combine V2 timed batch does not launch the operator");
    ok &= Require(timedBatch, "aclrtRecordEvent(stopEvent, stream)",
        "Combine V2 timed batch does not record one batch stop event");
    ok &= Require(timedBatch, "aclrtSynchronizeEvent(stopEvent)",
        "Combine V2 timed batch does not synchronize after all launches");
    ok &= Reject(timedBatch, "BarrierAll(",
        "Combine V2 timed batch still contains a rank barrier");
    ok &= Reject(timedBatch, "CaptureProfileSamples(",
        "Combine V2 timed batch still copies profiles between launches");
    ok &= Reject(timedBatch, "COMBINE_V2_SAMPLE",
        "Combine V2 timed batch still logs between launches");
    ok &= Require(multihostLauncher, "rank_average[rank_key]",
        "Combine V2 launcher does not aggregate rank batch averages");
    ok &= Reject(multihostLauncher, "sample_count[rank_key] != iterations",
        "Combine V2 launcher still requires per-iteration samples");

    ok &= Require(v2Cmake, "add_library(tilexr-moonep-combine-v2 SHARED",
        "Combine V2 does not own a standalone shared library");
    ok &= Require(v2Cmake, "tilexr_moonep_combine_v2_kernel.cpp",
        "Combine V2 kernel entry is not the build source");
    ok &= Require(v2Cmake, "tilexr_moonep_combine_v2_kernel.h",
        "Combine V2 kernel implementation is not a build dependency");
    ok &= Require(v2Cmake, "-DCATLASS_ARCH=3510",
        "Combine V2 CATLASS target is missing");
    ok &= Require(v2Cmake, "--cce-auto-sync",
        "Combine V2 auto synchronization is missing");
    ok &= Require(v2Cmake, "SOVERSION 2",
        "Combine V2 SONAME is not version 2");
    ok &= Reject(v2Cmake, "/devlib",
        "Combine V2 links the toolkit stub library directory");

    ok &= Require(kernelEntry, "#include \"tilexr_moonep_combine_v2_kernel.h\"",
        "Combine V2 entry does not include its implementation header");
    ok &= Require(kernelEntry, "tilexr_moonep_combine_v2_kernel",
        "Combine V2 operator entry is missing");
    ok &= Require(kernelEntry, "op.Init(",
        "Combine V2 entry does not initialize the implementation");
    ok &= Require(kernelEntry, "op.Process();",
        "Combine V2 entry does not execute the implementation");
    ok &= RequireBefore(kernelEntry,
        "GetBlockIdx()) >= activeCoreCount", "AscendC::TPipe pipe;",
        "Combine V2 inactive blocks do not exit before TPipe construction");
    ok &= Reject(kernelEntry, "TileXR::UDMA",
        "Combine V2 entry contains algorithm implementation");
    ok &= Reject(kernelEntry, "InitBuffers",
        "Combine V2 entry contains implementation helpers");
    ok &= Reject(kernelEntry, "<<<",
        "Combine V2 entry contains Host launch syntax");
    if (static_cast<size_t>(std::count(kernelEntry.begin(),
            kernelEntry.end(), '\n')) > 40U) {
        std::cerr << "Combine V2 kernel entry is no longer thin\n";
        ok = false;
    }

    ok &= Require(kernelImpl, "class MoonEpCombineV2",
        "Combine V2 implementation class is missing from the header");
    ok &= Require(kernelImpl, "TileXR::UDMA",
        "Combine V2 implementation does not use UDMA");
    ok &= Require(kernelImpl, "UDMA_SHARED_QP",
        "Combine V2 implementation does not require shared QPs");
    ok &= Require(kernelImpl,
        "Simt::VF_CALL<MoonEpCombineV2PrefillOperatorWqesVf>",
        "Combine V2 implementation does not prefill operator WQE fields");
    ok &= Require(kernelImpl,
        "Simt::VF_CALL<MoonEpCombineV2PrefillPeerWqesVf>",
        "Combine V2 implementation does not prefill peer WQE fields");
    ok &= Require(peerPrefill, "words[word] = 0U;",
        "Combine V2 peer prefill does not clear overwritten control WQEs");
    ok &= Require(peerPrefill,
        "sqe->opcode = static_cast<uint32_t>(TileXR::UDMAOpcode::WRITE);",
        "Combine V2 peer prefill does not restore the payload opcode");
    ok &= Require(peerPrefill, "sqe->flag = 0U;",
        "Combine V2 peer prefill does not restore the payload flag");
    ok &= Require(peerPrefill,
        "sge->len = static_cast<uint32_t>(fields->rowBytes);",
        "Combine V2 peer prefill does not restore the payload SGE length");
    ok &= Require(peerPrefill, "sge->tokenId = 0U;",
        "Combine V2 peer prefill does not restore the payload SGE token");
    ok &= Require(kernelImpl,
        "Simt::VF_CALL<MoonEpCombineV2BuildPayloadWqesVf>",
        "Combine V2 implementation does not use its SIMT WQE builder");
    ok &= Require(kernelImpl,
        "Simt::VF_CALL<MoonEpCombineV2PrefillFullmeshWqesVf>",
        "Combine V2 implementation does not prefill Fullmesh WQEs");
    ok &= Require(kernelImpl,
        "Simt::VF_CALL<MoonEpCombineV2BuildFullmeshPayloadWqesVf>",
        "Combine V2 implementation does not use its Fullmesh SIMT builder");
    ok &= Require(kernelImpl,
        "Simt::VF_CALL<MoonEpCombineV2BuildFullSyncWqesVf>",
        "Combine V2 full-sync path does not use a dedicated SIMT builder");
    ok &= Require(kernelImpl,
        "__simt_vf__ __aicore__ LAUNCH_BOUND(\n"
        "    TileXRMoonEp::kMoonEpCombineV2BuilderThreads)\n"
        "inline void MoonEpCombineV2BuildFullSyncWqesVf(",
        "Combine V2 full-sync WQE builder is not a SIMT VF");
    ok &= Require(fullSyncBuilder, "words[word] = 0U;",
        "Combine V2 full-sync builder does not clear the complete WQE");
    ok &= Require(fullSyncBuilder, "task + 1U == context->count",
        "Combine V2 full-sync completion is not limited to the terminal WQE");
    ok &= Require(fullSyncBuilder,
        "sge->len = TileXRMoonEp::kMoonEpCombineV2FullSyncSignalBytes;",
        "Combine V2 full-sync WQE does not send the 32-byte signal");
    ok &= Require(fullSyncBuilder, "sge->va = context->localSignalAddr;",
        "Combine V2 full-sync WQE lacks its local ping-pong source");
    ok &= Require(fullSyncBuilder, "remote->remoteRowBase",
        "Combine V2 full-sync WQE lacks its remote receive address");
    ok &= Reject(fullSyncBuilder, "__gm__",
        "Combine V2 full-sync SIMT builder accesses GM directly");
    ok &= Require(kernelImpl,
        "CreateVecIndex(slotIndexBuf_.Get<int16_t>()",
        "Combine V2 does not initialize reusable chunk indices");
    ok &= Require(vectorSelection, "Compares(lowerMask, dstSlots",
        "Combine V2 vector selection lacks the lower-bound comparison");
    ok &= Require(vectorSelection, "CMPMODE::GE",
        "Combine V2 vector selection does not include the peer lower bound");
    ok &= Require(vectorSelection, "Compares(upperMask, dstSlots",
        "Combine V2 vector selection lacks the upper-bound comparison");
    ok &= Require(vectorSelection, "CMPMODE::LT",
        "Combine V2 vector selection does not exclude the peer upper bound");
    ok &= Require(vectorSelection,
        "And(lowerMask.ReinterpretCast<uint16_t>()",
        "Combine V2 vector selection does not combine its masks");
    ok &= Require(vectorSelection, "GatherMask(selectedIndexBuf_.Get<int16_t>()",
        "Combine V2 vector selection does not compact chunk indices");
    ok &= Require(directBuilder,
        "context->batchOffset + task",
        "Combine V2 WQE builder does not advance through compacted batches");
    ok &= Require(directBuilder, "selectedIndices[densePosition]",
        "Combine V2 WQE builder does not read compacted chunk indices");
    ok &= Require(directBuilder, "dstSlots[relativeIndex]",
        "Combine V2 WQE builder does not decode the selected destination");
    ok &= Require(directBuilder, "context->chunkStart + relativeIndex",
        "Combine V2 WQE builder does not reconstruct the source slot");
    ok &= Require(directBuilder, "context->peerBase",
        "Combine V2 WQE builder does not decode target slots by subtraction");
    ok &= Reject(kernelImpl, "MoonEpCombineV2RouteEntry",
        "Combine V2 retained the obsolete RouteEntry representation");
    ok &= Reject(kernelImpl, "MoonEpCombineV2SelectPeerRoutesVf",
        "Combine V2 retained the obsolete SIMT route selector");
    ok &= Reject(kernelImpl, "asc_atomic_add",
        "Combine V2 retained atomic route reservation");
    ok &= Reject(kernelImpl, "simt_api/device_atomic_functions.h",
        "Combine V2 retained the obsolete SIMT atomic include");
    ok &= Require(kernelImpl,
        "constexpr uint32_t kSixPortPayloadCapacity = 96U;",
        "Combine V2 six-port payload capacity is missing");
    ok &= Require(kernelImpl,
        "constexpr uint32_t kTwoPortPayloadCapacity = 32U;",
        "Combine V2 two-port payload capacity is missing");
    ok &= Require(kernelImpl,
        "constexpr uint32_t kSixPortIssueCapacity = 98U;",
        "Combine V2 six-port issue capacity is missing");
    ok &= Require(kernelImpl,
        "constexpr uint32_t kTwoPortIssueCapacity = 34U;",
        "Combine V2 two-port issue capacity is missing");
    ok &= Require(kernelImpl, "TBuf<QuePosition::VECCALC> wqeIssueBuf_;",
        "Combine V2 continuous WQE issue buffer is missing");
    ok &= Require(kernelImpl,
        "issue[kSixPortIssueBytes].GetPhyAddr()",
        "Combine V2 two-port WQE view does not follow the six-port region");
    ok &= Require(kernelImpl,
        "pipe_->InitBuffer(selfCopyQueue_, 2U,",
        "Combine V2 Self copy queue is not double buffered");
    ok &= Require(selfStep, "SelectPeerIndices(peer, chunkElements)",
        "Combine V2 Self step does not reuse vector selection");
    ok &= Require(selfStep, "CopySelfSelectedIndices(selectedCount,",
        "Combine V2 Self step does not consume compacted indices");
    ok &= Require(selfStep, "SubmitSelfGrant(step)",
        "Combine V2 Self step does not publish its step grant");
    ok &= Require(selfStep, "PublishSelfDone(step)",
        "Combine V2 Self step does not publish local completion");
    ok &= RequireBefore(selfStep, "PublishSelfDone(step)",
        "SubmitSelfGrant(step)",
        "Combine V2 Self completion is not published before its grant");
    ok &= Require(selfGrant,
        "TileXR::TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION",
        "Combine V2 Self grant does not request ordered completion");
    ok &= Require(selfGrant, "if (successor == rank_)",
        "Combine V2 Self grant does not handle a local successor");
    ok &= Require(selfGrant, "PublishLocalGrant(step, lane)",
        "Combine V2 Self grant does not publish a local successor token");
    ok &= Require(localGrant,
        "kMoonEpCombineV2GrantReceiveOffsetBytes",
        "Combine V2 local grant does not target the receive slot");
    ok &= RequireBefore(selfGrant, "SyncFunc<HardEvent::S_MTE3>();",
        "CopyIssueToSq(",
        "Combine V2 Self grant does not establish MTE3 ordering");
    ok &= RequireBefore(selfGrant, "SyncFunc<HardEvent::MTE3_S>();",
        "st_dev(lane_[lane].head",
        "Combine V2 Self grant rings a doorbell before MTE3 completion");
    ok &= Require(kernelImpl,
        "MoonEpCombineV2SelfRowsPerBatch(rowBytes_)",
        "Combine V2 Self copy does not derive its runtime row batch");
    ok &= Require(kernelImpl,
        "selectedIndices.GetValue(selectedStart + row)",
        "Combine V2 Self copy does not consume compacted indices");
    ok &= Require(kernelImpl, "dstSlots.GetValue(relativeIndex)",
        "Combine V2 Self copy does not decode compacted destinations");
    ok &= RequireBefore(kernelImpl, "CopySelfRowsIn(",
        "CopySelfRowsOut(",
        "Combine V2 Self copy helpers are not ordered as copy-in/copy-out");
    ok &= RequireBefore(kernelImpl,
        "SyncFunc<HardEvent::MTE2_MTE3>();",
        "CopySelfRowsOut(pendingStart",
        "Combine V2 Self copy does not wait for MTE2 before MTE3");
    ok &= RequireBefore(kernelImpl,
        "CopySelfRowsOut(pendingStart",
        "selfCopyQueue_.FreeTensor(pending);",
        "Combine V2 Self copy frees its relay before copy-out");
    ok &= Reject(kernelImpl, "dstRankBuf_",
        "Combine V2 retained the obsolete dst-rank UB buffer");
    ok &= Require(kernelImpl, "TBuf<QuePosition::VECCALC> selectedIndexBuf_;",
        "Combine V2 compacted-index UB buffer is missing");
    ok &= Require(kernelImpl, "TBuf<QuePosition::VECCALC> lowerMaskBuf_;",
        "Combine V2 lower-bound mask buffer is missing");
    ok &= Require(kernelImpl, "TBuf<QuePosition::VECCALC> upperMaskBuf_;",
        "Combine V2 upper-bound mask buffer is missing");
    ok &= Reject(kernelImpl, "compareMaskBuf_",
        "Combine V2 retained the obsolete compare-mask UB buffer");
    ok &= Reject(kernelImpl, "routeEntryBuf_",
        "Combine V2 retained the obsolete route-entry UB buffer");
    ok &= Reject(kernelImpl, "selectStateBuf_",
        "Combine V2 retained obsolete selector state");
    ok &= Reject(kernelImpl, "threadMaxSlotIdxBuf_",
        "Combine V2 retained obsolete selector cursors");
    ok &= Reject(kernelImpl, "descriptorBuf_",
        "Combine V2 retained the obsolete descriptor UB buffer");
    ok &= Reject(kernelImpl, "CopyBytesGmToGm(",
        "Combine V2 retained the serialized Self-copy helper");
    ok &= RequireBefore(submitPair, "SyncFunc<HardEvent::S_MTE3>();",
        "CopyIssueToSq(",
        "Combine V2 does not establish scalar/SIMT-to-MTE3 ordering");
    ok &= RequireBefore(submitPair, "CopyIssueToSq(",
        "SyncFunc<HardEvent::MTE3_S>();",
        "Combine V2 does not wait for SQ MTE3 publication");
    ok &= RequireBefore(submitPair, "SyncFunc<HardEvent::MTE3_S>();",
        "st_dev(lane_[0].head",
        "Combine V2 rings a doorbell before SQ MTE3 completion");
    ok &= Require(submitPair,
        "*grantSource = TileXRMoonEp::MoonEpCombineV2GrantToken(",
        "Combine V2 remote path does not publish every step grant");
    ok &= Reject(submitPair, "step + 1U < stepCount_",
        "Combine V2 remote path still suppresses the terminal grant");
    ok &= Reject(kernelImpl, "WaitAdmission",
        "Combine V2 retained the pre-step admission wait");
    ok &= Reject(kernelImpl, "WaitFinalCqs",
        "Combine V2 retained the standalone final CQ wait");
    ok &= RequireBefore(process, "SendRemoteStep(peer, step)",
        "WaitStepCqs(step)",
        "Combine V2 step loop does not send before waiting for CQ");
    ok &= RequireBefore(process, "WaitStepCqs(step)",
        "WaitStepGrant(step)",
        "Combine V2 step loop does not wait CQ before grant");
    ok &= RequireBefore(process, "WaitStepGrant(step)",
        "WaitInboundDone()",
        "Combine V2 final grant wait does not precede finalization");
    ok &= Require(inboundDone,
        "bool ready[TileXRMoonEp::kMoonEpCombineV2RankCount]",
        "Combine V2 Done wait does not cover every source rank");
    ok &= Require(inboundDone, "sourceIndex < rankSize_",
        "Combine V2 Done wait stops at the per-core source partition");
    ok &= Require(inboundDone, "const uint32_t source = sourceIndex;",
        "Combine V2 Done wait does not address every source directly");
    ok &= Require(inboundDone,
        "const uint32_t conditionCount = rankSize_ *",
        "Combine V2 Done condition count is not derived from all sources");
    ok &= Require(inboundDone, "% conditionCount",
        "Combine V2 Done cursor assumes a power-of-two rank size");
    ok &= Reject(inboundDone, "source == rank_",
        "Combine V2 Done wait treats unfinished Self copies as ready");
    ok &= Reject(inboundDone, "MoonEpCombineV2SourceForCore(",
        "Combine V2 Done wait still observes only its core-owned sources");
    ok &= Require(kernelImpl, "st_dev(",
        "Combine V2 implementation does not ring device doorbells");
    ok &= Require(kernelImpl, "rank_, 0U, core_, rankSize_",
        "Combine V2 implementation does not use the runtime Ring schedule");
    const std::size_t scheduleCallCount =
        CountOccurrences(kernelImpl, "MoonEpCombineV2Peer(") +
        CountOccurrences(kernelImpl, "MoonEpCombineV2Successor(") +
        CountOccurrences(kernelImpl, "MoonEpCombineV2ReceiveStep(") +
        CountOccurrences(kernelImpl, "MoonEpCombineV2SenderCore(");
    if (CountOccurrences(kernelImpl, "kCombineV2ScheduleMode") !=
        scheduleCallCount + 1U) {
        std::cerr << "Combine V2 schedule mode is not passed to every "
                     "active schedule call\n";
        ok = false;
    }
    ok &= Require(kernelImpl,
        "core_, TileXRMoonEp::MOONEP_COMBINE_V2_SIX_PORT",
        "Combine V2 full-sync path does not select the core's six-port QP");
    ok &= RequireBefore(fullSyncPublisher,
        "SyncFunc<HardEvent::S_MTE3>();", "CopyIssueToSq(",
        "Combine V2 full-sync publisher lacks scalar-to-MTE3 ordering");
    ok &= RequireBefore(fullSyncPublisher, "CopyIssueToSq(",
        "SyncFunc<HardEvent::MTE3_S>();",
        "Combine V2 full-sync publisher does not await SQ publication");
    ok &= RequireBefore(fullSyncPublisher,
        "SyncFunc<HardEvent::MTE3_S>();", "state.sq->dbAddr",
        "Combine V2 full-sync publisher rings DB before MTE3 completion");
    if (CountOccurrences(fullSyncPublisher, "state.sq->headAddr") != 1U ||
        CountOccurrences(fullSyncPublisher, "state.sq->dbAddr") != 1U) {
        std::cerr << "Combine V2 full-sync publisher must update one head "
                     "and ring one doorbell\n";
        ok = false;
    }
    ok &= Require(fullSyncBuilder,
        "TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION",
        "Combine V2 full-sync terminal WQE does not request completion");
    ok &= Require(fullSyncPublisher, "state.completionCount",
        "Combine V2 full-sync publisher omits completion accounting");
    ok &= Require(fullSyncPublisher, "state.sq->wqeCntAddr",
        "Combine V2 full-sync publisher omits the device completion count");
    ok &= Require(fullSyncPublisher, "state.cqTarget",
        "Combine V2 full-sync publisher omits the CQ target");
    ok &= Require(fullSyncWait, "PollCqOnce(state)",
        "Combine V2 full-sync path does not consume its CQ");
    ok &= Require(fullSyncWait, "state.head != state.tail",
        "Combine V2 full-sync path does not require an empty SQ");
    ok &= Require(kernelImpl, "if (!laneStatesReady)",
        "Combine V2 required SQ/WQ validation is benchmark-optional");
    ok &= Require(process, "if (fullSyncWqeCount != 0U)",
        "Combine V2 full-sync path writes a source signal for self-only core");
    ok &= RequireBefore(process, "BuildFullSyncWqes(&fullSyncWqeCount)",
        "PrepareFullSyncSignal()",
        "Combine V2 full-sync source is prepared before peer compression");
    ok &= RequireBefore(process, "PrepareFullSyncSignal()",
        "PublishFullSyncBatch(fullSyncWqeCount)",
        "Combine V2 full-sync source is prepared after SQ publication");
    ok &= RequireBefore(process, "PublishFullSyncBatch(fullSyncWqeCount)",
        "WaitFullSyncCq()",
        "Combine V2 full-sync path does not wait for its submitted CQ");
    ok &= RequireBefore(process, "WaitFullSyncCq()",
        "WaitFullSyncSources()",
        "Combine V2 full-sync signals are consumed before local CQ success");

    ok &= Require(fullmeshPrefill, "sqe->flag = 0U;",
        "Combine V2 Fullmesh payload template unexpectedly completes each WQE");
    ok &= Require(fullmeshBuilder, "context->head[0] + task",
        "Combine V2 Fullmesh builder does not use the direct SQ head");
    ok &= Require(fullmeshSubmit, "AppendControlWqe(issue, count, fullmeshLane_, fullmeshInfo_",
        "Combine V2 Fullmesh path does not append done on the direct QP");
    ok &= RequireBefore(fullmeshSubmit,
        "AppendControlWqe(issue, count, fullmeshLane_, fullmeshInfo_",
        "MOONEP_COMBINE_V2_DIAG_FULLMESH_WQE_BUILD_END",
        "Combine V2 profiles Fullmesh WQE build before appending done");
    ok &= Require(fullmeshSubmit,
        "MOONEP_COMBINE_V2_SIX_PORT",
        "Combine V2 Fullmesh done does not reuse lane zero");
    ok &= Require(fullmeshSubmit,
        "TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION",
        "Combine V2 Fullmesh done does not request ordered completion");
    ok &= RequireBefore(fullmeshSubmit, "SyncFunc<HardEvent::S_MTE3>();",
        "CopyIssueToSq(issue, fullmeshLane_, count)",
        "Combine V2 Fullmesh WQEs lack scalar-to-MTE3 ordering");
    ok &= RequireBefore(fullmeshSubmit,
        "CopyIssueToSq(issue, fullmeshLane_, count)",
        "SyncFunc<HardEvent::MTE3_S>();",
        "Combine V2 Fullmesh SQ publication is not completed before doorbells");
    ok &= RequireBefore(fullmeshSubmit, "SyncFunc<HardEvent::MTE3_S>();",
        "st_dev(fullmeshLane_.head",
        "Combine V2 Fullmesh doorbell precedes MTE3 completion");
    ok &= Require(fullmeshSend, "selectedCount == 0U && lastChunk",
        "Combine V2 Fullmesh path skips done for empty payload");
    ok &= Require(fullmeshWait, "PollCqOnce(fullmeshLane_)",
        "Combine V2 Fullmesh path does not consume CQ");
    ok &= Reject(fullmeshWait, "kEnableSafetyChecks",
        "Combine V2 Fullmesh CQ validation is benchmark-optional");
    ok &= RequireBefore(process, "WaitFullmeshCq(step, peer)",
        "SubmitSelfGrant(step)",
        "Combine V2 publishes grant before Fullmesh CQ success");
    ok &= RequireBefore(process,
        "MOONEP_COMBINE_V2_DIAG_FULLMESH_CQ_SUCCESS",
        "SubmitSelfGrant(step)",
        "Combine V2 profiles Fullmesh CQ success after grant publication");
    ok &= RequireBefore(process,
        "MOONEP_COMBINE_V2_DIAG_FULLMESH_GRANT_SUBMIT",
        "WaitStepCqs(step)",
        "Combine V2 profiles deferred grant submit after its CQ wait");
    ok &= RequireBefore(process, "WaitStepCqs(step)",
        "MOONEP_COMBINE_V2_DIAG_FULLMESH_GRANT_CQ_SUCCESS",
        "Combine V2 profiles deferred grant CQ success before CQ wait");
    ok &= Require(host, "UDMA_FULLMESH",
        "Combine V2 Host does not require Fullmesh capability");
    ok &= Require(host, "TileXRUDMAFullmeshQuery",
        "Combine V2 Host does not validate the Fullmesh generation");
    ok &= Require(kernelImpl, "constexpr bool kEnableFullSync = false;",
        "Combine V2 Kernel full synchronization is not disabled by default");
    ok &= Require(process, "if (kEnableFullSync)",
        "Combine V2 Process does not use the Kernel-local full-sync switch");
    ok &= Reject(host, "TILEXR_MOONEP_COMBINE_V2_FULL_SYNC",
        "Combine V2 Host still exposes the full-sync environment switch");
    ok &= Reject(hostHeader, "bool fullSync",
        "Combine V2 launch context still stores the full-sync flag");
    ok &= Reject(launch, "uint64_t fullSync;",
        "Combine V2 launch ABI still carries the full-sync flag");
    ok &= Reject(launch, "context.fullSync",
        "Combine V2 launch context still carries the full-sync flag");
    ok &= Reject(kernelEntry, "uint64_t fullSync,",
        "Combine V2 Kernel entry still accepts the full-sync flag");
    ok &= Reject(kernelImpl, "bool fullSync,",
        "Combine V2 Kernel Init still accepts the full-sync flag");
    ok &= Reject(kernelImpl, "fullSync_",
        "Combine V2 Kernel still stores a runtime full-sync flag");
    ok &= Require(launch,
        "static_assert(sizeof(CombineV2KernelArgs) == 21U * sizeof(uint64_t)",
        "Combine V2 launch ABI does not contain exactly 21 64-bit slots");
    ok &= Require(kernelImpl,
        "TileXRMoonEp::MOONEP_COMBINE_V2_SINGLE_RING;",
        "Combine V2 default schedule is not the existing single ring");
    ok &= Require(kernelImpl,
        "peer == TileXRMoonEp::kMoonEpCombineV2InvalidPeer",
        "Combine V2 invalid bidirectional peer does not use Self processing");
    ok &= Require(kernelImpl, "SendSelfStep(rank_, step)",
        "Combine V2 Self processing does not use the local rank");
    ok &= Require(kernelImpl, "step < stepCount_",
        "Combine V2 implementation does not use the runtime step count");
    ok &= Require(kernelImpl, "core < activeCoreCount_",
        "Combine V2 failure convergence does not use active cores");
    ok &= Require(kernelImpl, "encoded, slots_, rankSize_",
        "Combine V2 destination validation does not use runtime rank size");
    ok &= Require(inboundDone, "kMoonEpCombineV2RankCount",
        "Combine V2 inbound Done storage is not rank generalized");
    ok &= Require(kernelImpl, "MOONEP_COMBINE_V2_METRIC_SELF_COPY",
        "Combine V2 detailed self-copy profiling is missing");
    ok &= Require(kernelImpl, "MOONEP_COMBINE_V2_METRIC_REMOTE_WQE_BUILD",
        "Combine V2 detailed remote WQE profiling is missing");
    ok &= Reject(kernelImpl, "SyncAll<true>();",
        "Combine V2 implementation still uses whole-launch barriers");
    ok &= Reject(kernelImpl, "extern \"C\" __global__",
        "Combine V2 implementation header contains the operator entry");
    ok &= Reject(kernelImpl, "<<<",
        "Combine V2 implementation contains Host launch syntax");

    ok &= Require(host, "UDMA_SHARED_QP",
        "Combine V2 Host does not validate shared-QP capability");
    ok &= Require(host, "TileXRUDMAGetQpCount",
        "Combine V2 Host does not validate the QP count");
    ok &= Require(launch, "LaunchRegisteredMoonEpKernel",
        "Combine V2 does not use registered Runtime launch");
    ok &= Require(launch, "ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE",
        "Combine V2 launch does not query the SIMT UB capacity");
    ok &= Require(launch, "cfgInfo.localMemorySize",
        "Combine V2 launch does not configure SIMT local memory");
    ok &= Require(launch, "params.aivCoreNum",
        "Combine V2 launch does not use the requested AIV count");
    ok &= Require(launch, "kCombineV2KernelSignature",
        "Combine V2 launch signature is missing");
    ok &= Reject(launch, "<<<",
        "Combine V2 Host contains compiler launch syntax");
    ok &= Require(registration, "kCombineV2KernelSignature",
        "Combine V2 stable registration signature is missing");
    ok &= Require(publicHeader, "TileXRMoonEpCombineGetWorkspaceSizeV2",
        "Combine V2 workspace API is missing");
    ok &= Require(publicHeader, "TileXRMoonEpCombineV2",
        "Combine V2 launch API is missing");
    ok &= Require(profileHeader,
        "kMoonEpCombineV2ProfileMetricCount = 8U",
        "Combine V2 profile metric ABI is missing");
    ok &= Require(profileHeader,
        "kMoonEpCombineV2ProfileVersion = 5U",
        "Combine V2 Fullmesh profile version is missing");
    ok &= Require(profileHeader,
        "MOONEP_COMBINE_V2_TIME_STEP0_READY_END",
        "Combine V2 profile does not expose post-grant readiness");
    ok &= Require(profileHeader,
        "MOONEP_COMBINE_V2_DIAG_FULLMESH_WQE_BUILD_END",
        "Combine V2 profile does not expose Fullmesh WQE build completion");
    ok &= Require(profileHeader,
        "MOONEP_COMBINE_V2_DIAG_FULLMESH_GRANT_CQ_SUCCESS",
        "Combine V2 profile does not expose deferred grant CQ completion");
    ok &= Require(profileHeader,
        "MoonEpCombineV2PackFullmeshProfileRoute",
        "Combine V2 profile does not encode its Fullmesh route");
    ok &= Require(hardwareProbe, "self_copy_us",
        "Combine V2 probe does not print self-copy profiling");
    ok &= Require(hardwareProbe, "remote_submit_us",
        "Combine V2 probe does not print remote-submit profiling");
    ok &= Require(hardwareProbe, "transport=",
        "Combine V2 probe does not print the profile transport");
    ok &= Require(hardwareProbe, "fm_logical_qp=",
        "Combine V2 probe does not print the Fullmesh logical QP");
    ok &= Require(hardwareProbe, "clos_grant_cq_success=",
        "Combine V2 probe does not print deferred grant completion");
    ok &= Require(traceTool, "append_fullmesh_events",
        "Combine V2 trace decoder does not emit Fullmesh boundary events");
    ok &= Require(traceTool, "invalid Fullmesh event order",
        "Combine V2 trace decoder does not validate Fullmesh ordering");
    ok &= Require(barrierServer,
        "close(listenFd);\n    const uint8_t release",
        "Combine V2 barrier releases clients before retiring its listener");
    ok &= Require(hardwareProbe, "enum class OutputCheckResult",
        "Combine V2 probe does not classify correctness failures");
    ok &= Require(hardwareProbe, "if (sourceRank != rank)",
        "Combine V2 probe does not inspect remote-output source ranks");
    ok &= Require(hardwareProbe, "return OutputCheckResult::Failed;",
        "Combine V2 probe does not reject remote-output mismatches");
    ok &= Require(hardwareProbe, "OutputCheckResult::SelfOnlyFailed",
        "Combine V2 probe does not identify disabled self-copy mismatches");
    ok &= Require(hardwareProbe,
        "BarrierAll(rank, world, \"correctness validation\")",
        "Combine V2 correctness check still gates timing on validation status");
    ok &= Reject(hardwareProbe, "--allow-self-only-failure",
        "Combine V2 probe still requires an option to time incorrect output");
    ok &= Require(clusterLauncher, "#!/usr/bin/env bash",
        "Combine V2 cluster launcher is not a Bash script");
    ok &= Require(clusterLauncher, "build_args+=(--enable-profiling)",
        "Combine V2 --profile does not enable profiling at build time");
    ok &= Reject(clusterLauncher, "--allow-self-only-failure",
        "Combine V2 cluster launcher still gates incorrect performance runs");
    ok &= Require(clusterLauncher,
        "bash \"${run_script}\" \"${run_args[@]}\"",
        "Combine V2 cluster launcher does not start the server-side launcher");
    ok &= Reject(Lower(clusterLauncher), "powershell",
        "Combine V2 cluster launcher must not depend on PowerShell");
    ok &= Reject(Lower(multihostLauncher), "powershell",
        "Combine V2 multihost launcher must not depend on PowerShell");
    ok &= Require(multihostLauncher, "correctness == \"failed\"",
        "Combine V2 launcher does not aggregate failed correctness results");
    ok &= Reject(multihostLauncher, "--allow-self-only-failure",
        "Combine V2 multihost launcher still gates incorrect performance runs");

    const std::vector<std::string> v2Files {
        root + "/src/include/tilexr_moonep_combine_v2.h",
        root + "/src/moonep/combine_v2/CMakeLists.txt",
        root + "/src/moonep/combine_v2/common/combine_v2_profile.h",
        root + "/src/moonep/combine_v2/common/combine_v2_schedule.h",
        root + "/src/moonep/combine_v2/common/combine_v2_wqe_batch.h",
        root + "/src/moonep/combine_v2/host/combine_v2_layout.h",
        root + "/src/moonep/combine_v2/host/combine_v2_layout.cpp",
        root + "/src/moonep/combine_v2/host/combine_v2_host.h",
        root + "/src/moonep/combine_v2/host/combine_v2_host.cpp",
        root + "/src/moonep/combine_v2/host/combine_v2_launch.h",
        root + "/src/moonep/combine_v2/host/combine_v2_launch.cpp",
        root + "/src/moonep/combine_v2/host/tilexr_moonep_combine_v2.cpp",
        root + "/src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h",
        root + "/src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.cpp",
        root + "/tools/moonep/run_combine_v2_perf_cluster.sh",
        root + "/tools/moonep/run_combine_v2_perf_multihost.sh",
    };
    std::string allV2;
    for (const std::string &path : v2Files) {
        allV2 += Lower(Read(path));
    }
    const std::string legacyKeyword = std::string("pair") + "wise";
    ok &= Reject(allV2, legacyKeyword,
        "Combine V2 sources contain the legacy algorithm keyword");
    return ok ? 0 : 1;
}
