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
    const std::string clusterLauncher = Read(root +
        "/tools/moonep/run_combine_v2_perf_cluster.sh");
    const std::string multihostLauncher = Read(root +
        "/tools/moonep/run_combine_v2_perf_multihost.sh");
    const std::string host = Read(root +
        "/src/moonep/combine_v2/host/combine_v2_host.cpp");
    const std::string launch = Read(root +
        "/src/moonep/combine_v2/host/combine_v2_launch.cpp");
    const std::string publicHeader = Read(root +
        "/src/include/tilexr_moonep_combine_v2.h");
    const std::string registration = Read(root +
        "/src/moonep/common/moonep_kernel_registration.h");
    const std::string peerPrefill = Section(kernelImpl,
        "inline void MoonEpCombineV2PrefillPeerWqesVf(",
        "inline void MoonEpCombineV2SelectPeerRoutesVf(");
    const std::string submitPair = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::SubmitPair(",
        "__aicore__ inline bool MoonEpCombineV2::SendRemoteStep(");
    const std::string selfStep = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::SendSelfStep(",
        "__aicore__ inline bool MoonEpCombineV2::WaitInboundDone(");
    const std::string selfGrant = Section(kernelImpl,
        "__aicore__ inline bool MoonEpCombineV2::SubmitSelfGrant(",
        "__aicore__ inline bool MoonEpCombineV2::SendSelfStep(");
    const std::string localGrant = Section(kernelImpl,
        "__aicore__ inline void MoonEpCombineV2::PublishLocalGrant(",
        "__aicore__ inline void MoonEpCombineV2::CopyIssueToSq(");
    const std::string process = Section(kernelImpl,
        "__aicore__ inline void MoonEpCombineV2::Process()",
        "} // namespace");

    bool ok = true;
    ok &= Require(rootCmake, "add_subdirectory(tests/moonep_combine_v2)",
        "Combine V2 tests are not wired at the root");
    ok &= Require(moonEpCmake, "add_subdirectory(combine_v2)",
        "Combine V2 source directory is not configured");
    ok &= Require(moonEpCmake, "tilexr-moonep-combine-v2",
        "Combine V2 is not linked into the MoonEP aggregate library");
    ok &= Reject(v1Cmake, "combine_v2",
        "Combine V1 build definition contains V2 changes");

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
        "Simt::VF_CALL<MoonEpCombineV2SelectPeerRoutesVf>",
        "Combine V2 implementation does not use SIMT route selection");
    ok &= Require(kernelImpl,
        "Simt::VF_CALL<MoonEpCombineV2BuildPayloadWqesVf>",
        "Combine V2 implementation does not use its SIMT WQE builder");
    ok &= Require(kernelImpl, "dstSlots[index - chunkStart]",
        "Combine V2 selector does not use a chunk-relative UB index");
    ok &= Require(kernelImpl,
        "const uint32_t old = asc_atomic_add(&state->curWqeNum, 1U);",
        "Combine V2 selector does not reserve routes with UB atomic add");
    ok &= Require(kernelImpl, "old - state->batchBase",
        "Combine V2 selector does not derive a batch-local route index");
    ok &= Require(kernelImpl,
        "constexpr uint32_t kSixPortPayloadCapacity = 192U;",
        "Combine V2 six-port payload capacity is missing");
    ok &= Require(kernelImpl,
        "constexpr uint32_t kTwoPortPayloadCapacity = 64U;",
        "Combine V2 two-port payload capacity is missing");
    ok &= Require(kernelImpl,
        "constexpr uint32_t kSixPortIssueCapacity = 194U;",
        "Combine V2 six-port issue capacity is missing");
    ok &= Require(kernelImpl,
        "constexpr uint32_t kTwoPortIssueCapacity = 66U;",
        "Combine V2 two-port issue capacity is missing");
    ok &= Require(kernelImpl, "TBuf<QuePosition::VECCALC> wqeIssueBuf_;",
        "Combine V2 continuous WQE issue buffer is missing");
    ok &= Require(kernelImpl,
        "issue[kSixPortIssueBytes].GetPhyAddr()",
        "Combine V2 two-port WQE view does not start after entry 194");
    ok &= Require(kernelImpl,
        "pipe_->InitBuffer(selfCopyQueue_, 2U,",
        "Combine V2 Self copy queue is not double buffered");
    ok &= Require(selfStep, "SelectPeerRoutes(",
        "Combine V2 Self step does not reuse SIMT route selection");
    ok &= Require(selfStep, "CopySelfRouteBatch(selectedCount)",
        "Combine V2 Self step does not consume compacted route batches");
    ok &= Require(selfStep, "SubmitSelfGrant(step)",
        "Combine V2 Self step does not publish its step grant");
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
        "routes[routeStart + row]",
        "Combine V2 Self copy does not consume RouteEntry records");
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
    ok &= Reject(kernelImpl, "selectedIndexBuf_",
        "Combine V2 retained the obsolete selected-index UB buffer");
    ok &= Reject(kernelImpl, "compareMaskBuf_",
        "Combine V2 retained the obsolete compare-mask UB buffer");
    ok &= Reject(kernelImpl, "GatherMask(",
        "Combine V2 retained the obsolete gather selection path");
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
    ok &= Require(kernelImpl, "st_dev(",
        "Combine V2 implementation does not ring device doorbells");
    ok &= Require(kernelImpl, "rank_, 0U, core_, rankSize_",
        "Combine V2 implementation does not use the runtime Ring schedule");
    if (CountOccurrences(kernelImpl, "kCombineV2ScheduleMode") != 7U) {
        std::cerr << "Combine V2 schedule mode is not passed to all six "
                     "active schedule calls\n";
        ok = false;
    }
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
    ok &= Require(kernelImpl, "kMoonEpCombineV2MaxSourcesPerCore",
        "Combine V2 inbound Done polling is not rank generalized");
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
        "kMoonEpCombineV2ProfileVersion = 3U",
        "Combine V2 full-grant profile version is missing");
    ok &= Require(profileHeader,
        "MOONEP_COMBINE_V2_TIME_STEP0_READY_END",
        "Combine V2 profile does not expose post-grant readiness");
    ok &= Require(hardwareProbe, "self_copy_us",
        "Combine V2 probe does not print self-copy profiling");
    ok &= Require(hardwareProbe, "remote_submit_us",
        "Combine V2 probe does not print remote-submit profiling");
    ok &= Require(hardwareProbe, "enum class OutputCheckResult",
        "Combine V2 probe does not classify correctness failures");
    ok &= Require(hardwareProbe, "if (sourceRank != rank)",
        "Combine V2 probe does not inspect remote-output source ranks");
    ok &= Require(hardwareProbe, "return OutputCheckResult::Failed;",
        "Combine V2 probe does not reject remote-output mismatches");
    ok &= Require(hardwareProbe, "OutputCheckResult::SelfOnlyFailed",
        "Combine V2 probe does not identify disabled self-copy mismatches");
    ok &= Require(clusterLauncher, "#!/usr/bin/env bash",
        "Combine V2 cluster launcher is not a Bash script");
    ok &= Require(clusterLauncher,
        "bash \"${run_script}\" \"${run_args[@]}\"",
        "Combine V2 cluster launcher does not start the server-side launcher");
    ok &= Reject(Lower(clusterLauncher), "powershell",
        "Combine V2 cluster launcher must not depend on PowerShell");
    ok &= Reject(Lower(multihostLauncher), "powershell",
        "Combine V2 multihost launcher must not depend on PowerShell");

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
