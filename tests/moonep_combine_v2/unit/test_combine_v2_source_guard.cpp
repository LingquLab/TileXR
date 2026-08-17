#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

std::size_t Count(const std::string &text, const std::string &needle)
{
    std::size_t count = 0U;
    std::size_t position = 0U;
    while ((position = text.find(needle, position)) != std::string::npos) {
        ++count;
        position += needle.size();
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
    const std::string v2Cmake = Read(root +
        "/src/moonep/combine_v2/CMakeLists.txt");
    const std::string kernelEntry = Read(root +
        "/src/moonep/combine_v2/kernels/"
        "tilexr_moonep_combine_v2_kernel.cpp");
    const std::string legacyKernel = Read(root +
        "/src/moonep/combine_v2/kernels/"
        "tilexr_moonep_combine_v2_kernel.h");
    const std::string groupKernel = Read(root +
        "/src/moonep/combine_v2/kernels/"
        "tilexr_moonep_combine_v2_group_kernel.h");
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
    const std::string schedule = Read(root +
        "/src/moonep/common/moonep_combine_schedule.h");

    const std::string process = Section(groupKernel,
        "__aicore__ inline void MoonEpCombineV2Group::Process()",
        "} // namespace");
    const std::string collectiveBegin = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::BeginCollectiveStage(",
        "__aicore__ inline bool MoonEpCombineV2Group::EndCollectiveStage(");
    const std::string collectiveEnd = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::EndCollectiveStage(",
        "__aicore__ inline void MoonEpCombineV2Group::InitReduceBuffers(");
    const std::string fullmeshSubmit = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::SubmitFullmeshBatch(",
        "__aicore__ inline bool MoonEpCombineV2Group::SendFullmeshStep(");
    const std::string closStep = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::SendClosStep(",
        "__aicore__ inline bool MoonEpCombineV2Group::SubmitFullmeshBatch(");
    const std::string singleDone = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::SubmitRemoteDone(",
        "__aicore__ inline bool MoonEpCombineV2Group::SendClosStep(");
    const std::string inboundDone = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::WaitSingleInboundDone(",
        "__aicore__ inline void MoonEpCombineV2Group::InitializeCreditSignal(");
    const std::string publishCredit = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::PublishNextCredit(",
        "__aicore__ inline bool MoonEpCombineV2Group::WaitStepCredit(");
    const std::string waitCredit = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::WaitStepCredit(",
        "__aicore__ inline bool MoonEpCombineV2Group::BeginCollectiveStage(");
    const std::string stepLoop = Section(process,
        "for (uint32_t step", "stageReady = BeginCollectiveStage(");

    bool ok = true;
    ok &= Require(rootCmake, "add_subdirectory(tests/moonep_combine_v2)",
        "Combine V2 tests are not wired at the root");
    ok &= Require(moonEpCmake, "add_subdirectory(combine_v2)",
        "Combine V2 source directory is not configured");
    ok &= Reject(v1Cmake, "combine_v2",
        "Combine V1 build definition contains V2 changes");
    ok &= Require(v2Cmake, "add_library(tilexr-moonep-combine-v2 SHARED",
        "Combine V2 standalone library is missing");
    ok &= Require(v2Cmake, "tilexr_moonep_combine_v2_kernel.h\"",
        "Combine V2 build does not depend on the legacy kernel header");
    ok &= Require(v2Cmake, "tilexr_moonep_combine_v2_group_kernel.h\"",
        "Combine V2 build does not depend on the Group kernel header");
    ok &= Require(v2Cmake, "-DCATLASS_ARCH=3510",
        "Combine V2 CATLASS target is missing");
    ok &= Reject(v2Cmake, "/devlib",
        "Combine V2 links the toolkit stub library directory");

    ok &= Require(kernelEntry,
        "#include \"tilexr_moonep_combine_v2_kernel.h\"",
        "Combine V2 entry does not include the legacy implementation");
    ok &= Require(kernelEntry,
        "#include \"tilexr_moonep_combine_v2_group_kernel.h\"",
        "Combine V2 entry does not include the Group implementation");
    ok &= Require(kernelEntry, "args != nullptr && args->rankSize == 128",
        "Combine V2 entry does not isolate the 128P Group path");
    ok &= Require(kernelEntry,
        "TileXRGroup128::MoonEpCombineV2Group op;",
        "Combine V2 entry does not select the namespaced Group kernel");
    ok &= Require(kernelEntry, "MoonEpCombineV2 op;",
        "Combine V2 entry does not retain the non-128P implementation");
    ok &= RequireBefore(kernelEntry,
        "TileXRGroup128::MoonEpCombineV2Group op;", "MoonEpCombineV2 op;",
        "Combine V2 fallback is placed before the 128P branch");
    ok &= Reject(kernelEntry, "<<<",
        "Combine V2 entry contains Host launch syntax");

    ok &= Require(legacyKernel, "class MoonEpCombineV2",
        "Legacy Combine V2 implementation is missing");
    ok &= Require(groupKernel, "namespace TileXRGroup128 {",
        "Group implementation is not isolated in its own namespace");
    ok &= Require(groupKernel, "class MoonEpCombineV2Group",
        "Group implementation class is missing");
    ok &= Require(groupKernel, "args_->rankSize != 128",
        "Group implementation does not reject non-128P direct use");
    ok &= Require(groupKernel, "(void)fullSyncReceiveOffset;",
        "Group Init does not explicitly ignore the retired receive offset");
    ok &= Require(groupKernel, "(void)fullSyncSourceOffset;",
        "Group Init does not explicitly ignore the retired source offset");
    ok &= Require(groupKernel, "fullSyncBarrierBase_",
        "Group kernel dropped the card-local collective status workspace");

    ok &= Require(groupKernel,
        "Simt::VF_CALL<MoonEpCombineV2PrefillOperatorWqesVf>",
        "Group kernel dropped operator WQE prefill");
    ok &= Require(groupKernel,
        "Simt::VF_CALL<MoonEpCombineV2PrefillPeerWqesVf>",
        "Group kernel dropped peer WQE prefill");
    ok &= Require(groupKernel,
        "Simt::VF_CALL<MoonEpCombineV2BuildPayloadWqesVf>",
        "Group kernel dropped the CLOS payload builder");
    ok &= Require(groupKernel,
        "Simt::VF_CALL<MoonEpCombineV2BuildFullmeshPayloadWqesVf>",
        "Group kernel dropped the Fullmesh payload builder");
    ok &= Require(groupKernel, "AppendControlWqe(",
        "Group kernel dropped the reusable control WQE builder");
    ok &= Require(groupKernel, "CopyIssueToSq(",
        "Group kernel dropped the reusable SQ publisher");
    ok &= Require(groupKernel, "PollCqOnce(",
        "Group kernel dropped CQ bookkeeping");
    ok &= Require(groupKernel, "CopySelfSelectedIndices(",
        "Group kernel dropped Self copy support");
    ok &= Require(groupKernel, "ReduceHidden()",
        "Group kernel dropped final reduction support");
    ok &= Require(groupKernel, "__gm__ uint8_t *doneBase_",
        "Group kernel dropped future Done workspace state");
    ok &= Require(groupKernel, "__gm__ uint8_t *grantBase_",
        "Group kernel dropped future credit workspace state");
    ok &= Require(groupKernel, "__gm__ uint8_t *controlSourceBase_",
        "Group kernel dropped future control source state");
    ok &= Require(groupKernel, "SyncFunc<HardEvent::S_MTE3>();",
        "Group kernel dropped UB-to-MTE3 ordering");
    ok &= Require(groupKernel, "SyncFunc<HardEvent::MTE3_S>();",
        "Group kernel dropped MTE3-to-doorbell ordering");
    ok &= Require(groupKernel, "st_dev(",
        "Group kernel dropped device doorbells");
    ok &= RequireBefore(fullmeshSubmit, "SyncFunc<HardEvent::MTE3_S>();",
        "st_dev(fullmeshLane_.head",
        "Fullmesh path rings its doorbell before MTE3 completion");
    ok &= Require(schedule, "MoonEpCombineV2GroupSendDstRank(",
        "Group send mapping helper is missing");
    ok &= Require(schedule, "MoonEpCombineV2GroupRecvSrcRank(",
        "Group receive mapping helper is missing");
    ok &= Require(schedule, "struct alignas(64) MoonEpCombineV2GroupCreditSignal",
        "Group Credit signal is missing");
    ok &= Require(schedule,
        "kMoonEpCombineV2GroupCreditWorkspaceBytes <=\n"
        "        kMoonEpCombineV2LegacyGrantWorkspaceBytes",
        "Group Credit is not bounded by the existing Grant workspace");

    ok &= RequireBefore(closStep, "WaitLaneCq(", "SubmitRemoteDone(peer, step)",
        "CLOS Done can be published before payload CQ completion");
    ok &= Require(singleDone, "SubmitClosControl(peer,",
        "CLOS path does not publish a standalone Done");
    ok &= Require(singleDone, "MOONEP_COMBINE_V2_SIX_PORT",
        "CLOS Done is not fixed to the six-port lane");
    ok &= Count(singleDone, "SubmitClosControl(") == 1U;
    if (Count(singleDone, "SubmitClosControl(") != 1U) {
        std::cerr << "CLOS path does not publish exactly one Done\n";
    }
    ok &= Count(inboundDone, "MoonEpCombineV2DoneIndex(") == 1U;
    if (Count(inboundDone, "MoonEpCombineV2DoneIndex(") != 1U) {
        std::cerr << "Group receiver does not read exactly one Done slot\n";
    }
    ok &= Require(inboundDone, "if (source == rank_)",
        "Group Self path does not treat local copy as Done");
    ok &= Require(publishCredit, "MoonEpCombineV2GroupRecvSrcRank(",
        "Group Credit does not target the next step sender");
    ok &= Require(publishCredit, "MoonEpCombineV2GroupInnerIndex(rank_)",
        "Group Credit does not target the destination inner core");
    ok &= Require(publishCredit, "SubmitClosControl(targetRank",
        "Remote Group Credit is not published through CLOS");
    ok &= Require(waitCredit, "signal->targetCore == core_",
        "Group sender does not validate its single Credit target");
    ok &= Require(waitCredit, "MOONEP_COMBINE_V2_CREDIT_TIMEOUT",
        "Group Credit wait is not bounded");

    const char *retired[] = {
        "kEnableFullSync",
        "MoonEpCombineV2FullSyncBuildContext",
        "MoonEpCombineV2BuildFullSyncWqesVf",
        "RunGlobalBarrier(",
        "WaitStepGrant(",
        "PublishLocalGrant(",
        "SubmitSelfGrant(",
        "ServerGrant",
        "RunServerGrantAdmission(",
        "WaitInboundDone(",
        "sourcesPerCore_",
        "SubmitPair(",
        "SendRemoteStep(",
        "MOONEP_COMBINE_V2_SERVER_PAIR_PARITY",
        "kCombineV2ScheduleMode",
        "MoonEpCombineV2PhaseBarrierAfterRound(",
        "fullSyncReceiveBase_",
        "fullSyncSourceBase_",
        "fullSyncReceiveOffset_",
        "fullSyncStartCycles_"
    };
    for (const char *symbol : retired) {
        if (!Reject(groupKernel, symbol,
                "Group kernel retained a retired protocol symbol")) {
            std::cerr << "retired symbol: " << symbol << '\n';
            ok = false;
        }
    }

    ok &= Require(process, "BeginCollectiveStage(kCollectiveInitStage)",
        "Group scaffold dropped card-local initialization convergence");
    ok &= Require(process,
        "BeginCollectiveStage(kCollectiveValidationStage)",
        "Group scaffold dropped card-local validation convergence");
    ok &= Require(process, "BeginCollectiveStage(kCollectiveDoneStage)",
        "Group implementation dropped final card-local convergence");
    ok &= Require(process, "for (uint32_t step",
        "Group implementation does not execute the 8-step schedule");
    ok &= Require(process, "MoonEpCombineV2GroupSendDstRank(",
        "Group process does not use the send mapping");
    ok &= Require(process, "MoonEpCombineV2GroupRecvSrcRank(",
        "Group process does not use the receive mapping");
    ok &= Require(process, "WaitStepCredit(step)",
        "Group process does not wait for one pre-step Credit");
    ok &= Require(process, "PublishNextCredit(step)",
        "Group process does not publish one next-step Credit");
    ok &= Require(process, "ReduceHidden()",
        "Group implementation dropped final reduction");
    ok &= Reject(stepLoop, "SyncAll<true>()",
        "Group hot loop contains a cross-core barrier");
    ok &= Require(collectiveBegin, "AscendC::SyncAll<true>();",
        "Group card-local stage does not align all launch blocks");
    ok &= Require(collectiveEnd, "AscendC::AtomicCas(&status->status",
        "Group card-local stage does not aggregate local failures");

    ok &= Require(host, "UDMA_SHARED_QP",
        "Combine V2 Host does not validate shared-QP capability");
    ok &= Require(host, "UDMA_FULLMESH",
        "Combine V2 Host does not validate Fullmesh capability");
    ok &= Require(launch, "LaunchRegisteredMoonEpKernel",
        "Combine V2 does not use registered Runtime launch");
    ok &= Require(launch,
        "static_assert(sizeof(CombineV2KernelArgs) == 21U * sizeof(uint64_t)",
        "Combine V2 launch ABI no longer has 21 64-bit slots");
    ok &= Reject(hostHeader, "bool fullSync",
        "Combine V2 launch context exposes a retired full-sync switch");
    ok &= Require(registration, "kCombineV2KernelSignature",
        "Combine V2 stable registration signature is missing");
    ok &= Require(publicHeader, "TileXRMoonEpCombineGetWorkspaceSizeV2",
        "Combine V2 workspace API is missing");
    ok &= Require(publicHeader, "TileXRMoonEpCombineV2",
        "Combine V2 launch API is missing");

    ok &= Reject(groupKernel, "extern \"C\" __global__",
        "Group implementation header contains the operator entry");
    ok &= Reject(groupKernel, "<<<",
        "Group implementation contains Host launch syntax");
    return ok ? 0 : 1;
}
