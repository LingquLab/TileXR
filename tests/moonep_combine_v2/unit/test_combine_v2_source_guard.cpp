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

    const std::string legacyProcess = Section(legacyKernel,
        "__aicore__ inline void MoonEpCombineV2::Process()",
        "} // namespace");
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
        "__aicore__ inline bool MoonEpCombineV2Group::PublishNextCredit(");
    const std::string publishCredit = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::PublishNextCredit(",
        "__aicore__ inline bool MoonEpCombineV2Group::WaitStepCredit(");
    const std::string waitCredit = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::WaitStepCredit(",
        "__aicore__ inline bool MoonEpCombineV2Group::BeginCollectiveStage(");
    const std::string stepLoop = Section(process,
        "for (uint32_t step", "stageReady = BeginCollectiveStage(");
    const std::string fullmeshStep = Section(stepLoop,
        "} else if (fullmeshStep) {", "} else {");
    const std::string legacyPublishCredit = Section(legacyKernel,
        "__aicore__ inline bool MoonEpCombineV2::PublishNextCredit(",
        "__aicore__ inline bool MoonEpCombineV2::BuildPayloadWqes(");
    const std::string legacyWaitCredit = Section(legacyKernel,
        "__aicore__ inline bool MoonEpCombineV2::WaitStepCredit(",
        "__aicore__ inline bool MoonEpCombineV2::PublishNextCredit(");
    const std::string legacyWeightSend = Section(legacyKernel,
        "__aicore__ inline bool MoonEpCombineV2::SendWeightMemoryStep(",
        "__aicore__ inline bool MoonEpCombineV2::WaitInboundWeightDone(");
    const std::string legacyInboundDone = Section(legacyKernel,
        "__aicore__ inline bool MoonEpCombineV2::WaitInboundDone(",
        "__aicore__ inline bool MoonEpCombineV2::BeginCollectiveStage(");
    const std::string legacyWeightOutput = Section(legacyKernel,
        "__aicore__ inline bool MoonEpCombineV2::CopyReceivedWeights()",
        "__aicore__ inline bool MoonEpCombineV2::WaitInboundDone(");
    const std::string groupWeightSend = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::SendWeightMemoryStep(",
        "__aicore__ inline bool MoonEpCombineV2Group::WaitInboundWeightDone(");
    const std::string groupWeightOutput = Section(groupKernel,
        "__aicore__ inline bool MoonEpCombineV2Group::CopyReceivedWeights()",
        "__aicore__ inline bool MoonEpCombineV2Group::WaitSingleInboundDone(");
    const std::string activeProtocolSource = legacyKernel + groupKernel +
        schedule;

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
    ok &= Require(groupKernel, "(void)reservedSyncReceiveOffset;",
        "Group Init does not explicitly ignore the retired receive offset");
    ok &= Require(groupKernel, "(void)reservedSyncSourceOffset;",
        "Group Init does not explicitly ignore the retired source offset");
    ok &= Require(groupKernel, "collectiveStatusBase_",
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
    ok &= Reject(groupKernel, "__gm__ uint8_t *grantBase_",
        "Group kernel retained the retired Grant workspace state");
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
    ok &= Require(schedule, "struct alignas(64) MoonEpCombineV2CreditSignal",
        "Combine V2 Credit signal is missing");
    ok &= Require(schedule, "kMoonEpCombineV2CreditBaseBytes",
        "Combine V2 Credit IPC base is missing");
    ok &= Require(schedule, "MoonEpCombineV2CreditReceiveOffset(",
        "Combine V2 Credit IPC offset helper is missing");
    ok &= Require(schedule, "MoonEpCombineV2ReceiveCore(",
        "Combine V2 Credit receiver-core inverse is missing");

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
    ok &= Require(publishCredit, "args_->creditMems[targetRank]",
        "Group Credit publish does not use the target Credit IPC mapping");
    ok &= Require(publishCredit,
        "DataCopy(remoteCreditGlobal, creditLocal",
        "Group Credit publish is not an MTE3 GM write");
    ok &= Require(publishCredit, "SyncFunc<HardEvent::S_MTE3>();",
        "Group Credit publish lacks scalar-to-MTE3 ordering");
    ok &= Require(publishCredit, "SyncFunc<HardEvent::MTE3_S>();",
        "Group Credit publish does not wait for MTE3 completion");
    ok &= Require(waitCredit, "args_->creditMems[rank_]",
        "Group Credit wait does not use the local Credit IPC mapping");
    ok &= Require(waitCredit, "DataCopy(creditLocal, creditGlobal",
        "Group Credit wait is not an MTE2 GM read");
    ok &= Require(waitCredit, "SyncFunc<HardEvent::S_MTE2>();",
        "Group Credit wait lacks scalar-to-MTE2 ordering");
    ok &= Require(waitCredit, "SyncFunc<HardEvent::MTE2_S>();",
        "Group Credit wait does not wait for MTE2 completion");
    ok &= Require(waitCredit, "creditLocal.GetValue(4U)",
        "Group sender does not validate its Credit target route");
    ok &= Require(waitCredit,
        "(static_cast<uint64_t>(core_) << 32U | rank_)",
        "Group sender does not validate its rank/core Credit target");
    ok &= Require(waitCredit, "MOONEP_COMBINE_V2_CREDIT_TIMEOUT",
        "Group Credit wait is not bounded");

    ok &= Require(legacyPublishCredit, "args_->creditMems[targetRank]",
        "Legacy Credit publish does not use the target Credit IPC mapping");
    ok &= Require(legacyPublishCredit,
        "DataCopy(remoteCreditGlobal, creditLocal",
        "Legacy Credit publish is not an MTE3 GM write");
    ok &= Require(legacyPublishCredit, "SyncFunc<HardEvent::S_MTE3>();",
        "Legacy Credit publish lacks scalar-to-MTE3 ordering");
    ok &= Require(legacyPublishCredit, "SyncFunc<HardEvent::MTE3_S>();",
        "Legacy Credit publish does not wait for MTE3 completion");
    ok &= Require(legacyWaitCredit, "args_->creditMems[rank_]",
        "Legacy Credit wait does not use the local Credit IPC mapping");
    ok &= Require(legacyWaitCredit, "DataCopy(creditLocal, creditGlobal",
        "Legacy Credit wait is not an MTE2 GM read");
    ok &= Require(legacyWaitCredit, "SyncFunc<HardEvent::S_MTE2>();",
        "Legacy Credit wait lacks scalar-to-MTE2 ordering");
    ok &= Require(legacyWaitCredit, "SyncFunc<HardEvent::MTE2_S>();",
        "Legacy Credit wait does not wait for MTE2 completion");
    ok &= Require(legacyWaitCredit, "MOONEP_COMBINE_V2_CREDIT_TIMEOUT",
        "Legacy Credit wait is not bounded");

    const std::string creditSections = publishCredit + waitCredit +
        legacyPublishCredit + legacyWaitCredit;
    const char *forbiddenCreditTransport[] = {
        "AppendControlWqe(",
        "SubmitClosControl(",
        "CopyIssueToSq(",
        "PollCqOnce(",
        "WaitLaneCq(",
        "st_dev(",
        "dbAddr",
        "cqTarget",
        "completionCount"
    };
    for (const char *symbol : forbiddenCreditTransport) {
        if (!Reject(creditSections, symbol,
                "Credit path retained a WQE, doorbell, or CQ dependency")) {
            std::cerr << "forbidden Credit transport symbol: " << symbol
                      << '\n';
            ok = false;
        }
    }

    const char *retiredProtocol[] = {
        "kEnableFullSync",
        "MoonEpCombineV2FullSyncBuildContext",
        "MoonEpCombineV2BuildFullSyncWqesVf",
        "RunGlobalBarrier(",
        "WaitStepGrant(",
        "PublishLocalGrant(",
        "SubmitSelfGrant(",
        "ServerGrant",
        "RunServerGrantAdmission(",
        "MoonEpCombineV2PhaseBarrierAfterRound(",
        "fullSyncReceiveBase_",
        "fullSyncSourceBase_",
        "fullSyncReceiveOffset_",
        "fullSyncStartCycles_"
    };
    for (const char *symbol : retiredProtocol) {
        if (!Reject(activeProtocolSource, symbol,
                "Combine V2 retained a retired protocol symbol")) {
            std::cerr << "retired symbol: " << symbol << '\n';
            ok = false;
        }
    }
    const char *retiredGroupSymbols[] = {
        "WaitInboundDone(",
        "sourcesPerCore_",
        "SubmitPair(",
        "SendRemoteStep(",
        "MOONEP_COMBINE_V2_SERVER_PAIR_PARITY",
        "kCombineV2ScheduleMode"
    };
    for (const char *symbol : retiredGroupSymbols) {
        if (!Reject(groupKernel, symbol,
                "Group kernel retained an obsolete implementation symbol")) {
            std::cerr << "retired Group symbol: " << symbol << '\n';
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
    ok &= RequireBefore(stepLoop, "WaitStepCredit(step)",
        "WaitSingleInboundDone(step, source)",
        "Group process waits for inbound Done before pre-step Credit");
    ok &= RequireBefore(stepLoop, "WaitSingleInboundDone(step, source)",
        "PublishNextCredit(step)",
        "Group process publishes Credit before inbound Done");
    ok &= RequireBefore(legacyProcess, "WaitStepCredit(step)",
        "WaitInboundDone(step)",
        "Legacy process waits for inbound Done before pre-step Credit");
    ok &= RequireBefore(legacyProcess, "WaitInboundDone(step)",
        "PublishNextCredit(step)",
        "Legacy process publishes Credit before inbound Done");
    ok &= RequireBefore(legacyProcess, "SendWeightMemoryStep(peer, step)",
        "WaitStepCqs(step)",
        "Legacy process does not send weight before CQ waiting");
    ok &= Require(legacyInboundDone,
        "return WaitInboundWeightDone(step, source);",
        "Legacy inbound completion does not include weight Done");
    ok &= RequireBefore(stepLoop, "SendWeightMemoryStep(peer, step)",
        "WaitSingleInboundDone(step, source)",
        "Group process does not send weight before inbound waiting");
    ok &= Require(inboundDone, "return WaitInboundWeightDone(step, source);",
        "Group inbound completion does not include weight Done");
    ok &= RequireBefore(legacyWeightSend, "DataCopyPad(destination, relay",
        "DataCopyPad(done, relay",
        "Legacy weight Done can be published before weight records");
    ok &= RequireBefore(groupWeightSend, "DataCopyPad(destination, relay",
        "DataCopyPad(done, relay",
        "Group weight Done can be published before weight records");
    ok &= Require(legacyWeightOutput, "words64.GetValue(1U) != magic_",
        "Legacy weight output does not filter stale generations");
    ok &= Require(legacyWeightOutput,
        "relay.ReinterpretCast<uint32_t>().SetValue(0U, 0U)",
        "Legacy stale weight output is not zeroed");
    ok &= Require(groupWeightOutput, "words64.GetValue(1U) != magic_",
        "Group weight output does not filter stale generations");
    if (Count(legacyKernel, "weightRecordEpochStride_ +") < 2U ||
        Count(legacyKernel, "weightDoneEpochStride_ +") < 2U) {
        std::cerr << "Legacy weight record/Done accesses are not epoch isolated\n";
        ok = false;
    }
    if (Count(groupKernel, "weightRecordEpochStride_ +") < 2U ||
        Count(groupKernel, "weightDoneEpochStride_ +") < 2U) {
        std::cerr << "Group weight record/Done accesses are not epoch isolated\n";
        ok = false;
    }
    ok &= Reject(legacyWeightSend, "st_dev(",
        "Legacy weight Memory path uses a device doorbell");
    ok &= Reject(groupWeightSend, "st_dev(",
        "Group weight Memory path uses a device doorbell");
    ok &= RequireBefore(fullmeshStep,
        "localSucceeded = WaitFullmeshCq(step, peer);",
        "MOONEP_COMBINE_V2_DIAG_FULLMESH_CQ_SUCCESS",
        "Group Fullmesh profile can report CQ success before final CQ wait");
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
        "static_assert(sizeof(CombineV2KernelArgs) == 29U * sizeof(uint64_t)",
        "Combine V2 launch ABI no longer has 29 64-bit slots");
    ok &= Count(launch, "        0U,") >= 3U;
    if (Count(launch, "        0U,") < 3U) {
        std::cerr << "Combine V2 launch does not zero all retired ABI slots\n";
    }
    ok &= Require(launch, "context.layout.collectiveStatusOffset",
        "Combine V2 launch dropped the card-local collective status offset");
    ok &= Require(host, "commArgs.creditMems[peer] == nullptr",
        "Combine V2 Host does not fail fast on a missing Credit mapping");
    ok &= Reject(hostHeader, "bool fullSync",
        "Combine V2 launch context exposes a retired full-sync switch");
    ok &= Require(registration, "kCombineV2KernelSignature",
        "Combine V2 stable registration signature is missing");
    ok &= Require(publicHeader, "TileXRMoonEpCombineGetWorkspaceSizeV2",
        "Combine V2 workspace API is missing");
    ok &= Require(publicHeader, "TileXRMoonEpCombineV2",
        "Combine V2 launch API is missing");
    ok &= Require(publicHeader, "TileXRMoonEpCombineStageV2Fused",
        "Combine V2 fused Stage API is missing");

    ok &= Reject(groupKernel, "extern \"C\" __global__",
        "Group implementation header contains the operator entry");
    ok &= Reject(groupKernel, "<<<",
        "Group implementation contains Host launch syntax");
    return ok ? 0 : 1;
}
