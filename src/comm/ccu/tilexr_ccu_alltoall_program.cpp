/*
 * Copyright (c) 2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "ccu/tilexr_ccu_alltoall_program.h"

#include <algorithm>
#include <initializer_list>
#include <set>

namespace TileXR {
namespace {

uint16_t PreSyncSignalMask(const TileXRCcuAllToAll2RankProgramSpec& spec)
{
    (void)spec;
    return static_cast<uint16_t>(1U << TILEXR_CCU_ALLTOALL_OUTPUT_XN_ID);
}

uint16_t PreSyncTokenMask(const TileXRCcuAllToAll2RankProgramSpec& spec)
{
    (void)spec;
    return static_cast<uint16_t>(1U << TILEXR_CCU_ALLTOALL_TOKEN_XN_ID);
}

uint16_t PostSyncSignalMask(const TileXRCcuAllToAll2RankProgramSpec& spec)
{
    return spec.ckeMask;
}

void ResetReport(TileXRCcuAllToAllProgramReport* report)
{
    if (report != nullptr) {
        *report = TileXRCcuAllToAllProgramReport{};
    }
}

int Fail(
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report,
    const std::string& message)
{
    if (program != nullptr) {
        program->clear();
    }
    if (report != nullptr) {
        report->message = message;
    }
    return TILEXR_ERROR_PARA_CHECK_FAIL;
}

int ValidateSpec(
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    if (program == nullptr) {
        return Fail(program, report, "missing output direct CCU alltoall program");
    }
    if (spec.localRank > 1U) {
        return Fail(program, report, "direct CCU alltoall localRank must be 0 or 1");
    }
    if (spec.localSendAddr == 0 || spec.localRecvAddr == 0 || spec.remoteRecvAddr == 0) {
        return Fail(program, report, "missing direct CCU alltoall address");
    }
    if (spec.localSendToken == 0 || spec.localRecvToken == 0 || spec.remoteRecvToken == 0) {
        return Fail(program, report, "missing direct CCU alltoall token");
    }
    if (spec.bytes == 0 || spec.bytes % TILEXR_CCU_ALLTOALL_MEMORY_SLICE_BYTES != 0) {
        return Fail(program, report, "direct CCU alltoall bytes must be nonzero and 4KB aligned");
    }
    if (spec.memorySliceBytes != TILEXR_CCU_ALLTOALL_MEMORY_SLICE_BYTES) {
        return Fail(program, report, "direct CCU alltoall memorySliceBytes must be 4096");
    }
    if (spec.memSlicePerBlock == 0 || spec.memSlicePerBlock > TILEXR_CCU_ALLTOALL_MEM_SLICE_PER_BLOCK) {
        return Fail(program, report, "direct CCU alltoall memSlicePerBlock must be in [1, 8]");
    }
    const uint64_t bytesPerBlock = static_cast<uint64_t>(spec.memorySliceBytes) * spec.memSlicePerBlock;
    if (spec.bytes % bytesPerBlock != 0) {
        return Fail(program, report, "direct CCU alltoall bytes must align to memSlicePerBlock");
    }
    if (spec.localGsa == 0 || spec.remoteGsa == 0 || spec.localXn == 0 || spec.remoteXn == 0 ||
        spec.lengthXn == 0) {
        return Fail(program, report, "missing direct CCU alltoall GSA/XN resource");
    }
    const uint16_t preSyncChannelId = spec.preSyncChannelId == 0 ? spec.channelId : spec.preSyncChannelId;
    const uint16_t copyChannelId = spec.copyChannelId == 0 ? spec.channelId : spec.copyChannelId;
    const uint16_t postSyncChannelId = spec.postSyncChannelId == 0 ? spec.channelId : spec.postSyncChannelId;
    if (preSyncChannelId == 0 || copyChannelId == 0 || postSyncChannelId == 0 ||
        spec.copyCompletionCke == 0 || spec.preSyncLocalWaitCke == 0 ||
        spec.preSyncRemoteNotifyCke == 0 || spec.postSyncLocalWaitCke == 0 ||
        spec.postSyncRemoteNotifyCke == 0 ||
        (spec.postSyncNotify && spec.sourceCke == 0) || spec.ckeMask == 0) {
        return Fail(program, report, "missing direct CCU alltoall CKE/channel resource");
    }
    if (spec.preSyncMarkerEnabled &&
        (spec.preSyncLocalMarkerXn == 0 || spec.preSyncRemoteMarkerXn == 0 ||
            spec.preSyncMarkerArgIndex >= TILEXR_CCU_SQE_ARGS_LEN)) {
        return Fail(program, report, "missing direct CCU alltoall loop marker resource");
    }
    return TILEXR_SUCCESS;
}

int AppendSetSourceCke(
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    uint16_t mask,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    TileXRCcuCkeSpec set;
    set.ckeId = spec.sourceCke;
    set.mask = mask;
    set.clearWait = true;

    TileXRCcuInstr instr;
    if (TileXRCcuEncodeSetCke(set, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to encode direct CCU alltoall source CKE set");
    }
    program->push_back(instr);
    return TILEXR_SUCCESS;
}

int AppendNotifyWait(
    uint16_t localWaitCke,
    uint16_t mask,
    const char* phase,
    bool clearCkeWait,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    TileXRCcuCkeSpec wait;
    wait.waitCkeId = localWaitCke;
    wait.waitMask = mask;
    wait.clearWait = true;

    TileXRCcuInstr instr;
    const int ret = clearCkeWait ?
        TileXRCcuEncodeClearCke(wait, &instr) :
        TileXRCcuEncodeSetCke(wait, &instr);
    if (ret != TILEXR_SUCCESS) {
        return Fail(program, report, std::string("failed to encode direct CCU alltoall ") + phase + " NotifyWait");
    }
    program->push_back(instr);
    return TILEXR_SUCCESS;
}

int AppendSyncPair(
    uint16_t remoteNotifyCke,
    uint16_t localWaitCke,
    uint16_t channelId,
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    uint16_t localMask,
    uint16_t waitMask,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    TileXRCcuSyncCkeSpec post;
    post.remoteCke = remoteNotifyCke;
    post.localCke = spec.sourceCke;
    post.localCkeMask = localMask;
    post.channelId = channelId;
    post.clearWait = true;

    TileXRCcuInstr instr;
    if (TileXRCcuEncodeSyncCke(post, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to encode direct CCU alltoall SyncCke");
    }
    program->push_back(instr);

    return AppendNotifyWait(localWaitCke, waitMask, "PostSync", true, program, report);
}

int AppendRemoteNotify(
    uint16_t remoteNotifyCke,
    uint16_t channelId,
    uint16_t localXn,
    uint16_t remoteXn,
    uint64_t value,
    uint16_t secFlag,
    uint16_t mask,
    const char* phase,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    TileXRCcuInstr instr;
    if (TileXRCcuEncodeLoadImdToXn(localXn, value, secFlag, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, std::string("failed to encode direct CCU alltoall ") + phase + " variable load");
    }
    program->push_back(instr);

    TileXRCcuSyncXnSpec notify;
    notify.remoteXn = remoteXn;
    notify.localXn = localXn;
    notify.channelId = channelId;
    notify.notifyCke = remoteNotifyCke;
    notify.notifyMask = mask;
    notify.clearWait = true;

    if (TileXRCcuEncodeSyncXn(notify, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, std::string("failed to encode direct CCU alltoall ") + phase + " SyncXn notify");
    }
    program->push_back(instr);
    return TILEXR_SUCCESS;
}

int AppendRemoteMarkerNotify(
    uint16_t remoteNotifyCke,
    uint16_t channelId,
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    TileXRCcuInstr instr;
    if (TileXRCcuEncodeLoadSqeArgsToX(
            spec.preSyncLocalMarkerXn,
            spec.preSyncMarkerArgIndex,
            &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to encode direct CCU alltoall PreSync loop marker load");
    }
    program->push_back(instr);

    TileXRCcuSyncXnSpec notify;
    notify.remoteXn = spec.preSyncRemoteMarkerXn;
    notify.localXn = spec.preSyncLocalMarkerXn;
    notify.channelId = channelId;
    notify.notifyCke = remoteNotifyCke;
    notify.notifyMask = TILEXR_CCU_ALLTOALL_LOOP_MARKER_MASK;
    notify.clearWait = true;
    if (TileXRCcuEncodeSyncXn(notify, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to encode direct CCU alltoall PreSync loop marker notify");
    }
    program->push_back(instr);
    return TILEXR_SUCCESS;
}

int AppendPreSyncPhase(
    uint16_t remoteNotifyCke,
    uint16_t localWaitCke,
    uint16_t outputChannelId,
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    const uint16_t outputMask = PreSyncSignalMask(spec);
    const uint16_t tokenMask = PreSyncTokenMask(spec);
    const uint16_t markerMask = spec.preSyncMarkerEnabled ? TILEXR_CCU_ALLTOALL_LOOP_MARKER_MASK : 0U;
    const uint16_t waitMask = static_cast<uint16_t>(markerMask | outputMask | tokenMask);
    const uint16_t localOutputXn =
        spec.preSyncLocalAddrXn == 0 ? spec.localXn : spec.preSyncLocalAddrXn;
    const uint16_t localTokenXn =
        spec.preSyncLocalTokenXn == 0 ? spec.lengthXn : spec.preSyncLocalTokenXn;
    const uint16_t tokenChannelId =
        spec.preSyncTokenChannelId == 0 ? outputChannelId : spec.preSyncTokenChannelId;
    const uint16_t tokenNotifyCke =
        spec.preSyncRemoteTokenNotifyCke == 0 ? remoteNotifyCke : spec.preSyncRemoteTokenNotifyCke;
    if (spec.preSyncMarkerEnabled &&
        AppendRemoteMarkerNotify(
            remoteNotifyCke,
            outputChannelId,
            spec,
            program,
            report) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (AppendRemoteNotify(
            remoteNotifyCke,
            outputChannelId,
            localOutputXn,
            spec.preSyncRemoteAddrXn,
            spec.localRecvAddr,
            0,
            outputMask,
            "PreSync output",
            program,
            report) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (AppendRemoteNotify(
            tokenNotifyCke,
            tokenChannelId,
            localTokenXn,
            spec.preSyncRemoteTokenXn,
            spec.localRecvToken,
            1,
            tokenMask,
            "PreSync token",
            program,
            report) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    if (!spec.preSyncWait) {
        return TILEXR_SUCCESS;
    }

    return AppendNotifyWait(localWaitCke, waitMask, "PreSync output", false, program, report);
}

int AppendPostSyncPhase(
    uint16_t remoteNotifyCke,
    uint16_t localWaitCke,
    uint16_t channelId,
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    const uint16_t notifyMask = PostSyncSignalMask(spec);
    const uint16_t waitMask = PostSyncSignalMask(spec);
    if (AppendSetSourceCke(spec, notifyMask, program, report) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!spec.postSyncWait) {
        TileXRCcuSyncCkeSpec post;
        post.remoteCke = remoteNotifyCke;
        post.localCke = spec.sourceCke;
        post.localCkeMask = notifyMask;
        post.channelId = channelId;
        post.clearWait = true;

        TileXRCcuInstr instr;
        if (TileXRCcuEncodeSyncCke(post, &instr) != TILEXR_SUCCESS) {
            return Fail(program, report, "failed to encode direct CCU alltoall PostSync notify-only SyncCke");
        }
        program->push_back(instr);
        return TILEXR_SUCCESS;
    }
    return AppendSyncPair(
        remoteNotifyCke,
        localWaitCke,
        channelId,
        spec,
        notifyMask,
        waitMask,
        program,
        report);
}

int AppendCopyBlock(
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    uint64_t offset,
    uint64_t bytesPerBlock,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    TileXRCcuMemoryCopySpec copy;
    copy.direction = TileXRCcuMemoryCopyDirection::LocalToRemote;
    copy.localGsa = spec.localGsa;
    copy.localXn = spec.localXn;
    copy.remoteGsa = spec.remoteGsa;
    copy.remoteXn = spec.remoteXn;
    copy.lengthXn = spec.lengthXn;
    copy.localAddr = spec.localSendAddr + offset;
    copy.localToken = spec.localSendToken;
    copy.remoteAddr = spec.remoteRecvAddr + offset;
    copy.remoteToken = spec.remoteRecvToken;
    copy.lengthBytes = bytesPerBlock;
    copy.channelId = spec.copyChannelId == 0 ? spec.channelId : spec.copyChannelId;
    copy.completionCke = spec.copyCompletionCke;
    copy.completionMask = spec.ckeMask;

    std::vector<TileXRCcuInstr> block;
    TileXRCcuMemoryProgramReport memoryReport;
    if (TileXRCcuBuildMemoryCopyProgram(copy, &block, &memoryReport) != TILEXR_SUCCESS) {
        return Fail(program, report, memoryReport.message);
    }
    program->insert(program->end(), block.begin(), block.end());
    return TILEXR_SUCCESS;
}

int AppendFinish(
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    TileXRCcuInstr instr;
    if (TileXRCcuEncodeLoadImdToXn(spec.localXn, 0, 0, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to encode direct CCU alltoall finish instruction");
    }
    program->push_back(instr);
    return TILEXR_SUCCESS;
}

void FillReport(
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    const std::vector<TileXRCcuInstr>& program,
    TileXRCcuAllToAllProgramReport* report)
{
    if (report == nullptr) {
        return;
    }
    const uint32_t bytesPerBlock = spec.memorySliceBytes * spec.memSlicePerBlock;
    const uint32_t markerInstructionCount = spec.preSyncMarkerEnabled ? 2U : 0U;
    report->preSyncInstructionCount =
        spec.preSyncNotify ? (spec.preSyncWait ? 5U : 4U) + markerInstructionCount : 0U;
    report->blockCount = static_cast<uint32_t>(spec.bytes / bytesPerBlock);
    report->bytesPerBlock = bytesPerBlock;
    report->copyInstructionCount = report->blockCount * 7U;
    report->postSyncInstructionCount = !spec.postSyncNotify ? 0U : (spec.postSyncWait ? 3U : 2U);
    report->finishInstructionCount = spec.emitFinish ? 1U : 0U;
    report->totalInstructionCount = static_cast<uint32_t>(program.size());
    report->message = "ok";
}

bool InsertUnique(std::set<uint16_t>* ids, std::initializer_list<uint16_t> values)
{
    for (const uint16_t value : values) {
        if (value == 0 || !ids->insert(value).second) {
            return false;
        }
    }
    return true;
}

bool InsertPeerUnique(std::set<uint16_t>* ids, std::initializer_list<uint16_t> values)
{
    std::set<uint16_t> peerIds;
    for (const uint16_t value : values) {
        if (value == 0) {
            return false;
        }
        peerIds.insert(value);
    }
    for (const uint16_t value : peerIds) {
        if (!ids->insert(value).second) {
            return false;
        }
    }
    return true;
}

int ValidateMeshSpec(
    const TileXRCcuAllToAllMeshProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    if (program == nullptr) {
        return Fail(program, report, "missing output direct CCU alltoall mesh program");
    }
    if (spec.rankSize != 4U || spec.localRank >= spec.rankSize || spec.peers.size() != 3U) {
        return Fail(program, report, "direct CCU alltoall mesh requires four ranks and three peers");
    }
    if (spec.localSendAddr == 0 || spec.localRecvAddr == 0 ||
        spec.localSendToken == 0 || spec.localRecvToken == 0 || spec.chunkBytes == 0 ||
        spec.chunkBytes % TILEXR_CCU_ALLTOALL_BLOCK_BYTES != 0) {
        return Fail(program, report, "invalid direct CCU alltoall mesh local buffer");
    }
    if (spec.selfSourceGsa == 0 || spec.selfDestinationGsa == 0 || spec.selfSourceXn == 0 ||
        spec.selfDestinationXn == 0 || spec.selfLengthXn == 0 || spec.selfChannelId == 0 ||
        spec.selfCompletionCke == 0) {
        return Fail(program, report, "missing direct CCU alltoall mesh self-copy resource");
    }
    bool peerRanks[4] = {};
    std::set<uint16_t> localXnIds;
    std::set<uint16_t> remoteXnIds;
    std::set<uint16_t> channelIds;
    std::set<uint16_t> localCkeIds;
    std::set<uint16_t> remoteCkeIds;
    for (const auto& peer : spec.peers) {
        if (peer.peerRank >= spec.rankSize || peer.peerRank == spec.localRank || peerRanks[peer.peerRank]) {
            return Fail(program, report, "invalid direct CCU alltoall mesh peer rank");
        }
        peerRanks[peer.peerRank] = true;
        if (peer.route.localRank != spec.localRank || peer.route.localSendAddr != spec.localSendAddr ||
            peer.route.localSendToken != spec.localSendToken || peer.route.localRecvAddr != spec.localRecvAddr ||
            peer.route.localRecvToken != spec.localRecvToken || peer.route.bytes != spec.chunkBytes ||
            !peer.route.preSyncMarkerEnabled || !peer.route.preSyncNotify || !peer.route.preSyncWait ||
            !peer.route.postSyncNotify || !peer.route.postSyncWait) {
            return Fail(program, report, "invalid direct CCU alltoall mesh peer route");
        }
        if (!InsertPeerUnique(&localXnIds,
                {peer.route.localXn, peer.route.lengthXn, peer.route.preSyncLocalAddrXn,
                    peer.route.preSyncLocalTokenXn, peer.route.preSyncLocalMarkerXn}) ||
            !InsertPeerUnique(&remoteXnIds,
                {peer.route.remoteXn, peer.route.preSyncRemoteAddrXn,
                    peer.route.preSyncRemoteTokenXn, peer.route.preSyncRemoteMarkerXn}) ||
            !InsertUnique(&channelIds,
                {peer.route.preSyncChannelId, peer.route.preSyncTokenChannelId, peer.route.copyChannelId}) ||
            !InsertPeerUnique(&localCkeIds,
                {peer.route.copyCompletionCke, peer.route.preSyncLocalWaitCke,
                    peer.route.postSyncLocalWaitCke}) ||
            !InsertPeerUnique(&remoteCkeIds,
                {peer.route.preSyncRemoteNotifyCke, peer.route.preSyncRemoteTokenNotifyCke,
                    peer.route.postSyncRemoteNotifyCke})) {
            return Fail(program, report, "duplicate direct CCU alltoall mesh peer resource");
        }
        TileXRCcuAllToAll2RankProgramSpec validationRoute = peer.route;
        validationRoute.localRank = 0;
        std::vector<TileXRCcuInstr> ignored;
        TileXRCcuAllToAllProgramReport ignoredReport;
        if (ValidateSpec(validationRoute, &ignored, &ignoredReport) != TILEXR_SUCCESS) {
            return Fail(program, report, ignoredReport.message);
        }
    }
    return TILEXR_SUCCESS;
}

int AppendMeshPeerPosts(
    const TileXRCcuAllToAll2RankProgramSpec& route,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    if (AppendRemoteMarkerNotify(
            route.preSyncRemoteNotifyCke,
            route.preSyncChannelId,
            route,
            program,
            report) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (AppendRemoteNotify(
            route.preSyncRemoteNotifyCke,
            route.preSyncChannelId,
            route.preSyncLocalAddrXn,
            route.preSyncRemoteAddrXn,
            route.localRecvAddr,
            0,
            PreSyncSignalMask(route),
            "mesh PreSync output",
            program,
            report) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    return AppendRemoteNotify(
        route.preSyncRemoteTokenNotifyCke,
        route.preSyncTokenChannelId,
        route.preSyncLocalTokenXn,
        route.preSyncRemoteTokenXn,
        route.localRecvToken,
        1,
        PreSyncTokenMask(route),
        "mesh PreSync token",
        program,
        report);
}

int AppendLocalCopyBlock(
    const TileXRCcuAllToAllMeshProgramSpec& spec,
    uint64_t offset,
    uint64_t bytesPerBlock,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    TileXRCcuInstr instr;
    if (TileXRCcuEncodeLoadImdToGsa(spec.selfSourceGsa, spec.localSendAddr + offset, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to load direct CCU alltoall mesh self source address");
    }
    program->push_back(instr);
    if (TileXRCcuEncodeLoadImdToXn(spec.selfSourceXn, spec.localSendToken, 1U, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to load direct CCU alltoall mesh self source token");
    }
    program->push_back(instr);
    if (TileXRCcuEncodeLoadImdToGsa(spec.selfDestinationGsa, spec.localRecvAddr + offset, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to load direct CCU alltoall mesh self destination address");
    }
    program->push_back(instr);
    if (TileXRCcuEncodeLoadImdToXn(spec.selfDestinationXn, spec.localRecvToken, 1U, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to load direct CCU alltoall mesh self destination token");
    }
    program->push_back(instr);
    if (TileXRCcuEncodeLoadImdToXn(spec.selfLengthXn, bytesPerBlock, 0, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to load direct CCU alltoall mesh self length");
    }
    program->push_back(instr);

    TileXRCcuMemTransferSpec transfer;
    transfer.localGsa = spec.selfSourceGsa;
    transfer.localXn = spec.selfSourceXn;
    transfer.remoteGsa = spec.selfDestinationGsa;
    transfer.remoteXn = spec.selfDestinationXn;
    transfer.lengthXn = spec.selfLengthXn;
    transfer.channelId = spec.selfChannelId;
    transfer.setCkeId = spec.selfCompletionCke;
    transfer.setCkeMask = 1U;
    if (TileXRCcuEncodeTransLocMemToLocMem(transfer, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to encode direct CCU alltoall mesh self transfer");
    }
    program->push_back(instr);

    TileXRCcuCkeSpec wait;
    wait.waitCkeId = spec.selfCompletionCke;
    wait.waitMask = 1U;
    if (TileXRCcuEncodeClearCke(wait, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to wait direct CCU alltoall mesh self transfer");
    }
    program->push_back(instr);
    return TILEXR_SUCCESS;
}

int AppendMeshPostNotify(
    const TileXRCcuAllToAll2RankProgramSpec& route,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    if (AppendSetSourceCke(route, route.ckeMask, program, report) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    TileXRCcuSyncCkeSpec post;
    post.remoteCke = route.postSyncRemoteNotifyCke;
    post.localCke = route.sourceCke;
    post.localCkeMask = route.ckeMask;
    post.channelId = route.postSyncChannelId;
    post.clearWait = true;
    TileXRCcuInstr instr;
    if (TileXRCcuEncodeSyncCke(post, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to notify direct CCU alltoall mesh completion");
    }
    program->push_back(instr);
    return TILEXR_SUCCESS;
}

} // namespace

int TileXRCcuBuildAllToAll2RankProgram(
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    ResetReport(report);
    if (program != nullptr) {
        program->clear();
    }
    int ret = ValidateSpec(spec, program, report);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }

    const uint64_t bytesPerBlock = static_cast<uint64_t>(spec.memorySliceBytes) * spec.memSlicePerBlock;
    const uint32_t blockCount = static_cast<uint32_t>(spec.bytes / bytesPerBlock);
    const uint16_t preSyncChannelId = spec.preSyncChannelId == 0 ? spec.channelId : spec.preSyncChannelId;
    const uint16_t postSyncChannelId = spec.postSyncChannelId == 0 ? spec.channelId : spec.postSyncChannelId;
    const uint32_t markerInstructionCount = spec.preSyncMarkerEnabled ? 2U : 0U;
    program->reserve(
        (spec.preSyncNotify ? (spec.preSyncWait ? 5U : 4U) + markerInstructionCount : 0U) + blockCount * 7U +
        (!spec.postSyncNotify ? 0U : (spec.postSyncWait ? 3U : 2U)) +
        (spec.emitFinish ? 1U : 0U));

    if (spec.preSyncNotify) {
        ret = AppendPreSyncPhase(
            spec.preSyncRemoteNotifyCke,
            spec.preSyncLocalWaitCke,
            preSyncChannelId,
            spec,
            program,
            report);
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
    }

    for (uint32_t block = 0; block < blockCount; ++block) {
        const uint64_t offset = static_cast<uint64_t>(block) * bytesPerBlock;
        ret = AppendCopyBlock(spec, offset, bytesPerBlock, program, report);
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
    }

    if (spec.postSyncNotify) {
        ret = AppendPostSyncPhase(
            spec.postSyncRemoteNotifyCke,
            spec.postSyncLocalWaitCke,
            postSyncChannelId,
            spec,
            program,
            report);
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
    }
    if (spec.emitFinish) {
        ret = AppendFinish(spec, program, report);
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
    }

    FillReport(spec, *program, report);
    return TILEXR_SUCCESS;
}

int TileXRCcuBuildAllToAllMeshProgram(
    const TileXRCcuAllToAllMeshProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    ResetReport(report);
    if (program != nullptr) {
        program->clear();
    }
    int ret = ValidateMeshSpec(spec, program, report);
    if (ret != TILEXR_SUCCESS) {
        return ret;
    }

    auto peers = spec.peers;
    std::sort(peers.begin(), peers.end(), [](const TileXRCcuAllToAllMeshPeerSpec& lhs,
                                             const TileXRCcuAllToAllMeshPeerSpec& rhs) {
        return lhs.peerRank < rhs.peerRank;
    });
    const uint64_t bytesPerBlock = TILEXR_CCU_ALLTOALL_BLOCK_BYTES;
    const uint32_t blocksPerChunk = static_cast<uint32_t>(spec.chunkBytes / bytesPerBlock);
    program->reserve(3U * 7U + 4U * blocksPerChunk * 7U + 3U * 3U + 1U);

    for (const auto& peer : peers) {
        ret = AppendMeshPeerPosts(peer.route, program, report);
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
    }
    for (const auto& peer : peers) {
        ret = AppendNotifyWait(peer.route.preSyncLocalWaitCke, 0x7U, "mesh PreSync", false, program, report);
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
    }

    for (const auto& peer : peers) {
        TileXRCcuAllToAll2RankProgramSpec route = peer.route;
        route.localSendAddr = spec.localSendAddr + static_cast<uint64_t>(peer.peerRank) * spec.chunkBytes;
        route.remoteRecvAddr += static_cast<uint64_t>(spec.localRank) * spec.chunkBytes;
        for (uint32_t block = 0; block < blocksPerChunk; ++block) {
            ret = AppendCopyBlock(route, static_cast<uint64_t>(block) * bytesPerBlock, bytesPerBlock, program, report);
            if (ret != TILEXR_SUCCESS) {
                return ret;
            }
        }
    }
    const uint64_t selfBaseOffset = static_cast<uint64_t>(spec.localRank) * spec.chunkBytes;
    for (uint32_t block = 0; block < blocksPerChunk; ++block) {
        ret = AppendLocalCopyBlock(
            spec,
            selfBaseOffset + static_cast<uint64_t>(block) * bytesPerBlock,
            bytesPerBlock,
            program,
            report);
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
    }

    for (const auto& peer : peers) {
        ret = AppendMeshPostNotify(peer.route, program, report);
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
    }
    for (const auto& peer : peers) {
        ret = AppendNotifyWait(
            peer.route.postSyncLocalWaitCke,
            peer.route.ckeMask,
            "mesh PostSync",
            true,
            program,
            report);
        if (ret != TILEXR_SUCCESS) {
            return ret;
        }
    }

    TileXRCcuInstr finish;
    if (TileXRCcuEncodeLoadImdToXn(spec.selfSourceXn, 0, 0, &finish) != TILEXR_SUCCESS) {
        return Fail(program, report, "failed to encode direct CCU alltoall mesh finish");
    }
    program->push_back(finish);

    if (report != nullptr) {
        report->preSyncInstructionCount = 21U;
        report->blockCount = blocksPerChunk;
        report->bytesPerBlock = static_cast<uint32_t>(bytesPerBlock);
        report->copyInstructionCount = 4U * blocksPerChunk * 7U;
        report->postSyncInstructionCount = 9U;
        report->finishInstructionCount = 1U;
        report->totalInstructionCount = static_cast<uint32_t>(program->size());
        report->peerCount = 3U;
        report->syncResourceCount = 9U;
        report->remoteBlockCount = 3U * blocksPerChunk;
        report->selfBlockCount = blocksPerChunk;
        report->message = "ok";
    }
    return TILEXR_SUCCESS;
}

} // namespace TileXR
