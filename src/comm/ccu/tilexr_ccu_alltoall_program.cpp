/*
 * Copyright (c) 2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "ccu/tilexr_ccu_alltoall_program.h"

namespace TileXR {
namespace {

uint16_t PreSyncSignalMask(const TileXRCcuAllToAll2RankProgramSpec& spec)
{
    (void)spec;
    return static_cast<uint16_t>(1U << TILEXR_CCU_ALLTOALL_OUTPUT_XN_ID);
}

uint16_t PostSyncSignalMask(const TileXRCcuAllToAll2RankProgramSpec& spec)
{
    (void)spec;
    return static_cast<uint16_t>(1U << TILEXR_CCU_ALLTOALL_POST_SYNC_ID);
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
    if (spec.localSendAddr == 0 || spec.localRecvAddr == 0 || spec.remoteSendAddr == 0) {
        return Fail(program, report, "missing direct CCU alltoall address");
    }
    if (spec.localSendToken == 0 || spec.localRecvToken == 0 || spec.remoteSendToken == 0) {
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
        spec.postSyncRemoteNotifyCke == 0 || spec.sourceCke == 0 || spec.ckeMask == 0) {
        return Fail(program, report, "missing direct CCU alltoall CKE/channel resource");
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
    const TileXRCcuAllToAll2RankProgramSpec& spec,
    uint16_t mask,
    const char* phase,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuAllToAllProgramReport* report)
{
    if (AppendSetSourceCke(spec, mask, program, report) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    TileXRCcuSyncCkeSpec post;
    post.remoteCke = remoteNotifyCke;
    post.localCke = spec.sourceCke;
    post.localCkeMask = mask;
    post.channelId = channelId;
    post.clearWait = true;

    TileXRCcuInstr instr;
    if (TileXRCcuEncodeSyncCke(post, &instr) != TILEXR_SUCCESS) {
        return Fail(program, report, std::string("failed to encode direct CCU alltoall ") + phase + " SyncCke notify");
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
    const uint16_t notifyMask = PreSyncSignalMask(spec);
    const uint16_t waitMask = PreSyncSignalMask(spec);
    if (AppendRemoteNotify(
            remoteNotifyCke,
            outputChannelId,
            spec,
            notifyMask,
            "PreSync output",
            program,
            report) != TILEXR_SUCCESS) {
        return TILEXR_ERROR_PARA_CHECK_FAIL;
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
    copy.direction = TileXRCcuMemoryCopyDirection::RemoteToLocal;
    copy.localGsa = spec.localGsa;
    copy.localXn = spec.localXn;
    copy.remoteGsa = spec.remoteGsa;
    copy.remoteXn = spec.remoteXn;
    copy.lengthXn = spec.lengthXn;
    copy.localAddr = spec.localRecvAddr + offset;
    copy.localToken = spec.localRecvToken;
    copy.remoteAddr = spec.remoteSendAddr + offset;
    copy.remoteToken = spec.remoteSendToken;
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
    report->preSyncInstructionCount = 3;
    report->blockCount = static_cast<uint32_t>(spec.bytes / bytesPerBlock);
    report->bytesPerBlock = bytesPerBlock;
    report->copyInstructionCount = report->blockCount * 7U;
    report->postSyncInstructionCount = !spec.postSyncNotify ? 0U : (spec.postSyncWait ? 3U : 2U);
    report->finishInstructionCount = spec.emitFinish ? 1U : 0U;
    report->totalInstructionCount = static_cast<uint32_t>(program.size());
    report->message = "ok";
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
    program->reserve(
        3U + blockCount * 7U +
        (!spec.postSyncNotify ? 0U : (spec.postSyncWait ? 3U : 2U)) +
        (spec.emitFinish ? 1U : 0U));

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

} // namespace TileXR
