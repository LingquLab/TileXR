#include "tilexr/checker/executor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tilexr/checker/collective_trace_runner.h"
#include "tilexr/checker/event.h"
#include "tilexr/checker/shim_runtime.h"

namespace tilexr {
namespace checker {

namespace {

const char *kEpDispatchSourceFile = "src/ep/kernels/tilexr_ep_dispatch_kernel.cpp";
const int kEpDispatchWindowHeaderLine = 143;
const int kEpDispatchSrcSlotHeaderLine = 159;
const int kEpDispatchProducerPayloadLine = 296;
const int kEpDispatchProducerAssistLine = 298;
const int kEpDispatchRecvCountLine = 327;
const int kEpDispatchOutputPayloadLine = 335;
const int kEpDispatchOutputAssistLine = 338;
const int kEpDispatchExpertTokenCountLine = 344;
const char *kEpCombineSourceFile = "src/ep/kernels/tilexr_ep_combine_kernel.cpp";
const int kEpCombineWindowHeaderLine = 197;
const int kEpCombineReturnSlotHeaderLine = 205;
const int kEpCombineSrcSlotHeaderLine = 211;
const int kEpCombinePayloadReadLine = 224;
const int kEpCombineAssistReadLine = 231;
const int kEpCombineOutputWriteLine = 247;
const size_t kEpAssistTupleBytes = sizeof(int32_t) * 4;

CheckerStatus RequireWorldAndSize(RankWorld *world, const CheckerCase &test_case) {
    if (world == nullptr) {
        return CheckerStatus::Fail("rank world is null");
    }
    if (world->rank_size() != test_case.rank_size) {
        return CheckerStatus::Fail("rank world size does not match checker case");
    }
    return CheckerStatus::Ok();
}

void AddUnsupportedFinding(const CheckerStatus &status, FindingSet *findings,
                           const std::string &source_file = std::string(),
                           int source_line = 0) {
    Finding finding;
    finding.id = "UNSUPPORTED_API:executor";
    finding.kind = FindingKind::kUnsupportedApi;
    finding.severity = Severity::kWarning;
    finding.message = status.message;
    finding.next_action =
        "Adjust the checker case to a supported collective/data type combination.";
    finding.buffer_role = BufferRole::kMetadata;
    finding.source_file = source_file;
    finding.source_line = source_line;
    finding.confidence = 0.9;
    findings->Add(finding);
}

bool HasFindingKind(const FindingSet &findings, FindingKind kind) {
    const std::vector<Finding> &items = findings.findings();
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].kind == kind) {
            return true;
        }
    }
    return false;
}

CheckerStatus PublishInputToCommData(RankWorld *world, int rank, size_t bytes,
                                     const std::string &detail) {
    ShimRuntime runtime(world);
    return runtime.Copy(rank, rank, BufferRole::kCommData, 0, BufferRole::kUserInput, 0, bytes,
                        TILEXR_CHECKER_HERE, detail);
}

CheckerStatus AddStoreEvents(RankWorld *world, const CheckerCase &test_case, int rank) {
    ShimRuntime runtime(world);
    for (int peer = 0; peer < test_case.rank_size; ++peer) {
        if (peer == rank) {
            continue;
        }
        CheckerStatus status =
            runtime.StoreFlag(rank, peer, rank, test_case.magic, TILEXR_CHECKER_HERE,
                              "publish comm-data ready");
        if (!status.ok()) {
            return status;
        }
    }
    return CheckerStatus::Ok();
}

CheckerStatus AddWaitEvents(RankWorld *world, const CheckerCase &test_case, int rank) {
    ShimRuntime runtime(world);
    for (int peer = 0; peer < test_case.rank_size; ++peer) {
        if (peer == rank) {
            continue;
        }
        CheckerStatus status =
            runtime.WaitFlag(rank, peer, peer, test_case.magic, TILEXR_CHECKER_HERE,
                             "wait for peer comm-data");
        if (!status.ok()) {
            return status;
        }
    }
    return CheckerStatus::Ok();
}

CheckerStatus AddCommDataReadEvent(RankWorld *world, int rank, int owner_rank, size_t bytes,
                                   const std::string &detail) {
    if (world == nullptr) {
        return CheckerStatus::Fail("rank world is null");
    }
    Event event;
    event.kind = EventKind::kRead;
    event.rank = rank;
    event.peer_rank = owner_rank;
    event.server = world->ServerOfRank(rank);
    event.peer_server = world->ServerOfRank(owner_rank);
    event.buffer_role = BufferRole::kCommData;
    event.slot = owner_rank;
    event.offset = 0;
    event.bytes = bytes;
    event.source_file = __FILE__;
    event.source_line = __LINE__;
    event.detail = detail;
    world->events().Add(event);
    return CheckerStatus::Ok();
}

CheckerStatus ReadAllGatherPeerToOutput(RankWorld *world, const CheckerCase &test_case, int rank,
                                        int peer) {
    ShimRuntime runtime(world);
    const size_t bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    std::vector<int32_t> scratch(static_cast<size_t>(test_case.count), 0);
    CheckerStatus read_event =
        AddCommDataReadEvent(world, rank, peer, bytes, "read peer comm-data");
    if (!read_event.ok()) {
        return read_event;
    }
    CheckerStatus read_status = world->CommData(peer, peer).ReadBytes(0, scratch.data(), bytes);
    if (!read_status.ok()) {
        return read_status;
    }
    CheckerStatus write_status =
        world->UserOutput(rank).WriteBytes(static_cast<size_t>(peer) * bytes, scratch.data(), bytes);
    if (!write_status.ok()) {
        return write_status;
    }
    return runtime.RecordWrite(rank, rank, BufferRole::kUserOutput,
                               static_cast<size_t>(peer) * bytes, bytes, TILEXR_CHECKER_HERE,
                               "write gathered user output");
}

CheckerStatus ExecuteAllGatherReadsAndWrites(RankWorld *world, const CheckerCase &test_case,
                                             int rank) {
    for (int peer = 0; peer < test_case.rank_size; ++peer) {
        CheckerStatus status = ReadAllGatherPeerToOutput(world, test_case, rank, peer);
        if (!status.ok()) {
            return status;
        }
    }
    return CheckerStatus::Ok();
}

CheckerStatus ExecuteAllReduceReadAndWrite(RankWorld *world, const CheckerCase &test_case,
                                           int rank) {
    ShimRuntime runtime(world);
    const size_t count = static_cast<size_t>(test_case.count);
    const size_t bytes = count * sizeof(int32_t);
    std::vector<int32_t> reduced(count, 0);
    for (int peer = 0; peer < test_case.rank_size; ++peer) {
        CheckerStatus read_event =
            AddCommDataReadEvent(world, rank, peer, bytes,
                                 "read peer comm-data for reduction");
        if (!read_event.ok()) {
            return read_event;
        }
        for (size_t i = 0; i < count; ++i) {
            int32_t value = 0;
            CheckerStatus read_status = world->CommData(peer, peer).ReadInt32(i, &value);
            if (!read_status.ok()) {
                return read_status;
            }
            reduced[i] += value;
        }
    }
    CheckerStatus write_status = world->UserOutput(rank).WriteBytes(0, reduced.data(), bytes);
    if (!write_status.ok()) {
        return write_status;
    }
    return runtime.RecordWrite(rank, rank, BufferRole::kUserOutput, 0, bytes,
                               TILEXR_CHECKER_HERE, "write reduced user output");
}

CheckerStatus AccumulateAllReducePeer(RankWorld *world, const CheckerCase &test_case, int rank,
                                      int peer, std::vector<int32_t> *reduced) {
    if (reduced == nullptr) {
        return CheckerStatus::Fail("allreduce accumulator is null");
    }
    const size_t bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    CheckerStatus read_event =
        AddCommDataReadEvent(world, rank, peer, bytes,
                             "read peer comm-data for reduction");
    if (!read_event.ok()) {
        return read_event;
    }
    for (size_t i = 0; i < reduced->size(); ++i) {
        int32_t value = 0;
        CheckerStatus read_status = world->CommData(peer, peer).ReadInt32(i, &value);
        if (!read_status.ok()) {
            return read_status;
        }
        (*reduced)[i] += value;
    }
    return CheckerStatus::Ok();
}

CheckerStatus WriteAllReduceOutput(RankWorld *world, int rank,
                                   const std::vector<int32_t> &reduced) {
    ShimRuntime runtime(world);
    const size_t bytes = reduced.size() * sizeof(int32_t);
    CheckerStatus write_status = world->UserOutput(rank).WriteBytes(0, reduced.data(), bytes);
    if (!write_status.ok()) {
        return write_status;
    }
    return runtime.RecordWrite(rank, rank, BufferRole::kUserOutput, 0, bytes,
                               TILEXR_CHECKER_HERE, "write reduced user output");
}

RunResult MakeResultFromStatus(const CheckerStatus &status, const FindingSet &findings,
                               const std::vector<OutputMismatch> &mismatches,
                               size_t event_count) {
    RunResult result;
    result.status = status;
    result.findings = findings;
    result.mismatches = mismatches;
    result.event_count = event_count;
    return result;
}

CheckerStatus FinalizeExecutorStatus(const FindingSet &findings,
                                     const std::vector<OutputMismatch> &mismatches) {
    if (!findings.findings().empty() || !mismatches.empty()) {
        if (HasFindingKind(findings, FindingKind::kUnsupportedApi)) {
            return CheckerStatus::Unsupported("checker executor encountered unsupported case");
        }
        return CheckerStatus::Fail("checker executor detected findings or mismatches");
    }
    return CheckerStatus::Ok();
}

size_t EpWindowSlotOffset(const EpDispatchWindowLayout &layout, int slot) {
    return kEpAssistTupleBytes == 0
               ? 0
               : static_cast<size_t>(64 + static_cast<int64_t>(slot) * layout.slot_bytes);
}

size_t EpWindowPayloadOffset(const EpDispatchWindowLayout &layout, int slot, size_t row) {
    return EpWindowSlotOffset(layout, slot) + 64 + row * static_cast<size_t>(layout.row_bytes);
}

size_t EpWindowAssistOffset(const EpDispatchWindowLayout &layout, int slot, size_t row) {
    return EpWindowSlotOffset(layout, slot) + 64 +
           static_cast<size_t>(layout.payload_bytes_per_slot) +
           row * kEpAssistTupleBytes;
}

CheckerStatus AddEpEventAt(RankWorld *world, EventKind kind, int rank, int peer_rank,
                           BufferRole role, int slot, size_t offset, size_t bytes,
                           const char *source_file, int source_line,
                           const std::string &detail) {
    if (world == nullptr) {
        return CheckerStatus::Fail("rank world is null");
    }

    Event event;
    event.kind = kind;
    event.rank = rank;
    event.peer_rank = peer_rank;
    event.server = world->ServerOfRank(rank);
    event.peer_server = peer_rank >= 0 ? world->ServerOfRank(peer_rank) : -1;
    event.buffer_role = role;
    event.slot = slot;
    event.offset = offset;
    event.bytes = bytes;
    event.source_file = source_file == nullptr ? "" : source_file;
    event.source_line = source_line;
    event.detail = detail;
    world->events().Add(event);
    return CheckerStatus::Ok();
}

CheckerStatus AddEpEvent(RankWorld *world, EventKind kind, int rank, int peer_rank,
                         BufferRole role, int slot, size_t offset, size_t bytes,
                         int source_line, const std::string &detail) {
    return AddEpEventAt(world, kind, rank, peer_rank, role, slot, offset, bytes,
                        kEpDispatchSourceFile, source_line, detail);
}

CheckerStatus RecordEpWindowHeaders(RankWorld *world, const CheckerCase &test_case,
                                    const EpDispatchWindowLayout &layout) {
    for (int rank = 0; rank < test_case.rank_size; ++rank) {
        CheckerStatus status =
            AddEpEvent(world, EventKind::kWrite, rank, -1,
                       BufferRole::kRegisteredCommBuffer, -1, 0, 64,
                       kEpDispatchWindowHeaderLine,
                       "ep dispatch window header");
        if (!status.ok()) {
            return status;
        }
        for (int src_rank = 0; src_rank < test_case.rank_size; ++src_rank) {
            const size_t slot_offset =
                64 + static_cast<size_t>(src_rank) * static_cast<size_t>(layout.slot_bytes);
            status = AddEpEvent(world, EventKind::kWrite, rank, src_rank,
                                BufferRole::kRegisteredCommBuffer, src_rank,
                                slot_offset, 64, kEpDispatchSrcSlotHeaderLine,
                                "ep dispatch src slot header");
            if (!status.ok()) {
                return status;
            }
        }
    }
    return CheckerStatus::Ok();
}

CheckerStatus WriteEpRoutePayload(RankWorld *world, const CheckerCase &test_case,
                                  const EpDispatchRoute &route, int dst_rank, size_t row_index,
                                  size_t slot_row_index, size_t row_bytes,
                                  const EpDispatchWindowLayout &layout) {
    std::vector<uint8_t> row(row_bytes, 0);
    const size_t input_offset = static_cast<size_t>(route.token_id) * row_bytes;
    CheckerStatus status =
        world->UserInput(route.src_rank).ReadBytes(input_offset, row.data(), row_bytes);
    if (!status.ok()) {
        return status;
    }

    status = world->UserOutput(dst_rank).WriteBytes(row_index * row_bytes, row.data(), row_bytes);
    if (!status.ok()) {
        return status;
    }

    status = AddEpEvent(world, EventKind::kCopy, route.src_rank, dst_rank,
                        BufferRole::kRegisteredCommBuffer, dst_rank,
                        EpWindowPayloadOffset(layout, dst_rank, slot_row_index),
                        row_bytes, kEpDispatchProducerPayloadLine,
                        "ep dispatch route payload to peer window");
    if (!status.ok()) {
        return status;
    }
    return AddEpEvent(world, EventKind::kCopy, dst_rank, route.src_rank,
                      BufferRole::kUserOutput, dst_rank, row_index * row_bytes, row_bytes,
                      kEpDispatchOutputPayloadLine,
                      "ep dispatch route payload to expand output");
}

CheckerStatus RecordEpAssist(RankWorld *world, const EpDispatchRoute &route, int dst_rank,
                             size_t output_row_index, size_t slot_row_index,
                             const EpDispatchWindowLayout &layout) {
    CheckerStatus status =
        AddEpEvent(world, EventKind::kWrite, route.src_rank, dst_rank,
                   BufferRole::kRegisteredCommBuffer, dst_rank,
                   EpWindowAssistOffset(layout, dst_rank, slot_row_index),
                   kEpAssistTupleBytes,
                   kEpDispatchProducerAssistLine,
                   "ep dispatch assist tuple in peer window");
    if (!status.ok()) {
        return status;
    }
    return AddEpEvent(world, EventKind::kWrite, dst_rank, route.src_rank,
                      BufferRole::kMetadata, -1, output_row_index * kEpAssistTupleBytes,
                      kEpAssistTupleBytes, kEpDispatchOutputAssistLine,
                      "ep dispatch assist output");
}

CheckerStatus WriteEpAssistTuple(RankWorld *world, const CheckerCase &test_case, int dst_rank,
                                 const EpDispatchRoute &route, size_t row_index) {
    const int32_t tuple[4] = {
        route.src_rank,
        route.token_id,
        route.top_k_id,
        route.expert_id,
    };
    const size_t offset =
        EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kAssist) +
        row_index * kEpAssistTupleBytes;
    return world->UserOutput(dst_rank).WriteBytes(offset, tuple, sizeof(tuple));
}

CheckerStatus WriteEpRecvCount(RankWorld *world, const CheckerCase &test_case, int dst_rank,
                               int src_rank, int32_t count) {
    const size_t offset =
        EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kRecvCounts) +
        static_cast<size_t>(src_rank) * sizeof(int32_t);
    return world->UserOutput(dst_rank).WriteBytes(offset, &count, sizeof(count));
}

CheckerStatus WriteEpExpertTokenCount(RankWorld *world, const CheckerCase &test_case,
                                      int dst_rank, int64_t local_expert, int64_t count) {
    const size_t offset =
        EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kExpertTokenNums) +
        static_cast<size_t>(local_expert) * sizeof(int64_t);
    return world->UserOutput(dst_rank).WriteBytes(offset, &count, sizeof(count));
}

CheckerStatus RecordEpCombineReturnWindowHeaders(RankWorld *world,
                                                 const CheckerCase &test_case,
                                                 const EpDispatchWindowLayout &layout) {
    for (int owner_rank = 0; owner_rank < test_case.rank_size; ++owner_rank) {
        CheckerStatus status =
            AddEpEventAt(world, EventKind::kWrite, owner_rank, -1,
                         BufferRole::kRegisteredCommBuffer, -1, 0, 64,
                         kEpCombineSourceFile, kEpCombineWindowHeaderLine,
                         "ep combine return window header");
        if (!status.ok()) {
            return status;
        }
        for (int src_rank = 0; src_rank < test_case.rank_size; ++src_rank) {
            status = AddEpEventAt(
                world, EventKind::kWrite, owner_rank, src_rank,
                BufferRole::kRegisteredCommBuffer, src_rank,
                EpWindowSlotOffset(layout, src_rank), 64, kEpCombineSourceFile,
                kEpCombineReturnSlotHeaderLine,
                "ep combine return src slot header");
            if (!status.ok()) {
                return status;
            }
        }
    }
    return CheckerStatus::Ok();
}

CheckerStatus RecordEpCombineReturnPayloadProducers(
    RankWorld *world, const CheckerCase &test_case,
    const EpDispatchWindowLayout &layout, const std::vector<int32_t> &expert_ids) {
    const size_t row_bytes = static_cast<size_t>(layout.row_bytes);
    for (int dst_rank = 0; dst_rank < test_case.rank_size; ++dst_rank) {
        const std::vector<EpDispatchRoute> routes =
            ExpectedEpDispatchRoutes(test_case, expert_ids, dst_rank);
        std::vector<size_t> slot_rows(static_cast<size_t>(test_case.rank_size), 0);
        for (size_t row = 0; row < routes.size(); ++row) {
            const EpDispatchRoute &route = routes[row];
            const size_t slot_row = slot_rows[static_cast<size_t>(route.src_rank)]++;
            CheckerStatus status = AddEpEventAt(
                world, EventKind::kCopy, dst_rank, route.src_rank,
                BufferRole::kRegisteredCommBuffer, route.src_rank,
                EpWindowPayloadOffset(layout, route.src_rank, slot_row), row_bytes,
                kEpCombineSourceFile, kEpCombinePayloadReadLine - 1,
                "ep combine publish return payload to source slot");
            if (!status.ok()) {
                return status;
            }
            status = AddEpEventAt(
                world, EventKind::kWrite, dst_rank, route.src_rank,
                BufferRole::kRegisteredCommBuffer, route.src_rank,
                EpWindowAssistOffset(layout, route.src_rank, slot_row),
                kEpAssistTupleBytes, kEpCombineSourceFile,
                kEpCombineAssistReadLine - 1,
                "ep combine publish return assist tuple");
            if (!status.ok()) {
                return status;
            }
        }
    }
    return CheckerStatus::Ok();
}

CheckerStatus ExecuteEpCombineRoute(RankWorld *world, const CheckerCase &test_case,
                                    const EpDispatchWindowLayout &layout,
                                    int dst_rank, const EpDispatchRoute &expected_route,
                                    size_t input_row_index, size_t slot_row_index) {
    const size_t row_bytes = static_cast<size_t>(layout.row_bytes);
    CheckerStatus status = AddEpEventAt(
        world, EventKind::kRead, expected_route.src_rank, dst_rank,
        BufferRole::kRegisteredCommBuffer, expected_route.src_rank,
        EpWindowSlotOffset(layout, expected_route.src_rank), 64,
        kEpCombineSourceFile, kEpCombineSrcSlotHeaderLine,
        "ep combine read src slot header");
    if (!status.ok()) {
        return status;
    }
    status = AddEpEventAt(
        world, EventKind::kRead, expected_route.src_rank, dst_rank,
        BufferRole::kRegisteredCommBuffer, expected_route.src_rank,
        EpWindowPayloadOffset(layout, expected_route.src_rank, slot_row_index),
        row_bytes, kEpCombineSourceFile, kEpCombinePayloadReadLine,
        "ep combine read return payload");
    if (!status.ok()) {
        return status;
    }
    status = AddEpEventAt(
        world, EventKind::kRead, expected_route.src_rank, dst_rank,
        BufferRole::kRegisteredCommBuffer, expected_route.src_rank,
        EpWindowAssistOffset(layout, expected_route.src_rank, slot_row_index),
        kEpAssistTupleBytes, kEpCombineSourceFile, kEpCombineAssistReadLine,
        "ep combine read return assist tuple");
    if (!status.ok()) {
        return status;
    }

    std::vector<uint8_t> row(row_bytes, 0);
    status = world->UserInput(dst_rank).ReadBytes(input_row_index * row_bytes,
                                                  row.data(), row_bytes);
    if (!status.ok()) {
        return status;
    }

    int32_t tuple[4] = {};
    const size_t tuple_offset =
        EpDispatchMetadataOffset(test_case, EpDispatchMetadataRole::kAssist) +
        input_row_index * kEpAssistTupleBytes;
    status = world->UserInput(dst_rank).ReadBytes(tuple_offset, tuple, sizeof(tuple));
    if (!status.ok()) {
        return status;
    }
    const int out_rank = tuple[0];
    const int token_id = tuple[1];
    const int top_k_id = tuple[2];
    if (out_rank < 0 || out_rank >= test_case.rank_size ||
        token_id < 0 || token_id >= test_case.bs ||
        top_k_id < 0 || top_k_id >= test_case.top_k) {
        return CheckerStatus::Fail("ep combine assist tuple is outside output bounds");
    }

    const size_t output_offset =
        static_cast<size_t>((static_cast<int64_t>(token_id) * test_case.top_k +
                             top_k_id) *
                            test_case.h) *
        ElementSize(test_case.data_type);
    status = world->UserOutput(out_rank).WriteBytes(output_offset, row.data(), row_bytes);
    if (!status.ok()) {
        return status;
    }
    return AddEpEventAt(world, EventKind::kWrite, out_rank, dst_rank,
                        BufferRole::kUserOutput, -1, output_offset, row_bytes,
                        kEpCombineSourceFile, kEpCombineOutputWriteLine,
                        "ep combine write restored payload");
}

}  // namespace

RunResult CollectiveExecutor::Run(RankWorld *world, const CheckerCase &test_case) {
    FindingSet findings;
    const std::vector<OutputMismatch> no_mismatches;

    CheckerStatus world_status = RequireWorldAndSize(world, test_case);
    if (!world_status.ok()) {
        return MakeResultFromStatus(world_status, findings, no_mismatches, 0);
    }

    CheckerStatus case_status = ValidateCase(test_case);
    if (!case_status.ok()) {
        if (case_status.code == CheckerStatusCode::kUnsupported) {
            AddUnsupportedFinding(case_status, &findings, CaseSourceFile(test_case), 0);
        }
        return MakeResultFromStatus(case_status, findings, no_mismatches, 0);
    }

    world->ConfigureServers(test_case.server_count);
    world->events().Clear();

    if (test_case.op == CollectiveOp::kEpDispatch ||
        test_case.op == CollectiveOp::kEpCombine) {
        RunResult result = test_case.op == CollectiveOp::kEpDispatch
                               ? RunEpDispatch(world, test_case)
                               : RunEpCombine(world, test_case);
        if (!result.status.ok() && result.status.code == CheckerStatusCode::kUnsupported &&
            result.findings.findings().empty()) {
            AddUnsupportedFinding(result.status, &result.findings);
        }
        result.event_count = world->events().events().size();
        return result;
    }

    CheckerStatus fill_status = FillRankIndexInt32Inputs(world, test_case);
    if (!fill_status.ok()) {
        return MakeResultFromStatus(fill_status, findings, no_mismatches,
                                    world->events().events().size());
    }

    RunResult result = test_case.op == CollectiveOp::kAllGather
        ? (test_case.algorithm == AlgorithmId::kAllGatherHierarchyDoubleRing
               ? RunAllGatherHierarchyDoubleRingInt32(world, test_case)
               : RunAllGatherInt32(world, test_case))
        : (test_case.algorithm == AlgorithmId::kAllReduceBigData
               ? RunAllReduceBigDataInt32(world, test_case)
               : RunAllReduceSumInt32(world, test_case));

    if (!result.status.ok() && result.status.code == CheckerStatusCode::kUnsupported &&
        result.findings.findings().empty()) {
        AddUnsupportedFinding(result.status, &result.findings);
    }

    result.event_count = world->events().events().size();
    if (!result.status.ok()) {
        return result;
    }

    const std::vector<OutputMismatch> mismatches =
        CompareInt32Output(*world, test_case, 8);
    const FindingSet output_findings = CheckOutputMismatches(mismatches);
    result.mismatches = mismatches;
    result.findings = MergeFindings(result.findings, output_findings);
    result.status = FinalizeExecutorStatus(result.findings, result.mismatches);
    return result;
}

RunResult CollectiveExecutor::RunAllGatherInt32(RankWorld *world, const CheckerCase &test_case) {
    FindingSet findings;
    const std::vector<OutputMismatch> mismatches;
    if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_INT32) {
        CheckerStatus status = CheckerStatus::Unsupported("only int32 allgather is supported");
        AddUnsupportedFinding(status, &findings);
        return MakeResultFromStatus(status, findings, mismatches, world->events().events().size());
    }

    const size_t bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    CheckerStatus exec_status = CheckerStatus::Ok();
    switch (test_case.scheduler) {
        case SchedulerMode::kSerial:
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status =
                    PublishInputToCommData(world, rank, bytes, "publish local input to comm-data");
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = AddStoreEvents(world, test_case, rank);
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = AddWaitEvents(world, test_case, rank);
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = ExecuteAllGatherReadsAndWrites(world, test_case, rank);
            }
            break;
        case SchedulerMode::kRoundRobin:
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status =
                    PublishInputToCommData(world, rank, bytes, "publish local input to comm-data");
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = AddStoreEvents(world, test_case, rank);
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = AddWaitEvents(world, test_case, rank);
            }
            for (int peer = 0; peer < test_case.rank_size && exec_status.ok(); ++peer) {
                for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                    exec_status = ReadAllGatherPeerToOutput(world, test_case, rank, peer);
                }
            }
            break;
    }
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches, world->events().events().size());
    }

    findings = CheckOrdering(world->events());
    return MakeResultFromStatus(CheckerStatus::Ok(), findings, mismatches,
                                world->events().events().size());
}

RunResult CollectiveExecutor::RunAllGatherHierarchyDoubleRingInt32(
    RankWorld *world, const CheckerCase &test_case) {
    FindingSet findings;
    const std::vector<OutputMismatch> mismatches;
    if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_INT32) {
        CheckerStatus status =
            CheckerStatus::Unsupported("only int32 allgather_hierarchy_double_ring is supported");
        AddUnsupportedFinding(status, &findings);
        return MakeResultFromStatus(status, findings, mismatches, world->events().events().size());
    }

    CheckerStatus exec_status = RunAllGatherHierarchyDoubleRingTrace(world, test_case);
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }

    findings = CheckOrdering(world->events());
    return MakeResultFromStatus(CheckerStatus::Ok(), findings, mismatches,
                                world->events().events().size());
}

RunResult CollectiveExecutor::RunAllReduceSumInt32(RankWorld *world,
                                                   const CheckerCase &test_case) {
    FindingSet findings;
    const std::vector<OutputMismatch> mismatches;
    if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_INT32 ||
        test_case.reduce_op != TileXR::TILEXR_REDUCE_SUM) {
        CheckerStatus status =
            CheckerStatus::Unsupported("only int32 allreduce sum is supported");
        AddUnsupportedFinding(status, &findings);
        return MakeResultFromStatus(status, findings, mismatches, world->events().events().size());
    }

    const size_t bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    CheckerStatus exec_status = CheckerStatus::Ok();
    switch (test_case.scheduler) {
        case SchedulerMode::kSerial:
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status =
                    PublishInputToCommData(world, rank, bytes, "publish local input to comm-data");
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = AddStoreEvents(world, test_case, rank);
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = AddWaitEvents(world, test_case, rank);
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = ExecuteAllReduceReadAndWrite(world, test_case, rank);
            }
            break;
        case SchedulerMode::kRoundRobin:
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status =
                    PublishInputToCommData(world, rank, bytes, "publish local input to comm-data");
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = AddStoreEvents(world, test_case, rank);
            }
            for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                exec_status = AddWaitEvents(world, test_case, rank);
            }
            {
                const size_t count = static_cast<size_t>(test_case.count);
                std::vector<std::vector<int32_t> > reduced(
                    static_cast<size_t>(test_case.rank_size), std::vector<int32_t>(count, 0));
                for (int peer = 0; peer < test_case.rank_size && exec_status.ok(); ++peer) {
                    for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                        exec_status =
                            AccumulateAllReducePeer(world, test_case, rank, peer, &reduced[rank]);
                    }
                }
                for (int rank = 0; rank < test_case.rank_size && exec_status.ok(); ++rank) {
                    exec_status = WriteAllReduceOutput(world, rank, reduced[rank]);
                }
            }
            break;
    }
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches, world->events().events().size());
    }

    findings = CheckOrdering(world->events());
    return MakeResultFromStatus(CheckerStatus::Ok(), findings, mismatches,
                                world->events().events().size());
}

RunResult CollectiveExecutor::RunAllReduceBigDataInt32(RankWorld *world,
                                                       const CheckerCase &test_case) {
    FindingSet findings;
    const std::vector<OutputMismatch> mismatches;
    if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_INT32 ||
        test_case.reduce_op != TileXR::TILEXR_REDUCE_SUM) {
        CheckerStatus status =
            CheckerStatus::Unsupported("only int32 allreduce big data sum is supported");
        AddUnsupportedFinding(status, &findings);
        return MakeResultFromStatus(status, findings, mismatches, world->events().events().size());
    }

    CheckerStatus exec_status = RunAllReduceBigDataTrace(world, test_case);
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }
    findings = CheckOrdering(world->events());
    return MakeResultFromStatus(CheckerStatus::Ok(), findings, mismatches,
                                world->events().events().size());
}

RunResult CollectiveExecutor::RunEpDispatch(RankWorld *world, const CheckerCase &test_case) {
    FindingSet findings;
    std::vector<OutputMismatch> mismatches;

    if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_FP16 &&
        test_case.data_type != TileXR::TILEXR_DATA_TYPE_BFP16) {
        CheckerStatus status = CheckerStatus::Unsupported(
            "only fp16 or bfp16 ep_dispatch is supported");
        AddUnsupportedFinding(status, &findings);
        return MakeResultFromStatus(status, findings, mismatches, world->events().events().size());
    }

    CheckerStatus exec_status = FillEpDispatchInputs(world, test_case);
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }

    EpDispatchWindowLayout window_layout;
    exec_status = ComputeEpDispatchWindowLayout(test_case, &window_layout);
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }
    exec_status = RecordEpWindowHeaders(world, test_case, window_layout);
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }

    const std::vector<int32_t> expert_ids = DefaultEpDispatchExpertIds(test_case);
    const size_t dtype_bytes = ElementSize(test_case.data_type);
    const size_t row_bytes = static_cast<size_t>(test_case.h) * dtype_bytes;
    const int64_t local_expert_num = test_case.moe_expert_num / test_case.rank_size;
    for (int dst_rank = 0; dst_rank < test_case.rank_size && exec_status.ok(); ++dst_rank) {
        const std::vector<EpDispatchRoute> routes =
            ExpectedEpDispatchRoutes(test_case, expert_ids, dst_rank);
        std::vector<int64_t> recv_counts(static_cast<size_t>(test_case.rank_size), 0);
        std::vector<int64_t> expert_counts(static_cast<size_t>(local_expert_num), 0);
        std::vector<size_t> slot_rows(static_cast<size_t>(test_case.rank_size), 0);

        for (size_t row = 0; row < routes.size() && exec_status.ok(); ++row) {
            const EpDispatchRoute &route = routes[row];
            const size_t slot_row = slot_rows[static_cast<size_t>(route.src_rank)]++;
            exec_status = WriteEpRoutePayload(world, test_case, route, dst_rank, row,
                                              slot_row, row_bytes, window_layout);
            if (!exec_status.ok()) {
                break;
            }
            exec_status = RecordEpAssist(world, route, dst_rank, row, slot_row,
                                         window_layout);
            if (!exec_status.ok()) {
                break;
            }
            exec_status = WriteEpAssistTuple(world, test_case, dst_rank, route, row);
            if (!exec_status.ok()) {
                break;
            }

            ++recv_counts[static_cast<size_t>(route.src_rank)];
            const int64_t local_expert =
                static_cast<int64_t>(route.expert_id) % local_expert_num;
            if (local_expert >= 0 && local_expert < local_expert_num) {
                ++expert_counts[static_cast<size_t>(local_expert)];
            }
        }

        for (int src_rank = 0; src_rank < test_case.rank_size && exec_status.ok(); ++src_rank) {
            exec_status = WriteEpRecvCount(
                world, test_case, dst_rank, src_rank,
                static_cast<int32_t>(recv_counts[static_cast<size_t>(src_rank)]));
            if (!exec_status.ok()) {
                break;
            }
            exec_status =
                AddEpEvent(world, EventKind::kWrite, dst_rank, src_rank,
                           BufferRole::kMetadata, -1, static_cast<size_t>(src_rank) *
                                                        sizeof(int32_t),
                           sizeof(int32_t), kEpDispatchRecvCountLine,
                           "ep dispatch recv count");
        }
        for (int64_t local_expert = 0;
             local_expert < local_expert_num && exec_status.ok(); ++local_expert) {
            exec_status = WriteEpExpertTokenCount(
                world, test_case, dst_rank, local_expert,
                expert_counts[static_cast<size_t>(local_expert)]);
            if (!exec_status.ok()) {
                break;
            }
            exec_status =
                AddEpEvent(world, EventKind::kWrite, dst_rank, -1, BufferRole::kMetadata, -1,
                           static_cast<size_t>(local_expert) * sizeof(int64_t),
                           sizeof(int64_t), kEpDispatchExpertTokenCountLine,
                           "ep dispatch expert token count");
        }
    }

    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }

    findings = CheckOrdering(world->events());
    mismatches = CompareEpDispatchOutput(*world, test_case, expert_ids, 8);
    findings = MergeFindings(findings, CheckOutputMismatches(mismatches));
    const CheckerStatus status = FinalizeExecutorStatus(findings, mismatches);
    return MakeResultFromStatus(status, findings, mismatches, world->events().events().size());
}

RunResult CollectiveExecutor::RunEpCombine(RankWorld *world, const CheckerCase &test_case) {
    FindingSet findings;
    std::vector<OutputMismatch> mismatches;

    if (test_case.data_type != TileXR::TILEXR_DATA_TYPE_FP16 &&
        test_case.data_type != TileXR::TILEXR_DATA_TYPE_BFP16) {
        CheckerStatus status = CheckerStatus::Unsupported(
            "only fp16 or bfp16 ep_combine is supported");
        AddUnsupportedFinding(status, &findings);
        return MakeResultFromStatus(status, findings, mismatches, world->events().events().size());
    }

    const std::vector<int32_t> expert_ids = DefaultEpDispatchExpertIds(test_case);
    CheckerStatus exec_status = FillEpCombineInputs(world, test_case, expert_ids);
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }

    EpDispatchWindowLayout window_layout;
    exec_status = ComputeEpDispatchWindowLayout(test_case, &window_layout);
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }
    exec_status = RecordEpCombineReturnWindowHeaders(world, test_case, window_layout);
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }
    exec_status =
        RecordEpCombineReturnPayloadProducers(world, test_case, window_layout, expert_ids);
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }

    for (int dst_rank = 0; dst_rank < test_case.rank_size && exec_status.ok(); ++dst_rank) {
        const std::vector<EpDispatchRoute> routes =
            ExpectedEpDispatchRoutes(test_case, expert_ids, dst_rank);
        std::vector<size_t> slot_rows(static_cast<size_t>(test_case.rank_size), 0);
        for (size_t row = 0; row < routes.size() && exec_status.ok(); ++row) {
            const EpDispatchRoute &route = routes[row];
            const size_t slot_row = slot_rows[static_cast<size_t>(route.src_rank)]++;
            exec_status = ExecuteEpCombineRoute(world, test_case, window_layout, dst_rank,
                                                route, row, slot_row);
        }
    }

    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches,
                                    world->events().events().size());
    }

    findings = CheckOrdering(world->events());
    mismatches = CompareEpCombineOutput(*world, test_case, expert_ids, 8);
    findings = MergeFindings(findings, CheckOutputMismatches(mismatches));
    const CheckerStatus status = FinalizeExecutorStatus(findings, mismatches);
    return MakeResultFromStatus(status, findings, mismatches, world->events().events().size());
}

}  // namespace checker
}  // namespace tilexr
