#include "tilexr/checker/executor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tilexr/checker/event.h"
#include "tilexr/checker/shim_runtime.h"

namespace tilexr {
namespace checker {

namespace {

CheckerStatus RequireWorldAndSize(RankWorld *world, const CheckerCase &test_case) {
    if (world == nullptr) {
        return CheckerStatus::Fail("rank world is null");
    }
    if (world->rank_size() != test_case.rank_size) {
        return CheckerStatus::Fail("rank world size does not match checker case");
    }
    return CheckerStatus::Ok();
}

void AddUnsupportedFinding(const CheckerStatus &status, FindingSet *findings) {
    Finding finding;
    finding.id = "UNSUPPORTED_API:executor";
    finding.kind = FindingKind::kUnsupportedApi;
    finding.severity = Severity::kWarning;
    finding.message = status.message;
    finding.next_action =
        "Adjust the checker case to a supported collective/data type combination.";
    finding.buffer_role = BufferRole::kMetadata;
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

CheckerStatus AddCopyEvent(RankWorld *world, int rank, size_t bytes, const std::string &detail) {
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

CheckerStatus ExecuteAllGatherReadsAndWrites(RankWorld *world, const CheckerCase &test_case,
                                             int rank) {
    ShimRuntime runtime(world);
    const size_t bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    std::vector<int32_t> scratch(static_cast<size_t>(test_case.count), 0);
    for (int peer = 0; peer < test_case.rank_size; ++peer) {
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
            world->UserOutput(rank).WriteBytes(static_cast<size_t>(peer) * bytes, scratch.data(),
                                               bytes);
        if (!write_status.ok()) {
            return write_status;
        }
        CheckerStatus write_event =
            runtime.RecordWrite(rank, rank, BufferRole::kUserOutput,
                                static_cast<size_t>(peer) * bytes, bytes, TILEXR_CHECKER_HERE,
                                "write gathered user output");
        if (!write_event.ok()) {
            return write_event;
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

template <typename PhaseFn>
CheckerStatus RunPhases(const CheckerCase &test_case, PhaseFn phase_fn) {
    const int phase_count = 4;
    for (int phase = 0; phase < phase_count; ++phase) {
        for (int rank = 0; rank < test_case.rank_size; ++rank) {
            CheckerStatus status = phase_fn(rank, phase);
            if (!status.ok()) {
                return status;
            }
        }
    }
    return CheckerStatus::Ok();
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
            AddUnsupportedFinding(case_status, &findings);
        }
        return MakeResultFromStatus(case_status, findings, no_mismatches, 0);
    }

    world->events().Clear();

    CheckerStatus fill_status = FillRankIndexInt32Inputs(world, test_case);
    if (!fill_status.ok()) {
        return MakeResultFromStatus(fill_status, findings, no_mismatches,
                                    world->events().events().size());
    }

    RunResult result =
        test_case.op == CollectiveOp::kAllGather ? RunAllGatherInt32(world, test_case)
                                                 : RunAllReduceSumInt32(world, test_case);

    if (!result.status.ok() && result.status.code == CheckerStatusCode::kUnsupported &&
        result.findings.findings().empty()) {
        AddUnsupportedFinding(result.status, &result.findings);
    }

    const std::vector<OutputMismatch> mismatches =
        CompareInt32Output(*world, test_case, 8);
    const FindingSet output_findings = CheckOutputMismatches(mismatches);
    result.mismatches = mismatches;
    result.findings = MergeFindings(result.findings, output_findings);
    result.event_count = world->events().events().size();

    if (result.findings.HasErrors() || !result.mismatches.empty()) {
        result.status = CheckerStatus::Fail("checker executor detected findings or mismatches");
        if (HasFindingKind(result.findings, FindingKind::kUnsupportedApi)) {
            result.status = CheckerStatus::Unsupported("checker executor encountered unsupported case");
        }
        return result;
    }

    result.status = CheckerStatus::Ok();
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
    CheckerStatus exec_status = RunPhases(
        test_case,
        [&](int rank, int phase) -> CheckerStatus {
            switch (phase) {
                case 0:
                    return AddCopyEvent(world, rank, bytes, "publish local input to comm-data");
                case 1:
                    return AddStoreEvents(world, test_case, rank);
                case 2:
                    return AddWaitEvents(world, test_case, rank);
                case 3:
                    return ExecuteAllGatherReadsAndWrites(world, test_case, rank);
            }
            return CheckerStatus::Fail("invalid executor phase");
        });
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches, world->events().events().size());
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
    CheckerStatus exec_status = RunPhases(
        test_case,
        [&](int rank, int phase) -> CheckerStatus {
            switch (phase) {
                case 0:
                    return AddCopyEvent(world, rank, bytes, "publish local input to comm-data");
                case 1:
                    return AddStoreEvents(world, test_case, rank);
                case 2:
                    return AddWaitEvents(world, test_case, rank);
                case 3:
                    return ExecuteAllReduceReadAndWrite(world, test_case, rank);
            }
            return CheckerStatus::Fail("invalid executor phase");
        });
    if (!exec_status.ok()) {
        return MakeResultFromStatus(exec_status, findings, mismatches, world->events().events().size());
    }

    findings = CheckOrdering(world->events());
    return MakeResultFromStatus(CheckerStatus::Ok(), findings, mismatches,
                                world->events().events().size());
}

}  // namespace checker
}  // namespace tilexr
