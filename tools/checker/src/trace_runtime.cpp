#include "tilexr/checker/trace_runtime.h"

#include <cstring>
#include <sstream>

namespace tilexr {
namespace checker {

namespace {

thread_local TraceRuntime *g_current_trace_runtime = nullptr;

bool PtrInRange(const void *ptr, const TraceRuntime::Range &range) {
    const uintptr_t byte_ptr = reinterpret_cast<uintptr_t>(ptr);
    if (range.begin_addr == 0 || range.bytes == 0 || byte_ptr < range.begin_addr) {
        return false;
    }
    return byte_ptr - range.begin_addr < range.bytes;
}

bool RangeContainsStorage(const TraceRuntime::Range &range, size_t offset, size_t bytes) {
    if (offset > range.storage_bytes) {
        return false;
    }
    return bytes <= range.storage_bytes - offset;
}

Event MakeTraceDiagnostic(TraceRuntime *runtime, SourceLocation loc,
                          const std::string &detail, size_t bytes) {
    Event event;
    event.kind = EventKind::kDiagnostic;
    event.rank = runtime == nullptr ? -1 : runtime->rank();
    event.peer_rank = -1;
    event.server = -1;
    event.peer_server = -1;
    event.core = runtime == nullptr ? -1 : runtime->block_idx();
    event.buffer_role = BufferRole::kMetadata;
    event.bytes = bytes;
    event.source_file = loc.file == nullptr ? "" : loc.file;
    event.source_line = loc.line;
    event.detail = detail;
    return event;
}

CheckerStatus MaterializeAllReduceBigDataSumInt32(TraceRuntime *runtime) {
    if (runtime == nullptr) {
        return CheckerStatus::Fail("trace materializer runtime is null");
    }
    RankWorld *world = runtime->world();
    const CheckerCase &test_case = runtime->test_case();
    if (world == nullptr) {
        return CheckerStatus::Fail("trace materializer world is null");
    }
    const size_t count = static_cast<size_t>(test_case.count);
    std::vector<int32_t> reduced(count, 0);
    for (size_t i = 0; i < count; ++i) {
        int32_t sum = 0;
        for (int rank = 0; rank < test_case.rank_size; ++rank) {
            int32_t value = 0;
            if (world->UserInput(rank).ReadInt32(i, &value).ok()) {
                sum += value;
            }
        }
        reduced[i] = sum;
    }
    const size_t bytes = reduced.size() * sizeof(int32_t);
    for (int rank = 0; rank < test_case.rank_size; ++rank) {
        CheckerStatus status = world->UserOutput(rank).WriteBytes(0, reduced.data(), bytes);
        if (!status.ok()) {
            return status;
        }
    }
    return CheckerStatus::Ok();
}

CheckerStatus MaterializeAllGatherInt32(TraceRuntime *runtime) {
    if (runtime == nullptr) {
        return CheckerStatus::Fail("trace materializer runtime is null");
    }
    RankWorld *world = runtime->world();
    const CheckerCase &test_case = runtime->test_case();
    if (world == nullptr) {
        return CheckerStatus::Fail("trace materializer world is null");
    }
    const size_t per_rank_bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    for (int dst_rank = 0; dst_rank < test_case.rank_size; ++dst_rank) {
        for (int src_rank = 0; src_rank < test_case.rank_size; ++src_rank) {
            CheckerStatus status =
                world->UserOutput(dst_rank).WriteBytes(
                    static_cast<size_t>(src_rank) * per_rank_bytes,
                    world->UserInput(src_rank).data_ptr(), per_rank_bytes);
            if (!status.ok()) {
                return status;
            }
        }
    }
    return CheckerStatus::Ok();
}

const TraceMaterializer *Materializers() {
    static const TraceMaterializer materializers[] = {
        {"allreduce_big_data_sum_int32",
         CollectiveOp::kAllReduce,
         AlgorithmId::kAllReduceBigData,
         TileXR::TILEXR_DATA_TYPE_INT32,
         TileXR::TILEXR_REDUCE_SUM,
         0,
         MaterializeAllReduceBigDataSumInt32},
        {"allgather_int32",
         CollectiveOp::kAllGather,
         AlgorithmId::kDefault,
         TileXR::TILEXR_DATA_TYPE_INT32,
         TileXR::TILEXR_REDUCE_SUM,
         0,
         MaterializeAllGatherInt32},
        {"allgather_hierarchy_double_ring_int32",
         CollectiveOp::kAllGather,
         AlgorithmId::kAllGatherHierarchyDoubleRing,
         TileXR::TILEXR_DATA_TYPE_INT32,
         TileXR::TILEXR_REDUCE_SUM,
         0,
         MaterializeAllGatherInt32},
    };
    return materializers;
}

size_t MaterializerCount() {
    return 3;
}

size_t MaterializerOutputSpanBytes(const TraceMaterializer &materializer,
                                   const CheckerCase &test_case) {
    if (materializer.op == CollectiveOp::kAllGather) {
        return static_cast<size_t>(test_case.rank_size) *
               static_cast<size_t>(test_case.count) * sizeof(int32_t);
    }
    if (materializer.op == CollectiveOp::kAllReduce) {
        return static_cast<size_t>(test_case.count) * sizeof(int32_t);
    }
    return materializer.output_span_bytes;
}

bool HasAppliedMaterializer(const std::vector<std::string> &applied,
                            const std::string &name) {
    for (size_t i = 0; i < applied.size(); ++i) {
        if (applied[i] == name) {
            return true;
        }
    }
    return false;
}

bool EventCoversRange(const Event &event, int rank, BufferRole role,
                      size_t offset, size_t bytes) {
    if ((event.kind != EventKind::kWrite && event.kind != EventKind::kCopy) ||
        event.rank != rank || event.buffer_role != role) {
        return false;
    }
    if (offset < event.offset) {
        return false;
    }
    const size_t relative = offset - event.offset;
    return relative <= event.bytes && bytes <= event.bytes - relative;
}

bool EventIntersectsCoverageFrontier(const Event &event, int rank, BufferRole role,
                                     size_t covered_until, size_t *next_covered_until) {
    if (next_covered_until == nullptr ||
        (event.kind != EventKind::kWrite && event.kind != EventKind::kCopy) ||
        event.rank != rank || event.buffer_role != role) {
        return false;
    }
    if (event.offset > covered_until) {
        return false;
    }
    const size_t event_end = event.offset + event.bytes;
    if (event_end <= covered_until) {
        return false;
    }
    *next_covered_until = event_end;
    return true;
}

bool HasUserOutputWriteCoverage(const EventLog &events, int rank,
                                size_t offset, size_t bytes) {
    if (bytes == 0) {
        return true;
    }
    const size_t target_end = offset + bytes;
    size_t covered_until = offset;
    const std::vector<Event> &items = events.events();
    bool advanced = true;
    while (covered_until < target_end && advanced) {
        advanced = false;
        for (size_t i = 0; i < items.size(); ++i) {
            size_t next_covered_until = covered_until;
            if (EventIntersectsCoverageFrontier(items[i], rank, BufferRole::kUserOutput,
                                                covered_until, &next_covered_until)) {
                if (next_covered_until > target_end) {
                    next_covered_until = target_end;
                }
                covered_until = next_covered_until;
                advanced = true;
            }
            if (covered_until >= target_end) {
                break;
            }
        }
    }
    if (covered_until >= target_end) {
        return true;
    }
    for (size_t i = 0; i < items.size(); ++i) {
        if (EventCoversRange(items[i], rank, BufferRole::kUserOutput, offset, bytes)) {
            return true;
        }
    }
    return false;
}

CheckerStatus RequireMaterializerOutputWriteCoverage(
        const TraceMaterializer &materializer,
        TraceRuntime *runtime) {
    if (runtime == nullptr || runtime->world() == nullptr) {
        return CheckerStatus::Fail("trace materializer runtime is null");
    }
    const size_t span = MaterializerOutputSpanBytes(materializer, runtime->test_case());
    if (span == 0) {
        return CheckerStatus::Unsupported(
            "trace materializer has no user-output span: " +
            std::string(materializer.name == nullptr ? "" : materializer.name));
    }
    const EventLog &events = runtime->world()->events();
    for (int rank = 0; rank < runtime->test_case().rank_size; ++rank) {
        if (!HasUserOutputWriteCoverage(events, rank, 0, span)) {
            std::ostringstream message;
            message << "trace materializer requires production user-output write coverage"
                    << " for rank " << rank << " span 0+" << span
                    << " before applying oracle materializer "
                    << (materializer.name == nullptr ? "" : materializer.name);
            return CheckerStatus::Unsupported(message.str());
        }
    }
    return CheckerStatus::Ok();
}

}  // namespace

TraceRuntime::TraceRuntime(RankWorld *world, const CheckerCase &test_case)
    : world_(world), test_case_(&test_case) {}

TraceRuntime *TraceRuntime::Current() {
    return g_current_trace_runtime;
}

void TraceRuntime::SetCurrent(TraceRuntime *runtime) {
    g_current_trace_runtime = runtime;
}

void TraceRuntime::AddRange(uint8_t *begin, size_t bytes, BufferRole role, int owner_rank,
                            int slot) {
    AddRange(begin, bytes, bytes, role, owner_rank, slot);
}

void TraceRuntime::AddRange(uint8_t *begin, size_t virtual_bytes, size_t storage_bytes,
                            BufferRole role, int owner_rank, int slot) {
    AddVirtualRange(reinterpret_cast<uintptr_t>(begin), begin, virtual_bytes, storage_bytes,
                    role, owner_rank, slot);
}

void TraceRuntime::AddVirtualRange(uintptr_t begin_addr, uint8_t *storage_begin,
                                   size_t virtual_bytes, size_t storage_bytes,
                                   BufferRole role, int owner_rank, int slot) {
    Range range;
    range.begin = storage_begin;
    range.begin_addr = begin_addr;
    range.bytes = virtual_bytes;
    range.storage_bytes = storage_bytes;
    range.role = role;
    range.owner_rank = owner_rank;
    range.slot = slot;
    range.sparse_virtual = begin_addr != reinterpret_cast<uintptr_t>(storage_begin);
    ranges_.push_back(range);
}

void TraceRuntime::SetKernelContext(int rank, int block_idx, int block_num) {
    rank_ = rank;
    block_idx_ = block_idx;
    block_num_ = block_num;
}

int TraceRuntime::rank() const {
    return rank_;
}

int TraceRuntime::rank_size() const {
    return test_case_ == nullptr ? 0 : test_case_->rank_size;
}

int TraceRuntime::block_idx() const {
    return block_idx_;
}

int TraceRuntime::block_num() const {
    return block_num_;
}

const CheckerCase &TraceRuntime::test_case() const {
    return *test_case_;
}

RankWorld *TraceRuntime::world() {
    return world_;
}

const RankWorld *TraceRuntime::world() const {
    return world_;
}

const TraceRuntime::Range *TraceRuntime::Resolve(const void *ptr) const {
    for (size_t i = 0; i < ranges_.size(); ++i) {
        if (PtrInRange(ptr, ranges_[i])) {
            return &ranges_[i];
        }
    }
    return nullptr;
}

size_t TraceRuntime::OffsetOf(const Range &range, const void *ptr) const {
    return static_cast<size_t>(reinterpret_cast<uintptr_t>(ptr) - range.begin_addr);
}

CheckerStatus TraceRuntime::RecordCopy(void *dst, const void *src, size_t bytes, int op,
                                       SourceLocation loc, const std::string &detail) {
    if (world_ == nullptr) {
        return CheckerStatus::Fail("trace runtime has no rank world");
    }
    (void)op;

    const Range *src_range = Resolve(src);
    const Range *dst_range = Resolve(dst);
    if (src_range == nullptr || dst_range == nullptr) {
        Event diagnostic =
            MakeTraceDiagnostic(this, loc,
                                "unresolved trace address in " + detail +
                                    ": register user/comm/output ranges before trusting this trace",
                                bytes);
        diagnostic.server = world_->ServerOfRank(rank_);
        world_->events().Add(diagnostic);
        return CheckerStatus::Ok();
    }

    Event read;
    read.kind = EventKind::kRead;
    read.rank = rank_;
    read.peer_rank = src_range->owner_rank;
    read.server = world_->ServerOfRank(rank_);
    read.peer_server = world_->ServerOfRank(src_range->owner_rank);
    read.core = block_idx_;
    read.buffer_role = src_range->role;
    read.slot = src_range->slot;
    read.offset = OffsetOf(*src_range, src);
    read.bytes = bytes;
    read.allow_future_producer = src_range->role == BufferRole::kCommData;
    read.source_file = loc.file == nullptr ? "" : loc.file;
    read.source_line = loc.line;
    read.detail = detail + " read";
    world_->events().Add(read);

    Event write;
    write.kind = dst_range->role == BufferRole::kCommData ? EventKind::kCopy : EventKind::kWrite;
    write.rank = rank_;
    write.peer_rank = dst_range->owner_rank;
    write.server = world_->ServerOfRank(rank_);
    write.peer_server = world_->ServerOfRank(dst_range->owner_rank);
    write.core = block_idx_;
    write.buffer_role = dst_range->role;
    write.slot = dst_range->slot;
    write.offset = OffsetOf(*dst_range, dst);
    write.bytes = bytes;
    write.source_file = loc.file == nullptr ? "" : loc.file;
    write.source_line = loc.line;
    write.detail = detail + " write";
    world_->events().Add(write);

    const size_t src_offset = OffsetOf(*src_range, src);
    const size_t dst_offset = OffsetOf(*dst_range, dst);
    const bool src_storage_exceeded =
        !src_range->sparse_virtual && !RangeContainsStorage(*src_range, src_offset, bytes);
    const bool dst_storage_exceeded =
        !dst_range->sparse_virtual && !RangeContainsStorage(*dst_range, dst_offset, bytes);
    if (src_storage_exceeded || dst_storage_exceeded) {
        Event diagnostic =
            MakeTraceDiagnostic(this, loc,
                                "trace copy exceeds registered storage in " + detail +
                                    ": virtual range is larger than the checker backing buffer",
                                bytes);
        diagnostic.server = world_->ServerOfRank(rank_);
        world_->events().Add(diagnostic);
        return CheckerStatus::Ok();
    }

    if (!src_range->sparse_virtual && !dst_range->sparse_virtual &&
        src_range->role == BufferRole::kUserInput && dst_range->role == BufferRole::kCommData) {
        std::memcpy(dst, src, bytes);
    }

    return CheckerStatus::Ok();
}

CheckerStatus TraceRuntime::RecordUnsupportedDataCopy(SourceLocation loc,
                                                      const std::string &detail) {
    if (world_ == nullptr) {
        return CheckerStatus::Fail("trace runtime has no rank world");
    }
    Event diagnostic =
        MakeTraceDiagnostic(this, loc,
                            "unsupported trace data copy in " + detail +
                                ": add a precise checker model or explicit unsupported gate",
                            0);
    diagnostic.server = world_->ServerOfRank(rank_);
    world_->events().Add(diagnostic);
    return CheckerStatus::Ok();
}

CheckerStatus TraceRuntime::RecordPipeEvent(EventKind kind, int pipe, int event_id,
                                            SourceLocation loc, const std::string &detail) {
    if (world_ == nullptr) {
        return CheckerStatus::Fail("trace runtime has no rank world");
    }
    Event event;
    event.kind = kind;
    event.rank = rank_;
    event.peer_rank = -1;
    event.server = world_->ServerOfRank(rank_);
    event.peer_server = -1;
    event.core = block_idx_;
    event.pipe = pipe;
    event.event_id = event_id;
    event.buffer_role = BufferRole::kMetadata;
    event.source_file = loc.file == nullptr ? "" : loc.file;
    event.source_line = loc.line;
    event.detail = detail;
    world_->events().Add(event);
    return CheckerStatus::Ok();
}

CheckerStatus TraceRuntime::StoreFlag(int producer_rank, int consumer_rank, int slot,
                                      uint64_t magic, SourceLocation loc,
                                      const std::string &detail) {
    if (world_ == nullptr) {
        return CheckerStatus::Fail("trace runtime has no rank world");
    }
    Event event;
    event.kind = EventKind::kFlagStore;
    event.rank = producer_rank;
    event.peer_rank = consumer_rank;
    event.server = world_->ServerOfRank(producer_rank);
    event.peer_server = consumer_rank >= 0 ? world_->ServerOfRank(consumer_rank) : -1;
    event.buffer_role = BufferRole::kCommFlag;
    event.slot = slot;
    event.magic = magic;
    event.offset = 0;
    event.bytes = sizeof(magic);
    event.source_file = loc.file == nullptr ? "" : loc.file;
    event.source_line = loc.line;
    event.detail = detail;
    world_->events().Add(event);
    return CheckerStatus::Ok();
}

CheckerStatus TraceRuntime::WaitFlag(int consumer_rank, int producer_rank, int slot,
                                     uint64_t magic, SourceLocation loc,
                                     const std::string &detail) {
    if (world_ == nullptr) {
        return CheckerStatus::Fail("trace runtime has no rank world");
    }
    Event event;
    event.kind = EventKind::kFlagWait;
    event.rank = consumer_rank;
    event.peer_rank = producer_rank;
    event.server = world_->ServerOfRank(consumer_rank);
    event.peer_server = world_->ServerOfRank(producer_rank);
    event.buffer_role = BufferRole::kCommFlag;
    event.slot = slot;
    event.magic = magic;
    event.offset = 0;
    event.bytes = sizeof(magic);
    event.source_file = loc.file == nullptr ? "" : loc.file;
    event.source_line = loc.line;
    event.detail = detail;
    world_->events().Add(event);
    return CheckerStatus::Ok();
}

const TraceMaterializer *TraceRuntime::FindMaterializer(const std::string &name) const {
    const TraceMaterializer *materializers = Materializers();
    for (size_t i = 0; i < MaterializerCount(); ++i) {
        if (materializers[i].name != nullptr && name == materializers[i].name) {
            materializer_view_ = materializers[i];
            if (test_case_ != nullptr) {
                materializer_view_.output_span_bytes =
                    MaterializerOutputSpanBytes(materializer_view_, *test_case_);
            }
            return &materializer_view_;
        }
    }
    return nullptr;
}

CheckerStatus TraceRuntime::ApplyMaterializer(const std::string &name) {
    const TraceMaterializer *materializer = FindMaterializer(name);
    if (materializer == nullptr) {
        return CheckerStatus::Unsupported("trace materializer is not registered: " + name);
    }
    if (HasAppliedMaterializer(applied_materializers_, name)) {
        return CheckerStatus::Ok();
    }
    if (materializer->apply == nullptr) {
        return CheckerStatus::Unsupported("trace materializer has no apply function: " + name);
    }
    CheckerStatus coverage_status =
        RequireMaterializerOutputWriteCoverage(*materializer, this);
    if (!coverage_status.ok()) {
        return coverage_status;
    }
    CheckerStatus status = materializer->apply(this);
    if (!status.ok()) {
        return status;
    }
    applied_materializers_.push_back(name);
    return CheckerStatus::Ok();
}

void TraceRuntime::MaterializeAllReduceBigData() {
    (void)ApplyMaterializer("allreduce_big_data_sum_int32");
}

void TraceRuntime::MaterializeAllGather() {
    if (test_case_ == nullptr) {
        return;
    }
    if (test_case_->algorithm == AlgorithmId::kAllGatherHierarchyDoubleRing) {
        (void)ApplyMaterializer("allgather_hierarchy_double_ring_int32");
    } else {
        (void)ApplyMaterializer("allgather_int32");
    }
}

}  // namespace checker
}  // namespace tilexr
