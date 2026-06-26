#ifndef TILEXR_CHECKER_TRACE_RUNTIME_H
#define TILEXR_CHECKER_TRACE_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tilexr/checker/case.h"
#include "tilexr/checker/shim_runtime.h"
#include "tilexr/checker/world.h"

namespace tilexr {
namespace checker {

class TraceRuntime;

typedef CheckerStatus (*TraceMaterializerFn)(TraceRuntime *runtime);

struct TraceMaterializer {
    const char *name = nullptr;
    CollectiveOp op = CollectiveOp::kAllGather;
    AlgorithmId algorithm = AlgorithmId::kDefault;
    TileXR::TileXRDataType data_type = TileXR::TILEXR_DATA_TYPE_RESERVED;
    TileXR::TileXRReduceOp reduce_op = TileXR::TILEXR_REDUCE_SUM;
    size_t output_span_bytes = 0;
    TraceMaterializerFn apply = nullptr;
};

class TraceRuntime {
public:
    struct Range {
        uint8_t *begin = nullptr;
        uintptr_t begin_addr = 0;
        size_t bytes = 0;
        size_t storage_bytes = 0;
        BufferRole role = BufferRole::kMetadata;
        int owner_rank = -1;
        int slot = -1;
        bool sparse_virtual = false;
    };

    TraceRuntime(RankWorld *world, const CheckerCase &test_case);

    static TraceRuntime *Current();
    static void SetCurrent(TraceRuntime *runtime);

    void AddRange(uint8_t *begin, size_t bytes, BufferRole role, int owner_rank, int slot);
    void AddRange(uint8_t *begin, size_t virtual_bytes, size_t storage_bytes, BufferRole role,
                  int owner_rank, int slot);
    void AddVirtualRange(uintptr_t begin_addr, uint8_t *storage_begin, size_t virtual_bytes,
                         size_t storage_bytes, BufferRole role, int owner_rank, int slot);
    void SetKernelContext(int rank, int block_idx, int block_num);

    int rank() const;
    int rank_size() const;
    int block_idx() const;
    int block_num() const;
    const CheckerCase &test_case() const;
    RankWorld *world();
    const RankWorld *world() const;

    CheckerStatus RecordCopy(void *dst, const void *src, size_t bytes, int op,
                             SourceLocation loc, const std::string &detail);
    CheckerStatus RecordUnsupportedDataCopy(SourceLocation loc,
                                            const std::string &detail);
    CheckerStatus RecordPipeEvent(EventKind kind, int pipe, int event_id,
                                  SourceLocation loc, const std::string &detail);
    CheckerStatus StoreFlag(int producer_rank, int consumer_rank, int slot, uint64_t magic,
                            SourceLocation loc, const std::string &detail);
    CheckerStatus WaitFlag(int consumer_rank, int producer_rank, int slot, uint64_t magic,
                           SourceLocation loc, const std::string &detail);
    const TraceMaterializer *FindMaterializer(const std::string &name) const;
    CheckerStatus ApplyMaterializer(const std::string &name);
    void MaterializeAllReduceBigData();
    void MaterializeAllGather();

private:
    const Range *Resolve(const void *ptr) const;
    size_t OffsetOf(const Range &range, const void *ptr) const;

    RankWorld *world_ = nullptr;
    const CheckerCase *test_case_ = nullptr;
    int rank_ = -1;
    int block_idx_ = 0;
    int block_num_ = 1;
    mutable TraceMaterializer materializer_view_;
    std::vector<std::string> applied_materializers_;
    std::vector<Range> ranges_;
};

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_TRACE_RUNTIME_H
