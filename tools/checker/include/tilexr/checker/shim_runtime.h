#ifndef TILEXR_CHECKER_SHIM_RUNTIME_H
#define TILEXR_CHECKER_SHIM_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "tilexr/checker/status.h"
#include "tilexr/checker/types.h"
#include "tilexr/checker/world.h"

namespace tilexr {
namespace checker {

struct SourceLocation {
    const char *file;
    int line;
};

#define TILEXR_CHECKER_HERE \
    ::tilexr::checker::SourceLocation{__FILE__, __LINE__}

class ShimRuntime {
public:
    explicit ShimRuntime(RankWorld *world);

    CheckerStatus Copy(int rank, int peer_rank, BufferRole dst_role, size_t dst_offset,
                       BufferRole src_role, size_t src_offset, size_t bytes,
                       SourceLocation loc, const std::string &detail);

    CheckerStatus RecordRead(int rank, int peer_rank, BufferRole role, size_t offset,
                             size_t bytes, SourceLocation loc,
                             const std::string &detail);

    CheckerStatus RecordWrite(int rank, int peer_rank, BufferRole role, size_t offset,
                              size_t bytes, SourceLocation loc,
                              const std::string &detail);

    CheckerStatus StoreFlag(int rank, int peer_rank, int slot, uint64_t magic,
                            SourceLocation loc, const std::string &detail);

    CheckerStatus WaitFlag(int rank, int peer_rank, int slot, uint64_t magic,
                           SourceLocation loc, const std::string &detail);

    CheckerStatus Barrier(int rank, int core, SourceLocation loc,
                          const std::string &detail);

private:
    RankWorld *world_;
};

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_SHIM_RUNTIME_H
