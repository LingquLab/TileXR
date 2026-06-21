#ifndef TILEXR_CHECKER_WORLD_H
#define TILEXR_CHECKER_WORLD_H

#include <cstddef>
#include <cstdint>
#include <vector>

#include "comm_args.h"
#include "tilexr/checker/event.h"
#include "tilexr/checker/memory.h"

namespace tilexr {
namespace checker {

class RankWorld {
public:
    static RankWorld Create(int rank_size, size_t user_input_bytes,
                            size_t user_output_bytes, size_t comm_data_bytes);

    int rank_size() const;
    int server_count() const;
    int ServerOfRank(int rank) const;
    void ConfigureServers(int server_count);
    uint64_t NextMagic();

    TileXR::CommArgs &HostArgs(int rank);
    const TileXR::CommArgs &HostArgs(int rank) const;

    ByteBuffer &UserInput(int rank);
    ByteBuffer &UserOutput(int rank);
    const ByteBuffer &UserOutput(int rank) const;
    ByteBuffer &CommData(int owner_rank, int slot);
    const ByteBuffer &CommData(int owner_rank, int slot) const;
    ByteBuffer &CommFlag(int owner_rank, int slot);
    EventLog &events();
    const EventLog &events() const;

private:
    int rank_size_ = 0;
    int server_count_ = 1;
    uint64_t next_magic_ = 1;
    std::vector<int> rank_servers_;
    std::vector<TileXR::CommArgs> args_;
    std::vector<ByteBuffer> user_inputs_;
    std::vector<ByteBuffer> user_outputs_;
    std::vector<std::vector<ByteBuffer> > comm_data_;
    std::vector<std::vector<ByteBuffer> > comm_flags_;
    EventLog events_;
};

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_WORLD_H
