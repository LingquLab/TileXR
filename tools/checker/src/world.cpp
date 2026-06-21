#include "tilexr/checker/world.h"

#include <cstring>

namespace tilexr {
namespace checker {

namespace {

ByteBuffer MakeFlagBuffer() {
    return ByteBuffer(sizeof(uint64_t));
}

void InitializeRankArgs(std::vector<TileXR::CommArgs> *args,
                        std::vector<std::vector<ByteBuffer> > *comm_data,
                        int server_count) {
    const int rank_size = static_cast<int>(args->size());
    const int local_rank_size =
        server_count <= 1 ? rank_size : rank_size / server_count;
    for (int rank = 0; rank < rank_size; ++rank) {
        TileXR::CommArgs &arg = (*args)[rank];
        arg.rank = rank;
        arg.localRank = server_count <= 1 ? rank : rank % local_rank_size;
        arg.rankSize = rank_size;
        arg.localRankSize = local_rank_size;
        arg.extraFlag = 0;
        arg.udmaInfoPtr = nullptr;
        arg.udmaRegistryPtr = nullptr;
        arg.sdmaWorkspacePtr = nullptr;
        std::memset(arg.sendCountMatrix, 0, sizeof(arg.sendCountMatrix));
        std::memset(arg.magics, 0, sizeof(arg.magics));
        for (int peer = 0; peer < rank_size; ++peer) {
            arg.peerMems[peer] = (*comm_data)[peer][rank].mutable_data().data();
        }
    }
}

}  // namespace

RankWorld RankWorld::Create(int rank_size, size_t user_input_bytes,
                            size_t user_output_bytes, size_t comm_data_bytes) {
    RankWorld world;
    world.rank_size_ = rank_size;
    world.server_count_ = 1;
    world.rank_servers_.resize(static_cast<size_t>(rank_size), 0);
    world.args_.resize(static_cast<size_t>(rank_size));
    world.user_inputs_.reserve(static_cast<size_t>(rank_size));
    world.user_outputs_.reserve(static_cast<size_t>(rank_size));
    world.comm_data_.resize(static_cast<size_t>(rank_size));
    world.comm_flags_.resize(static_cast<size_t>(rank_size));

    for (int rank = 0; rank < rank_size; ++rank) {
        world.user_inputs_.push_back(ByteBuffer(user_input_bytes));
        world.user_outputs_.push_back(ByteBuffer(user_output_bytes));

        std::vector<ByteBuffer> data_slots;
        std::vector<ByteBuffer> flag_slots;
        data_slots.reserve(static_cast<size_t>(rank_size));
        flag_slots.reserve(static_cast<size_t>(rank_size));
        for (int slot = 0; slot < rank_size; ++slot) {
            data_slots.push_back(ByteBuffer(comm_data_bytes));
            flag_slots.push_back(MakeFlagBuffer());
        }
        world.comm_data_[rank] = data_slots;
        world.comm_flags_[rank] = flag_slots;
    }

    InitializeRankArgs(&world.args_, &world.comm_data_, world.server_count_);
    return world;
}

int RankWorld::rank_size() const {
    return rank_size_;
}

int RankWorld::server_count() const {
    return server_count_;
}

int RankWorld::ServerOfRank(int rank) const {
    if (rank < 0 || rank >= rank_size_) {
        return -1;
    }
    return rank_servers_.at(static_cast<size_t>(rank));
}

void RankWorld::ConfigureServers(int server_count) {
    server_count_ = server_count <= 0 ? 1 : server_count;
    rank_servers_.resize(static_cast<size_t>(rank_size_), 0);
    const int local_rank_size =
        server_count_ <= 1 ? rank_size_ : rank_size_ / server_count_;
    for (int rank = 0; rank < rank_size_; ++rank) {
        int server = local_rank_size == 0 ? 0 : rank / local_rank_size;
        if (server >= server_count_) {
            server = server_count_ - 1;
        }
        rank_servers_[static_cast<size_t>(rank)] = server;
    }
    InitializeRankArgs(&args_, &comm_data_, server_count_);
}

uint64_t RankWorld::NextMagic() {
    return next_magic_++;
}

TileXR::CommArgs &RankWorld::HostArgs(int rank) {
    return args_.at(static_cast<size_t>(rank));
}

const TileXR::CommArgs &RankWorld::HostArgs(int rank) const {
    return args_.at(static_cast<size_t>(rank));
}

ByteBuffer &RankWorld::UserInput(int rank) {
    return user_inputs_.at(static_cast<size_t>(rank));
}

ByteBuffer &RankWorld::UserOutput(int rank) {
    return user_outputs_.at(static_cast<size_t>(rank));
}

const ByteBuffer &RankWorld::UserOutput(int rank) const {
    return user_outputs_.at(static_cast<size_t>(rank));
}

ByteBuffer &RankWorld::CommData(int owner_rank, int slot) {
    return comm_data_.at(static_cast<size_t>(owner_rank)).at(static_cast<size_t>(slot));
}

const ByteBuffer &RankWorld::CommData(int owner_rank, int slot) const {
    return comm_data_.at(static_cast<size_t>(owner_rank)).at(static_cast<size_t>(slot));
}

ByteBuffer &RankWorld::CommFlag(int owner_rank, int slot) {
    return comm_flags_.at(static_cast<size_t>(owner_rank)).at(static_cast<size_t>(slot));
}

EventLog &RankWorld::events() {
    return events_;
}

const EventLog &RankWorld::events() const {
    return events_;
}

}  // namespace checker
}  // namespace tilexr
