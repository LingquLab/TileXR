#include "tilexr/checker/shim_runtime.h"

#include <vector>

#include "tilexr/checker/event.h"
#include "tilexr/checker/memory.h"

namespace tilexr {
namespace checker {

namespace {

struct BufferView {
    ByteBuffer *buffer = nullptr;
    int slot = -1;
};

CheckerStatus RequireWorld(RankWorld *world) {
    if (world == nullptr) {
        return CheckerStatus::Fail("shim runtime requires a non-null rank world");
    }
    return CheckerStatus::Ok();
}

CheckerStatus ResolveRoleBuffer(RankWorld *world, int rank, int peer_rank, BufferRole role,
                                BufferView *view) {
    if (view == nullptr) {
        return CheckerStatus::Fail("buffer view destination is null");
    }

    switch (role) {
        case BufferRole::kUserInput:
            view->buffer = &world->UserInput(rank);
            view->slot = rank;
            return CheckerStatus::Ok();
        case BufferRole::kUserOutput:
            view->buffer = &world->UserOutput(rank);
            view->slot = rank;
            return CheckerStatus::Ok();
        case BufferRole::kCommData:
            // Copy-side peer-visible communication buffers use owner=peer_rank and slot=rank.
            view->buffer = &world->CommData(peer_rank, rank);
            view->slot = rank;
            return CheckerStatus::Ok();
        case BufferRole::kCommFlag:
            // Copy-side peer-visible flag buffers follow the same owner/slot rule as comm data.
            view->buffer = &world->CommFlag(peer_rank, rank);
            view->slot = rank;
            return CheckerStatus::Ok();
        case BufferRole::kLocalUb:
        case BufferRole::kMetadata:
        case BufferRole::kRegisteredCommBuffer:
            return CheckerStatus::Unsupported(std::string("unsupported checker buffer role: ") +
                                              ToString(role));
    }

    return CheckerStatus::Unsupported("unsupported checker buffer role");
}

Event MakeEvent(EventKind kind, int rank, int peer_rank, BufferRole role, int slot,
                uint64_t magic, size_t offset, size_t bytes, SourceLocation loc,
                const std::string &detail) {
    Event event;
    event.kind = kind;
    event.rank = rank;
    event.peer_rank = peer_rank;
    event.buffer_role = role;
    event.slot = slot;
    event.magic = magic;
    event.offset = offset;
    event.bytes = bytes;
    event.source_file = loc.file == nullptr ? "" : loc.file;
    event.source_line = loc.line;
    event.detail = detail;
    return event;
}

}  // namespace

ShimRuntime::ShimRuntime(RankWorld *world) : world_(world) {}

CheckerStatus ShimRuntime::Copy(int rank, int peer_rank, BufferRole dst_role, size_t dst_offset,
                                BufferRole src_role, size_t src_offset, size_t bytes,
                                SourceLocation loc, const std::string &detail) {
    CheckerStatus world_status = RequireWorld(world_);
    if (!world_status.ok()) {
        return world_status;
    }

    BufferView src;
    CheckerStatus src_status = ResolveRoleBuffer(world_, rank, peer_rank, src_role, &src);
    if (!src_status.ok()) {
        return src_status;
    }

    BufferView dst;
    CheckerStatus dst_status = ResolveRoleBuffer(world_, rank, peer_rank, dst_role, &dst);
    if (!dst_status.ok()) {
        return dst_status;
    }

    std::vector<uint8_t> temp(bytes, 0);
    CheckerStatus read_status = src.buffer->ReadBytes(src_offset, temp.data(), bytes);
    if (!read_status.ok()) {
        return read_status;
    }
    CheckerStatus write_status = dst.buffer->WriteBytes(dst_offset, temp.data(), bytes);
    if (!write_status.ok()) {
        return write_status;
    }

    world_->events().Add(MakeEvent(EventKind::kCopy, rank, peer_rank, dst_role, dst.slot, 0,
                                   dst_offset, bytes, loc, detail));
    return CheckerStatus::Ok();
}

CheckerStatus ShimRuntime::RecordRead(int rank, int peer_rank, BufferRole role, size_t offset,
                                      size_t bytes, SourceLocation loc,
                                      const std::string &detail) {
    CheckerStatus world_status = RequireWorld(world_);
    if (!world_status.ok()) {
        return world_status;
    }

    BufferView view;
    CheckerStatus status = ResolveRoleBuffer(world_, rank, peer_rank, role, &view);
    if (!status.ok()) {
        return status;
    }

    world_->events().Add(MakeEvent(EventKind::kRead, rank, peer_rank, role, view.slot, 0,
                                   offset, bytes, loc, detail));
    return CheckerStatus::Ok();
}

CheckerStatus ShimRuntime::RecordWrite(int rank, int peer_rank, BufferRole role, size_t offset,
                                       size_t bytes, SourceLocation loc,
                                       const std::string &detail) {
    CheckerStatus world_status = RequireWorld(world_);
    if (!world_status.ok()) {
        return world_status;
    }

    BufferView view;
    CheckerStatus status = ResolveRoleBuffer(world_, rank, peer_rank, role, &view);
    if (!status.ok()) {
        return status;
    }

    world_->events().Add(MakeEvent(EventKind::kWrite, rank, peer_rank, role, view.slot, 0,
                                   offset, bytes, loc, detail));
    return CheckerStatus::Ok();
}

CheckerStatus ShimRuntime::StoreFlag(int rank, int peer_rank, int slot, uint64_t magic,
                                     SourceLocation loc, const std::string &detail) {
    CheckerStatus world_status = RequireWorld(world_);
    if (!world_status.ok()) {
        return world_status;
    }

    CheckerStatus write_status =
        world_->CommFlag(rank, peer_rank).WriteBytes(0, &magic, sizeof(magic));
    if (!write_status.ok()) {
        return write_status;
    }

    world_->events().Add(MakeEvent(EventKind::kFlagStore, rank, peer_rank,
                                   BufferRole::kCommFlag, slot, magic, 0, sizeof(magic),
                                   loc, detail));
    return CheckerStatus::Ok();
}

CheckerStatus ShimRuntime::WaitFlag(int rank, int peer_rank, int slot, uint64_t magic,
                                    SourceLocation loc, const std::string &detail) {
    CheckerStatus world_status = RequireWorld(world_);
    if (!world_status.ok()) {
        return world_status;
    }

    world_->events().Add(MakeEvent(EventKind::kFlagWait, rank, peer_rank,
                                   BufferRole::kCommFlag, slot, magic, 0, sizeof(magic),
                                   loc, detail));
    return CheckerStatus::Ok();
}

CheckerStatus ShimRuntime::Barrier(int rank, int core, SourceLocation loc,
                                   const std::string &detail) {
    CheckerStatus world_status = RequireWorld(world_);
    if (!world_status.ok()) {
        return world_status;
    }

    Event event = MakeEvent(EventKind::kBarrier, rank, -1, BufferRole::kMetadata, -1, 0, 0, 0,
                            loc, detail);
    event.core = core;
    world_->events().Add(event);
    return CheckerStatus::Ok();
}

}  // namespace checker
}  // namespace tilexr
