#ifndef TILEXR_CHECKER_TYPES_H
#define TILEXR_CHECKER_TYPES_H

#include <string>

namespace tilexr {
namespace checker {

enum class CollectiveOp { kAllGather, kAllReduce };
enum class SchedulerMode { kSerial, kRoundRobin };
enum class BufferRole {
    kUserInput,
    kUserOutput,
    kCommFlag,
    kCommData,
    kLocalUb,
    kMetadata,
    kRegisteredCommBuffer
};

inline const char *ToString(CollectiveOp op) {
    switch (op) {
        case CollectiveOp::kAllGather:
            return "allgather";
        case CollectiveOp::kAllReduce:
            return "allreduce";
    }
    return "unknown";
}

inline const char *ToString(SchedulerMode mode) {
    switch (mode) {
        case SchedulerMode::kSerial:
            return "serial";
        case SchedulerMode::kRoundRobin:
            return "round_robin";
    }
    return "unknown";
}

inline const char *ToString(BufferRole role) {
    switch (role) {
        case BufferRole::kUserInput:
            return "user_input";
        case BufferRole::kUserOutput:
            return "user_output";
        case BufferRole::kCommFlag:
            return "comm_flag";
        case BufferRole::kCommData:
            return "comm_data";
        case BufferRole::kLocalUb:
            return "local_ub";
        case BufferRole::kMetadata:
            return "metadata";
        case BufferRole::kRegisteredCommBuffer:
            return "registered_comm_buffer";
    }
    return "unknown";
}

}  // namespace checker
}  // namespace tilexr

#endif  // TILEXR_CHECKER_TYPES_H
