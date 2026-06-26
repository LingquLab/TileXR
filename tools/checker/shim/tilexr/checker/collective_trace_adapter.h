#include "tilexr/checker/collective_trace_shim.h"

#ifndef TILEXR_CHECKER_TRACE_TARGET_HEADER
#error "TILEXR_CHECKER_TRACE_TARGET_HEADER must name the production header to trace"
#endif

#define CpGM2GMPingPong(dataSizeRemain, inputGT, outputGT, op, ...) \
    CpGM2GMPingPongTrace((dataSizeRemain), (inputGT), (outputGT), (op), \
                         tilexr::checker::TraceSourceLoc(__LINE__))
#define SetSyncFlag(magic, value, eventID, rank, ...) \
    SetSyncFlagTrace((magic), (value), (eventID), (rank), \
                     tilexr::checker::TraceSourceLoc(__LINE__))
#define WaitSyncFlag(magic, value, eventID, rank, ...) \
    WaitSyncFlagTrace((magic), (value), (eventID), (rank), \
                      tilexr::checker::TraceSourceLoc(__LINE__))

#include TILEXR_CHECKER_TRACE_TARGET_HEADER

#undef CpGM2GMPingPong
#undef SetSyncFlag
#undef WaitSyncFlag
