/*
 * Copyright (c) 2026 TileXR Project
 *
 * External-user compile probe for the public direct CCU API.
 *
 * This file intentionally includes only tilexr_api.h from TileXR. It is
 * compiled to an object file, not linked, so it verifies public declarations
 * without requiring Ascend runtime or NPU hardware.
 */

#include "tilexr_api.h"

#ifndef TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES
#error "TILEXR_DIRECT_CCU_REPORT_MESSAGE_BYTES must be public"
#endif

int TileXRPublicDirectCcuApiCompileProbe(void)
{
    TileXRCommPtr comm = 0;
    TileXRDirectCcuPreparedTasksPtr prepared = 0;
    TileXRDirectCcuPrepareOptions options = {0};
    TileXRDirectCcuPrepareReport prepareReport = {0};
    TileXRDirectCcuSubmitReport submitReport = {0};
    TileXRDirectCcuTaskInfo task = {0};

    int (*prepareFn)(
        TileXRCommPtr,
        const TileXRDirectCcuPrepareOptions *,
        TileXRDirectCcuPreparedTasksPtr *,
        TileXRDirectCcuPrepareReport *) = &TileXRCommPrepareDirectCcu;
    int (*getTaskFn)(
        TileXRDirectCcuPreparedTasksPtr,
        uint32_t,
        TileXRDirectCcuTaskInfo *) = &TileXRDirectCcuGetPreparedTask;
    int (*submitFn)(
        TileXRDirectCcuPreparedTasksPtr,
        void *,
        TileXRDirectCcuSubmitReport *) = &TileXRDirectCcuSubmitPrepared;
    int (*destroyFn)(TileXRDirectCcuPreparedTasksPtr) = &TileXRDirectCcuDestroyPrepared;

    options.syncResourceCount = 1;
    options.syncInstructionCount = 2;
    options.bindingsPerSyncResource = 1;
    options.gsaStartId = 510;
    options.provider = "tilexr-public-direct-ccu-compile-probe";

    prepareReport.message[0] = '\0';
    submitReport.message[0] = '\0';
    task.args[0] = 0;

    (void)comm;
    (void)prepared;
    (void)prepareFn;
    (void)getTaskFn;
    (void)submitFn;
    (void)destroyFn;
    return (int)(options.syncResourceCount + options.gsaStartId + prepareReport.taskCount +
        submitReport.taskCount + task.argSize);
}
