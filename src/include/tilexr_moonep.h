#ifndef TILEXR_MOONEP_H
#define TILEXR_MOONEP_H

#include <stddef.h>
#include <stdint.h>

#include "acl/acl_base.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *TileXRCommPtr;

#define TILEXR_MOONEP_ABI_VERSION_V1 UINT32_C(1)
#define TILEXR_MOONEP_ABI_VERSION_V2 UINT32_C(2)
#define TILEXR_MOONEP_MAX_TENSOR_RANK UINT32_C(4)
#define TILEXR_MOONEP_FLAG_NONE UINT64_C(0)
#define TILEXR_MOONEP_REDUCE_GRAD_UDMA_THRESHOLD_BYTES UINT64_C(1048576)
#define TILEXR_MOONEP_FLAG_BUILD_DEDUP (UINT64_C(1) << 0)
#define TILEXR_MOONEP_FLAG_SKIP_INTER_RANK_SYNC (UINT64_C(1) << 1)
#define TILEXR_MOONEP_FLAG_ZERO_COPY (UINT64_C(1) << 2)
#define TILEXR_MOONEP_FLAG_COMBINE_PUBLISH_ONLY (UINT64_C(1) << 3)
#define TILEXR_MOONEP_FLAG_COMBINE_CONSUME_ONLY (UINT64_C(1) << 4)

typedef enum TileXRMoonEpStatus {
    TILEXR_MOONEP_SUCCESS = 0,
    TILEXR_MOONEP_ERROR_INVALID_ARGUMENT = -3,
    TILEXR_MOONEP_ERROR_INTERNAL = -4,
    TILEXR_MOONEP_ERROR_NOT_SUPPORTED = -6
} TileXRMoonEpStatus;

typedef enum TileXRMoonEpDType {
    TILEXR_MOONEP_DTYPE_INT32 = 2,
    TILEXR_MOONEP_DTYPE_FLOAT16 = 3,
    TILEXR_MOONEP_DTYPE_FLOAT32 = 4,
    TILEXR_MOONEP_DTYPE_BFLOAT16 = 11
} TileXRMoonEpDType;

typedef enum TileXRMoonEpStage {
    TILEXR_MOONEP_STAGE_PLANNING = 1u << 0,
    TILEXR_MOONEP_STAGE_DISPATCH = 1u << 1,
    TILEXR_MOONEP_STAGE_PREFETCH_WEIGHT = 1u << 2,
    TILEXR_MOONEP_STAGE_COMBINE = 1u << 3,
    TILEXR_MOONEP_STAGE_REDUCE_GRAD = 1u << 4
} TileXRMoonEpStage;

typedef enum TileXRMoonEpReduceGradTransport {
    TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_NONE = 0,
    TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER = 1,
    TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_UDMA = 2
} TileXRMoonEpReduceGradTransport;

typedef enum TileXRMoonEpReduceGradDeviceStatus {
    TILEXR_MOONEP_REDUCE_GRAD_DEVICE_SUCCESS = 0,
    TILEXR_MOONEP_REDUCE_GRAD_DEVICE_INVALID_STATE = 1,
    TILEXR_MOONEP_REDUCE_GRAD_DEVICE_PEER_TIMEOUT = 2,
    TILEXR_MOONEP_REDUCE_GRAD_DEVICE_UDMA_CQ_ERROR = 3
} TileXRMoonEpReduceGradDeviceStatus;

typedef struct TileXRMoonEpTensorV1 {
    uint32_t structSize;
    uint32_t abiVersion;
    void *data;
    uint64_t elementCount;
    uint32_t dtype;
    uint32_t rank;
    int64_t shape[TILEXR_MOONEP_MAX_TENSOR_RANK];
} TileXRMoonEpTensorV1;

typedef struct TileXRMoonEpPlanV1 {
    uint32_t structSize;
    uint32_t abiVersion;
    int64_t n;
    int64_t r;
    int64_t e;
    int64_t b;
    int64_t nvS;
    int64_t k;
    void *dst;
    void *expertsToCopy;
    void *zeroFillRanges;
    void *remoteStats;
    void *dupGroups;
    void *dupLoffs;
    void *dupCounts;
    void *status;
} TileXRMoonEpPlanV1;

typedef struct TileXRMoonEpPlanningArgsV1 {
    uint32_t structSize;
    uint32_t abiVersion;
    TileXRCommPtr comm;
    const TileXRMoonEpTensorV1 *topkExperts;
    const TileXRMoonEpTensorV1 *tokensPerExpert;
    void *workspace;
    uint64_t workspaceBytes;
    TileXRMoonEpTensorV1 *cuSeqlens;
    TileXRMoonEpPlanV1 *plan;
    uint64_t waitIterations;
    uint64_t flags;
} TileXRMoonEpPlanningArgsV1;

typedef struct TileXRMoonEpDispatchArgsV1 {
    uint32_t structSize;
    uint32_t abiVersion;
    TileXRCommPtr comm;
    const TileXRMoonEpPlanV1 *plan;
    const TileXRMoonEpTensorV1 *hiddenSh;
    const TileXRMoonEpTensorV1 *routeWeightsSk;
    TileXRMoonEpTensorV1 *hiddenNvsh;
    TileXRMoonEpTensorV1 *routeWeightsNvs;
    uint64_t flags;
    void *registeredWorkspace;
    uint64_t registeredWorkspaceBytes;
} TileXRMoonEpDispatchArgsV1;

/*
 * V2 preserves the tensor and plan descriptors while requiring the registered
 * workspace path. It never falls back to the legacy peer-memory kernel.
 */
typedef TileXRMoonEpDispatchArgsV1 TileXRMoonEpDispatchArgsV2;

typedef struct TileXRMoonEpPrefetchWeightArgsV1 {
    uint32_t structSize;
    uint32_t abiVersion;
    TileXRCommPtr comm;
    const TileXRMoonEpPlanV1 *plan;
    const TileXRMoonEpTensorV1 *gate;
    const TileXRMoonEpTensorV1 *up;
    const TileXRMoonEpTensorV1 *down;
    uint64_t flags;
} TileXRMoonEpPrefetchWeightArgsV1;

typedef struct TileXRMoonEpCombineArgsV1 {
    uint32_t structSize;
    uint32_t abiVersion;
    TileXRCommPtr comm;
    const TileXRMoonEpPlanV1 *plan;
    const int32_t *dstLocal;
    const TileXRMoonEpTensorV1 *hiddenNvsh;
    const TileXRMoonEpTensorV1 *routeWeightsNvs;
    TileXRMoonEpTensorV1 *hiddenSh;
    TileXRMoonEpTensorV1 *routeWeightsSk;
    uint64_t flags;
} TileXRMoonEpCombineArgsV1;

typedef struct TileXRMoonEpReduceGradArgsV1 {
    uint32_t structSize;
    uint32_t abiVersion;
    TileXRCommPtr comm;
    const TileXRMoonEpPlanV1 *plan;
    const TileXRMoonEpTensorV1 *input;
    TileXRMoonEpTensorV1 *output;
    uint64_t flags;
} TileXRMoonEpReduceGradArgsV1;

typedef struct TileXRMoonEpReduceGradWorkspaceQueryV2 {
    uint32_t structSize;
    uint32_t abiVersion;
    TileXRCommPtr comm;
    const TileXRMoonEpPlanV1 *plan;
    const TileXRMoonEpTensorV1 *gate;
    const TileXRMoonEpTensorV1 *up;
    const TileXRMoonEpTensorV1 *down;
    uint64_t requestedUdmaChunkBytes;
    uint64_t flags;
} TileXRMoonEpReduceGradWorkspaceQueryV2;

typedef struct TileXRMoonEpReduceGradWorkspaceInfoV2 {
    uint32_t structSize;
    uint32_t abiVersion;
    uint64_t workspaceBytes;
    uint64_t workspaceAlignment;
    uint64_t udmaChunkBytes;
    uint64_t peerWindowBytes;
    uint64_t peerHalfBytes;
    uint64_t peerSlotStrideBytes;
    uint64_t rowBytes[3];
    uint32_t transports[3];
    uint32_t blockDim;
} TileXRMoonEpReduceGradWorkspaceInfoV2;

typedef struct TileXRMoonEpReduceGradArgsV2 {
    uint32_t structSize;
    uint32_t abiVersion;
    TileXRCommPtr comm;
    const TileXRMoonEpPlanV1 *plan;
    TileXRMoonEpTensorV1 *gate;
    TileXRMoonEpTensorV1 *up;
    TileXRMoonEpTensorV1 *down;
    void *workspace;
    uint64_t workspaceBytes;
    TileXRMoonEpTensorV1 *status;
    uint64_t waitIterations;
    uint64_t requestedUdmaChunkBytes;
    uint64_t flags;
} TileXRMoonEpReduceGradArgsV2;

uint32_t TileXRMoonEpGetAbiVersion(void);

int TileXRMoonEpGetCapabilitiesV1(uint64_t *nativeStages, uint64_t *stubStages);

int TileXRMoonEpGetCapabilitiesV2(uint64_t *nativeStages, uint64_t *stubStages);

int TileXRMoonEpPlanningGetWorkspaceSizeV1(TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t e, int64_t b, int64_t tokenPadding, uint64_t *workspaceBytes, int64_t *nvS);

int TileXRMoonEpPlanningV1(const TileXRMoonEpPlanningArgsV1 *args, aclrtStream stream);

int TileXRMoonEpDispatchGetWorkspaceSizeV1(TileXRCommPtr comm, int64_t s,
    int64_t k, int64_t h, uint32_t hiddenDtype, uint64_t *workspaceBytes,
    uint64_t *workspaceAlignment);

int TileXRMoonEpDispatchV1(const TileXRMoonEpDispatchArgsV1 *args, aclrtStream stream);

int TileXRMoonEpDispatchGetWorkspaceSizeV2(TileXRCommPtr comm, int64_t s,
    int64_t k, int64_t h, uint32_t hiddenDtype, uint64_t *workspaceBytes,
    uint64_t *workspaceAlignment);

int TileXRMoonEpDispatchV2(const TileXRMoonEpDispatchArgsV2 *args, aclrtStream stream);

int TileXRMoonEpPrefetchWeightV1(const TileXRMoonEpPrefetchWeightArgsV1 *args,
    aclrtStream stream);

int TileXRMoonEpCombineV1(const TileXRMoonEpCombineArgsV1 *args, aclrtStream stream);

int TileXRMoonEpReduceGradV1(const TileXRMoonEpReduceGradArgsV1 *args,
    aclrtStream stream);

int TileXRMoonEpReduceGradGetWorkspaceSizeV2(
    const TileXRMoonEpReduceGradWorkspaceQueryV2 *query,
    TileXRMoonEpReduceGradWorkspaceInfoV2 *info);

/*
 * Enqueues asynchronously and is non-reentrant per communicator. Successful
 * calls on one stream may be queued in order. Before switching streams,
 * unregistering workspace, destroying the communicator, or releasing device
 * buffers referenced by args, the caller must synchronize the last used stream.
 * If the status reset is queued but kernel launch fails, this function attempts
 * to drain that stream; a drain failure is reported as a runtime error.
 */
int TileXRMoonEpReduceGradV2(const TileXRMoonEpReduceGradArgsV2 *args,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // TILEXR_MOONEP_H
