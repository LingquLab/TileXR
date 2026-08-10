#include "reduce_grad_host.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

#include "acl/acl_rt.h"
#include "comm_args.h"
#include "tilexr_types.h"

namespace TileXRMoonEp {
namespace {

std::mutex g_reduceGradEnqueueMutex;

bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t *out)
{
    if (out == nullptr || rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool ValidatePlan(const TileXRMoonEpPlanV1 *plan, const TileXR::CommArgs &commArgs)
{
    if (plan == nullptr || plan->structSize < sizeof(*plan) ||
        plan->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || plan->n <= 0 ||
        plan->k <= 0 || plan->n % plan->k != 0 || plan->e <= 0 || plan->b <= 0 ||
        plan->r <= 0 || plan->r > TileXR::TILEXR_MAX_RANK_SIZE ||
        plan->r != commArgs.rankSize || plan->e % plan->r != 0 ||
        plan->b > plan->e / plan->r || plan->expertsToCopy == nullptr) {
        return false;
    }
    uint64_t capacity = 0;
    return CheckedMul(static_cast<uint64_t>(plan->n / plan->k),
        static_cast<uint64_t>(plan->k), &capacity) &&
        capacity == static_cast<uint64_t>(plan->n);
}

bool ValidateGradient(const TileXRMoonEpTensorV1 *tensor, int64_t rowCount,
    uint64_t *rowElements)
{
    if (tensor == nullptr || rowElements == nullptr || tensor->structSize < sizeof(*tensor) ||
        tensor->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || tensor->data == nullptr ||
        tensor->dtype != TILEXR_MOONEP_DTYPE_FLOAT32 || tensor->rank < 2 ||
        tensor->rank > TILEXR_MOONEP_MAX_TENSOR_RANK || tensor->shape[0] != rowCount) {
        return false;
    }

    uint64_t elements = 1;
    for (uint32_t dim = 0; dim < TILEXR_MOONEP_MAX_TENSOR_RANK; ++dim) {
        if (dim < tensor->rank) {
            if (tensor->shape[dim] <= 0 ||
                !CheckedMul(elements, static_cast<uint64_t>(tensor->shape[dim]), &elements)) {
                return false;
            }
        } else if (tensor->shape[dim] != 0) {
            return false;
        }
    }
    if (elements != tensor->elementCount || rowCount <= 0 ||
        elements % static_cast<uint64_t>(rowCount) != 0) {
        return false;
    }
    *rowElements = elements / static_cast<uint64_t>(rowCount);
    return *rowElements != 0;
}

bool ValidateStatus(const TileXRMoonEpTensorV1 *status)
{
    return status != nullptr && status->structSize >= sizeof(*status) &&
        status->abiVersion == TILEXR_MOONEP_ABI_VERSION_V1 && status->data != nullptr &&
        status->elementCount == 1 && status->dtype == TILEXR_MOONEP_DTYPE_INT32 &&
        status->rank == 1 && status->shape[0] == 1 && status->shape[1] == 0 &&
        status->shape[2] == 0 && status->shape[3] == 0;
}

bool IsA5(const TileXR::CommArgs &commArgs)
{
    return (commArgs.extraFlag & TileXR::ExtraFlag::TOPO_910A5) != 0;
}

bool LocalityValid(const TileXR::CommArgs &commArgs)
{
    return commArgs.rankSize > 0 && commArgs.rankSize <= TileXR::TILEXR_MAX_RANK_SIZE &&
        commArgs.rank >= 0 && commArgs.rank < commArgs.rankSize &&
        commArgs.localRankSize > 0 && commArgs.localRankSize <= commArgs.rankSize &&
        commArgs.rankSize % commArgs.localRankSize == 0 &&
        commArgs.localRank >= 0 && commArgs.localRank < commArgs.localRankSize;
}

bool UsesTransport(const ReduceGradLayout &layout, uint32_t transport)
{
    for (uint32_t q = 0; q < kReduceGradProjectionCount; ++q) {
        if (layout.transports[q] == transport) {
            return true;
        }
    }
    return false;
}

bool PeerWindowsReady(const TileXR::CommArgs &commArgs)
{
    for (int32_t rank = 0; rank < commArgs.rankSize; ++rank) {
        if (commArgs.peerMems[rank] == nullptr) {
            return false;
        }
    }
    return true;
}

int ValidateRegisteredWorkspace(const ReduceGradParams &params,
    const TileXR::CommArgs &commArgs, const ReduceGradLayout &layout,
    const TileXR::TileXRUDMARegistry **registryOut)
{
    if (registryOut == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *registryOut = nullptr;
    if (!UsesTransport(layout, TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_UDMA)) {
        return params.workspace == nullptr && params.workspaceBytes == 0 ?
            TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (params.workspace == nullptr || params.workspaceBytes < layout.workspaceBytes ||
        (commArgs.extraFlag & TileXR::ExtraFlag::UDMA) == 0 ||
        commArgs.udmaInfoPtr == nullptr || commArgs.udmaRegistryPtr == nullptr) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }

    const TileXR::TileXRUDMARegistry *registry = nullptr;
    const int ret = TileXRGetUDMARegistryHost(params.comm, &registry);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (!TileXR::UDMARegistryValid(registry, commArgs.rankSize) ||
        registry->regions[commArgs.rank].base != static_cast<GM_ADDR>(params.workspace) ||
        registry->regions[commArgs.rank].bytes != params.workspaceBytes) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    const size_t registeredBytes = registry->regions[0].bytes;
    for (int32_t rank = 0; rank < commArgs.rankSize; ++rank) {
        const auto &region = registry->regions[rank];
        if (reinterpret_cast<uintptr_t>(region.base) % kReduceGradUdmaWorkspaceAlignment != 0 ||
            region.bytes != registeredBytes ||
            !TileXR::UDMARegionContains(registry, rank, 0, layout.workspaceBytes)) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
    }
    *registryOut = registry;
    return TileXR::TILEXR_SUCCESS;
}

void FillWorkspaceInfo(const ReduceGradLayout &layout,
    TileXRMoonEpReduceGradWorkspaceInfoV2 *info)
{
    info->workspaceBytes = layout.workspaceBytes;
    info->workspaceAlignment = kReduceGradUdmaWorkspaceAlignment;
    info->udmaChunkBytes = layout.udmaChunkBytes;
    info->peerWindowBytes = layout.peerWindowBytes;
    info->peerHalfBytes = layout.peerHalfBytes;
    info->peerSlotStrideBytes = layout.peerSlotStrideBytes;
    info->blockDim = static_cast<uint32_t>(layout.blockDim);
    for (uint32_t q = 0; q < kReduceGradProjectionCount; ++q) {
        info->rowBytes[q] = layout.rowBytes[q];
        info->transports[q] = layout.transports[q];
    }
}

} // namespace

int TileXRMoonEpPrepareReduceGradLayout(TileXRCommPtr comm,
    const TileXRMoonEpPlanV1 *plan,
    const TileXRMoonEpTensorV1 *const gradients[kReduceGradProjectionCount],
    uint64_t requestedUdmaChunkBytes, ReduceGradLayout *layout)
{
    if (comm == nullptr || gradients == nullptr || layout == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    TileXR::CommArgs *commArgs = nullptr;
    int ret = TileXRGetCommArgsHost(comm, commArgs);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (commArgs == nullptr) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    if (!ValidatePlan(plan, *commArgs)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (!IsA5(*commArgs)) {
        return TileXR::TILEXR_ERROR_NOT_SUPPORT;
    }
    if (!LocalityValid(*commArgs)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    uint64_t rowElements[kReduceGradProjectionCount] = {};
    uint64_t rowCountValue = 0;
    if (!CheckedAdd(static_cast<uint64_t>(plan->e), static_cast<uint64_t>(plan->b),
            &rowCountValue) ||
        rowCountValue > static_cast<uint64_t>(INT64_MAX)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const int64_t rowCount = static_cast<int64_t>(rowCountValue);
    for (uint32_t q = 0; q < kReduceGradProjectionCount; ++q) {
        if (!ValidateGradient(gradients[q], rowCount, &rowElements[q])) {
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
        }
    }
    ret = TileXRMoonEpBuildReduceGradLayout(commArgs->rank, commArgs->rankSize,
        plan->e, plan->b, rowElements, TileXRMoonEpReduceGradPeerWindowBytes(),
        requestedUdmaChunkBytes, layout);
    if (ret != TileXR::TILEXR_SUCCESS ||
        !UsesTransport(*layout, TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_UDMA)) {
        return ret;
    }

    uint32_t qpCount = 0;
    ret = TileXRUDMAGetQpCount(comm, &qpCount);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (qpCount == 0 || qpCount > kReduceGradMaxUdmaQpCount) {
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    layout->udmaQpCount = qpCount;
    return TileXR::TILEXR_SUCCESS;
}

int TileXRMoonEpPrepareReduceGradLaunchContext(const ReduceGradParams &params,
    ReduceGradLaunchContext *context)
{
    if (context == nullptr || params.stream == nullptr || params.waitIterations == 0 ||
        !ValidateStatus(params.status)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *context = ReduceGradLaunchContext {};
    const TileXRMoonEpTensorV1 *gradients[kReduceGradProjectionCount] = {
        params.gradients[kReduceGradGate], params.gradients[kReduceGradUp],
        params.gradients[kReduceGradDown]};
    int ret = TileXRMoonEpPrepareReduceGradLayout(params.comm, params.plan, gradients,
        params.requestedUdmaChunkBytes, &context->layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    ret = TileXRGetCommArgsHost(params.comm, context->hostArgs);
    if (ret != TileXR::TILEXR_SUCCESS || context->hostArgs == nullptr) {
        *context = ReduceGradLaunchContext {};
        return ret == TileXR::TILEXR_SUCCESS ? TileXR::TILEXR_ERROR_NOT_INITIALIZED : ret;
    }
    if (UsesTransport(context->layout, TILEXR_MOONEP_REDUCE_GRAD_TRANSPORT_PEER) &&
        !PeerWindowsReady(*context->hostArgs)) {
        *context = ReduceGradLaunchContext {};
        return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
    }
    ret = ValidateRegisteredWorkspace(params, *context->hostArgs, context->layout,
        &context->registry);
    if (ret != TileXR::TILEXR_SUCCESS) {
        *context = ReduceGradLaunchContext {};
        return ret;
    }
    ret = TileXRGetCommArgsDev(params.comm, context->devArgs);
    if (ret != TileXR::TILEXR_SUCCESS || context->devArgs == nullptr) {
        *context = ReduceGradLaunchContext {};
        return ret == TileXR::TILEXR_SUCCESS ? TileXR::TILEXR_ERROR_NOT_INITIALIZED : ret;
    }
    return TileXR::TILEXR_SUCCESS;
}

} // namespace TileXRMoonEp

extern "C" int TileXRMoonEpReduceGradGetWorkspaceSizeV2(
    const TileXRMoonEpReduceGradWorkspaceQueryV2 *query,
    TileXRMoonEpReduceGradWorkspaceInfoV2 *info)
{
    if (query == nullptr || info == nullptr || query->structSize < sizeof(*query) ||
        query->abiVersion != TILEXR_MOONEP_ABI_VERSION_V2 ||
        info->structSize < sizeof(*info) || info->abiVersion != TILEXR_MOONEP_ABI_VERSION_V2 ||
        query->flags != TILEXR_MOONEP_FLAG_NONE) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const TileXRMoonEpTensorV1 *gradients[TileXRMoonEp::kReduceGradProjectionCount] = {
        query->gate, query->up, query->down};
    TileXRMoonEp::ReduceGradLayout layout {};
    const int ret = TileXRMoonEp::TileXRMoonEpPrepareReduceGradLayout(query->comm,
        query->plan, gradients, query->requestedUdmaChunkBytes, &layout);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    TileXRMoonEp::FillWorkspaceInfo(layout, info);
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRMoonEpReduceGradV2(const TileXRMoonEpReduceGradArgsV2 *args,
    aclrtStream stream)
{
    if (args == nullptr || args->structSize < sizeof(*args) ||
        args->abiVersion != TILEXR_MOONEP_ABI_VERSION_V2 || args->comm == nullptr ||
        args->plan == nullptr || args->gate == nullptr || args->up == nullptr ||
        args->down == nullptr || args->status == nullptr || args->waitIterations == 0 ||
        args->flags != TILEXR_MOONEP_FLAG_NONE || stream == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    TileXRMoonEp::ReduceGradParams params {};
    params.comm = args->comm;
    params.plan = args->plan;
    params.gradients[TileXRMoonEp::kReduceGradGate] = args->gate;
    params.gradients[TileXRMoonEp::kReduceGradUp] = args->up;
    params.gradients[TileXRMoonEp::kReduceGradDown] = args->down;
    params.workspace = args->workspace;
    params.workspaceBytes = args->workspaceBytes;
    params.status = args->status;
    params.waitIterations = args->waitIterations;
    params.requestedUdmaChunkBytes = args->requestedUdmaChunkBytes;
    params.stream = stream;

    TileXRMoonEp::ReduceGradLaunchContext context {};
    int ret = TileXRMoonEp::TileXRMoonEpPrepareReduceGradLaunchContext(params, &context);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }

    std::lock_guard<std::mutex> guard(TileXRMoonEp::g_reduceGradEnqueueMutex);
    const aclError memsetRet = aclrtMemsetAsync(args->status->data, sizeof(int32_t), 0,
        sizeof(int32_t), stream);
    if (memsetRet != ACL_SUCCESS) {
        return TileXR::TILEXR_ERROR_MKIRT;
    }
    ret = TileXRMoonEp::TileXRMoonEpLaunchReduceGradKernel(params, context);
    if (ret != TileXR::TILEXR_SUCCESS && aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        return TileXR::TILEXR_ERROR_MKIRT;
    }
    return ret;
}
