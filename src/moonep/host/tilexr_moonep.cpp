#include "tilexr_moonep.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "acl/acl_rt.h"
#include "tilexr_api.h"
#include "tilexr_moonep_planner.h"

namespace {

const uint64_t kNativeStages = static_cast<uint64_t>(TILEXR_MOONEP_STAGE_PLANNING);
const uint64_t kStubStages = static_cast<uint64_t>(TILEXR_MOONEP_STAGE_DISPATCH) |
    static_cast<uint64_t>(TILEXR_MOONEP_STAGE_PREFETCH_WEIGHT) |
    static_cast<uint64_t>(TILEXR_MOONEP_STAGE_COMBINE) |
    static_cast<uint64_t>(TILEXR_MOONEP_STAGE_REDUCE_GRAD);

bool CheckedMultiply(uint64_t lhs, uint64_t rhs, uint64_t *result)
{
    if (result == nullptr || (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool PositiveToUint64(int64_t value, uint64_t *result)
{
    if (value <= 0 || result == nullptr) {
        return false;
    }
    *result = static_cast<uint64_t>(value);
    return true;
}

bool ValidatePlan(const TileXRMoonEpPlanV1 *plan)
{
    if (plan == nullptr || plan->structSize < sizeof(TileXRMoonEpPlanV1) ||
        plan->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 ||
        plan->s <= 0 || plan->k <= 0 || plan->e <= 0 ||
        plan->b <= 0 || plan->world <= 0 || plan->rank < 0 || plan->rank >= plan->world ||
        plan->e % plan->world != 0 || plan->b != plan->e / plan->world ||
        plan->dst == nullptr || plan->cu == nullptr || plan->expertsToCopy == nullptr ||
        plan->remoteStats == nullptr || plan->status == nullptr) {
        return false;
    }

    uint64_t s = 0;
    uint64_t k = 0;
    uint64_t world = 0;
    uint64_t dispatchedCapacity = 0;
    uint64_t encodedCapacity = 0;
    if (!PositiveToUint64(plan->s, &s) || !PositiveToUint64(plan->k, &k) ||
        !PositiveToUint64(plan->world, &world) || !CheckedMultiply(s, k, &dispatchedCapacity) ||
        dispatchedCapacity > static_cast<uint64_t>(INT64_MAX) ||
        plan->dispatchedCapacity != static_cast<int64_t>(dispatchedCapacity) ||
        !CheckedMultiply(world, dispatchedCapacity, &encodedCapacity)) {
        return false;
    }

    return encodedCapacity <= static_cast<uint64_t>(INT32_MAX) + UINT64_C(1);
}

int ValidateCommPlan(TileXRCommPtr comm, const TileXRMoonEpPlanV1 *plan)
{
    if (comm == nullptr || !ValidatePlan(plan)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    TileXR::CommArgs *commArgs = nullptr;
    const int ret = TileXRGetCommArgsHost(comm, commArgs);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    if (commArgs == nullptr || plan->rank != static_cast<int64_t>(commArgs->rank) ||
        plan->world != static_cast<int64_t>(commArgs->rankSize)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    return TILEXR_MOONEP_SUCCESS;
}

bool DTypeSize(uint32_t dtype, uint64_t *bytes)
{
    if (bytes == nullptr) {
        return false;
    }
    switch (dtype) {
        case TILEXR_MOONEP_DTYPE_FLOAT16:
        case TILEXR_MOONEP_DTYPE_BFLOAT16:
            *bytes = 2;
            return true;
        case TILEXR_MOONEP_DTYPE_INT32:
        case TILEXR_MOONEP_DTYPE_FLOAT32:
            *bytes = 4;
            return true;
        default:
            return false;
    }
}

bool TensorBytes(const TileXRMoonEpTensorV1 *tensor, size_t *bytes)
{
    if (tensor == nullptr || bytes == nullptr ||
        tensor->structSize < sizeof(TileXRMoonEpTensorV1) ||
        tensor->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || tensor->data == nullptr ||
        tensor->elementCount == 0 || tensor->rank == 0 ||
        tensor->rank > TILEXR_MOONEP_MAX_TENSOR_RANK) {
        return false;
    }

    uint64_t elementBytes = 0;
    uint64_t shapeElements = 1;
    if (!DTypeSize(tensor->dtype, &elementBytes)) {
        return false;
    }
    for (uint32_t i = 0; i < TILEXR_MOONEP_MAX_TENSOR_RANK; ++i) {
        if (i < tensor->rank) {
            uint64_t dimension = 0;
            if (!PositiveToUint64(tensor->shape[i], &dimension) ||
                !CheckedMultiply(shapeElements, dimension, &shapeElements)) {
                return false;
            }
        } else if (tensor->shape[i] != 0) {
            return false;
        }
    }
    if (shapeElements != tensor->elementCount) {
        return false;
    }

    uint64_t byteCount = 0;
    if (!CheckedMultiply(tensor->elementCount, elementBytes, &byteCount) ||
        byteCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    *bytes = static_cast<size_t>(byteCount);
    return true;
}

bool TensorHasShape(const TileXRMoonEpTensorV1 *tensor, int64_t dim0, int64_t dim1)
{
    if (tensor == nullptr || tensor->dtype != TILEXR_MOONEP_DTYPE_INT32 ||
        tensor->shape[0] != dim0) {
        return false;
    }
    if (dim1 > 0) {
        return tensor->rank == 2 && tensor->shape[1] == dim1;
    }
    return tensor->rank == 1;
}

bool TensorDescriptorsEqual(const TileXRMoonEpTensorV1 &lhs,
                            const TileXRMoonEpTensorV1 &rhs)
{
    if (lhs.data != rhs.data || lhs.elementCount != rhs.elementCount ||
        lhs.dtype != rhs.dtype || lhs.rank != rhs.rank) {
        return false;
    }
    for (uint32_t i = 0; i < TILEXR_MOONEP_MAX_TENSOR_RANK; ++i) {
        if (lhs.shape[i] != rhs.shape[i]) {
            return false;
        }
    }
    return true;
}

enum class StubStage {
    Dispatch,
    PrefetchWeight,
    Combine,
    ReduceGrad,
};

bool ValidateStubShapes(StubStage stage, const TileXRMoonEpPlanV1 &plan,
                        const TileXRMoonEpTensorV1 &input,
                        const TileXRMoonEpTensorV1 &output)
{
    switch (stage) {
        case StubStage::Dispatch:
            if (input.dtype == TILEXR_MOONEP_DTYPE_FLOAT32) {
                return input.rank == 2 && input.shape[0] == plan.s &&
                    input.shape[1] == plan.k && output.rank == 1 &&
                    output.shape[0] == plan.dispatchedCapacity;
            }
            return input.rank == 2 && output.rank == 2 && input.shape[1] > 0 &&
                input.shape[1] == output.shape[1] && input.shape[0] == plan.s &&
                output.shape[0] == plan.dispatchedCapacity;
        case StubStage::Combine:
            if (input.dtype == TILEXR_MOONEP_DTYPE_FLOAT32) {
                return input.rank == 1 && input.shape[0] == plan.dispatchedCapacity &&
                    output.rank == 2 && output.shape[0] == plan.s &&
                    output.shape[1] == plan.k;
            }
            return input.rank == 2 && output.rank == 2 && input.shape[1] > 0 &&
                input.shape[1] == output.shape[1] &&
                input.shape[0] == plan.dispatchedCapacity && output.shape[0] == plan.s;
        case StubStage::PrefetchWeight:
        case StubStage::ReduceGrad:
            return input.rank == 2 && output.rank == 2 && input.shape[1] > 0 &&
                input.shape[1] == output.shape[1] && input.shape[0] == output.shape[0] &&
                input.elementCount == output.elementCount;
    }
    return false;
}

template <typename Args>
int RunLocalStub(const Args *args, aclrtStream stream, StubStage stage)
{
    if (args == nullptr || args->structSize < sizeof(Args) ||
        args->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || args->comm == nullptr ||
        args->flags != TILEXR_MOONEP_FLAG_NONE || stream == nullptr ||
        args->plan == nullptr || args->input == nullptr || args->output == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    const int commRet = ValidateCommPlan(args->comm, args->plan);
    if (commRet != TILEXR_MOONEP_SUCCESS) {
        return commRet;
    }

    size_t inputBytes = 0;
    size_t outputBytes = 0;
    if (!TensorBytes(args->input, &inputBytes) || !TensorBytes(args->output, &outputBytes) ||
        args->input->dtype != args->output->dtype ||
        !ValidateStubShapes(stage, *args->plan, *args->input, *args->output)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    if (args->input->data == args->output->data) {
        return TensorDescriptorsEqual(*args->input, *args->output) ?
            TILEXR_MOONEP_SUCCESS : TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    aclError ret = aclrtMemsetAsync(args->output->data, outputBytes, 0, outputBytes, stream);
    if (ret != ACL_SUCCESS) {
        return TILEXR_MOONEP_ERROR_INTERNAL;
    }
    const size_t copyBytes = std::min(inputBytes, outputBytes);
    ret = aclrtMemcpyAsync(args->output->data, outputBytes, args->input->data, copyBytes,
        ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    return ret == ACL_SUCCESS ? TILEXR_MOONEP_SUCCESS : TILEXR_MOONEP_ERROR_INTERNAL;
}

} // namespace

extern "C" uint32_t TileXRMoonEpGetAbiVersion(void)
{
    return TILEXR_MOONEP_ABI_VERSION_V1;
}

extern "C" int TileXRMoonEpGetCapabilitiesV1(uint64_t *nativeStages, uint64_t *stubStages)
{
    if (nativeStages == nullptr || stubStages == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }
    *nativeStages = kNativeStages;
    *stubStages = kStubStages;
    return TILEXR_MOONEP_SUCCESS;
}

extern "C" int TileXRMoonEpPlanningGetWorkspaceSizeV1(TileXRCommPtr comm, int64_t s,
    int64_t k, int64_t e, uint64_t *workspaceBytes, int64_t *dispatchedCapacity)
{
    if (comm == nullptr || s <= 0 || k <= 0 || e <= 0 || workspaceBytes == nullptr ||
        dispatchedCapacity == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    uint64_t sValue = 0;
    uint64_t kValue = 0;
    uint64_t expectedCapacity = 0;
    if (!PositiveToUint64(s, &sValue) || !PositiveToUint64(k, &kValue) ||
        !CheckedMultiply(sValue, kValue, &expectedCapacity) ||
        expectedCapacity > static_cast<uint64_t>(INT64_MAX)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    uint64_t plannerWorkspaceBytes = 0;
    int64_t plannerCapacity = 0;
    const int ret = TileXRMoonEpPlannerGetWorkspaceSizeV2(
        comm, s, k, e, &plannerWorkspaceBytes, &plannerCapacity);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    if (plannerWorkspaceBytes == 0 || plannerCapacity != static_cast<int64_t>(expectedCapacity)) {
        return TILEXR_MOONEP_ERROR_INTERNAL;
    }
    *workspaceBytes = plannerWorkspaceBytes;
    *dispatchedCapacity = plannerCapacity;
    return TILEXR_MOONEP_SUCCESS;
}

extern "C" int TileXRMoonEpPlanningV1(const TileXRMoonEpPlanningArgsV1 *args,
    aclrtStream stream)
{
    if (args == nullptr || args->structSize < sizeof(TileXRMoonEpPlanningArgsV1) ||
        args->abiVersion != TILEXR_MOONEP_ABI_VERSION_V1 || args->comm == nullptr ||
        args->topkExperts == nullptr || args->tokensPerExpert == nullptr ||
        args->workspace == nullptr || args->workspaceBytes == 0 || args->plan == nullptr ||
        args->waitIterations == 0 || args->flags != TILEXR_MOONEP_FLAG_NONE || stream == nullptr) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    const int commRet = ValidateCommPlan(args->comm, args->plan);
    if (commRet != TILEXR_MOONEP_SUCCESS) {
        return commRet;
    }

    size_t topkBytes = 0;
    size_t tpeBytes = 0;
    if (!TensorBytes(args->topkExperts, &topkBytes) ||
        !TensorBytes(args->tokensPerExpert, &tpeBytes) ||
        !TensorHasShape(args->topkExperts, args->plan->s, args->plan->k) ||
        !TensorHasShape(args->tokensPerExpert, args->plan->e, 0)) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    uint64_t requiredWorkspaceBytes = 0;
    int64_t dispatchedCapacity = 0;
    int ret = TileXRMoonEpPlanningGetWorkspaceSizeV1(args->comm, args->plan->s,
        args->plan->k, args->plan->e, &requiredWorkspaceBytes, &dispatchedCapacity);
    if (ret != TILEXR_MOONEP_SUCCESS) {
        return ret;
    }
    if (args->workspaceBytes < requiredWorkspaceBytes ||
        dispatchedCapacity != args->plan->dispatchedCapacity) {
        return TILEXR_MOONEP_ERROR_INVALID_ARGUMENT;
    }

    return TileXRMoonEpPlannerV2(
        static_cast<const int32_t *>(args->topkExperts->data),
        static_cast<const int32_t *>(args->tokensPerExpert->data),
        args->comm, args->plan->s, args->plan->k, args->plan->e,
        args->workspace, args->workspaceBytes,
        static_cast<int32_t *>(args->plan->dst),
        static_cast<int32_t *>(args->plan->cu),
        static_cast<int32_t *>(args->plan->expertsToCopy),
        static_cast<int32_t *>(args->plan->remoteStats),
        static_cast<int32_t *>(args->plan->status),
        args->waitIterations, stream);
}

extern "C" int TileXRMoonEpDispatchV1(const TileXRMoonEpDispatchArgsV1 *args,
    aclrtStream stream)
{
    return RunLocalStub(args, stream, StubStage::Dispatch);
}

extern "C" int TileXRMoonEpPrefetchWeightV1(
    const TileXRMoonEpPrefetchWeightArgsV1 *args, aclrtStream stream)
{
    return RunLocalStub(args, stream, StubStage::PrefetchWeight);
}

extern "C" int TileXRMoonEpCombineV1(const TileXRMoonEpCombineArgsV1 *args,
    aclrtStream stream)
{
    return RunLocalStub(args, stream, StubStage::Combine);
}

extern "C" int TileXRMoonEpReduceGradV1(const TileXRMoonEpReduceGradArgsV1 *args,
    aclrtStream stream)
{
    return RunLocalStub(args, stream, StubStage::ReduceGrad);
}
