#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl_rt.h"
#include "tilexr_api.h"
#include "tilexr_moonep.h"

#ifndef UINTPTR_C
#define UINTPTR_C(value) static_cast<uintptr_t>(value)
#endif

namespace {

TileXR::CommArgs MakeCommArgs()
{
    TileXR::CommArgs args {};
    args.rank = 0;
    args.localRank = 0;
    args.rankSize = 2;
    args.localRankSize = 2;
    return args;
}

int g_failures = 0;
int g_memsetReturn = ACL_SUCCESS;
int g_memcpyReturn = ACL_SUCCESS;
int g_plannerQueryReturn = TILEXR_MOONEP_SUCCESS;
int g_plannerLaunchReturn = TILEXR_MOONEP_SUCCESS;
uint64_t g_plannerWorkspaceBytes = 256;
int64_t g_plannerCapacity = 4;
int g_plannerQueryCalls = 0;
int g_plannerLaunchCalls = 0;
int g_streamSynchronizeCalls = 0;
int g_commArgsReturn = TILEXR_MOONEP_SUCCESS;
TileXR::CommArgs g_commArgs = MakeCommArgs();
std::vector<std::string> g_runtimeCalls;

struct MemsetCall {
    void *dst = nullptr;
    size_t maxBytes = 0;
    int value = -1;
    size_t bytes = 0;
    aclrtStream stream = nullptr;
} g_memsetCall;

struct MemcpyCall {
    void *dst = nullptr;
    size_t dstBytes = 0;
    const void *src = nullptr;
    size_t bytes = 0;
    aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
    aclrtStream stream = nullptr;
} g_memcpyCall;

struct PlannerCall {
    const int32_t *topk = nullptr;
    const int32_t *tpe = nullptr;
    TileXRCommPtr comm = nullptr;
    int64_t s = 0;
    int64_t k = 0;
    int64_t e = 0;
    void *workspace = nullptr;
    uint64_t workspaceBytes = 0;
    int32_t *dst = nullptr;
    int32_t *cu = nullptr;
    int32_t *expertsToCopy = nullptr;
    int32_t *remoteStats = nullptr;
    int32_t *status = nullptr;
    uint64_t waitIterations = 0;
    aclrtStream stream = nullptr;
} g_plannerCall;

void ResetFakes()
{
    g_memsetReturn = ACL_SUCCESS;
    g_memcpyReturn = ACL_SUCCESS;
    g_plannerQueryReturn = TILEXR_MOONEP_SUCCESS;
    g_plannerLaunchReturn = TILEXR_MOONEP_SUCCESS;
    g_plannerWorkspaceBytes = 256;
    g_plannerCapacity = 4;
    g_plannerQueryCalls = 0;
    g_plannerLaunchCalls = 0;
    g_streamSynchronizeCalls = 0;
    g_commArgsReturn = TILEXR_MOONEP_SUCCESS;
    g_commArgs = MakeCommArgs();
    g_runtimeCalls.clear();
    g_memsetCall = MemsetCall {};
    g_memcpyCall = MemcpyCall {};
    g_plannerCall = PlannerCall {};
}

void Check(bool condition, const std::string &message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        ++g_failures;
    }
}

void CheckStatus(const std::string &label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " returned " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

TileXRMoonEpPlanV1 ValidPlan()
{
    TileXRMoonEpPlanV1 plan {};
    plan.structSize = sizeof(plan);
    plan.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    plan.s = 2;
    plan.k = 2;
    plan.e = 4;
    plan.b = 2;
    plan.rank = 0;
    plan.world = 2;
    plan.dispatchedCapacity = 4;
    plan.dst = reinterpret_cast<void *>(UINTPTR_C(0x3000));
    plan.cu = reinterpret_cast<void *>(UINTPTR_C(0x3100));
    plan.expertsToCopy = reinterpret_cast<void *>(UINTPTR_C(0x3200));
    plan.remoteStats = reinterpret_cast<void *>(UINTPTR_C(0x3300));
    plan.status = reinterpret_cast<void *>(UINTPTR_C(0x3400));
    return plan;
}

TileXRMoonEpTensorV1 Tensor(void *data, uint64_t elements, uint32_t dtype,
                            int64_t dim0, int64_t dim1 = 0)
{
    TileXRMoonEpTensorV1 tensor {};
    tensor.structSize = sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.data = data;
    tensor.elementCount = elements;
    tensor.dtype = dtype;
    tensor.rank = dim1 > 0 ? 2 : 1;
    tensor.shape[0] = dim0;
    tensor.shape[1] = dim1;
    return tensor;
}

TileXRMoonEpDispatchArgsV1 DispatchArgs(const TileXRMoonEpPlanV1 *plan,
    const TileXRMoonEpTensorV1 *input, TileXRMoonEpTensorV1 *output)
{
    TileXRMoonEpDispatchArgsV1 args {};
    args.structSize = sizeof(args);
    args.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    args.comm = reinterpret_cast<TileXRCommPtr>(UINTPTR_C(0x1000));
    args.plan = plan;
    args.input = input;
    args.output = output;
    return args;
}

void CheckStubEnqueue(const char *name, int status, void *input, void *output,
                      size_t inputBytes, size_t outputBytes, aclrtStream stream)
{
    CheckStatus(name, status, TILEXR_MOONEP_SUCCESS);
    Check(g_runtimeCalls.size() == 2 && g_runtimeCalls[0] == "memset" &&
        g_runtimeCalls[1] == "memcpy", std::string(name) + " must enqueue memset then memcpy");
    Check(g_memsetCall.dst == output && g_memsetCall.maxBytes == outputBytes &&
        g_memsetCall.bytes == outputBytes && g_memsetCall.value == 0 &&
        g_memsetCall.stream == stream, std::string(name) + " memset arguments mismatch");
    Check(g_memcpyCall.dst == output && g_memcpyCall.dstBytes == outputBytes &&
        g_memcpyCall.src == input && g_memcpyCall.bytes == std::min(inputBytes, outputBytes) &&
        g_memcpyCall.kind == ACL_MEMCPY_DEVICE_TO_DEVICE && g_memcpyCall.stream == stream,
        std::string(name) + " memcpy arguments mismatch");
    Check(g_streamSynchronizeCalls == 0, std::string(name) + " synchronized the caller stream");
}

void TestCapabilities()
{
    uint64_t nativeStages = 0;
    uint64_t stubStages = 0;
    Check(TileXRMoonEpGetAbiVersion() == TILEXR_MOONEP_ABI_VERSION_V1,
        "ABI version query mismatch");
    CheckStatus("capability query", TileXRMoonEpGetCapabilitiesV1(&nativeStages, &stubStages),
        TILEXR_MOONEP_SUCCESS);
    Check(nativeStages == TILEXR_MOONEP_STAGE_PLANNING,
        "Planning must be the only native stage");
    const uint64_t expectedStubs = TILEXR_MOONEP_STAGE_DISPATCH |
        TILEXR_MOONEP_STAGE_PREFETCH_WEIGHT | TILEXR_MOONEP_STAGE_COMBINE |
        TILEXR_MOONEP_STAGE_REDUCE_GRAD;
    Check(stubStages == expectedStubs, "Stub capability mask mismatch");
    CheckStatus("null native mask", TileXRMoonEpGetCapabilitiesV1(nullptr, &stubStages),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    CheckStatus("null stub mask", TileXRMoonEpGetCapabilitiesV1(&nativeStages, nullptr),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
}

void TestWorkspaceQuery()
{
    ResetFakes();
    uint64_t workspaceBytes = 0;
    int64_t capacity = 0;
    TileXRCommPtr comm = reinterpret_cast<TileXRCommPtr>(UINTPTR_C(0x1000));
    CheckStatus("workspace query", TileXRMoonEpPlanningGetWorkspaceSizeV1(
        comm, 2, 2, 4, &workspaceBytes, &capacity), TILEXR_MOONEP_SUCCESS);
    Check(g_plannerQueryCalls == 1 && workspaceBytes == 256 && capacity == 4,
        "workspace query did not delegate to Planner V2");

    CheckStatus("null query comm", TileXRMoonEpPlanningGetWorkspaceSizeV1(
        nullptr, 2, 2, 4, &workspaceBytes, &capacity), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    CheckStatus("overflow query", TileXRMoonEpPlanningGetWorkspaceSizeV1(
        comm, std::numeric_limits<int64_t>::max(), 2, 4, &workspaceBytes, &capacity),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);

    g_plannerCapacity = 5;
    CheckStatus("padded capacity rejection", TileXRMoonEpPlanningGetWorkspaceSizeV1(
        comm, 2, 2, 4, &workspaceBytes, &capacity), TILEXR_MOONEP_ERROR_INTERNAL);
    g_plannerCapacity = 4;
    g_plannerWorkspaceBytes = 0;
    CheckStatus("zero planner workspace", TileXRMoonEpPlanningGetWorkspaceSizeV1(
        comm, 2, 2, 4, &workspaceBytes, &capacity), TILEXR_MOONEP_ERROR_INTERNAL);
    g_plannerQueryReturn = -77;
    CheckStatus("planner query error", TileXRMoonEpPlanningGetWorkspaceSizeV1(
        comm, 2, 2, 4, &workspaceBytes, &capacity), -77);
}

void TestPlanningDelegation()
{
    ResetFakes();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 topk = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x4000)),
        4, TILEXR_MOONEP_DTYPE_INT32, 2, 2);
    TileXRMoonEpTensorV1 tpe = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x4100)),
        4, TILEXR_MOONEP_DTYPE_INT32, 4);
    TileXRMoonEpPlanningArgsV1 args {};
    args.structSize = sizeof(args);
    args.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    args.comm = reinterpret_cast<TileXRCommPtr>(UINTPTR_C(0x1000));
    args.topkExperts = &topk;
    args.tokensPerExpert = &tpe;
    args.workspace = reinterpret_cast<void *>(UINTPTR_C(0x5000));
    args.workspaceBytes = 256;
    args.plan = &plan;
    args.waitIterations = 1234;
    aclrtStream stream = reinterpret_cast<aclrtStream>(UINTPTR_C(0x6000));

    CheckStatus("planning", TileXRMoonEpPlanningV1(&args, stream), TILEXR_MOONEP_SUCCESS);
    Check(g_plannerQueryCalls == 1 && g_plannerLaunchCalls == 1,
        "Planning V1 must query and launch Planner V2 exactly once");
    Check(g_plannerCall.topk == topk.data && g_plannerCall.tpe == tpe.data &&
        g_plannerCall.comm == args.comm && g_plannerCall.s == plan.s &&
        g_plannerCall.k == plan.k && g_plannerCall.e == plan.e &&
        g_plannerCall.workspace == args.workspace &&
        g_plannerCall.workspaceBytes == args.workspaceBytes &&
        g_plannerCall.dst == plan.dst && g_plannerCall.cu == plan.cu &&
        g_plannerCall.expertsToCopy == plan.expertsToCopy &&
        g_plannerCall.remoteStats == plan.remoteStats &&
        g_plannerCall.status == plan.status &&
        g_plannerCall.waitIterations == args.waitIterations &&
        g_plannerCall.stream == stream, "Planning V1 did not preserve Planner V2 arguments");
    Check(g_streamSynchronizeCalls == 0, "Planning V1 synchronized the caller stream");

    args.structSize -= 1;
    CheckStatus("planning struct size", TileXRMoonEpPlanningV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.structSize = sizeof(args);
    args.abiVersion = 2;
    CheckStatus("planning ABI", TileXRMoonEpPlanningV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    args.waitIterations = 0;
    CheckStatus("planning wait budget", TileXRMoonEpPlanningV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.waitIterations = 1;
    args.workspaceBytes = 255;
    CheckStatus("planning workspace", TileXRMoonEpPlanningV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.workspaceBytes = 256;
    topk.shape[1] = 3;
    CheckStatus("planning topk shape", TileXRMoonEpPlanningV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    topk.shape[1] = 2;
    tpe.dtype = TILEXR_MOONEP_DTYPE_FLOAT32;
    CheckStatus("planning tpe dtype", TileXRMoonEpPlanningV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    tpe.dtype = TILEXR_MOONEP_DTYPE_INT32;
    plan.dispatchedCapacity = 5;
    CheckStatus("planning NvS", TileXRMoonEpPlanningV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    plan.dispatchedCapacity = 4;
    g_plannerLaunchReturn = -88;
    CheckStatus("planner launch error", TileXRMoonEpPlanningV1(&args, stream), -88);
}

void TestStubEnqueueAndNoOp()
{
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 input = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x7000)),
        8, TILEXR_MOONEP_DTYPE_FLOAT16, 2, 4);
    TileXRMoonEpTensorV1 dispatched = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x8000)),
        16, TILEXR_MOONEP_DTYPE_FLOAT16, 4, 4);
    TileXRMoonEpTensorV1 sameShape = input;
    sameShape.data = reinterpret_cast<void *>(UINTPTR_C(0x8100));
    aclrtStream stream = reinterpret_cast<aclrtStream>(UINTPTR_C(0x9000));

    ResetFakes();
    TileXRMoonEpDispatchArgsV1 dispatch = DispatchArgs(&plan, &input, &dispatched);
    CheckStubEnqueue("dispatch", TileXRMoonEpDispatchV1(&dispatch, stream),
        input.data, dispatched.data, 16, 32, stream);

    TileXRMoonEpTensorV1 routeWeights = Tensor(
        reinterpret_cast<void *>(UINTPTR_C(0x8200)), 4,
        TILEXR_MOONEP_DTYPE_FLOAT32, 2, 2);
    TileXRMoonEpTensorV1 dispatchedWeights = Tensor(
        reinterpret_cast<void *>(UINTPTR_C(0x8300)), 4,
        TILEXR_MOONEP_DTYPE_FLOAT32, 4, 0);
    ResetFakes();
    dispatch = DispatchArgs(&plan, &routeWeights, &dispatchedWeights);
    CheckStubEnqueue("dispatch weights", TileXRMoonEpDispatchV1(&dispatch, stream),
        routeWeights.data, dispatchedWeights.data, 16, 16, stream);

    ResetFakes();
    TileXRMoonEpPrefetchWeightArgsV1 prefetch {};
    prefetch.structSize = sizeof(prefetch);
    prefetch.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    prefetch.comm = dispatch.comm;
    prefetch.plan = &plan;
    prefetch.input = &input;
    prefetch.output = &sameShape;
    CheckStubEnqueue("prefetch", TileXRMoonEpPrefetchWeightV1(&prefetch, stream),
        input.data, sameShape.data, 16, 16, stream);

    ResetFakes();
    TileXRMoonEpCombineArgsV1 combine {};
    combine.structSize = sizeof(combine);
    combine.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    combine.comm = dispatch.comm;
    combine.plan = &plan;
    combine.input = &dispatched;
    combine.output = &input;
    CheckStubEnqueue("combine", TileXRMoonEpCombineV1(&combine, stream),
        dispatched.data, input.data, 32, 16, stream);

    ResetFakes();
    combine.input = &dispatchedWeights;
    combine.output = &routeWeights;
    CheckStubEnqueue("combine weights", TileXRMoonEpCombineV1(&combine, stream),
        dispatchedWeights.data, routeWeights.data, 16, 16, stream);

    ResetFakes();
    TileXRMoonEpReduceGradArgsV1 reduce {};
    reduce.structSize = sizeof(reduce);
    reduce.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    reduce.comm = dispatch.comm;
    reduce.plan = &plan;
    reduce.input = &input;
    reduce.output = &sameShape;
    CheckStubEnqueue("reduce grad", TileXRMoonEpReduceGradV1(&reduce, stream),
        input.data, sameShape.data, 16, 16, stream);

    ResetFakes();
    sameShape = input;
    reduce.output = &sameShape;
    CheckStatus("exact in-place", TileXRMoonEpReduceGradV1(&reduce, stream),
        TILEXR_MOONEP_SUCCESS);
    Check(g_runtimeCalls.empty(), "exact in-place must not enqueue memset or copy");
    Check(g_streamSynchronizeCalls == 0, "exact in-place synchronized the stream");

    sameShape.elementCount += 1;
    sameShape.shape[1] += 1;
    CheckStatus("non-exact alias", TileXRMoonEpReduceGradV1(&reduce, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
}

void TestStubValidationAndRuntimeFailures()
{
    ResetFakes();
    TileXRMoonEpPlanV1 plan = ValidPlan();
    TileXRMoonEpTensorV1 input = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x7000)),
        8, TILEXR_MOONEP_DTYPE_FLOAT16, 2, 4);
    TileXRMoonEpTensorV1 output = Tensor(reinterpret_cast<void *>(UINTPTR_C(0x8000)),
        16, TILEXR_MOONEP_DTYPE_FLOAT16, 4, 4);
    TileXRMoonEpDispatchArgsV1 args = DispatchArgs(&plan, &input, &output);
    aclrtStream stream = reinterpret_cast<aclrtStream>(UINTPTR_C(0x9000));

    CheckStatus("null args", TileXRMoonEpDispatchV1(nullptr, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.structSize -= 1;
    CheckStatus("stub size", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.structSize = sizeof(args);
    args.abiVersion = 2;
    CheckStatus("stub ABI", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    args.structSize = sizeof(args) + 16;
    CheckStatus("larger stub size", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_SUCCESS);
    args.structSize = sizeof(args);
    ResetFakes();
    args.flags = 1;
    CheckStatus("stub flags", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.flags = 0;
    CheckStatus("null stream", TileXRMoonEpDispatchV1(&args, nullptr),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);

    plan.structSize += 16;
    input.structSize += 16;
    output.structSize += 16;
    CheckStatus("larger Plan/Tensor size", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_SUCCESS);
    plan.structSize = sizeof(plan);
    input.structSize = sizeof(input);
    output.structSize = sizeof(output);
    ResetFakes();

    plan.dispatchedCapacity = 5;
    CheckStatus("stub NvS", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    plan = ValidPlan();
    plan.abiVersion = 2;
    CheckStatus("stub plan ABI", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    plan = ValidPlan();
    plan.structSize -= 1;
    CheckStatus("stub plan size", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    plan = ValidPlan();
    plan.b = 1;
    CheckStatus("stub B", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    plan = ValidPlan();
    plan.dst = nullptr;
    CheckStatus("stub plan pointer", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    plan = ValidPlan();

    input.elementCount = 7;
    CheckStatus("stub element count", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    input.elementCount = 8;
    input.abiVersion = 2;
    CheckStatus("stub tensor ABI", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    input.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    input.structSize -= 1;
    CheckStatus("stub tensor size", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    input.structSize = sizeof(input);
    input.shape[2] = 1;
    CheckStatus("stub unused dimension", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    input.shape[2] = 0;
    input.dtype = 999;
    CheckStatus("stub dtype", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    input.dtype = TILEXR_MOONEP_DTYPE_FLOAT16;
    output.dtype = TILEXR_MOONEP_DTYPE_FLOAT32;
    CheckStatus("stub dtype mismatch", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    output.dtype = TILEXR_MOONEP_DTYPE_FLOAT16;

    g_commArgs.rank = 1;
    CheckStatus("comm rank mismatch", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    g_commArgs.rank = 0;
    g_commArgs.rankSize = 3;
    CheckStatus("comm world mismatch", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    g_commArgs.rankSize = 2;
    g_commArgsReturn = -77;
    CheckStatus("comm lookup failure", TileXRMoonEpDispatchV1(&args, stream), -77);
    g_commArgsReturn = TILEXR_MOONEP_SUCCESS;

    output.shape[0] = 3;
    output.elementCount = 12;
    CheckStatus("dispatch shape", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    output.shape[0] = 4;
    output.elementCount = 16;

    TileXRMoonEpCombineArgsV1 combine {};
    combine.structSize = sizeof(combine);
    combine.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    combine.comm = args.comm;
    combine.plan = &plan;
    combine.input = &input;
    combine.output = &output;
    CheckStatus("combine shape", TileXRMoonEpCombineV1(&combine, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);

    TileXRMoonEpPrefetchWeightArgsV1 prefetch {};
    prefetch.structSize = sizeof(prefetch);
    prefetch.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    prefetch.comm = args.comm;
    prefetch.plan = &plan;
    prefetch.input = &input;
    prefetch.output = &output;
    CheckStatus("prefetch shape", TileXRMoonEpPrefetchWeightV1(&prefetch, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);

    TileXRMoonEpReduceGradArgsV1 reduce {};
    reduce.structSize = sizeof(reduce);
    reduce.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    reduce.comm = args.comm;
    reduce.plan = &plan;
    reduce.input = &input;
    reduce.output = &output;
    CheckStatus("reduce grad shape", TileXRMoonEpReduceGradV1(&reduce, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);

    ResetFakes();
    g_memsetReturn = 9;
    CheckStatus("memset launch failure", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INTERNAL);
    Check(g_runtimeCalls.size() == 1 && g_runtimeCalls[0] == "memset",
        "memset failure must stop before memcpy");
    Check(g_streamSynchronizeCalls == 0, "memset failure synchronized the stream");

    ResetFakes();
    g_memcpyReturn = 10;
    CheckStatus("memcpy launch failure", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INTERNAL);
    Check(g_runtimeCalls.size() == 2 && g_runtimeCalls[1] == "memcpy",
        "memcpy failure call sequence mismatch");
    Check(g_streamSynchronizeCalls == 0, "memcpy failure synchronized the stream");

    plan = ValidPlan();
    plan.s = std::numeric_limits<int32_t>::max();
    plan.k = 2;
    plan.dispatchedCapacity = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) * 2;
    CheckStatus("plan encoding overflow", TileXRMoonEpDispatchV1(&args, stream),
        TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
}

} // namespace

extern "C" aclError aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value,
                                      size_t count, aclrtStream stream)
{
    g_runtimeCalls.push_back("memset");
    g_memsetCall = MemsetCall { devPtr, maxCount, value, count, stream };
    return g_memsetReturn;
}

extern "C" aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src,
                                      size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    g_runtimeCalls.push_back("memcpy");
    g_memcpyCall = MemcpyCall { dst, destMax, src, count, kind, stream };
    return g_memcpyReturn;
}

extern "C" aclError aclrtSynchronizeStream(aclrtStream)
{
    ++g_streamSynchronizeCalls;
    return ACL_SUCCESS;
}

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&commArgs)
{
    if (g_commArgsReturn != TILEXR_MOONEP_SUCCESS) {
        commArgs = nullptr;
        return g_commArgsReturn;
    }
    commArgs = &g_commArgs;
    return TILEXR_MOONEP_SUCCESS;
}

extern "C" int TileXRMoonEpPlannerGetWorkspaceSizeV2(TileXRCommPtr, int64_t s,
    int64_t k, int64_t, uint64_t *workspaceBytes, int64_t *dispatchedCapacity)
{
    ++g_plannerQueryCalls;
    if (g_plannerQueryReturn != TILEXR_MOONEP_SUCCESS) {
        return g_plannerQueryReturn;
    }
    *workspaceBytes = g_plannerWorkspaceBytes;
    *dispatchedCapacity = g_plannerCapacity >= 0 ? g_plannerCapacity : s * k;
    return TILEXR_MOONEP_SUCCESS;
}

extern "C" int TileXRMoonEpPlannerV2(const int32_t *topkExpertIds,
    const int32_t *tokensPerExpert, TileXRCommPtr comm, int64_t s, int64_t k,
    int64_t expertCount, void *workspace, uint64_t workspaceBytes, int32_t *dst,
    int32_t *cuSeqlens, int32_t *expertsToCopy, int32_t *remoteStats,
    int32_t *plannerStatus, uint64_t waitIterations, aclrtStream stream)
{
    ++g_plannerLaunchCalls;
    g_plannerCall = PlannerCall { topkExpertIds, tokensPerExpert, comm, s, k, expertCount,
        workspace, workspaceBytes, dst, cuSeqlens, expertsToCopy, remoteStats,
        plannerStatus, waitIterations, stream };
    return g_plannerLaunchReturn;
}

int main()
{
    TestCapabilities();
    TestWorkspaceQuery();
    TestPlanningDelegation();
    TestStubEnqueueAndNoOp();
    TestStubValidationAndRuntimeFailures();
    return g_failures == 0 ? 0 : 1;
}
