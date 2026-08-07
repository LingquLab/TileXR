#include <stdint.h>

#include "tilexr_moonep.h"

int main(void)
{
    TileXRMoonEpTensorV1 tensor = {0};
    TileXRMoonEpPlanV1 plan = {0};
    TileXRMoonEpDispatchArgsV1 dispatch = {0};
    TileXRMoonEpPrefetchWeightArgsV1 prefetch = {0};
    TileXRMoonEpCombineArgsV1 combine = {0};
    TileXRMoonEpReduceGradArgsV1 reduce = {0};

    tensor.structSize = (uint32_t)sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.dtype = TILEXR_MOONEP_DTYPE_FLOAT16;
    plan.structSize = (uint32_t)sizeof(plan);
    plan.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    plan.n = 1;
    plan.nvS = 1;
    dispatch.structSize = (uint32_t)sizeof(dispatch);
    dispatch.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    dispatch.hiddenSh = &tensor;
    prefetch.fullGateWeight = &tensor;
    combine.hiddenNvsh = &tensor;
    reduce.fullGateGrad = &tensor;

    return tensor.dtype == TILEXR_MOONEP_DTYPE_FLOAT16 &&
        plan.nvS == 1 && dispatch.hiddenSh == &tensor &&
        prefetch.fullGateWeight == &tensor && combine.hiddenNvsh == &tensor &&
        reduce.fullGateGrad == &tensor ? 0 : 1;
}
