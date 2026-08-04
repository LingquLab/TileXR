#include <stdint.h>

#include "tilexr_moonep.h"

int main(void)
{
    TileXRMoonEpTensorV1 tensor = {0};
    TileXRMoonEpPlanV1 plan = {0};
    TileXRMoonEpDispatchArgsV1 dispatch = {0};

    tensor.structSize = (uint32_t)sizeof(tensor);
    tensor.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    tensor.dtype = TILEXR_MOONEP_DTYPE_FLOAT16;
    plan.structSize = (uint32_t)sizeof(plan);
    plan.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;
    plan.dispatchedCapacity = 1;
    dispatch.structSize = (uint32_t)sizeof(dispatch);
    dispatch.abiVersion = TILEXR_MOONEP_ABI_VERSION_V1;

    return tensor.dtype == TILEXR_MOONEP_DTYPE_FLOAT16 &&
        plan.dispatchedCapacity == 1 && dispatch.structSize == sizeof(dispatch) ? 0 : 1;
}
