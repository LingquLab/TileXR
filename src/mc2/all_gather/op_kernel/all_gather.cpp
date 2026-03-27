/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_add.cpp
 * \brief
 */
#include "all_gather.h"
#include "kernel_operator.h"

using namespace AscendC;

template <uint32_t dType>
__global__ __aicore__ void all_gather(GM_ADDR aGM, GM_ADDR gatherGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    // 注册算子Tiling结构体
    REGISTER_TILING_DEFAULT(AllGatherTilingData);
    auto tiling = (__gm__ AllGatherTilingData*)tilingGM;
    GET_TILING_DATA(tilingData, tilingGM);

    TPipe pipe;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
    if (dType == DTYPE_TPL_FP16) {
        AllGather<half> allGather;
        allGather.Init(aGM, gatherGM, workspaceGM, contextGM, &tilingData, &pipe);
        allGather.Process();
    } else if (dType == DTYPE_TPL_FP32) {
        AllGather<float> allGather;
        allGather.Init(aGM, gatherGM, workspaceGM, contextGM, &tilingData, &pipe);
        allGather.Process();
    }
}