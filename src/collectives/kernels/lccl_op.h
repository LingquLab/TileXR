/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEXR_LCCL_OP_H
#define TILEXR_LCCL_OP_H

#if defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)

#include "op_def.h"
#include "allgather.h"
#include "91093/allgather_hierarchy_double_ring.h"
#include "91093/all2all_hierarchy.h"
#include "91093/all2all_hierarchy_small.h"

#include "kernels/lcal_allgather_910B2C.cce"
#include "kernels/lcal_allgather_big_data_910B2C.cce"
#include "kernels/lcal_allgather_2npu.cce"
#include "kernels/lcal_allgather_2npu_big_data_write.cce"
#include "kernels/lcal_allgather.cce"
#include "kernels/lcal_allgather_big_data.cce"
#include "kernels/lcal_all2all_transpose.cce"

extern "C" __global__ __aicore__ __attribute__((section("Attr_Section_TileXR"))) void TileXRDescriptor() {}

#define LCCL_ALLGATHER_FUNC_AUTO_DEF(type, suffix) \
extern "C" __global__ __aicore__ void TileXRAllGather_##type##suffix(KERNELS_ARGS_FUN()) { \
    if ASCEND_IS_AIV { \
    GET_COMM_ARGS; \
    constexpr int32_t quickOneshotRankSize = 2; \
    constexpr int32_t cceSmallDataSize = 2 * 1024 * 1024; \
    constexpr int32_t smallRankSize = 8; \
    constexpr int32_t smallDataSize910a3 = 32 * 1024 * 1024; \
    __gm__ type * shareAddrs[TILEXR_MAX_RANK_SIZE]; \
    GET_IPC_MEM_ARGS(type); \
    if ((extraFlag & ExtraFlag::TOPO_910B2C) != 0 && rankSize > smallRankSize) { \
        if (len * sizeof(type) < cceSmallDataSize) { \
            TileXRAllGather910B2C<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            TileXRAllGatherBigData910B2C<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } else if ((extraFlag & ExtraFlag::TOPO_PCIE) != 0) { \
        TileXRAllGather2npuBigDataWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
    } else if ((extraFlag & ExtraFlag::TOPO_910_93) != 0 && lcalBlockNum != rankSize && \
        (len > smallDataSize910a3 / sizeof(type) || rankSize > smallRankSize) && \
        rankSize > quickOneshotRankSize && rankSize % quickOneshotRankSize == 0) { \
        CLASS_OP_LAUNCH(AllGatherHierarchyDoubleRing, type); \
    } else { \
        if (rankSize == quickOneshotRankSize && len * sizeof(type) < SIZE_OF_8M && lcalBlockNum != rankSize) { \
            TileXRAllGather2npu<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else if (rankSize == quickOneshotRankSize && lcalBlockNum != rankSize) { \
            TileXRAllGather2npuBigDataWrite<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else if ((rankSize > quickOneshotRankSize && len * sizeof(type) < cceSmallDataSize) || \
            lcalBlockNum == rankSize) { \
            TileXRAllGather<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } else { \
            TileXRAllGatherBigData<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
        } \
    } \
    } \
}

#define LCCL_ALL2ALL_FUNC_AUTO_DEF(type, suffix) \
extern "C" __global__ __aicore__ void TileXRAll2All_##type##suffix(KERNELS_ARGS_FUN()) { \
    if ASCEND_IS_AIV { \
    GET_COMM_ARGS; \
    __gm__ type * shareAddrs[TILEXR_MAX_RANK_SIZE]; \
    GET_IPC_MEM_ARGS(type); \
    constexpr int32_t smallRankSize = 8; \
    if (op != 0 && root != 0) { \
        TileXRAll2AllTranspose<type>(ALLREDUCE_ARGS_CALL_16P(type)); \
    } else if ((extraFlag & ExtraFlag::TOPO_910_93) != 0) { \
        if (rankSize <= smallRankSize && len * sizeof(type) > SMALL_DATA_SIZE && \
            (len * sizeof(type)) % (smallRankSize * smallRankSize * rankSize) == 0) { \
            CLASS_OP_LAUNCH(All2AllHierarchySmall, type); \
        } else { \
            CLASS_OP_LAUNCH(All2AllHierarchy, type); \
        } \
    } \
    } \
}

#endif
#endif
