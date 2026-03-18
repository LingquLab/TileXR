/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TILEXR_CCL_KERNEL_ARGS_H
#define TILEXR_CCL_KERNEL_ARGS_H

#include "../include/tilexr_types.h"
#include "../include/comm_args.h"

namespace TileXR {
struct AscendCCLKernelArgs {
    const void *input = nullptr;  // input
    const void *output = nullptr; // output
    const void *commArgsPtr = nullptr;
    int64_t count = 0; // attr 数据长度
    int64_t magic = 0;   // attr 自增参数
    int op = 0;
    int root = 0;
    const void *scale = nullptr; // scale
    int64_t scaleCount = 0; // scale 数据长度
    const void *offset = nullptr; // offset
};

struct CCLGatherArgs {
    const void *embTable = nullptr; // emb表首地址
    const void *lookup = nullptr;   // lookup首地址
    const void *revData = nullptr;  // 输出连续地址
    int64_t lookupLen = 0;
    int64_t embTableLen = 0;
    int64_t embTableDim = 0;
};

}

#endif // TILEXR_CCL_KERNEL_ARGS_H
