/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILEXR_INTERNAL_H
#define TILEXR_INTERNAL_H

#include <string>
#include <unordered_map>
#include <acl/acl_base.h>
#include "../include/tilexr_types.h"
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

namespace TileXR {
// Common functions
int RegistKernel(const bool enableProfiling = false);

int64_t Count2Size(int64_t count, const HcclDataType &dataType);

int LoadMTE(TileXRType cclType, AscendCCLKernelArgs &args, uint32_t blockDim, HcclDataType dataType, aclrtStream stream);

ChipName GetChipName();

uint32_t GetCoreNum(ChipName chipName);
} // namespace TileXR
#endif // TILEXR_INTERNAL_H
