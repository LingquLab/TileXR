/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_tiling_key.h
 * \brief all_gather_tiling key declare
 */

#ifndef __ALL_GATHER_TILING_KEY_H__
#define __ALL_GATHER_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define DTYPE_TPL_FP16 0
#define DTYPE_TPL_FP32 1

ASCENDC_TPL_ARGS_DECL(AllGather,
    ASCENDC_TPL_DTYPE_DECL(dType, DTYPE_TPL_FP16, DTYPE_TPL_FP32)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, DTYPE_TPL_FP16)),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(dType, DTYPE_TPL_FP32))
);

#endif