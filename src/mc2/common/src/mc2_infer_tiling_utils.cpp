// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include "tiling/mc2_tiling_utils.h"

#include <cstdlib>
#include <string>

#include "mc2_log.h"

namespace {
constexpr char HCCL_BUFFSIZE[] = "HCCL_BUFFSIZE";
constexpr uint16_t DEFAULT_WINDOW_SIZE_MB = 200;
}

namespace mc2tiling {

uint64_t Mc2TilingUtils::GetMaxWindowSize()
{
  uint16_t defaultWindowSize = DEFAULT_WINDOW_SIZE_MB;
  if (getenv(HCCL_BUFFSIZE) == nullptr) {
    OP_LOGD("", "Env HCCL_BUFFSIZE don't set");
  } else {
    try {
      std::string envStr(getenv(HCCL_BUFFSIZE));
      defaultWindowSize = static_cast<uint16_t>(std::stoi(envStr));
    } catch (...) {
      OP_LOGE("", "Unknown Exception encountered when parser env HCCL_BUFFERSIZE");
    }
  }
  const uint64_t maxWindowSize = static_cast<uint64_t>(defaultWindowSize) * 1024UL * 1024UL;
  OP_LOGI("", "Get maxWindowSize is %lu", maxWindowSize);
  return maxWindowSize;
}

}  // namespace mc2tiling
