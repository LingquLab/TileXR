/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "tilexr_internal.h"
#include <string>
#include <unordered_map>
#include "tilexr_log.h"

#include "runtime/dev.h"

constexpr int AI_CORE_NUM_24 = 24;
constexpr int AI_CORE_NUM_20 = 20;
constexpr int AI_CORE_NUM_2 = 2;

using namespace std;

namespace TileXR {
const std::unordered_map<std::string, ChipName> CHIP_MAP = {
    {"Ascend310P", ChipName::CHIP_310P3},
    {"Ascend910B1", ChipName::CHIP_910B1},
    {"Ascend910B2", ChipName::CHIP_910B2},
    {"Ascend910B2C", ChipName::CHIP_910B2C},
    {"Ascend910B3", ChipName::CHIP_910B3},
    {"Ascend910B4", ChipName::CHIP_910B4},
    {"Ascend910B4-1", ChipName::CHIP_910B41},
    {"Ascend910_9391", ChipName::CHIP_910_9391},
    {"Ascend910_9381", ChipName::CHIP_910_9381},
    {"Ascend910_9392", ChipName::CHIP_910_9392},
    {"Ascend910_9382", ChipName::CHIP_910_9382},
    {"Ascend910_9372", ChipName::CHIP_910_9372},
    {"Ascend910_9361", ChipName::CHIP_910_9361},
    {"Ascend950", ChipName::CHIP_950},
    {"Ascend950DT", ChipName::CHIP_950},
    {"Ascend950DT_9581", ChipName::CHIP_950},
    {"Ascend950DT_9582", ChipName::CHIP_950},
    {"Ascend950DT_9584", ChipName::CHIP_950},
    {"Ascend950PR", ChipName::CHIP_950PR},
    {"Ascend950PR_9589", ChipName::CHIP_950PR},
    {"Ascend950PR_9599", ChipName::CHIP_950PR}
};

const std::unordered_map<std::string, ChipName> CHIP_PREFIX_MAP = {
    {"Ascend950DT_", ChipName::CHIP_950},
    {"Ascend950PR_", ChipName::CHIP_950PR}
};

ChipName ResolveChipNameFromSocVersion(const std::string& chipName)
{
    auto it = CHIP_MAP.find(chipName);
    if (it != CHIP_MAP.end()) {
        return it->second;
    }
    for (const auto& item : CHIP_PREFIX_MAP) {
        if (chipName.find(item.first) == 0) {
            return item.second;
        }
    }
    return ChipName::RESERVED;
}

/**
 * @brief 用于获取芯片名称
 */
ChipName GetChipName()
{
    // 在分配内存时用到
    static ChipName curChipName = ChipName::RESERVED;
    if (curChipName != ChipName::RESERVED) {
        return curChipName;
    }
    constexpr int socVerLength = 100; // asd没有相应的宏和常量，这里和asd测试代码中的长度保持一致
    char ver[socVerLength];
    auto ret = rtGetSocVersion(ver, socVerLength);
    if (ret != RT_ERROR_NONE) {
        TILEXR_LOG(ERROR) << "rtGetSocVersion failed, not sure whether the function is normal, please use it with caution";
        return ChipName::RESERVED;
    }
    string chipName(ver);
    TILEXR_LOG(DEBUG) << "rtGetSocVersion -- The result after converting ver to string is:" << chipName;

    curChipName = ResolveChipNameFromSocVersion(chipName);
    if (curChipName == ChipName::RESERVED) {
        TILEXR_LOG(WARN) << "There is no commitment to the supported chip types yet," <<
                      " and it is not certain whether the functions will work properly.";
    }
    return curChipName;
}

bool UseLegacyIpcPid(ChipName chipName)
{
    return chipName < ChipName::CHIP_910_9391 || chipName == ChipName::CHIP_950PR;
}

uint32_t GetCoreNum(ChipName chipName)
{
    switch (chipName) {
        case ChipName::CHIP_910B1:
        case ChipName::CHIP_910B2:
        case ChipName::CHIP_910_9391:
        case ChipName::CHIP_910_9381:
        case ChipName::CHIP_910_9392:
        case ChipName::CHIP_910_9382:
        case ChipName::CHIP_910B2C:
        case ChipName::CHIP_950PR:
        case ChipName::CHIP_950:
            return AI_CORE_NUM_24;
        case ChipName::CHIP_910B3:
        case ChipName::CHIP_910B4:
        case ChipName::CHIP_910B41:
        case ChipName::CHIP_910_9372:
        case ChipName::CHIP_910_9361:
        case ChipName::CHIP_910A5:
            return AI_CORE_NUM_20;
        case ChipName::CHIP_310P3:
            return AI_CORE_NUM_2;
        default:
            TILEXR_LOG(ERROR) << "Unknown chip name";
            return 0;
    }
}
}
