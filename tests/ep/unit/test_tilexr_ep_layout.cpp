#include <cstdint>
#include <iostream>

#include "ep_layout.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

void CheckInt64(const char *label, int64_t actual, int64_t expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckInt(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckBool(const char *label, bool actual, bool expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void TestExpertMapping()
{
    CheckInt("expert 0 dst", TileXREp::TileXREpDstRank(0, 4), 0);
    CheckInt("expert 0 local", TileXREp::TileXREpLocalExpert(0, 4), 0);
    CheckInt("expert 3 dst", TileXREp::TileXREpDstRank(3, 4), 0);
    CheckInt("expert 3 local", TileXREp::TileXREpLocalExpert(3, 4), 3);
    CheckInt("expert 4 dst", TileXREp::TileXREpDstRank(4, 4), 1);
    CheckInt("expert 4 local", TileXREp::TileXREpLocalExpert(4, 4), 0);
    CheckInt("expert 7 dst", TileXREp::TileXREpDstRank(7, 4), 1);
    CheckInt("expert 7 local", TileXREp::TileXREpLocalExpert(7, 4), 3);
    CheckInt("negative expert dst", TileXREp::TileXREpDstRank(-1, 4), TileXR::TILEXR_INVALID_VALUE);
    CheckInt("zero local expert dst", TileXREp::TileXREpDstRank(1, 0), TileXR::TILEXR_INVALID_VALUE);
}

void TestDataTypes()
{
    CheckBool("fp16 supported", TileXREp::TileXREpIsSupportedDataType(TileXR::TILEXR_DATA_TYPE_FP16), true);
    CheckBool("bf16 supported", TileXREp::TileXREpIsSupportedDataType(TileXR::TILEXR_DATA_TYPE_BFP16), true);
    CheckBool("fp32 unsupported", TileXREp::TileXREpIsSupportedDataType(TileXR::TILEXR_DATA_TYPE_FP32), false);
    CheckInt64("fp16 bytes", TileXREp::TileXREpDataTypeSize(TileXR::TILEXR_DATA_TYPE_FP16), 2);
    CheckInt64("bf16 bytes", TileXREp::TileXREpDataTypeSize(TileXR::TILEXR_DATA_TYPE_BFP16), 2);
    CheckInt64("int32 bytes invalid", TileXREp::TileXREpDataTypeSize(TileXR::TILEXR_DATA_TYPE_INT32),
        TileXR::TILEXR_INVALID_VALUE);
}

void TestWindowConfig()
{
    TileXREp::EpWindowConfig config {};
    const int ret = TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 8, TileXR::TILEXR_DATA_TYPE_FP16, &config);
    CheckInt("valid config ret", ret, TileXR::TILEXR_SUCCESS);
    CheckInt64("rank size", config.rankSize, 2);
    CheckInt64("bs", config.bs, 4);
    CheckInt64("hidden size", config.h, 8);
    CheckInt64("topk", config.topK, 2);
    CheckInt64("moe experts", config.moeExpertNum, 8);
    CheckInt64("local experts", config.localExpertNum, 4);
    CheckInt64("dtype bytes", config.dtypeBytes, 2);
    CheckInt64("max routes", config.maxRoutesPerSrc, 8);
    CheckInt64("row bytes", config.rowBytes, 16);
    CheckInt64("payload bytes", config.payloadBytesPerSlot, 128);
    CheckInt64("assist bytes", config.assistBytesPerSlot, 128);
    CheckInt64("slot bytes", config.slotBytes, 320);
    CheckInt64("total bytes", config.totalBytes, 704);
}

void TestRejectsInvalidConfig()
{
    TileXREp::EpWindowConfig config {};
    CheckInt("null out", TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 8, TileXR::TILEXR_DATA_TYPE_FP16, nullptr),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("non-divisible experts", TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 7, TileXR::TILEXR_DATA_TYPE_FP16, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("unsupported dtype", TileXREp::TileXREpBuildWindowConfig(
        2, 4, 8, 2, 8, TileXR::TILEXR_DATA_TYPE_FP32, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckInt("oversized window", TileXREp::TileXREpBuildWindowConfig(
        2, 1024 * 1024, 64, 8, 8, TileXR::TILEXR_DATA_TYPE_FP16, &config),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestExpertMapping();
    TestDataTypes();
    TestWindowConfig();
    TestRejectsInvalidConfig();
    return g_failures == 0 ? 0 : 1;
}
