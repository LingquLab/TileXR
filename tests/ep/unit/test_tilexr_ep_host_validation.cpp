#include <cstdint>
#include <iostream>

#include "comm_args.h"
#include "ep_dispatch_host.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

void CheckInt(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

TileXREp::EpDispatchParams ValidParams()
{
    static uint16_t x[32] = {};
    static int32_t expertIds[8] = {};
    static uint16_t expandXOut[64] = {};
    static int64_t expertTokenNumsOut[4] = {};
    static int32_t epRecvCountsOut[2] = {};
    static int32_t assistInfoForCombineOut[32] = {};

    TileXREp::EpDispatchParams params {};
    params.x = x;
    params.expertIds = expertIds;
    params.comm = reinterpret_cast<TileXRCommPtr>(0x1000);
    params.bs = 4;
    params.h = 8;
    params.topK = 2;
    params.moeExpertNum = 8;
    params.expandXOut = expandXOut;
    params.expertTokenNumsOut = expertTokenNumsOut;
    params.epRecvCountsOut = epRecvCountsOut;
    params.assistInfoForCombineOut = assistInfoForCombineOut;
    params.dtype = TileXR::TILEXR_DATA_TYPE_FP16;
    params.stream = reinterpret_cast<aclrtStream>(0x2000);
    return params;
}

TileXR::CommArgs ValidCommArgs()
{
    TileXR::CommArgs args {};
    args.rank = 0;
    args.rankSize = 2;
    args.peerMems[0] = reinterpret_cast<GM_ADDR>(0x10000000);
    args.peerMems[1] = reinterpret_cast<GM_ADDR>(0x20000000);
    return args;
}

void TestBasicValidation()
{
    TileXREp::EpDispatchParams params = ValidParams();
    CheckInt("valid basic params", TileXREp::TileXREpValidateBasicDispatchParams(params), TileXR::TILEXR_SUCCESS);

    params = ValidParams();
    params.x = nullptr;
    CheckInt("null x", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidParams();
    params.comm = nullptr;
    CheckInt("null comm", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidParams();
    params.bs = 0;
    CheckInt("zero bs", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    params = ValidParams();
    params.dtype = TileXR::TILEXR_DATA_TYPE_FP32;
    CheckInt("unsupported fp32", TileXREp::TileXREpValidateBasicDispatchParams(params),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

void TestCommValidation()
{
    TileXREp::EpDispatchParams params = ValidParams();
    TileXR::CommArgs commArgs = ValidCommArgs();
    TileXREp::EpWindowConfig window {};
    CheckInt("valid dispatch config", TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window),
        TileXR::TILEXR_SUCCESS);

    commArgs = ValidCommArgs();
    commArgs.rankSize = 0;
    CheckInt("rank size zero", TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    commArgs = ValidCommArgs();
    commArgs.peerMems[1] = nullptr;
    CheckInt("missing peer mem", TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window),
        TileXR::TILEXR_ERROR_NOT_INITIALIZED);

    commArgs = ValidCommArgs();
    params = ValidParams();
    params.moeExpertNum = 7;
    CheckInt("non-divisible experts", TileXREp::TileXREpValidateDispatchConfig(params, commArgs, &window),
        TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestBasicValidation();
    TestCommValidation();
    return g_failures == 0 ? 0 : 1;
}
