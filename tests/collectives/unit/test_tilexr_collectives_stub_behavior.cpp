#include <iostream>

#include "tilexr_collectives.h"

namespace {

int g_failures = 0;

struct StubCase {
    const char *name;
    void *sendBuf;
    void *recvBuf;
    int64_t sendCount;
    TileXR::TileXRDataType dataType;
};

void CheckStatus(const char *apiName, const StubCase& testCase, int status)
{
    if (status != TileXR::TILEXR_ERROR_INTERNAL) {
        std::cerr << apiName << " case \"" << testCase.name << "\" returned " << status
                  << ", expected " << TileXR::TILEXR_ERROR_INTERNAL << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    uint8_t sendStorage[4096] = {};
    uint8_t recvStorage[4096] = {};

    const StubCase cases[] = {
        { "all nulls, count 0, int32", nullptr, nullptr, 0, TileXR::TILEXR_DATA_TYPE_INT32 },
        { "dummy buffers, count 1, int32", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32 },
        { "dummy buffers, count -1, int32", sendStorage, recvStorage, -1, TileXR::TILEXR_DATA_TYPE_INT32 },
        { "dummy buffers, count 1024, invalid datatype",
          sendStorage, recvStorage, 1024, static_cast<TileXR::TileXRDataType>(999) },
        { "send null, recv dummy, count 16, fp16", nullptr, recvStorage, 16, TileXR::TILEXR_DATA_TYPE_FP16 },
        { "send dummy, recv null, count 16, fp16", sendStorage, nullptr, 16, TileXR::TILEXR_DATA_TYPE_FP16 },
    };

    for (const auto& testCase : cases) {
        CheckStatus("TileXRAllGather", testCase,
                    TileXRAllGather(testCase.sendBuf, testCase.recvBuf, testCase.sendCount,
                                    testCase.dataType, nullptr, nullptr));
        CheckStatus("TileXRAllToAll", testCase,
                    TileXRAllToAll(testCase.sendBuf, testCase.recvBuf, testCase.sendCount,
                                   testCase.dataType, nullptr, nullptr));
    }
    return g_failures == 0 ? 0 : 1;
}
