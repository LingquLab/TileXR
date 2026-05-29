#include <cstdint>
#include <iostream>

#include "tilexr_api.h"
#include "tilexr_collectives.h"

namespace {

int g_failures = 0;

struct ValidationCase {
    const char *name;
    void *sendBuf;
    void *recvBuf;
    int64_t sendCount;
    TileXR::TileXRDataType dataType;
    TileXRCommPtr comm;
};

void CheckStatus(const char *apiName, const ValidationCase& testCase, int status)
{
    if (status != TileXR::TILEXR_ERROR_PARA_CHECK_FAIL) {
        std::cerr << apiName << " case \"" << testCase.name << "\" returned " << status
                  << ", expected " << TileXR::TILEXR_ERROR_PARA_CHECK_FAIL << std::endl;
        ++g_failures;
    }
}

void CheckSetupStatus(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " returned " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    uint8_t sendStorage[4096] = {};
    uint8_t recvStorage[4096] = {};

    TileXRCommPtr comm = nullptr;
    CheckSetupStatus("TileXRCommInit(0, 1, &comm)",
                     TileXRCommInit(0, 1, &comm),
                     TileXR::TILEXR_SUCCESS);
    if (comm == nullptr) {
        std::cerr << "TileXRCommInit returned success but left comm null" << std::endl;
        return 1;
    }

    const ValidationCase cases[] = {
        { "null comm", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, nullptr },
        { "send null", nullptr, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_FP16, comm },
        { "recv null", sendStorage, nullptr, 1, TileXR::TILEXR_DATA_TYPE_FP16, comm },
        { "zero count", sendStorage, recvStorage, 0, TileXR::TILEXR_DATA_TYPE_INT32, comm },
        { "negative count", sendStorage, recvStorage, -1, TileXR::TILEXR_DATA_TYPE_INT32, comm },
        { "unsupported uint8 datatype", sendStorage, recvStorage, 1024, TileXR::TILEXR_DATA_TYPE_UINT8, comm },
        { "unknown datatype", sendStorage, recvStorage, 1024, static_cast<TileXR::TileXRDataType>(999), comm },
    };

    for (const auto& testCase : cases) {
        CheckStatus("TileXRAllGather", testCase,
                    TileXRAllGather(testCase.sendBuf, testCase.recvBuf, testCase.sendCount,
                                    testCase.dataType, testCase.comm, nullptr));
        CheckStatus("TileXRAllToAll", testCase,
                    TileXRAllToAll(testCase.sendBuf, testCase.recvBuf, testCase.sendCount,
                                   testCase.dataType, testCase.comm, nullptr));
    }

    CheckSetupStatus("TileXRCommDestroy(comm)",
                     TileXRCommDestroy(comm),
                     TileXR::TILEXR_SUCCESS);
    return g_failures == 0 ? 0 : 1;
}
