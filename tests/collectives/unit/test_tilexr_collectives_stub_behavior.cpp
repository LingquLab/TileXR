#include <cstddef>
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

struct BroadcastValidationCase {
    const char *name;
    void *buf;
    int64_t count;
    TileXR::TileXRDataType dataType;
    int root;
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

    for (const auto& testCase : cases) {
        CheckStatus("TileXRAllReduce", testCase,
                    TileXRAllReduce(testCase.sendBuf, testCase.recvBuf, testCase.sendCount,
                                    testCase.dataType, TileXR::TILEXR_REDUCE_SUM,
                                    testCase.comm, nullptr));
        CheckStatus("TileXRReduceScatter", testCase,
                    TileXRReduceScatter(testCase.sendBuf, testCase.recvBuf, testCase.sendCount,
                                        testCase.dataType, TileXR::TILEXR_REDUCE_SUM,
                                        testCase.comm, nullptr));
    }

    const ValidationCase unsupportedReduceOps[] = {
        { "prod reduce op", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm },
        { "max reduce op", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm },
        { "min reduce op", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm },
        { "reserved reduce op", sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm },
    };
    const TileXR::TileXRReduceOp reduceOps[] = {
        TileXR::TILEXR_REDUCE_PROD,
        TileXR::TILEXR_REDUCE_MAX,
        TileXR::TILEXR_REDUCE_MIN,
        TileXR::TILEXR_REDUCE_RESERVED,
    };
    for (size_t i = 0; i < sizeof(reduceOps) / sizeof(reduceOps[0]); ++i) {
        CheckStatus("TileXRAllReduce", unsupportedReduceOps[i],
                    TileXRAllReduce(unsupportedReduceOps[i].sendBuf, unsupportedReduceOps[i].recvBuf,
                                    unsupportedReduceOps[i].sendCount, unsupportedReduceOps[i].dataType,
                                    reduceOps[i], unsupportedReduceOps[i].comm, nullptr));
        CheckStatus("TileXRReduceScatter", unsupportedReduceOps[i],
                    TileXRReduceScatter(unsupportedReduceOps[i].sendBuf, unsupportedReduceOps[i].recvBuf,
                                        unsupportedReduceOps[i].sendCount, unsupportedReduceOps[i].dataType,
                                        reduceOps[i], unsupportedReduceOps[i].comm, nullptr));
    }

    const BroadcastValidationCase broadcastCases[] = {
        { "null comm", recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, 0, nullptr },
        { "buffer null", nullptr, 1, TileXR::TILEXR_DATA_TYPE_FP16, 0, comm },
        { "zero count", recvStorage, 0, TileXR::TILEXR_DATA_TYPE_INT32, 0, comm },
        { "negative count", recvStorage, -1, TileXR::TILEXR_DATA_TYPE_INT32, 0, comm },
        { "negative root", recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, -1, comm },
        { "root past rank size", recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, 1, comm },
        { "unsupported uint8 datatype", recvStorage, 1024, TileXR::TILEXR_DATA_TYPE_UINT8, 0, comm },
        { "unknown datatype", recvStorage, 1024, static_cast<TileXR::TileXRDataType>(999), 0, comm },
    };
    for (const auto& testCase : broadcastCases) {
        ValidationCase adapted { testCase.name, testCase.buf, testCase.buf, testCase.count,
            testCase.dataType, testCase.comm };
        CheckStatus("TileXRBroadcast", adapted,
                    TileXRBroadcast(testCase.buf, testCase.count, testCase.dataType,
                                    testCase.root, testCase.comm, nullptr));
    }

    CheckSetupStatus("TileXRCommDestroy(comm)",
                     TileXRCommDestroy(comm),
                     TileXR::TILEXR_SUCCESS);
    return g_failures == 0 ? 0 : 1;
}
