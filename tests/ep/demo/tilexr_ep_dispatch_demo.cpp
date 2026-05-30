#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "tilexr_api.h"
#include "tilexr_ep.h"
#include "tilexr_types.h"

namespace {

constexpr int64_t kBs = 4;
constexpr int64_t kH = 8;
constexpr int64_t kTopK = 2;
constexpr int64_t kMoeExpertNum = 4;
constexpr int64_t kRankSize = 2;
constexpr int64_t kRoutes = kBs * kTopK;
constexpr int64_t kXElements = kBs * kH;
constexpr int64_t kExpandedElements = kRoutes * kH;
constexpr int64_t kLocalExpertNum = kMoeExpertNum / kRankSize;
constexpr int64_t kAssistInts = 4;

int GetEnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

std::vector<int> ParseDeviceList(const char *value)
{
    std::vector<int> devices;
    if (value == nullptr || value[0] == '\0') {
        return devices;
    }

    std::string text(value);
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string part = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (!part.empty()) {
            devices.push_back(std::atoi(part.c_str()));
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return devices;
}

int GetDeviceIdFromEnv(int rank, int npuCount, int firstNpu)
{
    const std::vector<int> devices = ParseDeviceList(std::getenv("TILEXR_DEMO_DEVICES"));
    if (!devices.empty()) {
        return devices[rank % static_cast<int>(devices.size())];
    }
    return firstNpu + rank % std::max(npuCount, 1);
}

bool CheckAcl(aclError ret, const std::string &what)
{
    if (ret != ACL_SUCCESS) {
        std::cerr << what << " failed, acl ret=" << ret << std::endl;
        return false;
    }
    return true;
}

bool CheckTileXR(int ret, const std::string &what)
{
    if (ret != TileXR::TILEXR_SUCCESS) {
        std::cerr << what << " failed, TileXR ret=" << ret << std::endl;
        return false;
    }
    return true;
}

uint16_t XValue(int rank, int64_t token, int64_t h)
{
    return static_cast<uint16_t>(0x1000 + rank * 0x100 + token * 0x10 + h);
}

std::vector<int32_t> ExpertIds()
{
    return std::vector<int32_t> {0, 2, 1, 3, 2, 0, 3, 1};
}

bool RouteBelongsToRank(int32_t expertId, int rank)
{
    return expertId / static_cast<int32_t>(kLocalExpertNum) == rank;
}

struct ExpectedRoute {
    int32_t srcRank;
    int32_t tokenId;
    int32_t topKId;
    int32_t expertId;
};

std::vector<ExpectedRoute> BuildExpectedRoutes(int rank, int rankSize)
{
    const std::vector<int32_t> expertIds = ExpertIds();
    std::vector<ExpectedRoute> expected;
    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        for (int64_t token = 0; token < kBs; ++token) {
            for (int64_t topKId = 0; topKId < kTopK; ++topKId) {
                const int64_t route = token * kTopK + topKId;
                const int32_t expertId = expertIds[route];
                if (RouteBelongsToRank(expertId, rank)) {
                    expected.push_back(ExpectedRoute {srcRank, static_cast<int32_t>(token),
                        static_cast<int32_t>(topKId), expertId});
                }
            }
        }
    }
    return expected;
}

bool ValidateOutputs(int rank, int rankSize, const std::vector<uint16_t> &expandX,
    const std::vector<int64_t> &expertTokenNums, const std::vector<int32_t> &recvCounts,
    const std::vector<int32_t> &assist)
{
    const std::vector<ExpectedRoute> expected = BuildExpectedRoutes(rank, rankSize);
    int64_t expectedRecv[kRankSize] = {};
    int64_t expectedExpertCounts[kLocalExpertNum] = {};
    for (const ExpectedRoute &route : expected) {
        ++expectedRecv[route.srcRank];
        ++expectedExpertCounts[route.expertId % kLocalExpertNum];
    }

    for (int srcRank = 0; srcRank < rankSize; ++srcRank) {
        if (recvCounts[srcRank] != expectedRecv[srcRank]) {
            std::cerr << "rank " << rank << " recvCounts[" << srcRank << "] expected "
                      << expectedRecv[srcRank] << " got " << recvCounts[srcRank] << std::endl;
            return false;
        }
    }

    for (int localExpert = 0; localExpert < kLocalExpertNum; ++localExpert) {
        if (expertTokenNums[localExpert] != expectedExpertCounts[localExpert]) {
            std::cerr << "rank " << rank << " expertTokenNums[" << localExpert << "] expected "
                      << expectedExpertCounts[localExpert] << " got " << expertTokenNums[localExpert] << std::endl;
            return false;
        }
    }

    for (std::size_t row = 0; row < expected.size(); ++row) {
        const ExpectedRoute &route = expected[row];
        const std::size_t assistOffset = row * kAssistInts;
        if (assist[assistOffset] != route.srcRank || assist[assistOffset + 1] != route.tokenId ||
            assist[assistOffset + 2] != route.topKId || assist[assistOffset + 3] != route.expertId) {
            std::cerr << "rank " << rank << " assist row " << row << " mismatch: got {"
                      << assist[assistOffset] << ", " << assist[assistOffset + 1] << ", "
                      << assist[assistOffset + 2] << ", " << assist[assistOffset + 3] << "}" << std::endl;
            return false;
        }

        for (int64_t h = 0; h < kH; ++h) {
            const uint16_t expectedValue = XValue(route.srcRank, route.tokenId, h);
            const uint16_t actualValue = expandX[row * kH + h];
            if (actualValue != expectedValue) {
                std::cerr << "rank " << rank << " expandX[" << row << "][" << h << "] expected 0x"
                          << std::hex << expectedValue << " got 0x" << actualValue << std::dec << std::endl;
                return false;
            }
        }
    }

    return true;
}

void Cleanup(TileXRCommPtr comm, aclrtStream stream, int deviceId, bool deviceSet, bool aclReady,
    std::vector<void *> &buffers)
{
    for (void *buffer : buffers) {
        if (buffer != nullptr) {
            (void)aclrtFree(buffer);
        }
    }
    if (comm != nullptr) {
        (void)TileXRCommDestroy(comm);
    }
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    if (deviceSet) {
        (void)aclrtResetDevice(deviceId);
    }
    if (aclReady) {
        (void)aclFinalize();
    }
}

} // namespace

int main(int argc, char **argv)
{
    const int rankSize = argc > 1 ? std::atoi(argv[1]) : GetEnvInt("RANK_SIZE", 2);
    const int rank = argc > 2 ? std::atoi(argv[2]) : GetEnvInt("RANK", 0);
    const int npuCount = argc > 3 ? std::atoi(argv[3]) : GetEnvInt("TILEXR_DEMO_NPUS", rankSize);
    const int firstNpu = argc > 4 ? std::atoi(argv[4]) : GetEnvInt("TILEXR_DEMO_FIRST_NPU", 0);

    if (rankSize != kRankSize || rank < 0 || rank >= rankSize) {
        std::cerr << "This MVP demo expects rankSize=2 and valid rank, got rankSize=" << rankSize
                  << " rank=" << rank << std::endl;
        return 2;
    }

    const int deviceId = GetDeviceIdFromEnv(rank, npuCount, firstNpu);
    bool aclReady = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    TileXRCommPtr comm = nullptr;
    std::vector<void *> buffers;

    if (!CheckAcl(aclInit(nullptr), "aclInit")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    aclReady = true;
    if (!CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    deviceSet = true;
    if (!CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    if (!CheckTileXR(TileXRCommInitRankLocal(rankSize, rank, &comm), "TileXRCommInitRankLocal")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    std::vector<uint16_t> hostX(kXElements);
    for (int64_t token = 0; token < kBs; ++token) {
        for (int64_t h = 0; h < kH; ++h) {
            hostX[token * kH + h] = XValue(rank, token, h);
        }
    }
    const std::vector<int32_t> hostExpertIds = ExpertIds();

    void *xDev = nullptr;
    void *expertIdsDev = nullptr;
    void *expandXDev = nullptr;
    void *expertTokenNumsDev = nullptr;
    void *recvCountsDev = nullptr;
    void *assistDev = nullptr;

    const std::size_t xBytes = hostX.size() * sizeof(uint16_t);
    const std::size_t expertIdsBytes = hostExpertIds.size() * sizeof(int32_t);
    const std::size_t expandXBytes = kExpandedElements * sizeof(uint16_t);
    const std::size_t expertTokenNumsBytes = kLocalExpertNum * sizeof(int64_t);
    const std::size_t recvCountsBytes = rankSize * sizeof(int32_t);
    const std::size_t assistBytes = kRoutes * kAssistInts * sizeof(int32_t);

    if (!CheckAcl(aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc x") ||
        !CheckAcl(aclrtMalloc(&expertIdsDev, expertIdsBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc expertIds") ||
        !CheckAcl(aclrtMalloc(&expandXDev, expandXBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc expandX") ||
        !CheckAcl(aclrtMalloc(&expertTokenNumsDev, expertTokenNumsBytes, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc expertTokenNums") ||
        !CheckAcl(aclrtMalloc(&recvCountsDev, recvCountsBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc recvCounts") ||
        !CheckAcl(aclrtMalloc(&assistDev, assistBytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc assist")) {
        buffers = {xDev, expertIdsDev, expandXDev, expertTokenNumsDev, recvCountsDev, assistDev};
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }
    buffers = {xDev, expertIdsDev, expandXDev, expertTokenNumsDev, recvCountsDev, assistDev};

    if (!CheckAcl(aclrtMemcpy(xDev, xBytes, hostX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy x") ||
        !CheckAcl(aclrtMemcpy(expertIdsDev, expertIdsBytes, hostExpertIds.data(), expertIdsBytes,
            ACL_MEMCPY_HOST_TO_DEVICE), "copy expertIds") ||
        !CheckAcl(aclrtMemset(expandXDev, expandXBytes, 0, expandXBytes), "memset expandX") ||
        !CheckAcl(aclrtMemset(expertTokenNumsDev, expertTokenNumsBytes, 0, expertTokenNumsBytes),
            "memset expertTokenNums") ||
        !CheckAcl(aclrtMemset(recvCountsDev, recvCountsBytes, 0, recvCountsBytes), "memset recvCounts") ||
        !CheckAcl(aclrtMemset(assistDev, assistBytes, 0, assistBytes), "memset assist")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    if (!CheckTileXR(TileXRMoeEpDispatch(xDev, static_cast<int32_t *>(expertIdsDev), comm, kBs, kH, kTopK,
            kMoeExpertNum, expandXDev, static_cast<int64_t *>(expertTokenNumsDev),
            static_cast<int32_t *>(recvCountsDev), static_cast<int32_t *>(assistDev), TileXR::TILEXR_DATA_TYPE_FP16,
            stream), "TileXRMoeEpDispatch") ||
        !CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    std::vector<uint16_t> hostExpandX(kExpandedElements);
    std::vector<int64_t> hostExpertTokenNums(kLocalExpertNum);
    std::vector<int32_t> hostRecvCounts(rankSize);
    std::vector<int32_t> hostAssist(kRoutes * kAssistInts);

    if (!CheckAcl(aclrtMemcpy(hostExpandX.data(), expandXBytes, expandXDev, expandXBytes,
            ACL_MEMCPY_DEVICE_TO_HOST), "copy expandX") ||
        !CheckAcl(aclrtMemcpy(hostExpertTokenNums.data(), expertTokenNumsBytes, expertTokenNumsDev,
            expertTokenNumsBytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy expertTokenNums") ||
        !CheckAcl(aclrtMemcpy(hostRecvCounts.data(), recvCountsBytes, recvCountsDev, recvCountsBytes,
            ACL_MEMCPY_DEVICE_TO_HOST), "copy recvCounts") ||
        !CheckAcl(aclrtMemcpy(hostAssist.data(), assistBytes, assistDev, assistBytes,
            ACL_MEMCPY_DEVICE_TO_HOST), "copy assist")) {
        Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
        return 1;
    }

    const bool ok = ValidateOutputs(rank, rankSize, hostExpandX, hostExpertTokenNums, hostRecvCounts, hostAssist);
    std::cout << "rank " << rank << " validation " << (ok ? "PASS" : "FAIL") << std::endl;
    Cleanup(comm, stream, deviceId, deviceSet, aclReady, buffers);
    return ok ? 0 : 1;
}
