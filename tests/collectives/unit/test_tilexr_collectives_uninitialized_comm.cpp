#include <cstdint>
#include <iostream>

#include "tilexr_api.h"
#include "tilexr_collectives.h"

namespace {

int g_failures = 0;

void CheckStatus(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " returned " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

void CheckMagic(const char *label, int64_t actual, int64_t expected)
{
    if (actual != expected) {
        std::cerr << label << " wrote magic " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    uint8_t sendStorage[64] = {};
    uint8_t recvStorage[64] = {};

    TileXRCommPtr comm = nullptr;
    CheckStatus("TileXRCommInit(0, 1, &comm)",
                TileXRCommInit(0, 1, &comm),
                TileXR::TILEXR_SUCCESS);
    if (comm == nullptr) {
        std::cerr << "TileXRCommInit returned success but left comm null" << std::endl;
        return 1;
    }

    CheckStatus("TileXRAllGather(uninitialized comm)",
                TileXRAllGather(sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm, nullptr),
                TileXR::TILEXR_ERROR_NOT_INITIALIZED);
    CheckStatus("TileXRAllToAll(uninitialized comm)",
                TileXRAllToAll(sendStorage, recvStorage, 1, TileXR::TILEXR_DATA_TYPE_INT32, comm, nullptr),
                TileXR::TILEXR_ERROR_NOT_INITIALIZED);

    int64_t magic = -1;
    CheckStatus("TileXRCommNextMagic(comm, &magic)",
                TileXRCommNextMagic(comm, &magic),
                TileXR::TILEXR_SUCCESS);
    CheckMagic("TileXRCommNextMagic(comm, &magic)", magic, 1);

    CheckStatus("TileXRCommDestroy(comm)",
                TileXRCommDestroy(comm),
                TileXR::TILEXR_SUCCESS);
    return g_failures == 0 ? 0 : 1;
}
