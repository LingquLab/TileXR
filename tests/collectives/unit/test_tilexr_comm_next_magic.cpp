#include <cstdint>
#include <iostream>

#include "tilexr_api.h"
#include "tilexr_types.h"

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
    int64_t magic = -1;
    CheckStatus("TileXRCommNextMagic(nullptr, &magic)",
                TileXRCommNextMagic(nullptr, &magic),
                TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CheckMagic("TileXRCommNextMagic(nullptr, &magic)", magic, -1);

    TileXRCommPtr comm = nullptr;
    CheckStatus("TileXRCommInit(0, 1, &comm)",
                TileXRCommInit(0, 1, &comm),
                TileXR::TILEXR_SUCCESS);
    if (comm == nullptr) {
        std::cerr << "TileXRCommInit returned success but left comm null" << std::endl;
        return 1;
    }

    CheckStatus("TileXRCommNextMagic(comm, nullptr)",
                TileXRCommNextMagic(comm, nullptr),
                TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    magic = -1;
    CheckStatus("first TileXRCommNextMagic(comm, &magic)",
                TileXRCommNextMagic(comm, &magic),
                TileXR::TILEXR_SUCCESS);
    CheckMagic("first TileXRCommNextMagic(comm, &magic)", magic, 1);

    magic = -1;
    CheckStatus("second TileXRCommNextMagic(comm, &magic)",
                TileXRCommNextMagic(comm, &magic),
                TileXR::TILEXR_SUCCESS);
    CheckMagic("second TileXRCommNextMagic(comm, &magic)", magic, 2);

    CheckStatus("TileXRCommDestroy(comm)",
                TileXRCommDestroy(comm),
                TileXR::TILEXR_SUCCESS);
    return g_failures == 0 ? 0 : 1;
}
