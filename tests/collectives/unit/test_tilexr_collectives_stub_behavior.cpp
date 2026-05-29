#include <iostream>

#include "tilexr_collectives.h"

namespace {

int g_failures = 0;

void CheckStatus(const char *name, int status)
{
    if (status != TileXR::TILEXR_ERROR_INTERNAL) {
        std::cerr << name << " returned " << status << ", expected "
                  << TileXR::TILEXR_ERROR_INTERNAL << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    CheckStatus("TileXRAllGather",
                TileXRAllGather(nullptr, nullptr, 0, TileXR::TILEXR_DATA_TYPE_INT32, nullptr, nullptr));
    CheckStatus("TileXRAllToAll",
                TileXRAllToAll(nullptr, nullptr, 0, TileXR::TILEXR_DATA_TYPE_INT32, nullptr, nullptr));
    return g_failures == 0 ? 0 : 1;
}
