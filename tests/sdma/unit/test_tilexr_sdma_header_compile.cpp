#include <iostream>

#include "tilexr_sdma.h"
#include "tilexr_sdma_types.h"

int main()
{
    if (TileXR::TILEXR_SDMA_AUTO_CHANNEL_GROUP != 0xFFFFFFFFU) {
        std::cerr << "unexpected TILEXR_SDMA_AUTO_CHANNEL_GROUP" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA header compile check passed" << std::endl;
    return 0;
}
