#include <cstdint>
#include <iostream>

#include "tilexr_data_as_flag.h"

int main()
{
    static_assert(TileXR::DATA_AS_FLAG_BLOCK_BYTES == 512U, "DataAsFlag block size must be 512B");
    static_assert(TileXR::DATA_AS_FLAG_PAYLOAD_BYTES == 480U, "DataAsFlag payload size must be 480B");
    static_assert(TileXR::DATA_AS_FLAG_FLAG_BYTES == 32U, "DataAsFlag flag area must be 32B");
    static_assert(TileXR::DATA_AS_FLAG_FLAG_OFFSET_BYTES == 480U, "DataAsFlag flag offset must be 480B");
    static_assert(TileXR::DATA_AS_FLAG_FLAG_FLOATS == 8U, "DataAsFlag flag area must hold 8 floats");
    static_assert(TileXR::DATA_AS_FLAG_READY_VALUE == 1.0f, "DataAsFlag ready value must be float 1.0");

    if (TileXR::DataAsFlagBlockCountForPayloadBytes(0) != 0U) {
        std::cerr << "expected 0 payload bytes to require 0 DataAsFlag blocks" << std::endl;
        return 1;
    }
    if (TileXR::DataAsFlagBlockCountForPayloadBytes(480) != 1U) {
        std::cerr << "expected 480 payload bytes to require 1 DataAsFlag block" << std::endl;
        return 1;
    }
    if (TileXR::DataAsFlagBlockCountForPayloadBytes(481) != 2U) {
        std::cerr << "expected 481 payload bytes to require 2 DataAsFlag blocks" << std::endl;
        return 1;
    }

    std::cout << "TileXR DataAsFlag header compile check passed" << std::endl;
    return 0;
}
