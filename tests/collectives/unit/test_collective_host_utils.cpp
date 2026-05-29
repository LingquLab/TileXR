#include <cstdint>
#include <iostream>
#include <limits>

#include "collective_kernel.h"
#include "collective_utils.h"

namespace {

int g_failures = 0;

void CheckBool(const char *label, bool actual, bool expected)
{
    if (actual != expected) {
        std::cerr << label << " returned " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

void CheckInt64(const char *label, int64_t actual, int64_t expected)
{
    if (actual != expected) {
        std::cerr << label << " returned " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

void CheckUint32(const char *label, uint32_t actual, uint32_t expected)
{
    if (actual != expected) {
        std::cerr << label << " returned " << actual << ", expected " << expected << std::endl;
        ++g_failures;
    }
}

TileXR::CommArgs Args(int rankSize, uint32_t extraFlag)
{
    TileXR::CommArgs args {};
    args.rankSize = rankSize;
    args.extraFlag = extraFlag;
    return args;
}

void TestDataTypeSupport()
{
    using TileXRCollectives::Host::IsSupportedDataType;

    CheckBool("INT8 supported", IsSupportedDataType(TileXR::TILEXR_DATA_TYPE_INT8), true);
    CheckBool("INT16 supported", IsSupportedDataType(TileXR::TILEXR_DATA_TYPE_INT16), true);
    CheckBool("INT32 supported", IsSupportedDataType(TileXR::TILEXR_DATA_TYPE_INT32), true);
    CheckBool("INT64 supported", IsSupportedDataType(TileXR::TILEXR_DATA_TYPE_INT64), true);
    CheckBool("FP16 supported", IsSupportedDataType(TileXR::TILEXR_DATA_TYPE_FP16), true);
    CheckBool("FP32 supported", IsSupportedDataType(TileXR::TILEXR_DATA_TYPE_FP32), true);
    CheckBool("BFP16 supported", IsSupportedDataType(TileXR::TILEXR_DATA_TYPE_BFP16), true);

    CheckBool("UINT8 unsupported", IsSupportedDataType(TileXR::TILEXR_DATA_TYPE_UINT8), false);
    CheckBool("unknown datatype unsupported",
              IsSupportedDataType(static_cast<TileXR::TileXRDataType>(999)), false);
}

void TestCountToBytes()
{
    using TileXRCollectives::Host::CountToBytes;

    CheckInt64("INT8 byte count", CountToBytes(7, TileXR::TILEXR_DATA_TYPE_INT8), 7);
    CheckInt64("INT16 byte count", CountToBytes(7, TileXR::TILEXR_DATA_TYPE_INT16), 14);
    CheckInt64("FP16 byte count", CountToBytes(7, TileXR::TILEXR_DATA_TYPE_FP16), 14);
    CheckInt64("BFP16 byte count", CountToBytes(7, TileXR::TILEXR_DATA_TYPE_BFP16), 14);
    CheckInt64("INT32 byte count", CountToBytes(7, TileXR::TILEXR_DATA_TYPE_INT32), 28);
    CheckInt64("FP32 byte count", CountToBytes(7, TileXR::TILEXR_DATA_TYPE_FP32), 28);
    CheckInt64("INT64 byte count", CountToBytes(7, TileXR::TILEXR_DATA_TYPE_INT64), 56);
    CheckInt64("negative count rejected",
               CountToBytes(-1, TileXR::TILEXR_DATA_TYPE_INT32),
               TileXR::TILEXR_INVALID_VALUE);
    CheckInt64("unsupported datatype rejected",
               CountToBytes(7, TileXR::TILEXR_DATA_TYPE_UINT8),
               TileXR::TILEXR_INVALID_VALUE);
    CheckInt64("overflow rejected",
               CountToBytes(std::numeric_limits<int64_t>::max() / 2 + 1, TileXR::TILEXR_DATA_TYPE_INT16),
               TileXR::TILEXR_INVALID_VALUE);
}

void TestAllGatherBlockNum()
{
    using TileXRCollectives::Host::GetAllGatherBlockNum;

    CheckUint32("rank2 allgather uses two blocks per rank",
                GetAllGatherBlockNum(Args(2, 0), 1024),
                4);
    CheckUint32("small rank4 allgather uses one block per rank",
                GetAllGatherBlockNum(Args(4, 0), 1024),
                4);
    CheckUint32("large rank4 allgather uses two blocks per rank",
                GetAllGatherBlockNum(Args(4, 0), 2 * 1024 * 1024),
                8);
    CheckUint32("pcie allgather uses two blocks per rank",
                GetAllGatherBlockNum(Args(4, TileXR::ExtraFlag::TOPO_PCIE), 1024),
                8);
    CheckUint32("910B2C 16-rank allgather uses AX block count",
                GetAllGatherBlockNum(Args(16, TileXR::ExtraFlag::TOPO_910B2C), 1024),
                10);
    CheckUint32("910_93 large allgather uses HDB ring block count",
                GetAllGatherBlockNum(Args(8, TileXR::ExtraFlag::TOPO_910_93), 33LL * 1024 * 1024),
                32);
}

void TestAllToAllBlockNum()
{
    using TileXRCollectives::Host::GetAllToAllBlockNum;

    CheckUint32("default alltoall uses two blocks per rank",
                GetAllToAllBlockNum(Args(4, 0), 1024),
                8);
    CheckUint32("910_93 small alltoall uses two blocks per rank",
                GetAllToAllBlockNum(Args(8, TileXR::ExtraFlag::TOPO_910_93), 1024),
                16);
    CheckUint32("910_93 aligned large alltoall uses two-step block count",
                GetAllToAllBlockNum(Args(8, TileXR::ExtraFlag::TOPO_910_93), 2LL * 1024 * 1024),
                32);
    CheckUint32("910_93 large rank alltoall caps at two-step block count",
                GetAllToAllBlockNum(Args(32, TileXR::ExtraFlag::TOPO_910_93), 1024),
                32);
}

void TestKernelShimWaitsForRegistration()
{
    TileXR::CommArgs args {};
    args.rankSize = 2;
    TileXRCollectives::Host::HostLaunchContext context;
    context.hostArgs = &args;
    context.devArgs = reinterpret_cast<GM_ADDR>(0x1000);
    context.magic = 7;
    uint8_t sendStorage[16] = {};
    uint8_t recvStorage[16] = {};

    CheckInt64("LaunchCollectiveKernel without registered kernels",
               TileXRCollectives::Host::LaunchCollectiveKernel(TileXR::TileXRType::ALL_GATHER, context,
                   sendStorage, recvStorage, 4, TileXR::TILEXR_DATA_TYPE_INT32, 2, nullptr),
               TileXR::TILEXR_ERROR_NOT_INITIALIZED);
}

} // namespace

int main()
{
    TestDataTypeSupport();
    TestCountToBytes();
    TestAllGatherBlockNum();
    TestAllToAllBlockNum();
    TestKernelShimWaitsForRegistration();
    return g_failures == 0 ? 0 : 1;
}
