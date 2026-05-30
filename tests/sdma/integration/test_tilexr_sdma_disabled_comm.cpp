#include <cstdlib>
#include <iostream>

#include <acl/acl.h>
#include <acl/acl_rt.h>

#include "tilexr_api.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK_TRUE failed at line " << __LINE__ << ": " #expr << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_EQ(lhs, rhs) \
    do { \
        auto lhsValue = (lhs); \
        auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs \
                      << " (" << lhsValue << " vs " << rhsValue << ")" << std::endl; \
            ++g_failures; \
        } \
    } while (0)

} // namespace

int main()
{
    unsetenv("TILEXR_ENABLE_SDMA");

    CHECK_EQ(aclInit(nullptr), ACL_SUCCESS);
    CHECK_EQ(aclrtSetDevice(0), ACL_SUCCESS);

    TileXRCommPtr comm = nullptr;
    CHECK_EQ(TileXRCommInitRankLocal(1, 0, &comm), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(comm != nullptr);

    bool available = true;
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(0x1234);
    CHECK_EQ(TileXRSDMAAvailable(comm, &available), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(TileXRGetSDMAWorkspaceDev(comm, &workspace), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(!available);
    CHECK_TRUE(workspace == nullptr);

    TileXR::CommArgs* args = nullptr;
    CHECK_EQ(TileXRGetCommArgsHost(comm, args), TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(args != nullptr);
    CHECK_TRUE((args->extraFlag & TileXR::ExtraFlag::SDMA) == 0);
    CHECK_TRUE(args->sdmaWorkspacePtr == nullptr);

    CHECK_EQ(TileXRCommDestroy(comm), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(aclrtResetDevice(0), ACL_SUCCESS);
    CHECK_EQ(aclFinalize(), ACL_SUCCESS);

    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA disabled communicator checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA disabled communicator checks passed" << std::endl;
    return 0;
}
