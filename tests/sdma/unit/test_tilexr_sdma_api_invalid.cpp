#include <iostream>

#include "tilexr_api.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

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

void TestAvailableRejectsInvalidArgs()
{
    bool available = true;
    CHECK_EQ(TileXRSDMAAvailable(nullptr, &available), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CHECK_EQ(available, true);
    CHECK_EQ(TileXRSDMAAvailable(reinterpret_cast<TileXRCommPtr>(0x1), nullptr),
             TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

void TestWorkspaceRejectsInvalidArgs()
{
    GM_ADDR workspace = reinterpret_cast<GM_ADDR>(0x1234);
    CHECK_EQ(TileXRGetSDMAWorkspaceDev(nullptr, &workspace), TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
    CHECK_EQ(workspace, reinterpret_cast<GM_ADDR>(0x1234));
    CHECK_EQ(TileXRGetSDMAWorkspaceDev(reinterpret_cast<TileXRCommPtr>(0x1), nullptr),
             TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);
}

} // namespace

int main()
{
    TestAvailableRejectsInvalidArgs();
    TestWorkspaceRejectsInvalidArgs();
    if (g_failures != 0) {
        std::cerr << g_failures << " SDMA API invalid-argument checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR SDMA API invalid-argument checks passed" << std::endl;
    return 0;
}
