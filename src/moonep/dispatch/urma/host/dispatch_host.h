#ifndef TILEXR_MOONEP_DISPATCH_URMA_HOST_H
#define TILEXR_MOONEP_DISPATCH_URMA_HOST_H

#include "tilexr_moonep.h"

namespace TileXRMoonEp {

int TileXRMoonEpQueryDispatchUrmaWorkspace(TileXRCommPtr comm, int64_t s,
    int64_t k, int64_t h, uint32_t hiddenDtype, uint64_t *workspaceBytes,
    uint64_t *workspaceAlignment);

int TileXRMoonEpRunDispatchUrmaV1(const TileXRMoonEpDispatchArgsV1 *args,
    aclrtStream stream);

int TileXRMoonEpRunDispatchUrmaV2(const TileXRMoonEpDispatchArgsV2 *args,
    aclrtStream stream);

} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_DISPATCH_URMA_HOST_H
