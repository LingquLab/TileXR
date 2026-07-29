#ifndef TILEXR_EP_HOST_EP_TRANSPORT_ROUTE_H
#define TILEXR_EP_HOST_EP_TRANSPORT_ROUTE_H

#include <cstdint>

#include "comm_args.h"
#include "tilexr_transport.h"

namespace TileXREp {

enum class EpTransportMode : uint8_t {
    AUTO = 0,
    MEMORY = 1,
    DIRECT_URMA = 2,
};

int TileXREpParseTransportMode(const char *value, EpTransportMode *mode);
bool TileXREpShouldRegisterWorkspace(EpTransportMode mode, const TileXR::CommArgs &args);
int TileXREpResolveTransport(EpTransportMode mode, const TileXR::CommArgs &args, uint64_t bytes,
    TileXR::TileXRTransportKind *transport);
int TileXREpResolveTransportFromEnv(const TileXR::CommArgs &args, uint64_t bytes,
    TileXR::TileXRTransportKind *transport);
int TileXREpResolveTransportForWorkspaceFromEnv(const TileXR::CommArgs &args, uint64_t bytes,
    const void *workspace, TileXR::TileXRTransportKind *transport);

} // namespace TileXREp

#endif // TILEXR_EP_HOST_EP_TRANSPORT_ROUTE_H
