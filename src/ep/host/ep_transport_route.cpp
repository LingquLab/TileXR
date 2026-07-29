#include "ep_transport_route.h"

#include <cstdlib>
#include <cstring>

#include "tilexr_types.h"

namespace TileXREp {

int TileXREpParseTransportMode(const char *value, EpTransportMode *mode)
{
    if (mode == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0) {
        *mode = EpTransportMode::AUTO;
        return TileXR::TILEXR_SUCCESS;
    }
    if (std::strcmp(value, "memory") == 0) {
        *mode = EpTransportMode::MEMORY;
        return TileXR::TILEXR_SUCCESS;
    }
    if (std::strcmp(value, "direct_urma") == 0) {
        *mode = EpTransportMode::DIRECT_URMA;
        return TileXR::TILEXR_SUCCESS;
    }
    return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
}

bool TileXREpShouldRegisterWorkspace(EpTransportMode mode, const TileXR::CommArgs &args)
{
    return mode != EpTransportMode::MEMORY && TileXR::TileXRDirectUrmaCapable(&args);
}

int TileXREpResolveTransport(EpTransportMode mode, const TileXR::CommArgs &args, uint64_t bytes,
    TileXR::TileXRTransportKind *transport)
{
    if (transport == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }

    switch (mode) {
        case EpTransportMode::AUTO: {
            *transport = TileXR::TileXRSelectAutoTransport(&args, bytes);
            return TileXR::TILEXR_SUCCESS;
        }
        case EpTransportMode::MEMORY:
            *transport = TileXR::TileXRTransportKind::MEMORY;
            return TileXR::TILEXR_SUCCESS;
        case EpTransportMode::DIRECT_URMA:
            if (!TileXR::TileXRDirectUrmaAvailable(&args)) {
                return TileXR::TILEXR_ERROR_NOT_INITIALIZED;
            }
            *transport = TileXR::TileXRTransportKind::DIRECT_URMA;
            return TileXR::TILEXR_SUCCESS;
        default:
            return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
}

int TileXREpResolveTransportFromEnv(const TileXR::CommArgs &args, uint64_t bytes,
    TileXR::TileXRTransportKind *transport)
{
    EpTransportMode mode = EpTransportMode::AUTO;
    int ret = TileXREpParseTransportMode(std::getenv("TILEXR_TRANSPORT_MODE"), &mode);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    return TileXREpResolveTransport(mode, args, bytes, transport);
}

int TileXREpResolveTransportForWorkspaceFromEnv(const TileXR::CommArgs &args, uint64_t bytes,
    const void *workspace, TileXR::TileXRTransportKind *transport)
{
    EpTransportMode mode = EpTransportMode::AUTO;
    int ret = TileXREpParseTransportMode(std::getenv("TILEXR_TRANSPORT_MODE"), &mode);
    if (ret != TileXR::TILEXR_SUCCESS) {
        return ret;
    }
    if (mode == EpTransportMode::AUTO && workspace == nullptr) {
        mode = EpTransportMode::MEMORY;
    }
    return TileXREpResolveTransport(mode, args, bytes, transport);
}

} // namespace TileXREp
