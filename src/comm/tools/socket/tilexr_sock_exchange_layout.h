#ifndef TILEXR_SOCK_EXCHANGE_LAYOUT_H
#define TILEXR_SOCK_EXCHANGE_LAYOUT_H

#include <algorithm>

namespace TileXR {

struct SockExchangeGroupLayout {
    int groupCount = 0;
    int groupIndex = 0;
    int groupBegin = 0;
    int groupEnd = 0;
    int groupLeader = 0;
};

inline SockExchangeGroupLayout BuildSockExchangeGroupLayout(
    int rank, int rankSize, int groupSize)
{
    SockExchangeGroupLayout layout {};
    if (rank < 0 || rank >= rankSize || rankSize <= 0 || groupSize <= 0) {
        return layout;
    }
    layout.groupCount = (rankSize + groupSize - 1) / groupSize;
    layout.groupIndex = rank / groupSize;
    layout.groupBegin = layout.groupIndex * groupSize;
    layout.groupEnd = std::min(rankSize, layout.groupBegin + groupSize);
    layout.groupLeader = layout.groupBegin;
    return layout;
}

} // namespace TileXR

#endif
