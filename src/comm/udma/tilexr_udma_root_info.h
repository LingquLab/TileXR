/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_UDMA_ROOT_INFO_H
#define TILEXR_UDMA_ROOT_INFO_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "udma/tilexr_hccp_defs.h"

namespace TileXR {

constexpr char TILEXR_UDMA_ROOTINFO_PATH_ENV[] = "TILEXR_UDMA_ROOTINFO_PATH";
constexpr char TILEXR_UDMA_DEFAULT_ROOTINFO_PATH[] = "/etc/hccl_rootinfo.json";

struct UDMARootInfo {
    std::string topoPath;
    uint32_t deviceIdOffset = 0;
    uint32_t eidCount = 0;
    std::map<uint32_t, uint32_t> deviceToLocalId;
    std::map<uint32_t, std::map<std::string, uint32_t>> portToEidByLocalId;
    std::map<uint32_t, std::map<uint32_t, HccpEid>> eidByLocalId;
    std::map<uint32_t, std::map<uint32_t, uint32_t>> portCountByEidByLocalId;
};

struct UDMATopologyEdge {
    uint32_t localA = 0;
    uint32_t localB = 0;
    std::vector<std::string> localAPorts;
    std::vector<std::string> localBPorts;
};

bool ParseUDMARootInfoJson(
    const std::string& json, UDMARootInfo& rootInfo, std::string* error = nullptr);

bool LoadUDMARootInfoFromPath(
    const std::string& path, UDMARootInfo& rootInfo, std::string* error = nullptr);

bool LoadUDMARootInfo(UDMARootInfo& rootInfo, std::string* error = nullptr);

bool ParseUDMATopologyJson(
    const std::string& json, std::vector<UDMATopologyEdge>& edges,
    std::string* error = nullptr);

bool LoadUDMATopologyFromPath(
    const std::string& path, std::vector<UDMATopologyEdge>& edges,
    std::string* error = nullptr);

bool ParseUDMAEidHex(const std::string& text, HccpEid& eid);

bool ResolveUDMALocalId(
    const UDMARootInfo& rootInfo, uint32_t deviceId, uint32_t& localId);

bool ResolveUDMATopologyEid(
    const UDMARootInfo& rootInfo, const std::vector<UDMATopologyEdge>& edges,
    uint32_t localId, uint32_t peerLocalId, uint32_t& eidIndex);

bool ResolveUDMAPortCountEid(
    const UDMARootInfo& rootInfo, uint32_t localId, uint32_t portCount,
    uint32_t& eidIndex);

bool ResolveUDMAAggregateEid(
    const UDMARootInfo& rootInfo, uint32_t localId, uint32_t& eidIndex);

} // namespace TileXR

#endif // TILEXR_UDMA_ROOT_INFO_H
