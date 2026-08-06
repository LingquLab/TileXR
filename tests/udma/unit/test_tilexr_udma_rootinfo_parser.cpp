#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

#include "udma/tilexr_udma_root_info.h"

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
        const auto lhsValue = (lhs); \
        const auto rhsValue = (rhs); \
        if (lhsValue != rhsValue) { \
            std::cerr << "CHECK_EQ failed at line " << __LINE__ << ": " #lhs " != " #rhs << std::endl; \
            ++g_failures; \
        } \
    } while (0)

const char* kValidRootInfo = R"JSON(
{
  "ignored": {"value": true, "nothing": null},
  "topo_file_path": "\/tmp\/topo\u002descaped.json",
  "rank_list": [
    {
      "device_id": "8",
      "local_id": 0,
      "eids": [
        {"addr": "000102030405060708090a0b0c0d0e0f", "ports": ["p0", "p1"]},
        {"addr": "101112131415161718191A1B1C1D1E1F", "ports": ["p2"]}
      ]
    },
    {
      "device_id": 9,
      "local_id": "1",
      "eids": [
        {"addr": "202122232425262728292a2b2c2d2e2f", "ports": ["q0", "q1"]},
        {"addr": "303132333435363738393a3b3c3d3e3f", "ports": ["q2"]}
      ]
    }
  ]
}
)JSON";

void ExpectInvalid(const std::string& json)
{
    TileXR::UDMARootInfo root;
    std::string error;
    CHECK_TRUE(!TileXR::ParseUDMARootInfoJson(json, root, &error));
    CHECK_TRUE(!error.empty());
    CHECK_TRUE(root.deviceToLocalId.empty());
}

void TestRootInfoFieldsAndRoutes()
{
    TileXR::UDMARootInfo root;
    std::string error;
    CHECK_TRUE(TileXR::ParseUDMARootInfoJson(kValidRootInfo, root, &error));
    CHECK_TRUE(error.empty());
    CHECK_EQ(root.topoPath, std::string("/tmp/topo-escaped.json"));
    CHECK_EQ(root.deviceIdOffset, 8U);
    CHECK_EQ(root.eidCount, 2U);
    CHECK_EQ(root.deviceToLocalId.at(8), 0U);
    CHECK_EQ(root.deviceToLocalId.at(9), 1U);
    CHECK_EQ(root.portToEidByLocalId.at(0).at("p0"), 0U);
    CHECK_EQ(root.portToEidByLocalId.at(0).at("p2"), 1U);
    CHECK_EQ(root.portCountByEidByLocalId.at(0).at(0), 2U);
    CHECK_EQ(root.portCountByEidByLocalId.at(0).at(1), 1U);
    CHECK_EQ(root.eidByLocalId.at(0).at(0).raw[15], static_cast<uint8_t>(0x0f));

    uint32_t localId = 99;
    CHECK_TRUE(TileXR::ResolveUDMALocalId(root, 0, localId));
    CHECK_EQ(localId, 0U);
    CHECK_TRUE(TileXR::ResolveUDMALocalId(root, 9, localId));
    CHECK_EQ(localId, 1U);
    CHECK_TRUE(!TileXR::ResolveUDMALocalId(root, 100, localId));

    uint32_t eidIndex = 99;
    CHECK_TRUE(TileXR::ResolveUDMAPortCountEid(root, 0, 2, eidIndex));
    CHECK_EQ(eidIndex, 0U);
    CHECK_TRUE(TileXR::ResolveUDMAPortCountEid(root, 0, 1, eidIndex));
    CHECK_EQ(eidIndex, 1U);
    CHECK_TRUE(!TileXR::ResolveUDMAPortCountEid(root, 0, 6, eidIndex));

    root.portCountByEidByLocalId[0][1] = 2;
    CHECK_TRUE(TileXR::ResolveUDMAPortCountEid(root, 0, 2, eidIndex));
    CHECK_EQ(eidIndex, 0U);

    CHECK_TRUE(TileXR::ResolveUDMAAggregateEid(root, 0, eidIndex));
    CHECK_EQ(eidIndex, 0U);
    root.portCountByEidByLocalId[0][0] = 1;
    root.portCountByEidByLocalId[0][1] = 1;
    CHECK_TRUE(!TileXR::ResolveUDMAAggregateEid(root, 0, eidIndex));
    root.portCountByEidByLocalId[0][7] = 6;
    root.eidByLocalId[0][7] = root.eidByLocalId[0][0];
    CHECK_TRUE(TileXR::ResolveUDMAAggregateEid(root, 0, eidIndex));
    CHECK_EQ(eidIndex, 7U);
}

void TestTopologyFieldsAndResolution()
{
    const std::string topology = R"JSON(
    {
      "network_edges": [
        {"local_a": 0, "local_a_ports": ["aggregate"]}
      ],
      "links": [
        {"local_a": "0", "local_b": 1,
         "local_a_ports": ["p2"], "local_b_ports": ["q2"]}
      ]
    })JSON";
    std::vector<TileXR::UDMATopologyEdge> edges;
    std::string error;
    CHECK_TRUE(TileXR::ParseUDMATopologyJson(topology, edges, &error));
    CHECK_EQ(edges.size(), static_cast<size_t>(1));
    CHECK_EQ(edges[0].localA, 0U);
    CHECK_EQ(edges[0].localBPorts[0], std::string("q2"));

    TileXR::UDMARootInfo root;
    CHECK_TRUE(TileXR::ParseUDMARootInfoJson(kValidRootInfo, root));
    uint32_t eidIndex = 99;
    CHECK_TRUE(TileXR::ResolveUDMATopologyEid(root, edges, 0, 1, eidIndex));
    CHECK_EQ(eidIndex, 1U);
    CHECK_TRUE(TileXR::ResolveUDMATopologyEid(root, edges, 1, 0, eidIndex));
    CHECK_EQ(eidIndex, 1U);
    CHECK_TRUE(!TileXR::ResolveUDMATopologyEid(root, edges, 0, 7, eidIndex));
}

void TestMalformedAndMissingFields()
{
    ExpectInvalid("{");
    ExpectInvalid("[]");
    ExpectInvalid(R"JSON({"note":"\\\"device_id\\\": 8"})JSON");
    ExpectInvalid(R"JSON({"topo_file_path":"/tmp/t", "rank_list":[]})JSON");
    ExpectInvalid(R"JSON({"topo_file_path":"/tmp/t", "rank_list":[{"device_id":8}]})JSON");
    ExpectInvalid(R"JSON({"topo_file_path":"/tmp/t", "rank_list":[{
        "device_id":8,"local_id":0,"eids":[{"addr":"bad","ports":[]}]}]})JSON");
    ExpectInvalid(R"JSON({"topo_file_path":"/tmp/t", "rank_list":[{
        "device_id":8,"local_id":0,"eids":[{
        "addr":"000102030405060708090a0b0c0d0e0f","ports":[1]}]}]})JSON");
    ExpectInvalid(R"JSON({"topo_file_path":"/tmp/t", "rank_list":[{
        "device_id":4294967296,"local_id":0,"eids":[{
        "addr":"000102030405060708090a0b0c0d0e0f","ports":[]}]}]})JSON");

    std::vector<TileXR::UDMATopologyEdge> edges;
    std::string error;
    CHECK_TRUE(!TileXR::ParseUDMATopologyJson(
        R"JSON({"links":[{"local_a":0,"local_b":1,"local_a_ports":[],"local_b_ports":["q"]}]})JSON",
        edges, &error));
    CHECK_TRUE(!error.empty());
}

void TestEnvironmentPathOverride()
{
    const std::string path = "/tmp/tilexr_udma_rootinfo_" + std::to_string(getpid()) + ".json";
    {
        std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
        output << kValidRootInfo;
    }
    CHECK_EQ(setenv(TileXR::TILEXR_UDMA_ROOTINFO_PATH_ENV, path.c_str(), 1), 0);
    TileXR::UDMARootInfo root;
    std::string error;
    CHECK_TRUE(TileXR::LoadUDMARootInfo(root, &error));
    CHECK_EQ(root.topoPath, std::string("/tmp/topo-escaped.json"));
    CHECK_EQ(unsetenv(TileXR::TILEXR_UDMA_ROOTINFO_PATH_ENV), 0);
    CHECK_EQ(std::remove(path.c_str()), 0);
}

void TestEidHex()
{
    TileXR::HccpEid eid {};
    CHECK_TRUE(TileXR::ParseUDMAEidHex("00112233445566778899aAbBcCdDeEfF", eid));
    CHECK_EQ(eid.raw[0], static_cast<uint8_t>(0x00));
    CHECK_EQ(eid.raw[15], static_cast<uint8_t>(0xff));
    CHECK_TRUE(!TileXR::ParseUDMAEidHex("0011", eid));
    CHECK_TRUE(!TileXR::ParseUDMAEidHex("00112233445566778899aabbccddeezz", eid));
}

} // namespace

int main()
{
    TestRootInfoFieldsAndRoutes();
    TestTopologyFieldsAndResolution();
    TestMalformedAndMissingFields();
    TestEnvironmentPathOverride();
    TestEidHex();
    if (g_failures != 0) {
        std::cerr << g_failures << " UDMA RootInfo parser checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA RootInfo parser checks passed" << std::endl;
    return 0;
}
