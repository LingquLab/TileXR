#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "udma/tilexr_udma_layout.h"

#ifndef TILEXR_SOURCE_ROOT
#define TILEXR_SOURCE_ROOT "."
#endif

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

#define CHECK_CONTAINS(text, needle) \
    do { \
        if ((text).find(needle) == std::string::npos) { \
            std::cerr << "CHECK_CONTAINS failed at line " << __LINE__ << ": " << needle << std::endl; \
            ++g_failures; \
        } \
    } while (0)

#define CHECK_NOT_CONTAINS(text, needle) \
    do { \
        if ((text).find(needle) != std::string::npos) { \
            std::cerr << "CHECK_NOT_CONTAINS failed at line " << __LINE__ << ": " << needle << std::endl; \
            ++g_failures; \
        } \
    } while (0)

std::string ReadFile(const std::string& path)
{
    std::ifstream in(path.c_str());
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void TestHostLayoutUsesDeviceRelativePointers()
{
    std::vector<TileXR::UDMAWQCtx> sq(2);
    std::vector<TileXR::UDMAWQCtx> rq(2);
    std::vector<TileXR::UDMACQCtx> scq(2);
    std::vector<TileXR::UDMACQCtx> rcq(2);
    std::vector<TileXR::UDMAMemInfo> mem(2);

    sq[0].bufAddr = 0x1000;
    sq[1].bufAddr = 0x2000;
    scq[1].dbAddr = 0x3000;
    mem[1].addr = 0x4000;
    mem[1].tid = 7;
    mem[1].tpn = 9;
    std::vector<uint32_t> weights = {3, 1};

    constexpr uintptr_t deviceBase = 0x100000000ULL;
    TileXR::UDMAInfo info = {};
    std::vector<uint8_t> bytes;
    const int ret = TileXR::BuildUDMAInfoImage(deviceBase, 1, sq, rq, scq, rcq, mem, weights, info, bytes);

    CHECK_EQ(ret, TileXR::TILEXR_UDMA_LAYOUT_SUCCESS);
    CHECK_EQ(info.qpNum, 1U);
    CHECK_TRUE(info.sqPtr >= deviceBase);
    CHECK_TRUE(info.rqPtr > info.sqPtr);
    CHECK_TRUE(info.scqPtr > info.rqPtr);
    CHECK_TRUE(info.rcqPtr > info.scqPtr);
    CHECK_TRUE(info.memPtr > info.rcqPtr);
    CHECK_TRUE(info.qpWeightPtr > info.memPtr);
    CHECK_EQ(bytes.size(),
             sizeof(TileXR::UDMAInfo) + 2 * sizeof(TileXR::UDMAWQCtx) +
                 2 * sizeof(TileXR::UDMAWQCtx) + 2 * sizeof(TileXR::UDMACQCtx) +
                 2 * sizeof(TileXR::UDMACQCtx) + 2 * sizeof(TileXR::UDMAMemInfo) +
                 2 * sizeof(uint32_t));

    const auto* imageInfo = reinterpret_cast<const TileXR::UDMAInfo*>(bytes.data());
    CHECK_EQ(imageInfo->sqPtr, info.sqPtr);
    const auto* imageSq = reinterpret_cast<const TileXR::UDMAWQCtx*>(
        bytes.data() + (info.sqPtr - deviceBase));
    const auto* imageMem = reinterpret_cast<const TileXR::UDMAMemInfo*>(
        bytes.data() + (info.memPtr - deviceBase));
    const auto* imageWeights = reinterpret_cast<const uint32_t*>(
        bytes.data() + (info.qpWeightPtr - deviceBase));
    CHECK_EQ(imageSq[1].bufAddr, static_cast<uint64_t>(0x2000));
    CHECK_EQ(imageMem[1].addr, static_cast<uint64_t>(0x4000));
    CHECK_EQ(imageMem[1].tid, 7U);
    CHECK_EQ(imageMem[1].tpn, 9U);
    CHECK_EQ(imageWeights[0], 3U);
    CHECK_EQ(imageWeights[1], 1U);
}

void TestRejectsMismatchedArrays()
{
    std::vector<TileXR::UDMAWQCtx> sq(2);
    std::vector<TileXR::UDMAWQCtx> rq(1);
    std::vector<TileXR::UDMACQCtx> scq(2);
    std::vector<TileXR::UDMACQCtx> rcq(2);
    std::vector<TileXR::UDMAMemInfo> mem(2);

    TileXR::UDMAInfo info = {};
    std::vector<uint8_t> bytes;
    const int ret = TileXR::BuildUDMAInfoImage(0x1000, sq, rq, scq, rcq, mem, info, bytes);
    CHECK_EQ(ret, TileXR::TILEXR_UDMA_LAYOUT_INVALID);
}

void TestMultiRouteQpMappingRepeatsEachRoute()
{
    const std::vector<uint32_t> routeEids = {7, 8};
    const std::vector<uint32_t> qpToEid = TileXR::BuildUDMAMultiRouteQpToEid(routeEids, 2);

    CHECK_EQ(qpToEid.size(), static_cast<size_t>(4));
    CHECK_EQ(qpToEid[0], 7U);
    CHECK_EQ(qpToEid[1], 7U);
    CHECK_EQ(qpToEid[2], 8U);
    CHECK_EQ(qpToEid[3], 8U);
}

void TestMultiRouteQpMappingRejectsEmptyInputs()
{
    CHECK_TRUE(TileXR::BuildUDMAMultiRouteQpToEid({}, 1).empty());
    CHECK_TRUE(TileXR::BuildUDMAMultiRouteQpToEid({7}, 0).empty());
}

void TestMultiRouteQpWeightsUseRouteBandwidth()
{
    const std::vector<uint32_t> routeEids = {7, 8};
    const std::map<uint32_t, uint32_t> routeWeights = {{7, 6}, {8, 2}};
    const std::vector<uint32_t> qpWeights = TileXR::BuildUDMAMultiRouteQpWeights(routeEids, routeWeights, 1);

    CHECK_EQ(qpWeights.size(), static_cast<size_t>(2));
    CHECK_EQ(qpWeights[0], 6U);
    CHECK_EQ(qpWeights[1], 2U);
}

void TestExplicitRouteSelectionKeepsRequestedCandidateOrder()
{
    const std::vector<uint32_t> candidates = {7, 8};
    const std::vector<uint32_t> selected = TileXR::SelectExplicitUDMARouteEids("8,7,9,bad", candidates);

    CHECK_EQ(selected.size(), static_cast<size_t>(2));
    CHECK_EQ(selected[0], 8U);
    CHECK_EQ(selected[1], 7U);
}

void TestExplicitRouteSelectionRejectsMissingInputs()
{
    const std::vector<uint32_t> candidates = {7, 8};
    CHECK_TRUE(TileXR::SelectExplicitUDMARouteEids("", candidates).empty());
    CHECK_TRUE(TileXR::SelectExplicitUDMARouteEids("9", candidates).empty());
    CHECK_TRUE(TileXR::SelectExplicitUDMARouteEids("8", {}).empty());
}

void TestTransportUsesPerPeerQueues()
{
    const std::string transport =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/comm/udma/tilexr_udma_transport.cpp");
    CHECK_CONTAINS(transport, "struct PeerQueueState");
    CHECK_CONTAINS(transport, "std::map<int, PeerQueueState> peerQueues");
    CHECK_CONTAINS(transport, "CreatePeerQueue(");
    CHECK_CONTAINS(transport, "state.peerQueues[peer]");
    CHECK_CONTAINS(transport, "localPeerImports[peer].in.key");
    CHECK_CONTAINS(transport, "allImports[(peer * options_.rankSize + options_.rank)");
    CHECK_CONTAINS(transport, "queue.localWq");
    CHECK_CONTAINS(transport, "queue.localCq");
    CHECK_CONTAINS(transport, "queue.tpn");
    CHECK_NOT_CONTAINS(transport, "void* qpHandle = nullptr;\n    CqInfoT cqInfo");
}

void TestRootInfoEidBytesSelectRuntimeContexts()
{
    const std::string transport =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/comm/udma/tilexr_udma_transport.cpp");

    CHECK_CONTAINS(transport, "ReadTextFile(\"/etc/hccl_rootinfo.json\")");
    CHECK_CONTAINS(transport, "root.eidByLocalId[localId][eidIndex] = eid");
    CHECK_CONTAINS(transport, "root.portToEidByLocalId[localId][port] = eidIndex");
    CHECK_CONTAINS(transport, "localEidByEid_ = localEids->second");
    CHECK_CONTAINS(transport, "auto targetEidIt = localEidByEid_.find(eidIndex)");
    CHECK_CONTAINS(transport,
        "matched = std::memcmp(infoList[i].eid.raw, targetEidIt->second.raw, sizeof(infoList[i].eid.raw)) == 0");
    CHECK_CONTAINS(transport, "attr.ub.eidIndex = infoList[i].eidIndex");
    CHECK_CONTAINS(transport, "ctxHandleByEid_[eidIndex] = ctxHandle");
}

void TestMemoryRegistrationUsesOfficialUbFlags()
{
    const std::string transport =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/comm/udma/tilexr_udma_transport.cpp");
    CHECK_CONTAINS(transport, "mrInfo.in.ub.flags.bs.cacheable = 0");
    CHECK_CONTAINS(transport, "mrInfo.in.ub.flags.bs.nonPin = 0");
    CHECK_CONTAINS(transport, "mrInfo.in.ub.flags.bs.userIova = 0");
    CHECK_CONTAINS(transport, "mrInfo.in.ub.flags.bs.tokenIdValid = 1");
}

void TestDeviceSgeUsesPerPeerLocalTokenId()
{
    const std::string types =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/include/tilexr_udma_types.h");
    const std::string device =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/include/tilexr_udma.h");
    const std::string transport =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/comm/udma/tilexr_udma_transport.cpp");

    CHECK_CONTAINS(types, "uint32_t localTokenId");
    CHECK_CONTAINS(device, "sgeCtx->tokenId = qpCtxEntry->localTokenId");
    CHECK_NOT_CONTAINS(device, "sgeCtx->tokenId = 0;");
    CHECK_CONTAINS(transport, "queue.localWq.localTokenId = localMrIt->second.tokenId");
}

void TestDeviceSqeInitializesOfficialFields()
{
    const std::string device =
        ReadFile(std::string(TILEXR_SOURCE_ROOT) + "/src/include/tilexr_udma.h");

    CHECK_CONTAINS(device, "sqeCtx->sqeBbIdx = curHead % depth");
    CHECK_CONTAINS(device, "sqeCtx->rsv0 = 0");
    CHECK_CONTAINS(device, "sqeCtx->rsv1 = 0");
    CHECK_CONTAINS(device, "sqeCtx->rsv2 = 0");
    CHECK_CONTAINS(device, "sqeCtx->rsv3 = 0");
    CHECK_CONTAINS(device, "UDMAFillSqeCtx(sqeCtx, remoteAddr, remoteMemInfo, curHead, qpCtxEntry->depth");
}

} // namespace

int main()
{
    TestHostLayoutUsesDeviceRelativePointers();
    TestRejectsMismatchedArrays();
    TestMultiRouteQpMappingRepeatsEachRoute();
    TestMultiRouteQpMappingRejectsEmptyInputs();
    TestMultiRouteQpWeightsUseRouteBandwidth();
    TestExplicitRouteSelectionKeepsRequestedCandidateOrder();
    TestExplicitRouteSelectionRejectsMissingInputs();
    TestTransportUsesPerPeerQueues();
    TestRootInfoEidBytesSelectRuntimeContexts();
    TestMemoryRegistrationUsesOfficialUbFlags();
    TestDeviceSgeUsesPerPeerLocalTokenId();
    TestDeviceSqeInitializesOfficialFields();
    if (g_failures != 0) {
        std::cerr << g_failures << " UDMA transport layout checks failed" << std::endl;
        return 1;
    }
    std::cout << "TileXR UDMA transport layout checks passed" << std::endl;
    return 0;
}
