#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "prefetch_weight_host.h"
#include "prefetch_weight_launch.h"
#include "tilexr_types.h"

namespace {
int failures = 0;
int hostRet = 0, devRet = 0, magicRet = 0, registryRet = 0, qpRet = 0, launchRet = 0;
int launchCalls = 0;
int64_t nextMagic = 71;
uint32_t qpNum = 4;
TileXR::CommArgs commArgs {};
TileXR::TileXRUDMARegistry registry {};
GM_ADDR devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
TileXRMoonEp::PrefetchWeightParams seenParams {};
TileXRMoonEp::PrefetchWeightLaunchContext seenContext {};

void Check(bool value, const std::string &message) { if (!value) { std::cerr << message << '\n'; ++failures; } }
void Status(const char *name, int value, int expected) { if (value != expected) { std::cerr << name << ": " << value << " != " << expected << '\n'; ++failures; } }

void Reset()
{
    hostRet = devRet = magicRet = registryRet = qpRet = launchRet = 0;
    launchCalls = 0;
    nextMagic = 71;
    qpNum = 4;
    commArgs = TileXR::CommArgs {};
    commArgs.rank = 0; commArgs.localRank = 0; commArgs.rankSize = 2; commArgs.localRankSize = 2;
    commArgs.extraFlag = TileXR::ExtraFlag::TOPO_910A5 | TileXR::ExtraFlag::UDMA;
    commArgs.peerMems[0] = reinterpret_cast<GM_ADDR>(uintptr_t {0x100000});
    commArgs.peerMems[1] = reinterpret_cast<GM_ADDR>(uintptr_t {0x200000});
    commArgs.udmaInfoPtr = reinterpret_cast<GM_ADDR>(uintptr_t {0x8000});
    commArgs.udmaRegistryPtr = reinterpret_cast<GM_ADDR>(uintptr_t {0x8100});
    registry = TileXR::TileXRUDMARegistry {};
    registry.rankSize = 2; registry.regionCount = 1;
    registry.regions[0].base = reinterpret_cast<GM_ADDR>(uintptr_t {0x100000});
    registry.regions[0].bytes = 0x10000;
    registry.regions[1].base = reinterpret_cast<GM_ADDR>(uintptr_t {0x200000});
    registry.regions[1].bytes = 0x10000;
    devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
    seenParams = TileXRMoonEp::PrefetchWeightParams {};
    seenContext = TileXRMoonEp::PrefetchWeightLaunchContext {};
}

TileXRMoonEpPlanV1 Plan()
{
    TileXRMoonEpPlanV1 p {};
    p.structSize = sizeof(p); p.abiVersion = 1; p.n = 4; p.r = 2; p.e = 8; p.b = 4; p.nvS = 8; p.k = 2;
    p.dst = reinterpret_cast<void *>(uintptr_t {0x3000});
    p.expertsToCopy = reinterpret_cast<void *>(uintptr_t {0x3100});
    p.zeroFillRanges = reinterpret_cast<void *>(uintptr_t {0x3200});
    p.remoteStats = reinterpret_cast<void *>(uintptr_t {0x3300});
    p.dupGroups = reinterpret_cast<void *>(uintptr_t {0x3400});
    p.dupLoffs = reinterpret_cast<void *>(uintptr_t {0x3500});
    p.dupCounts = reinterpret_cast<void *>(uintptr_t {0x3600});
    p.status = reinterpret_cast<void *>(uintptr_t {0x3700});
    return p;
}

TileXRMoonEpTensorV1 Weight(uintptr_t ptr, int64_t h, int64_t hp)
{
    TileXRMoonEpTensorV1 t {};
    t.structSize = sizeof(t); t.abiVersion = 1; t.data = reinterpret_cast<void *>(ptr);
    t.dtype = TILEXR_MOONEP_DTYPE_BFLOAT16; t.rank = 3;
    t.shape[0] = 8; t.shape[1] = h; t.shape[2] = hp;
    t.elementCount = static_cast<uint64_t>(8 * h * hp);
    return t;
}

TileXRMoonEpPrefetchWeightArgsV1 Args(TileXRMoonEpPlanV1 *plan,
    TileXRMoonEpTensorV1 *gate, TileXRMoonEpTensorV1 *up, TileXRMoonEpTensorV1 *down)
{
    TileXRMoonEpPrefetchWeightArgsV1 a {};
    a.structSize = sizeof(a); a.abiVersion = 1;
    a.comm = reinterpret_cast<TileXRCommPtr>(uintptr_t {0x1000}); a.plan = plan;
    a.gate = gate; a.up = up; a.down = down;
    return a;
}

void TestLaunch()
{
    Reset();
    auto plan = Plan(); auto gate = Weight(0x100000, 4, 8); auto up = Weight(0x101000, 4, 16); auto down = Weight(0x102000, 8, 8);
    auto args = Args(&plan, &gate, &up, &down);
    auto stream = reinterpret_cast<aclrtStream>(uintptr_t {0x7000});
    Status("prefetch launch", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), 0);
    Check(launchCalls == 1 && seenParams.expertsToCopy == plan.expertsToCopy && seenParams.gate == gate.data && seenParams.up == up.data && seenParams.down == down.data && seenParams.status == plan.status && seenParams.stream == stream, "prefetch launch arguments mismatch");
    Check(seenContext.devArgs == devArgs &&
        seenContext.layout.gate.rowBytes == 64 && seenContext.layout.up.rowBytes == 128 &&
        seenContext.layout.down.rowBytes == 128 &&
        seenContext.layout.gate.registryOffset == 0 &&
        seenContext.layout.up.registryOffset == 0x1000 &&
        seenContext.layout.down.registryOffset == 0x2000 &&
        seenContext.layout.qpNum == 4 && seenContext.layout.blockDim == 4 &&
        seenContext.layout.physicalQpMap == UINT64_C(0x03020100) &&
        seenContext.layout.expertsPerRank == 4,
        "prefetch UDMA layout mismatch");

    Reset();
    qpNum = 48;
    commArgs.extraFlag |= TileXR::ExtraFlag::UDMA_SHARED_QP;
    plan = Plan();
    gate = Weight(0x100000, 4, 8);
    up = Weight(0x101000, 4, 16);
    down = Weight(0x102000, 8, 8);
    args = Args(&plan, &gate, &up, &down);
    Status("prefetch shared-domain QPs",
        TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream),
        TILEXR_MOONEP_SUCCESS);
    Check(launchCalls == 1 && seenContext.layout.qpNum == 48 &&
        seenContext.layout.blockDim == 4 &&
        TileXRMoonEp::TileXRMoonEpPrefetchWeightPhysicalQp(
            seenContext.layout.physicalQpMap, 0) == 0 &&
        TileXRMoonEp::TileXRMoonEpPrefetchWeightPhysicalQp(
            seenContext.layout.physicalQpMap, 1) == 1 &&
        TileXRMoonEp::TileXRMoonEpPrefetchWeightPhysicalQp(
            seenContext.layout.physicalQpMap, 2) == 2 &&
        TileXRMoonEp::TileXRMoonEpPrefetchWeightPhysicalQp(
            seenContext.layout.physicalQpMap, 3) == 16,
        "prefetch must cap workers without rejecting the shared-domain QP count");

    uint64_t physicalQpMap = 0;
    Status("prefetch eight-worker shared QP map",
        TileXRMoonEp::TileXRMoonEpBuildPrefetchWeightQpMap(
            8, 48, true, &physicalQpMap), TILEXR_MOONEP_SUCCESS);
    const uint32_t expectedSharedQps[8] = {0, 1, 2, 3, 4, 5, 16, 17};
    for (uint32_t worker = 0; worker < 8; ++worker) {
        Check(TileXRMoonEp::TileXRMoonEpPrefetchWeightPhysicalQp(
            physicalQpMap, worker) == expectedSharedQps[worker],
            "prefetch eight-worker shared QP mapping mismatch");
    }
    Status("prefetch malformed shared QP domain",
        TileXRMoonEp::TileXRMoonEpBuildPrefetchWeightQpMap(
            4, 4, true, &physicalQpMap), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);

    Reset();
    qpNum = 3;
    plan = Plan(); gate = Weight(0x100000, 4, 8);
    up = Weight(0x101000, 4, 16); down = Weight(0x102000, 8, 8);
    args = Args(&plan, &gate, &up, &down);
    Status("prefetch three QPs",
        TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream),
        TILEXR_MOONEP_SUCCESS);
    Check(launchCalls == 1 && seenContext.layout.qpNum == 3 &&
        seenContext.layout.blockDim == 2,
        "prefetch three-QP layout must use two workers");

    gate.dtype = TILEXR_MOONEP_DTYPE_FLOAT32;
    Status("prefetch dtype", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    gate = Weight(0x100000, 4, 8); args.flags = 1;
    Status("prefetch flags", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.flags = 0; commArgs.localRankSize = 1; commArgs.peerMems[0] = nullptr;
    commArgs.peerMems[1] = nullptr; magicRet = -1;
    Status("prefetch cross-node registered UDMA",
        TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), 0);
    Reset();
    plan = Plan(); gate = Weight(0x100000, 4, 8); up = Weight(0x101000, 4, 16);
    down = Weight(0x102000, 8, 8); args = Args(&plan, &gate, &up, &down);
    gate.dtype = up.dtype = down.dtype = TILEXR_MOONEP_DTYPE_FLOAT16;
    gate.rank = 2; gate.shape[1] = 32; gate.shape[2] = 0;
    up.rank = 4; up.shape[1] = 2; up.shape[2] = 2; up.shape[3] = 16;
    Status("prefetch fp16 rank2-rank4",
        TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), 0);
    Reset();
    plan = Plan(); plan.b = 2; gate = Weight(0x100000, 4, 8);
    up = Weight(0x101000, 4, 16); down = Weight(0x102000, 8, 8);
    args = Args(&plan, &gate, &up, &down);
    Status("prefetch compact B",
        TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream),
        TILEXR_MOONEP_SUCCESS);
    Check(launchCalls == 1 && seenContext.layout.expertsPerRank == 4 &&
        seenContext.layout.prefetchSlots == 2,
        "prefetch compact B must preserve the owner expert stride");
    Reset(); plan = Plan(); gate = Weight(0x100000, 4, 8);
    up = Weight(0x101000, 4, 16); down = Weight(0x102000, 8, 8);
    args = Args(&plan, &gate, &up, &down); registryRet = -1;
    Status("prefetch unavailable UDMA", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), TILEXR_MOONEP_ERROR_NOT_SUPPORTED);
    Reset(); launchRet = -77;
    Status("prefetch launch failure", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), -77);
}

}

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&out) { out = hostRet == 0 ? &commArgs : nullptr; return hostRet; }
extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &out) { out = devRet == 0 ? devArgs : nullptr; return devRet; }
extern "C" int TileXRCommNextMagic(TileXRCommPtr, int64_t *out) { if (magicRet == 0) *out = nextMagic; return magicRet; }
extern "C" int TileXRGetUDMARegistryHost(TileXRCommPtr, const TileXR::TileXRUDMARegistry **out) { *out = registryRet == 0 ? &registry : nullptr; return registryRet; }
extern "C" int TileXRUDMAGetQpCount(TileXRCommPtr, uint32_t *out) { if (qpRet == 0) *out = qpNum; return qpRet; }
namespace TileXRMoonEp { int TileXRMoonEpLaunchPrefetchWeightKernel(const PrefetchWeightParams &p, const PrefetchWeightLaunchContext &c) { ++launchCalls; seenParams = p; seenContext = c; return launchRet; } }

int main() { TestLaunch(); return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }
