#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "prefetch_weight_host.h"
#include "prefetch_weight_launch.h"
#include "moonep_peer_window.h"
#include "tilexr_types.h"

namespace {
int failures = 0;
int hostRet = 0, devRet = 0, magicRet = 0, launchRet = 0;
int launchCalls = 0;
int64_t nextMagic = 71;
TileXR::CommArgs commArgs {};
GM_ADDR devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
TileXRMoonEp::PrefetchWeightParams seenParams {};
TileXRMoonEp::PrefetchWeightLaunchContext seenContext {};

void Check(bool value, const std::string &message) { if (!value) { std::cerr << message << '\n'; ++failures; } }
void Status(const char *name, int value, int expected) { if (value != expected) { std::cerr << name << ": " << value << " != " << expected << '\n'; ++failures; } }

void Reset()
{
    hostRet = devRet = magicRet = launchRet = 0;
    launchCalls = 0;
    nextMagic = 71;
    commArgs = TileXR::CommArgs {};
    commArgs.rank = 0; commArgs.localRank = 0; commArgs.rankSize = 2; commArgs.localRankSize = 2;
    commArgs.extraFlag = TileXR::ExtraFlag::TOPO_910A5;
    commArgs.peerMems[0] = reinterpret_cast<GM_ADDR>(uintptr_t {0x100000});
    commArgs.peerMems[1] = reinterpret_cast<GM_ADDR>(uintptr_t {0x200000});
    devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
    seenParams = TileXRMoonEp::PrefetchWeightParams {};
    seenContext = TileXRMoonEp::PrefetchWeightLaunchContext {};
}

TileXRMoonEpPlanV1 Plan()
{
    TileXRMoonEpPlanV1 p {};
    p.structSize = sizeof(p); p.abiVersion = 1; p.n = 4; p.r = 2; p.e = 8; p.b = 2; p.nvS = 8; p.k = 2;
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
    t.shape[0] = 10; t.shape[1] = h; t.shape[2] = hp;
    t.elementCount = static_cast<uint64_t>(10 * h * hp);
    return t;
}

TileXRMoonEpPrefetchWeightArgsV1 Args(TileXRMoonEpPlanV1 *plan,
    TileXRMoonEpTensorV1 *gate, TileXRMoonEpTensorV1 *up, TileXRMoonEpTensorV1 *down)
{
    TileXRMoonEpPrefetchWeightArgsV1 a {};
    a.structSize = sizeof(a); a.abiVersion = 1;
    a.comm = reinterpret_cast<TileXRCommPtr>(uintptr_t {0x1000}); a.plan = plan;
    a.fullGateWeight = gate; a.fullUpWeight = up; a.fullDownWeight = down;
    return a;
}

void TestLaunch()
{
    Reset();
    auto plan = Plan(); auto gate = Weight(0x4000, 2, 3); auto up = Weight(0x5000, 4, 2); auto down = Weight(0x6000, 2, 5);
    auto args = Args(&plan, &gate, &up, &down);
    auto stream = reinterpret_cast<aclrtStream>(uintptr_t {0x7000});
    Status("prefetch launch", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), 0);
    Check(launchCalls == 1 && seenParams.expertsToCopy == plan.expertsToCopy && seenParams.fullGateWeight == gate.data && seenParams.fullUpWeight == up.data && seenParams.fullDownWeight == down.data && seenParams.status == plan.status && seenParams.stream == stream, "prefetch launch arguments mismatch");
    Check(seenContext.devArgs == devArgs && seenContext.magic == nextMagic && seenContext.layout.gate.rowBytes == 12 && seenContext.layout.up.rowBytes == 16 && seenContext.layout.down.rowBytes == 20 && seenContext.layout.iterationCount == 6, "prefetch layout mismatch");

    gate.dtype = TILEXR_MOONEP_DTYPE_FLOAT32;
    Status("prefetch dtype", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    gate = Weight(0x4000, 2, 3); args.flags = 1;
    Status("prefetch flags", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    args.flags = 0; commArgs.localRankSize = 1;
    Status("prefetch cross-node", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), TILEXR_MOONEP_ERROR_NOT_SUPPORTED);
    Reset(); launchRet = -77;
    Status("prefetch launch failure", TileXRMoonEp::TileXRMoonEpRunPrefetchWeightV1(&args, stream), -77);
}
}

extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&out) { out = hostRet == 0 ? &commArgs : nullptr; return hostRet; }
extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &out) { out = devRet == 0 ? devArgs : nullptr; return devRet; }
extern "C" int TileXRCommNextMagic(TileXRCommPtr, int64_t *out) { if (magicRet == 0) *out = nextMagic; return magicRet; }
namespace TileXRMoonEp { int TileXRMoonEpLaunchPrefetchWeightKernel(const PrefetchWeightParams &p, const PrefetchWeightLaunchContext &c) { ++launchCalls; seenParams = p; seenContext = c; return launchRet; } }

int main() { TestLaunch(); return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }
