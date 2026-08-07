#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "reduce_grad_host.h"
#include "reduce_grad_launch.h"
#include "moonep_peer_window.h"
#include "tilexr_types.h"

namespace {
int failures = 0;
int hostRet = 0, devRet = 0, magicRet = 0, launchRet = 0;
int launchCalls = 0;
int64_t nextMagic = 81;
TileXR::CommArgs commArgs {};
GM_ADDR devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000});
TileXRMoonEp::ReduceGradParams seenParams {};
TileXRMoonEp::ReduceGradLaunchContext seenContext {};
void Check(bool value, const std::string &message) { if (!value) { std::cerr << message << '\n'; ++failures; } }
void Status(const char *name, int value, int expected) { if (value != expected) { std::cerr << name << ": " << value << " != " << expected << '\n'; ++failures; } }
void Reset() { hostRet = devRet = magicRet = launchRet = 0; launchCalls = 0; nextMagic = 81; commArgs = TileXR::CommArgs {}; commArgs.rank = 0; commArgs.localRank = 0; commArgs.rankSize = 2; commArgs.localRankSize = 2; commArgs.extraFlag = TileXR::ExtraFlag::TOPO_910A5; commArgs.peerMems[0] = reinterpret_cast<GM_ADDR>(uintptr_t {0x100000}); commArgs.peerMems[1] = reinterpret_cast<GM_ADDR>(uintptr_t {0x200000}); devArgs = reinterpret_cast<GM_ADDR>(uintptr_t {0x9000}); }
TileXRMoonEpPlanV1 Plan() { TileXRMoonEpPlanV1 p {}; p.structSize = sizeof(p); p.abiVersion = 1; p.n = 4; p.r = 2; p.e = 8; p.b = 2; p.nvS = 8; p.k = 2; p.dst = reinterpret_cast<void *>(uintptr_t {0x3000}); p.expertsToCopy = reinterpret_cast<void *>(uintptr_t {0x3100}); p.zeroFillRanges = reinterpret_cast<void *>(uintptr_t {0x3200}); p.remoteStats = reinterpret_cast<void *>(uintptr_t {0x3300}); p.dupGroups = reinterpret_cast<void *>(uintptr_t {0x3400}); p.dupLoffs = reinterpret_cast<void *>(uintptr_t {0x3500}); p.dupCounts = reinterpret_cast<void *>(uintptr_t {0x3600}); p.status = reinterpret_cast<void *>(uintptr_t {0x3700}); return p; }
TileXRMoonEpTensorV1 Full(uintptr_t ptr, int64_t h, int64_t hp) { TileXRMoonEpTensorV1 t {}; t.structSize = sizeof(t); t.abiVersion = 1; t.data = reinterpret_cast<void *>(ptr); t.dtype = TILEXR_MOONEP_DTYPE_FLOAT32; t.rank = 3; t.shape[0] = 10; t.shape[1] = h; t.shape[2] = hp; t.elementCount = static_cast<uint64_t>(10 * h * hp); return t; }
TileXRMoonEpTensorV1 Buffer(uintptr_t ptr, int64_t h, int64_t hp) { TileXRMoonEpTensorV1 t {}; t.structSize = sizeof(t); t.abiVersion = 1; t.data = reinterpret_cast<void *>(ptr); t.dtype = TILEXR_MOONEP_DTYPE_FLOAT32; t.rank = 4; t.shape[0] = 2; t.shape[1] = 2; t.shape[2] = h; t.shape[3] = hp; t.elementCount = static_cast<uint64_t>(4 * h * hp); return t; }

void TestLaunch()
{
    Reset(); auto plan = Plan(); auto gate = Full(0x4000, 2, 3); auto up = Full(0x5000, 4, 2); auto down = Full(0x6000, 2, 5); auto gateBuf = Buffer(0x7000, 2, 3); auto upBuf = Buffer(0x8000, 4, 2); auto downBuf = Buffer(0xa000, 2, 5);
    TileXRMoonEpReduceGradArgsV1 a {}; a.structSize = sizeof(a); a.abiVersion = 1; a.comm = reinterpret_cast<TileXRCommPtr>(uintptr_t {0x1000}); a.plan = &plan; a.fullGateGrad = &gate; a.fullUpGrad = &up; a.fullDownGrad = &down; a.gateReduceBuffer = &gateBuf; a.upReduceBuffer = &upBuf; a.downReduceBuffer = &downBuf;
    auto stream = reinterpret_cast<aclrtStream>(uintptr_t {0xb000});
    Status("reduce launch", TileXRMoonEp::TileXRMoonEpRunReduceGradV1(&a, stream), 0);
    Check(launchCalls == 1 && seenParams.expertsToCopy == plan.expertsToCopy && seenParams.fullGateGrad == gate.data && seenParams.gateReduceBuffer == gateBuf.data && seenParams.fullUpGrad == up.data && seenParams.upReduceBuffer == upBuf.data && seenParams.fullDownGrad == down.data && seenParams.downReduceBuffer == downBuf.data && seenParams.status == plan.status, "reduce launch arguments mismatch");
    Check(seenContext.layout.gate.rowBytes == 24 && seenContext.layout.gate.chunkStride == 32 && seenContext.layout.gate.payloadBytes == 64 && seenContext.layout.up.rowBytes == 32 && seenContext.layout.down.rowBytes == 40 && seenContext.layout.iterationCount == 3, "reduce layout mismatch");
    gateBuf.shape[1] = 1;
    Status("reduce buffer shape", TileXRMoonEp::TileXRMoonEpRunReduceGradV1(&a, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    gateBuf = Buffer(0x7000, 2, 3); a.flags = 1;
    Status("reduce flags", TileXRMoonEp::TileXRMoonEpRunReduceGradV1(&a, stream), TILEXR_MOONEP_ERROR_INVALID_ARGUMENT);
    a.flags = 0; commArgs.extraFlag = 0;
    Status("reduce topology", TileXRMoonEp::TileXRMoonEpRunReduceGradV1(&a, stream), TILEXR_MOONEP_ERROR_NOT_SUPPORTED);
    Reset(); launchRet = -88;
    Status("reduce launch failure", TileXRMoonEp::TileXRMoonEpRunReduceGradV1(&a, stream), -88);
}
}
extern "C" int TileXRGetCommArgsHost(TileXRCommPtr, TileXR::CommArgs *&out) { out = hostRet == 0 ? &commArgs : nullptr; return hostRet; }
extern "C" int TileXRGetCommArgsDev(TileXRCommPtr, GM_ADDR &out) { out = devRet == 0 ? devArgs : nullptr; return devRet; }
extern "C" int TileXRCommNextMagic(TileXRCommPtr, int64_t *out) { if (magicRet == 0) *out = nextMagic; return magicRet; }
namespace TileXRMoonEp { int TileXRMoonEpLaunchReduceGradKernel(const ReduceGradParams &p, const ReduceGradLaunchContext &c) { ++launchCalls; seenParams = p; seenContext = c; return launchRet; } }
int main() { TestLaunch(); return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE; }
