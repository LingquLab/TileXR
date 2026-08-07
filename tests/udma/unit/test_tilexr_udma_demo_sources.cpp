#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int g_failures = 0;

std::string RepoPath(const std::string &path)
{
#ifdef TILEXR_SOURCE_ROOT
    return std::string(TILEXR_SOURCE_ROOT) + "/" + path;
#else
    return path;
#endif
}

std::string ReadFile(const std::string &path)
{
    const std::string fullPath = RepoPath(path);
    std::ifstream input(fullPath.c_str());
    if (!input) {
        std::cerr << "failed to open " << fullPath << std::endl;
        ++g_failures;
        return {};
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

void CheckContains(const std::string &path, const std::string &text, const std::string &needle)
{
    if (text.find(needle) == std::string::npos) {
        std::cerr << path << " missing: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckNotContains(const std::string &path, const std::string &text, const std::string &needle)
{
    if (text.find(needle) != std::string::npos) {
        std::cerr << path << " should not contain: " << needle << std::endl;
        ++g_failures;
    }
}

void CheckOrdered(
    const std::string &path, const std::string &text, const std::string &first, const std::string &second)
{
    const size_t firstPos = text.find(first);
    const size_t secondPos = text.find(second);
    if (firstPos == std::string::npos || secondPos == std::string::npos || firstPos >= secondPos) {
        std::cerr << path << " expected '" << first << "' before '" << second << "'" << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    const std::string demoPath = "tests/udma/demo/tilexr_udma_demo.cpp";
    const std::string runnerPath = "tests/udma/demo/run_tilexr_udma_data_channel_probe_mpi.sh";
    const std::string hccpDefsPath = "src/comm/udma/tilexr_hccp_defs.h";
    const std::string demo = ReadFile(demoPath);
    const std::string runner = ReadFile(runnerPath);
    const std::string hccpDefs = ReadFile(hccpDefsPath);
    CheckContains(demoPath, demo, "TileXRUDMARegister");
    CheckContains(demoPath, demo, "TileXRUDMAUnregister");
    CheckContains(demoPath, demo, "TileXRUDMAGetQpCount");
    CheckContains(demoPath, demo, "ExtraFlag::UDMA");
    CheckContains(demoPath, demo, "launch_tilexr_udma_all_gather");
    CheckContains(demoPath, demo, "launch_tilexr_udma_put_signal");
    CheckContains(demoPath, demo, "DemoBarrierAll");
    CheckContains(demoPath, demo, "TILEXR_DEMO_BARRIER_ADDR");
    CheckContains(demoPath, demo, "inet_pton(AF_INET, host.c_str()");
    CheckContains(demoPath, demo, "htonl(INADDR_ANY)");
    CheckNotContains(demoPath, demo, "addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK)");
    CheckNotContains(demoPath, demo, "aclshmem");
    CheckNotContains(demoPath, demo, "shmem_");
    CheckContains(demoPath, demo, "TILEXR_DEMO_EXPECT_UDMA");
    CheckContains(demoPath, demo, "TILEXR_DEMO_EXPECT_QP_COUNT");
    CheckContains(demoPath, demo, "TILEXR_DEMO_REGISTERED_BYTES");
    CheckContains(demoPath, demo, "TILEXR_DEMO_REREGISTER");
    CheckContains(demoPath, demo, "TILEXR_DEMO_WARMUP_ITERS");
    CheckContains(demoPath, demo, "TILEXR_DEMO_TIMED_ITERS");
    CheckContains(demoPath, demo, "\"TILEXR_DEMO_TIMED_ITERS\", 1,");
    CheckContains(demoPath, demo, "TILEXR_DEMO_TIMED_ITERS must be positive");
    CheckContains(demoPath, demo, "warmupIters > 0U || timedIters > 1U");
    CheckContains(demoPath, demo, "aclrtEventElapsedTime");
    CheckContains(demoPath, demo, "std::chrono::steady_clock");
    CheckContains(demoPath, demo, "wall_start_ns=");
    CheckContains(demoPath, demo, "wall_stop_ns=");
    CheckContains(demoPath, demo, "wall_tx_GBps=");
    CheckContains(demoPath, demo, "TILEXR_DEMO_PERF_BARRIER_ADDR");
    CheckContains(demoPath, demo, "all communicators ready for timed perf");
    CheckContains(demoPath, demo, "TILEXR_UDMA_PERF");
    CheckContains(demoPath, demo, "kDefaultRegisteredBytes = 2097152");
    CheckContains(demoPath, demo, "TileXR UDMA unavailable as expected");
    CheckContains(demoPath, demo, "qpCountRet != TileXR::TILEXR_ERROR_NOT_SUPPORT");
    CheckContains(demoPath, demo, "actualQpCount != 0");
    CheckContains(demoPath, demo, "registeredBytes < payloadBytes");
    CheckContains(demoPath, demo, "TileXR UDMA re-register lifecycle success");
    CheckContains(demoPath, demo, "TileXRUDMARegister(re-register)");
    CheckContains(demoPath, demo, "TileXRUDMAUnregister(re-register)");
    CheckContains(demoPath, demo, "ok = unregisterOk && ok");
    CheckOrdered(demoPath, demo, "if (!expectUdma)", "if (!CheckAcl(rank, \"aclrtMalloc debug\"");

    CheckContains(runnerPath, runner, "--expect-udma <0|1>");
    CheckContains(runnerPath, runner, "--case <name>");
    CheckContains(runnerPath, runner, "--mismatch-qp-route-spec <spec>");
    CheckContains(runnerPath, runner, "--expect-qp-count <N>");
    CheckContains(runnerPath, runner, "--registered-bytes <N>");
    CheckContains(runnerPath, runner, "--reregister <0|1>");
    CheckContains(runnerPath, runner, "--no-reregister");
    CheckContains(runnerPath, runner, "--warmup-iters <N>");
    CheckContains(runnerPath, runner, "--iterations <N>");
    CheckContains(runnerPath, runner, "--perf-barrier-addr <ip:port>");
    CheckContains(runnerPath, runner, "--perf-barrier-rank-base <N>");
    CheckContains(runnerPath, runner, "--perf-barrier-size <N>");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_EXPECT_UDMA");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_EXPECT_QP_COUNT");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_REGISTERED_BYTES");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_REREGISTER");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_WARMUP_ITERS");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_TIMED_ITERS");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_PERF_BARRIER_ADDR");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_PERF_BARRIER_RANK_BASE");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_PERF_BARRIER_SIZE");
    CheckContains(runnerPath, runner, "-genv TILEXR_DEMO_EXPECT_UDMA");
    CheckContains(runnerPath, runner, "-genv TILEXR_DEMO_EXPECT_QP_COUNT");
    CheckContains(runnerPath, runner, "-genv TILEXR_DEMO_REGISTERED_BYTES");
    CheckContains(runnerPath, runner, "-genv TILEXR_DEMO_REREGISTER");
    CheckContains(runnerPath, runner, "-genv TILEXR_DEMO_WARMUP_ITERS");
    CheckContains(runnerPath, runner, "-genv TILEXR_DEMO_TIMED_ITERS");
    CheckContains(runnerPath, runner, "-genv TILEXR_DEMO_PERF_BARRIER_ADDR");
    CheckContains(runnerPath, runner, "-genv TILEXR_DEMO_PERF_BARRIER_RANK_BASE");
    CheckContains(runnerPath, runner, "-genv TILEXR_DEMO_PERF_BARRIER_SIZE");
    CheckContains(runnerPath, runner, "REGISTERED_BYTES=\"${TILEXR_DEMO_REGISTERED_BYTES:-2097152}\"");
    CheckContains(runnerPath, runner, "WARMUP_ITERS=\"${TILEXR_DEMO_WARMUP_ITERS:-0}\"");
    CheckContains(runnerPath, runner, "TIMED_ITERS=\"${TILEXR_DEMO_TIMED_ITERS:-1}\"");
    CheckContains(runnerPath, runner, "--warmup-iters must be a non-negative decimal integer");
    CheckContains(runnerPath, runner, "--iterations must be a positive decimal integer");
    CheckContains(runnerPath, runner, "timeout --signal=TERM --kill-after=30");
    CheckContains(runnerPath, runner, "legacy)");
    CheckContains(runnerPath, runner, "exact-2m)");
    CheckContains(runnerPath, runner, "two-qp)");
    CheckContains(runnerPath, runner, "three-qp)");
    CheckContains(runnerPath, runner, "missing-route)");
    CheckContains(runnerPath, runner, "mismatch)");
    CheckContains(runnerPath, runner, "reuse)");
    CheckContains(runnerPath, runner, "TILEXR_DEMO_MISMATCH_QP_ROUTE_SPEC");
    CheckContains(runnerPath, runner, "UNSET_QP_ROUTE_SPEC=1");
    CheckContains(runnerPath, runner, "unset TILEXR_UDMA_QP_ROUTE_SPEC");
    CheckContains(runnerPath, runner, "${PMI_RANK:-${OMPI_COMM_WORLD_RANK");
    CheckContains(runnerPath, runner, "MV2_COMM_WORLD_RANK");
    CheckNotContains(runnerPath, runner, "if [[ -n \"${QP_ROUTE_SPEC}\" ]]");

    const std::string kernelPath = "tests/udma/demo/tilexr_udma_demo_kernel.cpp";
    const std::string kernel = ReadFile(kernelPath);
    CheckContains(kernelPath, kernel, "UDMARegistryEnabled");
    CheckContains(kernelPath, kernel, "UDMAPutSignalNbi<int32_t>");
    CheckContains(kernelPath, kernel, "UDMAQuiet");
    CheckContains(kernelPath, kernel, "UDMAQpCount(args)");
    CheckContains(kernelPath, kernel, "UDMAPutNbiOnQpWithFlagDeferred<int32_t>");
    CheckContains(kernelPath, kernel, "UDMAFlushQpDoorbell(args, peer, qpIdx)");
    CheckContains(kernelPath, kernel, "UDMAQuietStatusOnQp(args, peer, qpIdx)");
    CheckContains(kernelPath, kernel, "return allQpsSucceeded;");
    CheckContains(kernelPath, kernel, "TILEXR_UDMA_SQE_FLAG_ORDERED_COMPLETION");
    CheckContains(kernelPath, kernel, "qpIdx * elementsPerQp");
    CheckContains(demoPath, demo, "ExpectedDataValue(qpIdx, srcRank)");
    CheckContains(demoPath, demo, "per-QP completion status");
    CheckContains(demoPath, demo, "ValidateKernelDebug");
    CheckNotContains(kernelPath, kernel, "aclshmem");
    CheckNotContains(kernelPath, kernel, "shmem_");

    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_LOCAL_ONLY = 1");
    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_READ = (1 << 1)");
    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_WRITE = (1 << 2)");
    CheckContains(hccpDefsPath, hccpDefs, "MEM_SEG_ACCESS_ATOMIC = (1 << 3)");
    CheckContains(hccpDefsPath, hccpDefs,
        "MEM_SEG_ACCESS_DEFAULT = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC");

    const std::string transportPath = "src/comm/udma/tilexr_udma_transport.cpp";
    const std::string transport = ReadFile(transportPath);
    CheckContains(transportPath, transport, "ResolveUDMATopologyEid(");
    CheckContains(transportPath, transport, "ResolveUDMAPortCountEid(");
    CheckNotContains(transportPath, transport, "std::regex");

    if (g_failures == 0) {
        std::cout << "TileXR UDMA demo source checks passed" << std::endl;
    }
    return g_failures == 0 ? 0 : 1;
}
