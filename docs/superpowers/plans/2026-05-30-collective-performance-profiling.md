# Collective Performance Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a low-overhead, opt-in TileXR collective profiling path that records per-rank/per-core/per-stage kernel timing and emits bottleneck-first reports with timeline and AI export.

**Architecture:** Add a versioned generic trace schema in `src/include`, a collectives host session/reporting layer, and a profile-gated AscendC kernel helper. The first complete kernel instrumentation target is `TileXRAllGatherBigData`; other collective paths only receive common launch plumbing.

**Tech Stack:** C++14 host code, AscendC CCE kernel code, ACL runtime APIs, CMake/CTest, existing TileXR collectives test style.

---

## File Map

- Create `src/include/tilexr_perf_trace.h`: stable public trace structs, constants, stage ids, layout helpers, cycle conversion helpers.
- Create `src/include/tilexr_collectives_perf.h`: public optional C API for collectives perf sessions.
- Create `src/collectives/host/perf_trace_report.h`: pure host summary/report API used by tests and session code.
- Create `src/collectives/host/perf_trace_report.cpp`: summary aggregation, deterministic analysis, JSON/CSV/HTML/prompt writing.
- Create `src/collectives/host/perf_trace_session.h`: internal session object and active-session accessors.
- Create `src/collectives/host/perf_trace_session.cpp`: ACL buffer allocation, launch preparation, report copy-back.
- Modify `src/collectives/host/collective_kernel.h`: add a trailing perf trace pointer to private kernel launch args.
- Modify `src/collectives/host/collective_kernel.cpp`: pass the active session trace control pointer into kernel launch args.
- Modify `src/collectives/host/tilexr_collectives.cpp`: no public API signature changes; keep existing collective calls unchanged.
- Create `src/collectives/kernels/perf_trace_kernel.h`: AscendC helper that compiles away when profiling is disabled.
- Modify `src/collectives/kernels/collectives.h`: add private trailing `perfTrace` kernel argument.
- Modify `src/collectives/kernels/kernels/collectives.cce`: pass `perfTrace` through internal AllGather/AllToAll helper macro argument lists.
- Modify `src/collectives/kernels/kernels/lcal_allgather_big_data.cce`: insert complete stage instrumentation for the big-data AllGather branch.
- Modify `src/collectives/kernels/CMakeLists.txt`: add `TILEXR_COLLECTIVES_ENABLE_PROFILING` CCE compile definition when enabled.
- Modify `src/collectives/CMakeLists.txt`: compile new host perf sources and install new public headers.
- Modify `tests/collectives/CMakeLists.txt`: add host-only tests for trace layout, report generation, session API, and source wiring.
- Create `tests/collectives/unit/test_tilexr_perf_trace_layout.cpp`: validates trace structs and helpers.
- Create `tests/collectives/unit/test_collective_perf_report.cpp`: validates aggregation, bottleneck analysis, and artifact generation.
- Create `tests/collectives/unit/test_collective_perf_session.cpp`: validates public perf session config and active-session lifecycle.
- Modify `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`: source-check CLI profile flags and report hooks.
- Modify `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`: source-check profile build gate and kernel instrumentation ownership.
- Modify `tests/collectives/unit/test_tilexr_collectives_header_compile.cpp`: include and type-check the new public perf headers after the perf API exists.
- Modify `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`: add profile CLI flags and report generation calls.
- Modify `tests/collectives/README.md`: document profile build and report artifacts.

## Task 1: Public Trace Schema

**Files:**
- Create: `src/include/tilexr_perf_trace.h`
- Create: `tests/collectives/unit/test_tilexr_perf_trace_layout.cpp`
- Modify: `tests/collectives/CMakeLists.txt`

- [ ] **Step 1: Write the failing layout test**

Create `tests/collectives/unit/test_tilexr_perf_trace_layout.cpp`:

```cpp
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "tilexr_perf_trace.h"

namespace {

int g_failures = 0;

template <typename T>
void CheckEq(const char *label, T actual, T expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}

void CheckTrue(const char *label, bool value)
{
    if (!value) {
        std::cerr << label << " expected true" << std::endl;
        ++g_failures;
    }
}

} // namespace

int main()
{
    CheckEq("trace magic", TileXR::TILEXR_PERF_TRACE_MAGIC, 0x54585054u);
    CheckEq("trace version", TileXR::TILEXR_PERF_TRACE_VERSION, 1u);
    CheckEq("stage count", TileXR::TILEXR_PERF_STAGE_COUNT, 7u);
    CheckEq("a5 divisor", TileXR::PerfTraceCycleDivisor(TileXR::PerfChipClass::A5), 1000u);
    CheckEq("generic divisor", TileXR::PerfTraceCycleDivisor(TileXR::PerfChipClass::GENERIC), 50u);
    CheckEq("cycles to us", TileXR::PerfTraceCyclesToUs(2500, 50), 50.0);
    CheckEq("stats offset",
            TileXR::PerfTraceStatsOffset(1, 3, 2, 8, TileXR::TILEXR_PERF_STAGE_COUNT),
            static_cast<size_t>(1 * 8 * TileXR::TILEXR_PERF_STAGE_COUNT +
                                3 * TileXR::TILEXR_PERF_STAGE_COUNT + 2));

    TileXR::TileXRPerfTraceHeader header {};
    header.magic = TileXR::TILEXR_PERF_TRACE_MAGIC;
    header.version = TileXR::TILEXR_PERF_TRACE_VERSION;
    header.headerSize = sizeof(TileXR::TileXRPerfTraceHeader);
    header.stageDescSize = sizeof(TileXR::TileXRPerfStageDesc);
    header.coreStageStatsSize = sizeof(TileXR::TileXRPerfCoreStageStats);
    CheckTrue("header size is recorded", header.headerSize >= sizeof(uint32_t) * 8);
    CheckTrue("stats carry raw cycles", offsetof(TileXR::TileXRPerfCoreStageStats, sumCycles) > 0);
    CheckTrue("stats carry timeline bounds", offsetof(TileXR::TileXRPerfCoreStageStats, lastEndCycle) >
        offsetof(TileXR::TileXRPerfCoreStageStats, firstStartCycle));
    return g_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Wire the test target**

In `tests/collectives/CMakeLists.txt`, add the target near the existing unit executables:

```cmake
add_executable(test_tilexr_perf_trace_layout
    unit/test_tilexr_perf_trace_layout.cpp
)
```

Add the test registration:

```cmake
add_test(NAME test_tilexr_perf_trace_layout COMMAND test_tilexr_perf_trace_layout)
```

Add it to the `install(TARGETS ...)` list:

```cmake
    test_tilexr_perf_trace_layout
```

- [ ] **Step 3: Run the new test and verify it fails**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B build -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON -DBUILD_TESTING=ON
cmake --build build --target test_tilexr_perf_trace_layout -j"$(nproc)"
```

Expected: build fails with `fatal error: tilexr_perf_trace.h: No such file or directory`.

- [ ] **Step 4: Add the public trace header**

Create `src/include/tilexr_perf_trace.h`:

```cpp
#ifndef TILEXR_PERF_TRACE_H
#define TILEXR_PERF_TRACE_H

#include <cstddef>
#include <cstdint>

namespace TileXR {

constexpr uint32_t TILEXR_PERF_TRACE_MAGIC = 0x54585054u; // TXPT
constexpr uint32_t TILEXR_PERF_TRACE_VERSION = 1u;
constexpr uint32_t TILEXR_PERF_MAX_STAGE_NAME = 32u;
constexpr uint32_t TILEXR_PERF_STAGE_COUNT = 7u;

enum class PerfChipClass : uint32_t {
    GENERIC = 0,
    A5 = 1,
};

enum class PerfStageCategory : uint32_t {
    TOTAL = 0,
    COPY = 1,
    WAIT = 2,
    SYNC = 3,
    BARRIER = 4,
};

enum class PerfBarrierPolicy : uint32_t {
    NO_BARRIER = 0,
    END_BARRIER_ONLY = 1,
    BARRIERED = 2,
};

enum class PerfStageId : uint32_t {
    KERNEL_TOTAL = 0,
    CHUNK_TOTAL = 1,
    POST_SYNC = 2,
    LOCAL_INPUT_TO_IPC = 3,
    FLAG_POLL_WAIT = 4,
    PEER_IPC_TO_OUTPUT = 5,
    CHUNK_BARRIER = 6,
};

struct TileXRPerfTraceHeader {
    uint32_t magic = TILEXR_PERF_TRACE_MAGIC;
    uint32_t version = TILEXR_PERF_TRACE_VERSION;
    uint32_t headerSize = sizeof(TileXRPerfTraceHeader);
    uint32_t stageDescSize = sizeof(uint32_t) * 4 + TILEXR_PERF_MAX_STAGE_NAME;
    uint32_t coreStageStatsSize = sizeof(uint32_t) * 4 + sizeof(uint64_t) * 8;
    uint32_t flags = 0;
    uint32_t rank = 0;
    uint32_t rankSize = 0;
    uint32_t blockDim = 0;
    uint32_t maxCoreCount = 0;
    uint32_t stageCount = TILEXR_PERF_STAGE_COUNT;
    uint32_t cycleToUsDivisor = 50;
    uint64_t launchId = 0;
    uint64_t messageBytes = 0;
    uint32_t opType = 0;
    uint32_t dataType = 255;
    uint64_t statsOffset = 0;
    uint64_t statsBytes = 0;
};

struct TileXRPerfStageDesc {
    uint32_t stageId = 0;
    PerfStageCategory category = PerfStageCategory::TOTAL;
    PerfBarrierPolicy barrierPolicy = PerfBarrierPolicy::NO_BARRIER;
    uint32_t displayOrder = 0;
    char name[TILEXR_PERF_MAX_STAGE_NAME] = {};
};

struct TileXRPerfCoreStageStats {
    uint32_t rank = 0;
    uint32_t core = 0;
    uint32_t stageId = 0;
    uint32_t reserved = 0;
    uint64_t count = 0;
    uint64_t sumCycles = 0;
    uint64_t minCycles = 0;
    uint64_t maxCycles = 0;
    uint64_t firstStartCycle = 0;
    uint64_t lastEndCycle = 0;
    uint64_t aux0 = 0;
    uint64_t aux1 = 0;
};

inline size_t PerfTraceStatsOffset(uint32_t rank, uint32_t core, uint32_t stage,
                                   uint32_t maxCoreCount, uint32_t stageCount)
{
    return (static_cast<size_t>(rank) * maxCoreCount * stageCount) +
        (static_cast<size_t>(core) * stageCount) + stage;
}

inline uint32_t PerfTraceCycleDivisor(PerfChipClass chipClass)
{
    return chipClass == PerfChipClass::A5 ? 1000u : 50u;
}

inline double PerfTraceCyclesToUs(uint64_t cycles, uint32_t divisor)
{
    return divisor == 0 ? 0.0 : static_cast<double>(cycles) / static_cast<double>(divisor);
}

} // namespace TileXR

#endif // TILEXR_PERF_TRACE_H
```

- [ ] **Step 5: Run the layout test and commit**

Run:

```bash
cmake --build build --target test_tilexr_perf_trace_layout -j"$(nproc)"
./build/tests/collectives/test_tilexr_perf_trace_layout
```

Expected: executable exits with status `0`.

Commit:

```bash
git add src/include/tilexr_perf_trace.h tests/collectives/CMakeLists.txt \
    tests/collectives/unit/test_tilexr_perf_trace_layout.cpp
git commit -m "feat: add perf trace schema"
```

## Task 2: Pure Host Summary and Report Writer

**Files:**
- Create: `src/collectives/host/perf_trace_report.h`
- Create: `src/collectives/host/perf_trace_report.cpp`
- Create: `tests/collectives/unit/test_collective_perf_report.cpp`
- Modify: `src/collectives/CMakeLists.txt`
- Modify: `tests/collectives/CMakeLists.txt`

- [ ] **Step 1: Write the failing report test**

Create `tests/collectives/unit/test_collective_perf_report.cpp`:

```cpp
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "perf_trace_report.h"
#include "tilexr_perf_trace.h"

namespace {

int g_failures = 0;

void CheckTrue(const std::string &label, bool value)
{
    if (!value) {
        std::cerr << label << " expected true" << std::endl;
        ++g_failures;
    }
}

std::string ReadFile(const std::string &path)
{
    std::ifstream in(path.c_str());
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

TileXR::TileXRPerfCoreStageStats Stat(uint32_t rank, uint32_t core, TileXR::PerfStageId stage,
                                      uint64_t count, uint64_t sum, uint64_t max)
{
    TileXR::TileXRPerfCoreStageStats stat {};
    stat.rank = rank;
    stat.core = core;
    stat.stageId = static_cast<uint32_t>(stage);
    stat.count = count;
    stat.sumCycles = sum;
    stat.minCycles = count == 0 ? 0 : sum / count;
    stat.maxCycles = max;
    stat.firstStartCycle = 100;
    stat.lastEndCycle = 100 + sum;
    return stat;
}

} // namespace

int main()
{
    TileXR::TileXRPerfTraceHeader header {};
    header.rankSize = 2;
    header.maxCoreCount = 2;
    header.blockDim = 2;
    header.stageCount = TileXR::TILEXR_PERF_STAGE_COUNT;
    header.cycleToUsDivisor = 50;
    header.opType = static_cast<uint32_t>(TileXR::TileXRType::ALL_GATHER);
    header.messageBytes = 64 * 1024 * 1024;

    std::vector<TileXR::TileXRPerfCoreStageStats> stats(
        header.rankSize * header.maxCoreCount * header.stageCount);
    stats[TileXR::PerfTraceStatsOffset(0, 0, static_cast<uint32_t>(TileXR::PerfStageId::FLAG_POLL_WAIT),
        header.maxCoreCount, header.stageCount)] =
        Stat(0, 0, TileXR::PerfStageId::FLAG_POLL_WAIT, 3, 3000, 1200);
    stats[TileXR::PerfTraceStatsOffset(1, 1, static_cast<uint32_t>(TileXR::PerfStageId::PEER_IPC_TO_OUTPUT),
        header.maxCoreCount, header.stageCount)] =
        Stat(1, 1, TileXR::PerfStageId::PEER_IPC_TO_OUTPUT, 2, 8000, 5000);

    const auto summaries = TileXRCollectives::Host::SummarizePerfTrace(header, stats);
    CheckTrue("summary is non-empty", !summaries.empty());
    CheckTrue("peer copy is top stage", summaries.front().stageId ==
        static_cast<uint32_t>(TileXR::PerfStageId::PEER_IPC_TO_OUTPUT));

    const auto findings = TileXRCollectives::Host::AnalyzePerfTrace(header, summaries);
    CheckTrue("analysis mentions bottleneck", findings[0].find("bottleneck") != std::string::npos);

    const std::string dir = "/tmp/tilexr_perf_report_test";
    TileXRCollectives::Host::PerfReportOptions options {};
    options.outputDir = dir;
    options.emitAiPrompt = true;
    CheckTrue("write artifacts",
        TileXRCollectives::Host::WritePerfTraceReports(header, stats, options) == TileXR::TILEXR_SUCCESS);

    CheckTrue("trace json", ReadFile(dir + "/trace.json").find("\"raw_cycles\"") != std::string::npos);
    CheckTrue("summary csv", ReadFile(dir + "/summary.csv").find("stage,rank,core") != std::string::npos);
    CheckTrue("analysis md", ReadFile(dir + "/analysis.md").find("bottleneck") != std::string::npos);
    CheckTrue("html", ReadFile(dir + "/report.html").find("Bottleneck First") != std::string::npos);
    CheckTrue("ai prompt", ReadFile(dir + "/ai_prompt.md").find("TileXR collective profiling") != std::string::npos);
    return g_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Wire sources and test target**

In `src/collectives/CMakeLists.txt`, add `host/perf_trace_report.cpp` to `TILEXR_COLLECTIVES_SOURCE_FILE`:

```cmake
        host/perf_trace_report.cpp
```

In `tests/collectives/CMakeLists.txt`, add:

```cmake
add_executable(test_collective_perf_report
    unit/test_collective_perf_report.cpp
)

target_include_directories(test_collective_perf_report PRIVATE
    ${TILEXR_ROOT}/src/collectives/host
)

target_link_libraries(test_collective_perf_report
    ${TILEXR_COLLECTIVES_TEST_TARGET}
)

add_test(NAME test_collective_perf_report COMMAND test_collective_perf_report)
```

Add `test_collective_perf_report` to the `install(TARGETS ...)` list.

- [ ] **Step 3: Run the test and verify it fails**

Run:

```bash
cmake --build build --target test_collective_perf_report -j"$(nproc)"
```

Expected: build fails with `fatal error: perf_trace_report.h: No such file or directory`.

- [ ] **Step 4: Add report API and implementation**

Create `src/collectives/host/perf_trace_report.h`:

```cpp
#ifndef TILEXR_COLLECTIVES_HOST_PERF_TRACE_REPORT_H
#define TILEXR_COLLECTIVES_HOST_PERF_TRACE_REPORT_H

#include <cstdint>
#include <string>
#include <vector>

#include "tilexr_perf_trace.h"
#include "tilexr_types.h"

namespace TileXRCollectives {
namespace Host {

struct PerfStageSummary {
    uint32_t stageId = 0;
    std::string stageName;
    uint64_t count = 0;
    uint64_t sumCycles = 0;
    uint64_t maxCycles = 0;
    uint32_t maxRank = 0;
    uint32_t maxCore = 0;
    double sumUs = 0.0;
};

struct PerfReportOptions {
    std::string outputDir;
    bool emitAiPrompt = false;
};

const char *PerfStageName(uint32_t stageId);
std::vector<PerfStageSummary> SummarizePerfTrace(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<TileXR::TileXRPerfCoreStageStats> &stats);
std::vector<std::string> AnalyzePerfTrace(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<PerfStageSummary> &summaries);
int WritePerfTraceReports(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<TileXR::TileXRPerfCoreStageStats> &stats,
    const PerfReportOptions &options);

} // namespace Host
} // namespace TileXRCollectives

#endif // TILEXR_COLLECTIVES_HOST_PERF_TRACE_REPORT_H
```

Create `src/collectives/host/perf_trace_report.cpp` with these required functions:

```cpp
#include "perf_trace_report.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

namespace TileXRCollectives {
namespace Host {
namespace {

bool EnsureDirectory(const std::string &path)
{
    if (path.empty()) {
        return false;
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

void WriteTextFile(const std::string &path, const std::string &text, bool &ok)
{
    std::ofstream out(path.c_str());
    if (!out.is_open()) {
        ok = false;
        return;
    }
    out << text;
}

} // namespace

const char *PerfStageName(uint32_t stageId)
{
    switch (static_cast<TileXR::PerfStageId>(stageId)) {
        case TileXR::PerfStageId::KERNEL_TOTAL: return "kernel_total";
        case TileXR::PerfStageId::CHUNK_TOTAL: return "chunk_total";
        case TileXR::PerfStageId::POST_SYNC: return "post_sync";
        case TileXR::PerfStageId::LOCAL_INPUT_TO_IPC: return "local_input_to_ipc";
        case TileXR::PerfStageId::FLAG_POLL_WAIT: return "flag_poll_wait";
        case TileXR::PerfStageId::PEER_IPC_TO_OUTPUT: return "peer_ipc_to_output";
        case TileXR::PerfStageId::CHUNK_BARRIER: return "chunk_barrier";
        default: return "unknown";
    }
}

std::vector<PerfStageSummary> SummarizePerfTrace(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<TileXR::TileXRPerfCoreStageStats> &stats)
{
    std::vector<PerfStageSummary> summaries(header.stageCount);
    for (uint32_t stage = 0; stage < header.stageCount; ++stage) {
        summaries[stage].stageId = stage;
        summaries[stage].stageName = PerfStageName(stage);
    }
    for (const auto &stat : stats) {
        if (stat.stageId >= summaries.size() || stat.count == 0) {
            continue;
        }
        auto &summary = summaries[stat.stageId];
        summary.count += stat.count;
        summary.sumCycles += stat.sumCycles;
        summary.sumUs += TileXR::PerfTraceCyclesToUs(stat.sumCycles, header.cycleToUsDivisor);
        if (stat.maxCycles > summary.maxCycles) {
            summary.maxCycles = stat.maxCycles;
            summary.maxRank = stat.rank;
            summary.maxCore = stat.core;
        }
    }
    summaries.erase(std::remove_if(summaries.begin(), summaries.end(),
        [](const PerfStageSummary &summary) { return summary.count == 0; }), summaries.end());
    std::sort(summaries.begin(), summaries.end(),
        [](const PerfStageSummary &lhs, const PerfStageSummary &rhs) {
            return lhs.sumCycles > rhs.sumCycles;
        });
    return summaries;
}

std::vector<std::string> AnalyzePerfTrace(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<PerfStageSummary> &summaries)
{
    (void)header;
    std::vector<std::string> findings;
    if (summaries.empty()) {
        findings.push_back("No profiling samples were recorded.");
        return findings;
    }
    std::ostringstream top;
    top << "Top bottleneck stage is " << summaries.front().stageName
        << " with " << std::fixed << std::setprecision(3) << summaries.front().sumUs
        << " us aggregated across recorded rank/core slots.";
    findings.push_back(top.str());
    for (const auto &summary : summaries) {
        if (summary.stageName == "flag_poll_wait") {
            findings.push_back("flag_poll_wait recorded time; inspect rank progress skew and peer readiness.");
        }
        if (summary.stageName == "peer_ipc_to_output") {
            findings.push_back("peer_ipc_to_output recorded time; inspect peer IPC read bandwidth.");
        }
    }
    return findings;
}

int WritePerfTraceReports(
    const TileXR::TileXRPerfTraceHeader &header,
    const std::vector<TileXR::TileXRPerfCoreStageStats> &stats,
    const PerfReportOptions &options)
{
    if (!EnsureDirectory(options.outputDir)) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    const auto summaries = SummarizePerfTrace(header, stats);
    const auto findings = AnalyzePerfTrace(header, summaries);
    bool ok = true;

    std::ostringstream json;
    json << "{\n  \"raw_cycles\": true,\n  \"rank_size\": " << header.rankSize
         << ",\n  \"cycle_to_us_divisor\": " << header.cycleToUsDivisor << "\n}\n";
    WriteTextFile(options.outputDir + "/trace.json", json.str(), ok);

    std::ostringstream csv;
    csv << "stage,rank,core,count,sum_cycles,max_cycles,sum_us\n";
    for (const auto &stat : stats) {
        if (stat.count == 0) {
            continue;
        }
        csv << PerfStageName(stat.stageId) << ',' << stat.rank << ',' << stat.core << ','
            << stat.count << ',' << stat.sumCycles << ',' << stat.maxCycles << ','
            << TileXR::PerfTraceCyclesToUs(stat.sumCycles, header.cycleToUsDivisor) << '\n';
    }
    WriteTextFile(options.outputDir + "/summary.csv", csv.str(), ok);

    std::ostringstream analysis;
    for (const auto &finding : findings) {
        analysis << "- " << finding << '\n';
    }
    WriteTextFile(options.outputDir + "/analysis.md", analysis.str(), ok);

    std::ostringstream html;
    html << "<!doctype html><html><body><h1>Bottleneck First</h1><pre>"
         << analysis.str() << "</pre><h2>Timeline</h2></body></html>\n";
    WriteTextFile(options.outputDir + "/report.html", html.str(), ok);

    if (options.emitAiPrompt) {
        std::ostringstream prompt;
        prompt << "TileXR collective profiling summary\n" << analysis.str();
        WriteTextFile(options.outputDir + "/ai_prompt.md", prompt.str(), ok);
    }
    return ok ? TileXR::TILEXR_SUCCESS : TileXR::TILEXR_ERROR_INTERNAL;
}

} // namespace Host
} // namespace TileXRCollectives
```

- [ ] **Step 5: Run the report test and commit**

Run:

```bash
cmake --build build --target test_collective_perf_report -j"$(nproc)"
./build/tests/collectives/test_collective_perf_report
```

Expected: executable exits with status `0`; `/tmp/tilexr_perf_report_test/report.html` exists and contains `Bottleneck First`.

Commit:

```bash
git add src/collectives/host/perf_trace_report.h src/collectives/host/perf_trace_report.cpp \
    src/collectives/CMakeLists.txt tests/collectives/CMakeLists.txt \
    tests/collectives/unit/test_collective_perf_report.cpp
git commit -m "feat: add collective perf report writer"
```

## Task 3: Public Perf Session API

**Files:**
- Create: `src/include/tilexr_collectives_perf.h`
- Create: `src/collectives/host/perf_trace_session.h`
- Create: `src/collectives/host/perf_trace_session.cpp`
- Create: `tests/collectives/unit/test_collective_perf_session.cpp`
- Modify: `src/collectives/CMakeLists.txt`
- Modify: `tests/collectives/CMakeLists.txt`
- Modify: `tests/collectives/unit/test_tilexr_collectives_header_compile.cpp`

- [ ] **Step 1: Write the failing session API test**

Create `tests/collectives/unit/test_collective_perf_session.cpp`:

```cpp
#include <iostream>

#include "tilexr_collectives_perf.h"
#include "tilexr_types.h"

namespace {
int g_failures = 0;

void CheckStatus(const char *label, int actual, int expected)
{
    if (actual != expected) {
        std::cerr << label << " actual=" << actual << " expected=" << expected << std::endl;
        ++g_failures;
    }
}
} // namespace

int main()
{
    TileXRCollectivePerfSession session = nullptr;
    CheckStatus("create null config", TileXRCollectivePerfSessionCreate(nullptr, &session),
                TileXR::TILEXR_ERROR_PARA_CHECK_FAIL);

    TileXRCollectivePerfConfig config {};
    config.enabled = 1;
    config.outputDir = "/tmp/tilexr_perf_session_test";
    config.emitAiPrompt = 1;
    config.sampleEveryN = 1;
    CheckStatus("create", TileXRCollectivePerfSessionCreate(&config, &session), TileXR::TILEXR_SUCCESS);
    CheckStatus("set active", TileXRCollectivePerfSetActiveSession(session), TileXR::TILEXR_SUCCESS);
    CheckStatus("clear active", TileXRCollectivePerfSetActiveSession(nullptr), TileXR::TILEXR_SUCCESS);
    CheckStatus("destroy", TileXRCollectivePerfSessionDestroy(session), TileXR::TILEXR_SUCCESS);
    return g_failures == 0 ? 0 : 1;
}
```

- [ ] **Step 2: Wire the test target**

In `tests/collectives/CMakeLists.txt`, add:

```cmake
add_executable(test_collective_perf_session
    unit/test_collective_perf_session.cpp
)

target_link_libraries(test_collective_perf_session
    ${TILEXR_COLLECTIVES_TEST_TARGET}
)

add_test(NAME test_collective_perf_session COMMAND test_collective_perf_session)
```

Add `test_collective_perf_session` to the `install(TARGETS ...)` list.

- [ ] **Step 3: Run the test and verify it fails**

Run:

```bash
cmake --build build --target test_collective_perf_session -j"$(nproc)"
```

Expected: build fails with `fatal error: tilexr_collectives_perf.h: No such file or directory`.

- [ ] **Step 4: Add public header and internal session implementation**

Create `src/include/tilexr_collectives_perf.h`:

```cpp
#ifndef TILEXR_COLLECTIVES_PERF_H
#define TILEXR_COLLECTIVES_PERF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void *TileXRCollectivePerfSession;

typedef struct TileXRCollectivePerfConfig {
    int enabled;
    const char *outputDir;
    unsigned int sampleEveryN;
    int emitAiPrompt;
    const char *aiCommand;
} TileXRCollectivePerfConfig;

int TileXRCollectivePerfSessionCreate(const TileXRCollectivePerfConfig *config,
                                      TileXRCollectivePerfSession *session);
int TileXRCollectivePerfSessionDestroy(TileXRCollectivePerfSession session);
int TileXRCollectivePerfSetActiveSession(TileXRCollectivePerfSession session);
int TileXRCollectivePerfWriteReport(TileXRCollectivePerfSession session);

#ifdef __cplusplus
}
#endif

#endif // TILEXR_COLLECTIVES_PERF_H
```

Create `src/collectives/host/perf_trace_session.h`:

```cpp
#ifndef TILEXR_COLLECTIVES_HOST_PERF_TRACE_SESSION_H
#define TILEXR_COLLECTIVES_HOST_PERF_TRACE_SESSION_H

#include <string>
#include <vector>

#include "tilexr_collectives_perf.h"
#include "tilexr_perf_trace.h"

namespace TileXRCollectives {
namespace Host {

struct PerfTraceSession {
    TileXRCollectivePerfConfig config {};
    std::string outputDir;
    std::vector<TileXR::TileXRPerfCoreStageStats> hostStats;
    TileXR::TileXRPerfTraceHeader header {};
    void *deviceBuffer = nullptr;
    size_t deviceBufferBytes = 0;
};

PerfTraceSession *GetActivePerfTraceSession();
void SetActivePerfTraceSessionForHost(PerfTraceSession *session);

} // namespace Host
} // namespace TileXRCollectives

#endif // TILEXR_COLLECTIVES_HOST_PERF_TRACE_SESSION_H
```

Create `src/collectives/host/perf_trace_session.cpp`:

```cpp
#include "perf_trace_session.h"

#include <string>

#include "perf_trace_report.h"
#include "tilexr_types.h"

namespace TileXRCollectives {
namespace Host {
namespace {

PerfTraceSession *g_activeSession = nullptr;

} // namespace

PerfTraceSession *GetActivePerfTraceSession()
{
    return g_activeSession;
}

void SetActivePerfTraceSessionForHost(PerfTraceSession *session)
{
    g_activeSession = session;
}

} // namespace Host
} // namespace TileXRCollectives

extern "C" int TileXRCollectivePerfSessionCreate(const TileXRCollectivePerfConfig *config,
                                                  TileXRCollectivePerfSession *session)
{
    if (config == nullptr || session == nullptr || config->enabled == 0 || config->outputDir == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    auto *impl = new TileXRCollectives::Host::PerfTraceSession();
    impl->config = *config;
    impl->outputDir = config->outputDir;
    impl->config.outputDir = impl->outputDir.c_str();
    *session = impl;
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRCollectivePerfSessionDestroy(TileXRCollectivePerfSession session)
{
    delete static_cast<TileXRCollectives::Host::PerfTraceSession *>(session);
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRCollectivePerfSetActiveSession(TileXRCollectivePerfSession session)
{
    TileXRCollectives::Host::SetActivePerfTraceSessionForHost(
        static_cast<TileXRCollectives::Host::PerfTraceSession *>(session));
    return TileXR::TILEXR_SUCCESS;
}

extern "C" int TileXRCollectivePerfWriteReport(TileXRCollectivePerfSession session)
{
    auto *impl = static_cast<TileXRCollectives::Host::PerfTraceSession *>(session);
    if (impl == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    TileXRCollectives::Host::PerfReportOptions options {};
    options.outputDir = impl->outputDir;
    options.emitAiPrompt = impl->config.emitAiPrompt != 0;
    return TileXRCollectives::Host::WritePerfTraceReports(impl->header, impl->hostStats, options);
}
```

- [ ] **Step 5: Compile new source and install header**

In `src/collectives/CMakeLists.txt`, add to `TILEXR_COLLECTIVES_SOURCE_FILE`:

```cmake
        host/perf_trace_session.cpp
```

Change the install header list:

```cmake
install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_collectives.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_collectives_perf.h
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/tilexr_perf_trace.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
```

- [ ] **Step 6: Update the public header compile test**

Modify `tests/collectives/unit/test_tilexr_collectives_header_compile.cpp`:

```cpp
#include "tilexr_collectives.h"
#include "tilexr_collectives_perf.h"
#include "tilexr_perf_trace.h"

namespace {

using AllGatherFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXRCommPtr, aclrtStream);
using AllToAllFn = int (*)(void *, void *, int64_t, TileXR::TileXRDataType, TileXRCommPtr, aclrtStream);

} // namespace

int main()
{
    AllGatherFn allGather = &TileXRAllGather;
    AllToAllFn allToAll = &TileXRAllToAll;
    TileXR::TileXRPerfTraceHeader header {};
    TileXRCollectivePerfConfig config {};
    config.enabled = 1;
    return (allGather != nullptr && allToAll != nullptr &&
            header.magic == TileXR::TILEXR_PERF_TRACE_MAGIC &&
            config.enabled == 1) ? 0 : 1;
}
```

- [ ] **Step 7: Run the session and header tests, then commit**

Run:

```bash
cmake --build build --target test_collective_perf_session test_tilexr_collectives_header_compile -j"$(nproc)"
./build/tests/collectives/test_collective_perf_session
./build/tests/collectives/test_tilexr_collectives_header_compile
```

Expected: both executables exit with status `0`.

Commit:

```bash
git add src/include/tilexr_collectives_perf.h src/collectives/host/perf_trace_session.h \
    src/collectives/host/perf_trace_session.cpp src/collectives/CMakeLists.txt \
    tests/collectives/CMakeLists.txt tests/collectives/unit/test_collective_perf_session.cpp \
    tests/collectives/unit/test_tilexr_collectives_header_compile.cpp
git commit -m "feat: add collective perf session api"
```

## Task 4: Launch Plumbing and Device Trace Buffer

**Files:**
- Modify: `src/collectives/host/collective_kernel.h`
- Modify: `src/collectives/host/collective_kernel.cpp`
- Modify: `src/collectives/host/perf_trace_session.h`
- Modify: `src/collectives/host/perf_trace_session.cpp`
- Modify: `tests/collectives/unit/test_prepare_host_launch_context.cpp`
- Modify: `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`

- [ ] **Step 1: Add failing source checks for launch plumbing**

In `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`, extend `TestHostRegistrationLivesInCollectives()`:

```cpp
    CheckContains(kernelPath, kernel, "perfTrace");
    CheckContains(kernelPath, kernel, "PreparePerfTraceLaunch");
    CheckContains(kernelPath, kernel, "GetActivePerfTraceSession");
```

In `tests/collectives/unit/test_prepare_host_launch_context.cpp`, add an include and a layout check:

```cpp
#include "collective_kernel.h"

void CheckKernelArgsHasPerfTrace()
{
    TileXRCollectives::Host::AscendCCLKernelArgs args {};
    CheckPointer("args.perfTrace", args.perfTrace, nullptr);
}
```

Call `CheckKernelArgsHasPerfTrace();` before returning from `main()`.

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
cmake --build build --target test_tilexr_collectives_kernel_ownership test_prepare_host_launch_context -j"$(nproc)"
./build/tests/collectives/test_tilexr_collectives_kernel_ownership
./build/tests/collectives/test_prepare_host_launch_context
```

Expected: build or test fails because `perfTrace` and `PreparePerfTraceLaunch` are absent.

- [ ] **Step 3: Add the private kernel arg field**

Modify `src/collectives/host/collective_kernel.h`:

```cpp
struct AscendCCLKernelArgs {
    const void *input = nullptr;
    const void *output = nullptr;
    const void *commArgsPtr = nullptr;
    int64_t count = 0;
    int64_t magic = 0;
    int op = 0;
    int root = 0;
    int cycleCount = 0;
    const void *scale = nullptr;
    int64_t scaleCount = 0;
    const void *offset = nullptr;
    const void *perfTrace = nullptr;
};
```

- [ ] **Step 4: Add launch preparation hooks**

Add declarations to `src/collectives/host/perf_trace_session.h`:

```cpp
int PreparePerfTraceLaunch(PerfTraceSession *session, const TileXR::CommArgs &commArgs,
                           TileXR::TileXRType opType, TileXR::TileXRDataType dataType,
                           uint32_t blockDim, int64_t count, aclrtStream stream,
                           const void **deviceTrace);
```

Include ACL stream type:

```cpp
#include "acl/acl_base.h"
```

Add a first implementation in `src/collectives/host/perf_trace_session.cpp`:

```cpp
int PreparePerfTraceLaunch(PerfTraceSession *session, const TileXR::CommArgs &commArgs,
                           TileXR::TileXRType opType, TileXR::TileXRDataType dataType,
                           uint32_t blockDim, int64_t count, aclrtStream stream,
                           const void **deviceTrace)
{
    (void)stream;
    if (deviceTrace == nullptr) {
        return TileXR::TILEXR_ERROR_PARA_CHECK_FAIL;
    }
    *deviceTrace = nullptr;
    if (session == nullptr || session->config.enabled == 0) {
        return TileXR::TILEXR_SUCCESS;
    }
    session->header = TileXR::TileXRPerfTraceHeader {};
    session->header.rank = static_cast<uint32_t>(commArgs.rank);
    session->header.rankSize = static_cast<uint32_t>(commArgs.rankSize);
    session->header.blockDim = blockDim;
    session->header.maxCoreCount = blockDim;
    session->header.opType = static_cast<uint32_t>(opType);
    session->header.dataType = static_cast<uint32_t>(dataType);
    session->header.messageBytes = static_cast<uint64_t>(count);
    session->header.cycleToUsDivisor =
        (commArgs.extraFlag & TileXR::ExtraFlag::TOPO_910A5) != 0 ? 1000u : 50u;
    session->hostStats.assign(static_cast<size_t>(session->header.rankSize) *
        session->header.maxCoreCount * session->header.stageCount, TileXR::TileXRPerfCoreStageStats {});
    *deviceTrace = session->deviceBuffer;
    return TileXR::TILEXR_SUCCESS;
}
```

This step records host metadata. ACL device allocation is added in the next step so this step remains testable without hardware.

- [ ] **Step 5: Use the active session in launch args**

Modify `src/collectives/host/collective_kernel.cpp` includes:

```cpp
#include "perf_trace_session.h"
```

Before `rtTaskCfgInfo_t cfgInfo {};`, add:

```cpp
    const void *perfTrace = nullptr;
    const int perfRet = PreparePerfTraceLaunch(GetActivePerfTraceSession(), *context.hostArgs,
        type, dataType, blockDim, kernelCount, stream, &perfTrace);
    if (perfRet != TileXR::TILEXR_SUCCESS) {
        return perfRet;
    }
    args.perfTrace = perfTrace;
```

- [ ] **Step 6: Run launch plumbing tests and commit**

Run:

```bash
cmake --build build --target test_tilexr_collectives_kernel_ownership test_prepare_host_launch_context -j"$(nproc)"
./build/tests/collectives/test_tilexr_collectives_kernel_ownership
./build/tests/collectives/test_prepare_host_launch_context
```

Expected: both executables exit with status `0`.

Commit:

```bash
git add src/collectives/host/collective_kernel.h src/collectives/host/collective_kernel.cpp \
    src/collectives/host/perf_trace_session.h src/collectives/host/perf_trace_session.cpp \
    tests/collectives/unit/test_prepare_host_launch_context.cpp \
    tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp
git commit -m "feat: plumb perf trace through collective launch"
```

## Task 5: Profile Build Gate and Kernel Helper

**Files:**
- Create: `src/collectives/kernels/perf_trace_kernel.h`
- Modify: `src/collectives/kernels/CMakeLists.txt`
- Modify: `src/collectives/kernels/collectives.h`
- Modify: `src/collectives/kernels/kernels/collectives.cce`
- Modify: `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`

- [ ] **Step 1: Add failing source checks for the profile gate**

In `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`, inside `TestCollectivesKernelSourcesAreScoped()`, add:

```cpp
    const std::string perfHeaderPath = "src/collectives/kernels/perf_trace_kernel.h";
    const auto perfHeader = ReadFile(perfHeaderPath);
    CheckContains(perfHeaderPath, perfHeader, "TILEXR_COLLECTIVES_ENABLE_PROFILING");
    CheckContains(perfHeaderPath, perfHeader, "TileXRPerfStageBegin");
    CheckContains(perfHeaderPath, perfHeader, "TileXRPerfStageEnd");
    CheckContains(perfHeaderPath, perfHeader, "TileXRPerfAccumulateDuration");
```

Inside `TestCollectivesOwnsCceBuild()`, add:

```cpp
    CheckContains(kernelsCmakePath, kernelsCmake, "TILEXR_COLLECTIVES_ENABLE_PROFILING");
```

- [ ] **Step 2: Run source test and verify it fails**

Run:

```bash
cmake --build build --target test_tilexr_collectives_kernel_ownership -j"$(nproc)"
./build/tests/collectives/test_tilexr_collectives_kernel_ownership
```

Expected: test fails because `perf_trace_kernel.h` is absent and the CMake option is absent.

- [ ] **Step 3: Add the CMake option**

In `src/collectives/kernels/CMakeLists.txt`, add near the binary size option:

```cmake
option(TILEXR_COLLECTIVES_ENABLE_PROFILING "Enable TileXR collectives kernel profiling helpers" OFF)
```

After `target_compile_options(tilexr_collectives_op_tmp PRIVATE ...)`, add:

```cmake
if(TILEXR_COLLECTIVES_ENABLE_PROFILING)
    target_compile_definitions(tilexr_collectives_op_tmp PRIVATE TILEXR_COLLECTIVES_ENABLE_PROFILING=1)
endif()
```

- [ ] **Step 4: Add kernel helper header**

Create `src/collectives/kernels/perf_trace_kernel.h`:

```cpp
#ifndef TILEXR_COLLECTIVES_KERNEL_PERF_TRACE_KERNEL_H
#define TILEXR_COLLECTIVES_KERNEL_PERF_TRACE_KERNEL_H

#include "kernel_operator.h"
#include "tilexr_perf_trace.h"

namespace TileXR {

struct TileXRPerfStageToken {
    uint64_t startCycle = 0;
};

#if defined(TILEXR_COLLECTIVES_ENABLE_PROFILING)
__attribute__((always_inline)) inline __aicore__ __gm__ TileXRPerfCoreStageStats *
TileXRPerfStatsSlot(GM_ADDR trace, uint32_t rank, uint32_t core, uint32_t stage)
{
    if (trace == nullptr) {
        return nullptr;
    }
    auto header = reinterpret_cast<__gm__ TileXRPerfTraceHeader *>(trace);
    auto stats = reinterpret_cast<__gm__ TileXRPerfCoreStageStats *>(trace + header->statsOffset);
    const size_t offset = PerfTraceStatsOffset(rank, core, stage, header->maxCoreCount, header->stageCount);
    return stats + offset;
}

__attribute__((always_inline)) inline __aicore__ TileXRPerfStageToken
TileXRPerfStageBegin(GM_ADDR trace, PerfStageId stage, PerfBarrierPolicy policy)
{
    (void)stage;
    if (trace != nullptr && policy == PerfBarrierPolicy::BARRIERED) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    TileXRPerfStageToken token {};
    token.startCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    return token;
}

__attribute__((always_inline)) inline __aicore__ void
TileXRPerfAccumulateDuration(GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage,
                             uint64_t startCycle, uint64_t endCycle)
{
    auto slot = TileXRPerfStatsSlot(trace, rank, core, static_cast<uint32_t>(stage));
    if (slot == nullptr || endCycle < startCycle) {
        return;
    }
    const uint64_t duration = endCycle - startCycle;
    slot->rank = rank;
    slot->core = core;
    slot->stageId = static_cast<uint32_t>(stage);
    slot->count += 1;
    slot->sumCycles += duration;
    slot->minCycles = (slot->minCycles == 0 || duration < slot->minCycles) ? duration : slot->minCycles;
    slot->maxCycles = duration > slot->maxCycles ? duration : slot->maxCycles;
    slot->firstStartCycle = (slot->firstStartCycle == 0 || startCycle < slot->firstStartCycle) ?
        startCycle : slot->firstStartCycle;
    slot->lastEndCycle = endCycle > slot->lastEndCycle ? endCycle : slot->lastEndCycle;
}

__attribute__((always_inline)) inline __aicore__ void
TileXRPerfStageEnd(GM_ADDR trace, uint32_t rank, uint32_t core, PerfStageId stage,
                   TileXRPerfStageToken token, PerfBarrierPolicy policy)
{
    if (trace != nullptr &&
        (policy == PerfBarrierPolicy::BARRIERED || policy == PerfBarrierPolicy::END_BARRIER_ONLY)) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    const uint64_t endCycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
    TileXRPerfAccumulateDuration(trace, rank, core, stage, token.startCycle, endCycle);
}
#else
__attribute__((always_inline)) inline __aicore__ TileXRPerfStageToken
TileXRPerfStageBegin(GM_ADDR, PerfStageId, PerfBarrierPolicy)
{
    return TileXRPerfStageToken {};
}

__attribute__((always_inline)) inline __aicore__ void
TileXRPerfAccumulateDuration(GM_ADDR, uint32_t, uint32_t, PerfStageId, uint64_t, uint64_t)
{
}

__attribute__((always_inline)) inline __aicore__ void
TileXRPerfStageEnd(GM_ADDR, uint32_t, uint32_t, PerfStageId, TileXRPerfStageToken, PerfBarrierPolicy)
{
}
#endif

} // namespace TileXR

#endif // TILEXR_COLLECTIVES_KERNEL_PERF_TRACE_KERNEL_H
```

- [ ] **Step 5: Pass `perfTrace` through kernel macros**

Modify `src/collectives/kernels/collectives.h`:

```cpp
#include "perf_trace_kernel.h"
```

Change `KERNELS_ARGS_FUN()` and `KERNELS_ARGS_CALL()`:

```cpp
#define KERNELS_ARGS_FUN() \
GM_ADDR input, GM_ADDR output, GM_ADDR commArgs, int64_t len, int64_t magic, int op, int root, int cycleCount, \
GM_ADDR scale, int64_t scaleCount, GM_ADDR offset, GM_ADDR perfTrace

#define KERNELS_ARGS_CALL() \
input, output, commArgs, len, magic, op, root, cycleCount, scale, scaleCount, offset, perfTrace
```

Modify `src/collectives/kernels/kernels/collectives.cce` by appending `GM_ADDR perfTrace` to `ALLREDUCE_ARGS_FUN`, `ALLREDUCE_ARGS_FUN_16P`, and `ALLREDUCE_ARGS_FUN_16P_Origin`, and by appending `perfTrace` to `ALLREDUCE_ARGS_CALL`, `ALLREDUCE_ARGS_CALL_16P`, `ALLREDUCE_ARGS_CALL_16P_Origin`, `MODIFIABLE_MAGIC_PROCESSED_NUM_ALLREDUCE_ARGS_CALL_16P_Origin`, and `MODIFIABLE_MAGIC_ALLREDUCE_ARGS_CALL_16P`.

The `ALLREDUCE_ARGS_FUN_16P` macro must end with:

```cpp
__gm__ T *buff10, __gm__ T *buff11, __gm__ T *buff12, __gm__ T *buff13, __gm__ T *buff14, __gm__ T *buff15, \
GM_ADDR perfTrace
```

The `ALLREDUCE_ARGS_CALL_16P` macro must end with:

```cpp
shareAddrs[10], shareAddrs[11], shareAddrs[12], shareAddrs[13], shareAddrs[14], shareAddrs[15], perfTrace
```

- [ ] **Step 6: Run source checks and commit**

Run:

```bash
cmake --build build --target test_tilexr_collectives_kernel_ownership -j"$(nproc)"
./build/tests/collectives/test_tilexr_collectives_kernel_ownership
```

Expected: executable exits with status `0`.

Commit:

```bash
git add src/collectives/kernels/perf_trace_kernel.h src/collectives/kernels/CMakeLists.txt \
    src/collectives/kernels/collectives.h src/collectives/kernels/kernels/collectives.cce \
    tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp
git commit -m "feat: add profile-gated kernel trace helpers"
```

## Task 6: Instrument Big-Data AllGather

**Files:**
- Modify: `src/collectives/kernels/kernels/lcal_allgather_big_data.cce`
- Modify: `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`

- [ ] **Step 1: Add failing source checks for stage points**

In `tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp`, add a new function:

```cpp
void TestBigDataAllGatherPerfStages()
{
    const std::string path = "src/collectives/kernels/kernels/lcal_allgather_big_data.cce";
    const auto text = ReadFile(path);
    CheckContains(path, text, "PerfStageId::KERNEL_TOTAL");
    CheckContains(path, text, "PerfStageId::CHUNK_TOTAL");
    CheckContains(path, text, "PerfStageId::POST_SYNC");
    CheckContains(path, text, "PerfStageId::LOCAL_INPUT_TO_IPC");
    CheckContains(path, text, "PerfStageId::FLAG_POLL_WAIT");
    CheckContains(path, text, "PerfStageId::PEER_IPC_TO_OUTPUT");
    CheckContains(path, text, "PerfStageId::CHUNK_BARRIER");
    CheckContains(path, text, "TileXRPerfStageBegin");
    CheckContains(path, text, "TileXRPerfStageEnd");
    CheckContains(path, text, "TileXRPerfAccumulateDuration");
}
```

Call it from `main()`:

```cpp
    TestBigDataAllGatherPerfStages();
```

- [ ] **Step 2: Run source test and verify it fails**

Run:

```bash
cmake --build build --target test_tilexr_collectives_kernel_ownership -j"$(nproc)"
./build/tests/collectives/test_tilexr_collectives_kernel_ownership
```

Expected: test fails because the big-data AllGather branch has no stage helper calls.

- [ ] **Step 3: Instrument producer path**

In `TileXRAllGatherBigDataOrigin`, wrap the producer copy path:

```cpp
    const uint32_t perfCore = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t perfRank = static_cast<uint32_t>(rank);
    if (GetBlockIdx() < blockNumPerGroup) {
        int64_t ipcBuffOffsetNum = GetBlockIdx() * avgNumDMAPerCore;
        int64_t inputOffsetNum = GetBlockIdx() * avgNumDMAPerCore;
        auto localToIpc = TileXR::TileXRPerfStageBegin(
            perfTrace, TileXR::PerfStageId::LOCAL_INPUT_TO_IPC, TileXR::PerfBarrierPolicy::BARRIERED);
        input2BuffRankMagic(dataNumRemain * sizeof(T), inputUB[0], receiveBuff, ipcBuffOffsetNum,
                            sendBuff, inputOffsetNum, ctrlFlagsUB, ctrlFlagsGM, magic);
        TileXR::TileXRPerfStageEnd(
            perfTrace, perfRank, perfCore, TileXR::PerfStageId::LOCAL_INPUT_TO_IPC,
            localToIpc, TileXR::PerfBarrierPolicy::END_BARRIER_ONLY);
        return;
    }
```

- [ ] **Step 4: Instrument consumer wait and peer copy**

Around peer flag polling and peer copy, use explicit durations:

```cpp
            const uint64_t pollStart = static_cast<uint64_t>(AscendC::GetSystemCycle());
            ctrlFlagsGMX = (__gm__ int64_t*)buff[x] + (blockGroup0Idx) * MEM_DMA_UNIT_INT_NUM;
            CpGM2UB(ctrlFlagsUB2[blockGroup0Idx], ctrlFlagsGMX, sizeof(int64_t));
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint64_t pollEnd = static_cast<uint64_t>(AscendC::GetSystemCycle());
            TileXR::TileXRPerfAccumulateDuration(
                perfTrace, perfRank, perfCore, TileXR::PerfStageId::FLAG_POLL_WAIT, pollStart, pollEnd);
```

Wrap peer IPC copy:

```cpp
            auto peerCopy = TileXR::TileXRPerfStageBegin(
                perfTrace, TileXR::PerfStageId::PEER_IPC_TO_OUTPUT, TileXR::PerfBarrierPolicy::BARRIERED);
            GM2GMPingPong<T>(dataSizeRemain, inputUB, receiveBuff, revBuffOffsetNum, sendBuff, sendBuffOffsetNum);
            TileXR::TileXRPerfStageEnd(
                perfTrace, perfRank, perfCore, TileXR::PerfStageId::PEER_IPC_TO_OUTPUT,
                peerCopy, TileXR::PerfBarrierPolicy::END_BARRIER_ONLY);
```

- [ ] **Step 5: Instrument outer kernel/chunk/post-sync/barrier stages**

In `TileXRAllGatherBigData`, add:

```cpp
    const uint32_t perfCore = static_cast<uint32_t>(GetBlockIdx());
    const uint32_t perfRank = static_cast<uint32_t>(rank);
    auto kernelTotal = TileXR::TileXRPerfStageBegin(
        perfTrace, TileXR::PerfStageId::KERNEL_TOTAL, TileXR::PerfBarrierPolicy::NO_BARRIER);
```

Inside the chunk loop, wrap chunk total:

```cpp
        auto chunkTotal = TileXR::TileXRPerfStageBegin(
            perfTrace, TileXR::PerfStageId::CHUNK_TOTAL, TileXR::PerfBarrierPolicy::NO_BARRIER);
```

Wrap `PostSyncBigData`:

```cpp
        auto postSync = TileXR::TileXRPerfStageBegin(
            perfTrace, TileXR::PerfStageId::POST_SYNC, TileXR::PerfBarrierPolicy::BARRIERED);
        PostSyncBigData<T>(ctrlFlagsUB, buff, rank, rankSize, dataOffsetNum, ipcBuffMaxNum, magic, i);
        TileXR::TileXRPerfStageEnd(
            perfTrace, perfRank, perfCore, TileXR::PerfStageId::POST_SYNC,
            postSync, TileXR::PerfBarrierPolicy::END_BARRIER_ONLY);
```

Wrap the final chunk barrier:

```cpp
        auto chunkBarrier = TileXR::TileXRPerfStageBegin(
            perfTrace, TileXR::PerfStageId::CHUNK_BARRIER, TileXR::PerfBarrierPolicy::NO_BARRIER);
        AscendC::PipeBarrier<PIPE_ALL>();
        TileXR::TileXRPerfStageEnd(
            perfTrace, perfRank, perfCore, TileXR::PerfStageId::CHUNK_BARRIER,
            chunkBarrier, TileXR::PerfBarrierPolicy::NO_BARRIER);
        TileXR::TileXRPerfStageEnd(
            perfTrace, perfRank, perfCore, TileXR::PerfStageId::CHUNK_TOTAL,
            chunkTotal, TileXR::PerfBarrierPolicy::NO_BARRIER);
```

Before function exit:

```cpp
    TileXR::TileXRPerfStageEnd(
        perfTrace, perfRank, perfCore, TileXR::PerfStageId::KERNEL_TOTAL,
        kernelTotal, TileXR::PerfBarrierPolicy::NO_BARRIER);
```

- [ ] **Step 6: Run source checks and profile compile**

Run source check:

```bash
cmake --build build --target test_tilexr_collectives_kernel_ownership -j"$(nproc)"
./build/tests/collectives/test_tilexr_collectives_kernel_ownership
```

Expected: executable exits with status `0`.

Run profile compile:

```bash
source scripts/common_env.sh
cmake -S . -B build-profile -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON
cmake --build build-profile --target tilexr_collectives_op -j"$(nproc)"
```

Expected: `tilexr_collectives_op` builds successfully.

- [ ] **Step 7: Commit**

```bash
git add src/collectives/kernels/kernels/lcal_allgather_big_data.cce \
    tests/collectives/unit/test_tilexr_collectives_kernel_ownership.cpp
git commit -m "feat: instrument big-data allgather profiling stages"
```

## Task 7: Perf Tool CLI and Report Emission

**Files:**
- Modify: `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`
- Modify: `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`
- Modify: `tests/collectives/README.md`

- [ ] **Step 1: Add failing source checks for CLI flags**

In `tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp`, extend `TestPerfToolSource()`:

```cpp
    CheckContains(path, text, "--profile");
    CheckContains(path, text, "--profile-dir");
    CheckContains(path, text, "--profile-ai-prompt");
    CheckContains(path, text, "--profile-sample-every");
    CheckContains(path, text, "TileXRCollectivePerfSessionCreate");
    CheckContains(path, text, "TileXRCollectivePerfSetActiveSession");
    CheckContains(path, text, "TileXRCollectivePerfWriteReport");
    CheckContains(path, text, "TileXRCollectivePerfSessionDestroy");
```

- [ ] **Step 2: Run source test and verify it fails**

Run:

```bash
cmake --build build --target test_tilexr_collectives_tools_sources -j"$(nproc)"
./build/tests/collectives/test_tilexr_collectives_tools_sources
```

Expected: test fails because the perf tool has no profile flags.

- [ ] **Step 3: Include the perf API and extend options**

In `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp`, add:

```cpp
#include "tilexr_collectives_perf.h"
```

Extend `Options`:

```cpp
    bool profile = false;
    std::string profileDir;
    bool profileAiPrompt = false;
    int profileSampleEvery = 1;
```

Extend `PrintUsage()`:

```cpp
        << "  [--profile 0|1] [--profile-dir path]\n"
        << "  [--profile-ai-prompt 0|1] [--profile-sample-every N]\n";
```

- [ ] **Step 4: Parse profile flags**

In `ParseOptions`, add branches:

```cpp
        } else if (arg == "--profile") {
            const char *value = requireValue(arg);
            if (value == nullptr || !ParseBool(value, options.profile)) {
                std::cerr << "ERROR: --profile must be 0 or 1" << std::endl;
                return false;
            }
        } else if (arg == "--profile-dir") {
            const char *value = requireValue(arg);
            if (value == nullptr) {
                return false;
            }
            options.profileDir = value;
        } else if (arg == "--profile-ai-prompt") {
            const char *value = requireValue(arg);
            if (value == nullptr || !ParseBool(value, options.profileAiPrompt)) {
                std::cerr << "ERROR: --profile-ai-prompt must be 0 or 1" << std::endl;
                return false;
            }
        } else if (arg == "--profile-sample-every") {
            const char *value = requireValue(arg);
            if (value == nullptr || !ParseInt(value, options.profileSampleEvery)) {
                std::cerr << "ERROR: invalid --profile-sample-every" << std::endl;
                return false;
            }
```

Add validation:

```cpp
    if (options.profileSampleEvery <= 0) {
        std::cerr << "ERROR: --profile-sample-every must be positive" << std::endl;
        return false;
    }
```

- [ ] **Step 5: Create and activate session in main**

After successful communicator initialization:

```cpp
    TileXRCollectivePerfSession perfSession = nullptr;
    if (options.profile) {
        const std::string defaultDir = options.profileDir.empty() ?
            ("run/prof/collectives/rank" + std::to_string(options.rank)) : options.profileDir;
        TileXRCollectivePerfConfig perfConfig {};
        perfConfig.enabled = 1;
        perfConfig.outputDir = defaultDir.c_str();
        perfConfig.emitAiPrompt = options.profileAiPrompt ? 1 : 0;
        perfConfig.sampleEveryN = static_cast<unsigned int>(options.profileSampleEvery);
        if (!CheckTileXR(options.rank, "TileXRCollectivePerfSessionCreate",
                TileXRCollectivePerfSessionCreate(&perfConfig, &perfSession)) ||
            !CheckTileXR(options.rank, "TileXRCollectivePerfSetActiveSession",
                TileXRCollectivePerfSetActiveSession(perfSession))) {
            Cleanup(comm, stream, deviceId, deviceSet);
            return 1;
        }
    }
```

Before cleanup:

```cpp
    if (perfSession != nullptr) {
        TileXRCollectivePerfSetActiveSession(nullptr);
        if (TileXRCollectivePerfWriteReport(perfSession) != TileXR::TILEXR_SUCCESS) {
            std::cerr << "[rank " << options.rank << "] ERROR: TileXRCollectivePerfWriteReport failed" << std::endl;
            totalErrors += 1;
        }
        TileXRCollectivePerfSessionDestroy(perfSession);
    }
```

- [ ] **Step 6: Update README**

Add to `tests/collectives/README.md` under the perf run section:

```markdown
### Operator-Internal Profiling

Build collectives with profiling enabled:

```bash
cmake -S . -B build-profile -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON
cmake --build build-profile --target tilexr_collective_perf -j"$(nproc)"
```

Run the perf tool with profiling:

```bash
./run_collective_perf.sh 2 0 ../../build-profile/tests/collectives \
  --op allgather --min-bytes 67108864 --max-bytes 67108864 \
  --profile 1 --profile-dir run/prof/collectives --profile-ai-prompt 1
```

The report directory contains `trace.json`, `summary.csv`, `analysis.md`, `report.html`, and `ai_prompt.md` when prompt export is enabled.
```

- [ ] **Step 7: Run tests and commit**

Run:

```bash
cmake --build build --target test_tilexr_collectives_tools_sources tilexr_collective_perf -j"$(nproc)"
./build/tests/collectives/test_tilexr_collectives_tools_sources
```

Expected: source test exits with status `0`, and `tilexr_collective_perf` links successfully.

Commit:

```bash
git add tests/collectives/tilexr-tests/tilexr_collective_perf.cpp \
    tests/collectives/unit/test_tilexr_collectives_tools_sources.cpp \
    tests/collectives/README.md
git commit -m "feat: add collectives perf profiling cli"
```

## Task 8: ACL Device Buffer Copy-Back and Final Verification

**Files:**
- Modify: `src/collectives/host/perf_trace_session.cpp`
- Modify: `tests/collectives/unit/test_collective_perf_session.cpp`
- Modify: `docs/superpowers/specs/2026-05-30-collective-performance-profiling-design.md`

- [ ] **Step 1: Add host-only validation for inactive report**

In `tests/collectives/unit/test_collective_perf_session.cpp`, after `TileXRCollectivePerfSetActiveSession(session)` add:

```cpp
    CheckStatus("write empty report", TileXRCollectivePerfWriteReport(session), TileXR::TILEXR_SUCCESS);
```

Run:

```bash
cmake --build build --target test_collective_perf_session -j"$(nproc)"
./build/tests/collectives/test_collective_perf_session
```

Expected: this still passes before ACL allocation is added.

- [ ] **Step 2: Allocate and clear device trace buffer during launch preparation**

In `src/collectives/host/perf_trace_session.cpp`, include ACL runtime:

```cpp
#include "acl/acl_rt.h"
```

Replace the metadata-only body of `PreparePerfTraceLaunch` after `session->hostStats.assign(...)` with:

```cpp
    session->header.statsOffset = sizeof(TileXR::TileXRPerfTraceHeader);
    session->header.statsBytes = static_cast<uint64_t>(session->hostStats.size() *
        sizeof(TileXR::TileXRPerfCoreStageStats));
    const size_t requiredBytes = static_cast<size_t>(session->header.statsOffset + session->header.statsBytes);
    if (session->deviceBufferBytes < requiredBytes) {
        if (session->deviceBuffer != nullptr) {
            aclrtFree(session->deviceBuffer);
            session->deviceBuffer = nullptr;
            session->deviceBufferBytes = 0;
        }
        aclError allocRet = aclrtMalloc(&session->deviceBuffer, requiredBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (allocRet != ACL_SUCCESS) {
            return TileXR::TILEXR_ERROR_INTERNAL;
        }
        session->deviceBufferBytes = requiredBytes;
    }
    aclError copyRet = aclrtMemcpyAsync(session->deviceBuffer, sizeof(session->header),
        &session->header, sizeof(session->header), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (copyRet != ACL_SUCCESS) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    void *statsDevice = static_cast<uint8_t *>(session->deviceBuffer) + session->header.statsOffset;
    aclError memsetRet = aclrtMemsetAsync(statsDevice, static_cast<size_t>(session->header.statsBytes),
        0, static_cast<size_t>(session->header.statsBytes), stream);
    if (memsetRet != ACL_SUCCESS) {
        return TileXR::TILEXR_ERROR_INTERNAL;
    }
    *deviceTrace = session->deviceBuffer;
```

In `TileXRCollectivePerfSessionDestroy`, free the device buffer:

```cpp
    auto *impl = static_cast<TileXRCollectives::Host::PerfTraceSession *>(session);
    if (impl != nullptr && impl->deviceBuffer != nullptr) {
        aclrtFree(impl->deviceBuffer);
        impl->deviceBuffer = nullptr;
    }
    delete impl;
```

- [ ] **Step 3: Copy stats back before writing reports**

At the start of `TileXRCollectivePerfWriteReport`, before `PerfReportOptions options {}`:

```cpp
    if (impl->deviceBuffer != nullptr && impl->header.statsBytes > 0 && !impl->hostStats.empty()) {
        const void *statsDevice = static_cast<const uint8_t *>(impl->deviceBuffer) + impl->header.statsOffset;
        aclError copyRet = aclrtMemcpy(impl->hostStats.data(), static_cast<size_t>(impl->header.statsBytes),
            statsDevice, static_cast<size_t>(impl->header.statsBytes), ACL_MEMCPY_DEVICE_TO_HOST);
        if (copyRet != ACL_SUCCESS) {
            return TileXR::TILEXR_ERROR_INTERNAL;
        }
    }
```

- [ ] **Step 4: Run host-only tests**

Run:

```bash
cmake --build build --target test_collective_perf_session test_collective_perf_report -j"$(nproc)"
./build/tests/collectives/test_collective_perf_session
./build/tests/collectives/test_collective_perf_report
```

Expected: both executables exit with status `0`. `TileXRCollectivePerfWriteReport` must skip `aclrtMemcpy` whenever `impl->deviceBuffer == nullptr`, so an empty host-only session still writes deterministic empty report artifacts without requiring an ACL device.

- [ ] **Step 5: Run full host test set**

Run:

```bash
cmake --build build --target test_tilexr_collectives_api test_tilexr_collectives_kernel_ownership \
    test_tilexr_collectives_tools_sources test_tilexr_collectives_header_compile \
    test_collective_perf_report test_collective_perf_session test_tilexr_perf_trace_layout \
    test_tilexr_collectives_stub_behavior test_tilexr_collectives_uninitialized_comm \
    test_collective_host_utils test_prepare_host_launch_context -j"$(nproc)"
ctest --test-dir build/tests/collectives --output-on-failure
```

Expected: all host-only collectives tests pass.

- [ ] **Step 6: Run profile compile verification**

Run:

```bash
source scripts/common_env.sh
cmake -S . -B build-profile -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON
cmake --build build-profile --target tilexr-collectives tilexr_collective_perf -j"$(nproc)"
```

Expected: `tilexr-collectives` and `tilexr_collective_perf` build successfully.

- [ ] **Step 7: Document hardware verification result**

If NPU devices are available, run:

```bash
cd tests/collectives
./run_collective_perf.sh 2 0 ../../build-profile/tests/collectives \
  --op allgather --min-bytes 67108864 --max-bytes 67108864 \
  --profile 1 --profile-dir ../../run/prof/collectives --profile-ai-prompt 1
```

Expected: each rank writes report artifacts under `run/prof/collectives`, and `summary.csv` contains nonzero rows for `flag_poll_wait` or `peer_ipc_to_output` when the big-data branch is selected.

If hardware is not available, record the skipped hardware verification in the final implementation summary. Do not mark hardware verification as passing.

- [ ] **Step 8: Update spec status and commit**

Append this verification note to `docs/superpowers/specs/2026-05-30-collective-performance-profiling-design.md`:

```markdown

## Implementation Verification

Implementation plan verification requires host-only collectives tests, profile-build compilation, and an optional hardware run of `tilexr_collective_perf --profile 1` on the big-data AllGather path. Hardware verification is reported as skipped when no usable NPU devices are available.
```

Commit:

```bash
git add src/collectives/host/perf_trace_session.cpp \
    tests/collectives/unit/test_collective_perf_session.cpp \
    docs/superpowers/specs/2026-05-30-collective-performance-profiling-design.md
git commit -m "feat: complete collective perf trace runtime"
```

## Final Verification

- [ ] Run default build host tests:

```bash
source scripts/common_env.sh
cmake -S . -B build -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON -DBUILD_TESTING=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

Expected: CTest reports all configured tests passing. Hardware-dependent tests may be absent from CTest and remain manual.

- [ ] Run profile build compile:

```bash
source scripts/common_env.sh
cmake -S . -B build-profile -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON
cmake --build build-profile --target tilexr-collectives tilexr_collective_perf -j"$(nproc)"
```

Expected: both targets build successfully.

- [ ] Check git status:

```bash
git status --short
```

Expected: only intentionally untracked local artifacts are listed. `.superpowers/` must not be staged or committed.
