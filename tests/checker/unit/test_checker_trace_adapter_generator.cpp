#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <unistd.h>

#include "tilexr/checker/trace_adapter_generator.h"

namespace {

int g_failures = 0;

std::string SourcePath(const std::string &relative_path) {
    return std::string(TILEXR_SOURCE_ROOT) + "/" + relative_path;
}

void ExpectTrue(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << "\n";
        ++g_failures;
    }
}

void ExpectContains(const std::string &text, const std::string &needle,
                    const char *message) {
    if (text.find(needle) == std::string::npos) {
        std::cerr << message << ": missing " << needle << "\n";
        ++g_failures;
    }
}

void ExpectNotContains(const std::string &text, const std::string &needle,
                       const char *message) {
    if (text.find(needle) != std::string::npos) {
        std::cerr << message << ": unexpectedly found " << needle << "\n";
        ++g_failures;
    }
}

bool IsBalancedJsonObject(const std::string &text) {
    bool in_string = false;
    bool escaped = false;
    int depth = 0;
    bool saw_open = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const char ch = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            ++depth;
            saw_open = true;
        } else if (ch == '}') {
            --depth;
            if (depth < 0) {
                return false;
            }
            if (depth == 0 && i + 1 != text.size()) {
                return false;
            }
        }
    }
    return saw_open && depth == 0 && !in_string;
}

std::string ReadFile(const std::string &path) {
    std::ifstream input(path.c_str());
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

void WriteFile(const std::string &path, const std::string &content) {
    std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
    output << content;
    if (!output) {
        std::cerr << "write failed: " << path << "\n";
        ++g_failures;
    }
}

std::string MakeTempDir() {
    char dir_template[] = "/tmp/tilexr-checker-adapter-XXXXXX";
    char *created = mkdtemp(dir_template);
    if (created == nullptr) {
        std::cerr << "mkdtemp failed: " << std::strerror(errno) << "\n";
        ++g_failures;
        return std::string();
    }
    return std::string(created);
}

tilexr::checker::TraceAdapterSpec MakeSpec() {
    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "alltoall_demo";
    spec.source_file = "src/collectives/kernels/alltoall_demo.h";
    spec.target_header = "alltoall_demo.h";
    return spec;
}

void TestNormalizeTraceAdapterSpecInfersCollectiveNameAndTargetHeader() {
    tilexr::checker::TraceAdapterSpec spec;
    spec.source_file =
        "src/collectives/kernels/91093/allgather_hierarchy_double_ring.h";

    const tilexr::checker::CheckerStatus status =
        tilexr::checker::NormalizeTraceAdapterSpec(&spec);

    ExpectTrue(status.ok(), "normalize trace adapter spec status");
    ExpectTrue(spec.adapter_name == "allgather_hierarchy_double_ring",
               "normalize trace adapter inferred name");
    ExpectTrue(spec.target_header == "91093/allgather_hierarchy_double_ring.h",
               "normalize trace adapter inferred collective target header");
}

void TestRenderTraceAdapterIsThinAndSourcePreserving() {
    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapter(MakeSpec(), &output);

    ExpectTrue(status.ok(), "render trace adapter status");
    ExpectContains(output, "TILEXR_CHECKER_ALLTOALL_DEMO_TRACE_SHIM_H",
                   "adapter include guard");
    ExpectContains(output,
                   "#define TILEXR_CHECKER_TRACE_SOURCE_FILE \"src/collectives/kernels/alltoall_demo.h\"",
                   "adapter source file define");
    ExpectContains(output,
                   "#define TILEXR_CHECKER_TRACE_TARGET_HEADER \"alltoall_demo.h\"",
                   "adapter target header define");
    ExpectContains(output, "#include \"tilexr/checker/collective_trace_adapter.h\"",
                   "adapter includes generic adapter");
    ExpectContains(output, "namespace trace_adapter_alltoall_demo",
                   "adapter emits metadata namespace");
    ExpectContains(output, "TraceAdapterMetadata",
                   "adapter emits metadata API");
    ExpectContains(output, "\"alltoall_demo\"",
                   "adapter metadata names adapter");
    ExpectContains(output, "\"src/collectives/kernels/alltoall_demo.h\"",
                   "adapter metadata records production source");
    ExpectContains(output, "\"alltoall_demo.h\"",
                   "adapter metadata records target header");
    ExpectContains(output, "\"TILEXR_CHECKER_ALLTOALL_DEMO_TRACE_SHIM_H\"",
                   "adapter metadata records include guard");
    ExpectContains(output, "inline bool Audit(const char **reason)",
                   "adapter emits audit helper");
    ExpectContains(output, "AuditTraceAdapterMetadata(Metadata(), reason)",
                   "adapter audit delegates to common metadata audit");
    ExpectNotContains(output, "#define CpGM2GMPingPong",
                      "adapter should not duplicate trace macros");
    ExpectNotContains(output, "#include \"allreduce_big_data.h\"",
                      "adapter should not hard-code another production header");
}

void TestWriteTraceAdapterRejectsProductionSourcePath() {
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteTraceAdapterFile(
            MakeSpec(), "src/collectives/kernels/alltoall_demo_trace_shim.h");

    ExpectTrue(!status.ok(), "adapter generator rejects production source output");
    ExpectContains(status.message, "checker shim", "adapter rejection message");
}

void TestWriteTraceAdapterFile() {
    const std::string dir = MakeTempDir();
    const std::string path = dir + "/alltoall_demo_trace_shim.h";
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteTraceAdapterFile(MakeSpec(), path);

    ExpectTrue(status.ok(), "write trace adapter status");
    const std::string output = ReadFile(path);
    ExpectContains(output, "TILEXR_CHECKER_ALLTOALL_DEMO_TRACE_SHIM_H",
                   "written adapter include guard");
    ExpectContains(output, "collective_trace_adapter.h",
                   "written adapter includes generic adapter");
}

void TestRenderTraceAdapterManifestDocumentsIntegrationPlan() {
    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            MakeSpec(), "tools/checker/shim/tilexr/checker/alltoall_demo_trace_shim.h",
            &output);

    ExpectTrue(status.ok(), "render trace adapter manifest status");
    ExpectContains(output, "\"adapter_name\":\"alltoall_demo\"",
                   "manifest adapter name");
    ExpectContains(output, "\"source_file\":\"src/collectives/kernels/alltoall_demo.h\"",
                   "manifest source file");
    ExpectContains(output,
                   "\"shim_header\":\"tools/checker/shim/tilexr/checker/alltoall_demo_trace_shim.h\"",
                   "manifest shim header");
    ExpectContains(output, "\"source_preservation\"",
                   "manifest source preservation section");
    ExpectContains(output, "\"runner_integration\"",
                   "manifest runner integration section");
    ExpectContains(output, "\"checklist\"",
                   "manifest runner checklist section");
    ExpectContains(output, "\"block_schedule_defined\"",
                   "manifest runner block schedule checklist");
    ExpectContains(output, "\"operator_init_process_wired\"",
                   "manifest runner operator checklist");
    ExpectContains(output, "\"oracle_materializer_reviewed\"",
                   "manifest runner materializer checklist");
    ExpectContains(output, "\"include_guard\":\"TILEXR_CHECKER_ALLTOALL_DEMO_TRACE_SHIM_H\"",
                   "manifest include guard");
    ExpectContains(output, "\"generated_outputs\"",
                   "manifest generated outputs section");
    ExpectContains(output, "\"adapter_header\":\"tools/checker/shim/tilexr/checker/alltoall_demo_trace_shim.h\"",
                   "manifest generated adapter header");
    ExpectContains(output, "\"header_probe\":\"generated when --probe-output is set\"",
                   "manifest generated header probe");
    ExpectContains(output, "\"trace_hooks\"",
                   "manifest trace hook section");
    ExpectContains(output, "\"symbol\":\"CpGM2GMPingPong\"",
                   "manifest gm2gm hook");
    ExpectContains(output, "\"symbol\":\"AscendC::PipeBarrier\"",
                   "manifest pipe barrier hook");
    ExpectContains(output, "\"metadata_api\"",
                   "manifest metadata api section");
    ExpectContains(output, "\"namespace\":\"tilexr::checker::trace_adapter_alltoall_demo\"",
                   "manifest metadata namespace");
    ExpectContains(output, "\"metadata\":\"Metadata\"",
                   "manifest metadata function");
    ExpectContains(output, "\"audit\":\"Audit\"",
                   "manifest audit function");
    ExpectContains(output, "\"validation_commands\"",
                   "manifest validation command section");
    ExpectContains(output, "git diff -- src/collectives src/ep",
                   "manifest source preservation validation command");
}

void TestRenderTraceAdapterManifestIsBalancedJsonObject() {
    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            MakeSpec(), "tools/checker/shim/tilexr/checker/alltoall_demo_trace_shim.h",
            &output);

    ExpectTrue(status.ok(), "render trace adapter manifest json status");
    ExpectTrue(IsBalancedJsonObject(output),
               "render trace adapter manifest should be balanced json object");
}

void TestRenderTraceAdapterManifestReportsObservedSourceHooks() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/demo_collective.h";
    WriteFile(source_path,
              "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n"
              "sync.SetSyncFlag(magic, value, event, rank);\n"
              "sync.WaitSyncFlag(magic, value, event, rank);\n"
              "AscendC::PipeBarrier<PIPE_ALL>();\n"
              "DataCopy(dst, src, 32);\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "demo_collective";
    spec.source_file = source_path;
    spec.target_header = "demo_collective.h";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec, "tools/checker/shim/tilexr/checker/demo_collective_trace_shim.h",
            &output);

    ExpectTrue(status.ok(), "render trace adapter manifest source hook status");
    ExpectContains(output, "\"source_observation\":{\"available\":true",
                   "manifest source observation available");
    ExpectContains(output, "\"known_hook_hits\"",
                   "manifest known hook hits");
    ExpectContains(output, "\"symbol\":\"CpGM2GMPingPong\"",
                   "manifest observed gm2gm hook");
    ExpectContains(output, "\"lines\":[1]",
                   "manifest observed gm2gm hook line");
    ExpectContains(output, "\"occurrences\"",
                   "manifest observed hook occurrences");
    ExpectContains(output, "\"line\":1",
                   "manifest observed gm2gm occurrence line");
    ExpectContains(output, "\"snippet\":\"Collectives::CpGM2GMPingPong(bytes, input, output, op);\"",
                   "manifest observed gm2gm snippet");
    ExpectContains(output, "\"symbol\":\"WaitSyncFlag\"",
                   "manifest observed wait hook");
    ExpectContains(output, "\"symbol\":\"AscendC::PipeBarrier\"",
                   "manifest observed pipe barrier hook");
    ExpectContains(output, "\"manual_review_candidates\"",
                   "manifest manual review candidates");
    ExpectContains(output, "\"symbol\":\"DataCopy\"",
                   "manifest raw data copy candidate");
    ExpectContains(output, "\"snippet\":\"DataCopy(dst, src, 32);\"",
                   "manifest raw data copy snippet");
    ExpectContains(output, "raw AscendC DataCopy",
                   "manifest raw data copy candidate reason");
    ExpectContains(output, "\"manual_review_required\":true",
                   "manifest requires manual review for raw candidate");
    ExpectContains(output, "\"required_actions\"",
                   "manifest manual review required actions");
    ExpectContains(output, "add a trace wrapper or explicit unsupported gate for DataCopy",
                   "manifest raw data copy action");
    ExpectContains(output, "\"status\":\"pending\"",
                   "manifest manual review default pending status");
    ExpectContains(output, "\"installed_trace_seed\"",
                   "manifest installed trace seed section");
    ExpectContains(output, "\"required_event_coverage\"",
                   "manifest required event coverage seed");
    ExpectContains(output,
                   "{\"kind\":\"COPY\",\"source_file\":\"" + source_path +
                       "\",\"source_line\":1}",
                   "manifest gm2gm copy coverage seed");
    ExpectContains(output,
                   "{\"kind\":\"FLAG_STORE\",\"source_file\":\"" + source_path +
                       "\",\"source_line\":2}",
                   "manifest sync store coverage seed");
    ExpectContains(output,
                   "{\"kind\":\"FLAG_WAIT\",\"source_file\":\"" + source_path +
                       "\",\"source_line\":3}",
                   "manifest sync wait coverage seed");
    ExpectContains(output,
                   "{\"kind\":\"PIPE_BARRIER\",\"source_file\":\"" + source_path +
                       "\",\"source_line\":4}",
                   "manifest pipe barrier coverage seed");
}

void TestRenderTraceAdapterManifestIncludesSemiAutomaticTraceStubPlan() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/stub_plan_collective.h";
    WriteFile(source_path,
              "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n"
              "sync.SetSyncFlag(magic, value, event, rank);\n"
              "DataCopy(dst, src, 32);\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "stub_plan_collective";
    spec.source_file = "src/collectives/kernels/stub_plan_collective.h";
    spec.source_scan_file = source_path;
    spec.target_header = "stub_plan_collective.h";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec,
            "tools/checker/shim/tilexr/checker/stub_plan_collective_trace_shim.h",
            &output);

    ExpectTrue(status.ok(), "render trace adapter manifest stub plan status");
    ExpectContains(output, "\"semi_automatic_trace\"",
                   "manifest semi automatic trace section");
    ExpectContains(output, "\"header_stub\"",
                   "manifest header stub section");
    ExpectContains(output,
                   "\"shim_header\":\"tools/checker/shim/tilexr/checker/stub_plan_collective_trace_shim.h\"",
                   "manifest header stub shim path");
    ExpectContains(output, "\"generic_adapter\":\"tilexr/checker/collective_trace_adapter.h\"",
                   "manifest header stub generic adapter");
    ExpectContains(output, "\"source_define\":\"TILEXR_CHECKER_TRACE_SOURCE_FILE\"",
                   "manifest header stub source define");
    ExpectContains(output, "\"target_header_define\":\"TILEXR_CHECKER_TRACE_TARGET_HEADER\"",
                   "manifest header stub target define");
    ExpectContains(output, "\"auto_hook_candidates\"",
                   "manifest auto hook candidates section");
    ExpectContains(output,
                   "{\"symbol\":\"CpGM2GMPingPong\",\"event\":\"COPY\",\"source_file\":\"src/collectives/kernels/stub_plan_collective.h\",\"line\":1",
                   "manifest gm2gm auto hook candidate");
    ExpectContains(output,
                   "\"macro\":\"CpGM2GMPingPong\"",
                   "manifest gm2gm macro candidate");
    ExpectContains(output,
                   "\"status\":\"auto_traceable\"",
                   "manifest auto hook status");
    ExpectContains(output,
                   "{\"symbol\":\"SetSyncFlag\",\"event\":\"FLAG_STORE\",\"source_file\":\"src/collectives/kernels/stub_plan_collective.h\",\"line\":2",
                   "manifest sync store auto hook candidate");
    ExpectContains(output, "\"manual_review_actions\"",
                   "manifest semi automatic manual review section");
    ExpectContains(output,
                   "\"action\":\"add a trace wrapper or explicit unsupported gate for DataCopy\"",
                   "manifest semi automatic raw data copy action");
}

void TestRenderTraceAdapterManifestMarksNoManualReviewWhenSourceIsCovered() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/covered_collective.h";
    WriteFile(source_path,
              "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n"
              "sync.SetSyncFlag(magic, value, event, rank);\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "covered_collective";
    spec.source_file = source_path;
    spec.target_header = "covered_collective.h";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec, "tools/checker/shim/tilexr/checker/covered_collective_trace_shim.h",
            &output);

    ExpectTrue(status.ok(), "render covered manifest status");
    ExpectContains(output, "\"manual_review_required\":false",
                   "covered manifest no manual review required");
    ExpectContains(output, "\"required_actions\":[]",
                   "covered manifest no required actions");
}

void TestRenderTraceAdapterEmitsObservedHookMetadata() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/demo_collective.h";
    WriteFile(source_path,
              "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n"
              "sync.SetSyncFlag(magic, value, event, rank);\n"
              "sync.WaitSyncFlag(magic, value, event, rank);\n"
              "AscendC::PipeBarrier<PIPE_ALL>();\n"
              "DataCopy(dst, src, 32);\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "demo_collective";
    spec.source_file = source_path;
    spec.target_header = "demo_collective.h";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapter(spec, &output);

    ExpectTrue(status.ok(), "render trace adapter observed metadata status");
    ExpectContains(output, "namespace trace_adapter_demo_collective",
                   "adapter metadata namespace for observed source");
    ExpectContains(output, "static const int kKnownHook0Lines[] = {1};",
                   "adapter metadata gm2gm hook lines");
    ExpectContains(output, "{\"CpGM2GMPingPong\", kKnownHook0Lines, 1}",
                   "adapter metadata gm2gm hook observation");
    ExpectContains(output, "static const int kKnownHook1Lines[] = {2};",
                   "adapter metadata sync store hook lines");
    ExpectContains(output, "{\"SetSyncFlag\", kKnownHook1Lines, 1}",
                   "adapter metadata sync store hook observation");
    ExpectContains(output, "static const int kKnownHook2Lines[] = {3};",
                   "adapter metadata sync wait hook lines");
    ExpectContains(output, "{\"WaitSyncFlag\", kKnownHook2Lines, 1}",
                   "adapter metadata sync wait hook observation");
    ExpectContains(output, "static const int kKnownHook3Lines[] = {4};",
                   "adapter metadata pipe barrier hook lines");
    ExpectContains(output, "{\"AscendC::PipeBarrier\", kKnownHook3Lines, 1}",
                   "adapter metadata pipe barrier hook observation");
    ExpectContains(output, "static const int kManualReviewCandidate0Lines[] = {5};",
                   "adapter metadata raw data copy lines");
    ExpectContains(output, "{\"DataCopy\", kManualReviewCandidate0Lines, 1}",
                   "adapter metadata manual review candidate");
    ExpectContains(output, "kKnownHooks, 4",
                   "adapter metadata known hook count");
    ExpectContains(output, "kManualReviewCandidates, 1",
                   "adapter metadata manual review count");
}

void TestStrictTraceAdapterManifestRejectsManualReviewCandidates() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/raw_copy_collective.h";
    WriteFile(source_path,
              "DataCopy(dst, src, 32);\n"
              "SetAtomicAdd<int32_t>();\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "raw_copy_collective";
    spec.source_file = source_path;
    spec.target_header = "raw_copy_collective.h";
    spec.strict_source_observation = true;

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec, "tools/checker/shim/tilexr/checker/raw_copy_collective_trace_shim.h",
            &output);

    ExpectTrue(!status.ok(), "strict manifest rejects manual review candidates");
    ExpectContains(status.message, "manual review candidates",
                   "strict manifest rejection message");
    ExpectContains(status.message, "DataCopy",
                   "strict manifest rejection names raw copy");
    ExpectContains(status.message, "SetAtomicAdd",
                   "strict manifest rejection names atomic");
}

void TestTraceAdapterManifestRequiresCoverageDecisionWhenNoHooksObserved() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/opaque_collective.h";
    WriteFile(source_path,
              "class OpaqueCollective {\n"
              "public:\n"
              "    void Process() { int local = 0; (void)local; }\n"
              "};\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "opaque_collective";
    spec.source_file = source_path;
    spec.target_header = "opaque_collective.h";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec, "tools/checker/shim/tilexr/checker/opaque_collective_trace_shim.h",
            &output);

    ExpectTrue(status.ok(), "opaque manifest render status");
    ExpectContains(output, "\"known_hook_hits\":[]",
                   "opaque manifest no known hooks");
    ExpectContains(output, "\"manual_review_required\":true",
                   "opaque manifest requires manual review");
    ExpectContains(output, "\"symbol\":\"coverage_decision\"",
                   "opaque manifest coverage decision action");
    ExpectContains(output, "\"status\":\"pending\"",
                   "opaque manifest coverage decision pending");
    ExpectContains(output, "no recognized checker trace hook was observed",
                   "opaque manifest coverage decision reason");
}

void TestStrictTraceAdapterManifestRejectsMissingHookCoverage() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/opaque_collective_strict.h";
    WriteFile(source_path,
              "class OpaqueCollectiveStrict { void Process() {} };\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "opaque_collective_strict";
    spec.source_file = source_path;
    spec.target_header = "opaque_collective_strict.h";
    spec.strict_source_observation = true;

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec,
            "tools/checker/shim/tilexr/checker/opaque_collective_strict_trace_shim.h",
            &output);

    ExpectTrue(!status.ok(), "strict opaque manifest rejects missing hooks");
    ExpectContains(status.message, "no recognized checker trace hook",
                   "strict opaque manifest rejection message");
}

void TestTraceAdapterManifestRequiresSourceUnavailableDecision() {
    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "missing_source";
    spec.source_file = "/tmp/tilexr-checker-missing-source-does-not-exist.h";
    spec.target_header = "missing_source.h";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec, "tools/checker/shim/tilexr/checker/missing_source_trace_shim.h",
            &output);

    ExpectTrue(status.ok(), "missing source manifest render status");
    ExpectContains(output, "\"available\":false",
                   "missing source manifest records unavailable source");
    ExpectContains(output, "\"manual_review_required\":true",
                   "missing source manifest requires manual review");
    ExpectContains(output, "\"symbol\":\"source_unavailable\"",
                   "missing source manifest required action symbol");
    ExpectContains(output, "\"status\":\"pending\"",
                   "missing source manifest required action pending");
    ExpectContains(output, "source file could not be read",
                   "missing source manifest required action reason");
}

void TestAnalyzeTraceSourceReportsCoverageDecisionAction() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/opaque_trace_source.h";
    WriteFile(source_path,
              "class OpaqueTraceSource {\n"
              "public:\n"
              "    void Process() { int local = 0; (void)local; }\n"
              "};\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "opaque_trace_source";
    spec.source_file = "src/collectives/kernels/opaque_trace_source.h";
    spec.source_scan_file = source_path;
    spec.target_header = "opaque_trace_source.h";

    tilexr::checker::TraceSourceAnalysis analysis;
    const tilexr::checker::CheckerStatus analyze_status =
        tilexr::checker::AnalyzeTraceSource(
            spec, "tools/checker/shim/tilexr/checker/opaque_trace_source_trace_shim.h",
            &analysis);
    ExpectTrue(analyze_status.ok(), "analyze opaque source status");

    std::string json;
    const tilexr::checker::CheckerStatus json_status =
        tilexr::checker::RenderTraceSourceAnalysisJson(analysis, &json);
    ExpectTrue(json_status.ok(), "render opaque source analysis json status");
    ExpectContains(json, "\"manual_review_required\":true",
                   "opaque source analysis requires manual review");
    ExpectContains(json, "\"symbol\":\"coverage_decision\"",
                   "opaque source analysis coverage decision symbol");
    ExpectContains(json, "\"line\":0",
                   "opaque source analysis coverage decision line");
    ExpectContains(json, "no recognized checker trace hook was observed",
                   "opaque source analysis coverage decision reason");

    std::string text;
    const tilexr::checker::CheckerStatus text_status =
        tilexr::checker::RenderTraceSourceAnalysisText(analysis, &text);
    ExpectTrue(text_status.ok(), "render opaque source analysis text status");
    ExpectContains(text, "manual review actions: 1",
                   "opaque source analysis text action count");
    ExpectContains(text, "coverage_decision line 0",
                   "opaque source analysis text decision line");
}

void TestAnalyzeTraceSourceReportsSourceUnavailableAction() {
    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "missing_trace_source";
    spec.source_file = "src/collectives/kernels/missing_trace_source.h";
    spec.source_scan_file = "/tmp/tilexr-checker-missing-trace-source-does-not-exist.h";
    spec.target_header = "missing_trace_source.h";

    tilexr::checker::TraceSourceAnalysis analysis;
    const tilexr::checker::CheckerStatus analyze_status =
        tilexr::checker::AnalyzeTraceSource(
            spec, "tools/checker/shim/tilexr/checker/missing_trace_source_trace_shim.h",
            &analysis);
    ExpectTrue(analyze_status.ok(), "analyze missing source status");

    std::string json;
    const tilexr::checker::CheckerStatus json_status =
        tilexr::checker::RenderTraceSourceAnalysisJson(analysis, &json);
    ExpectTrue(json_status.ok(), "render missing source analysis json status");
    ExpectContains(json, "\"available\":false",
                   "missing source analysis records unavailable source");
    ExpectContains(json, "\"symbol\":\"source_unavailable\"",
                   "missing source analysis source unavailable symbol");
    ExpectContains(json, "source file could not be read",
                   "missing source analysis unavailable reason");

    std::string text;
    const tilexr::checker::CheckerStatus text_status =
        tilexr::checker::RenderTraceSourceAnalysisText(analysis, &text);
    ExpectTrue(text_status.ok(), "render missing source analysis text status");
    ExpectContains(text, "manual review actions: 1",
                   "missing source analysis text action count");
    ExpectContains(text, "source_unavailable line 0",
                   "missing source analysis text unavailable line");
}

void TestStrictTraceAdapterManifestRejectsUnavailableSource() {
    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "missing_source_strict";
    spec.source_file = "/tmp/tilexr-checker-missing-source-strict-does-not-exist.h";
    spec.target_header = "missing_source_strict.h";
    spec.strict_source_observation = true;

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec,
            "tools/checker/shim/tilexr/checker/missing_source_strict_trace_shim.h",
            &output);

    ExpectTrue(!status.ok(), "strict missing source manifest status");
    ExpectContains(status.message, "could not read source file",
                   "strict missing source manifest rejection message");
}

void TestSourceObservationIgnoresCommentsAndStrings() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/comment_only_candidates.h";
    WriteFile(source_path,
              "// DataCopy(dst, src, 32);\n"
              "const char *name = \"SetAtomicAdd should not count\";\n"
              "/* DataCopyPad(dst, src, 32);\n"
              "   SetAtomicMax<int32_t>(); */\n"
              "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "comment_only_candidates";
    spec.source_file = source_path;
    spec.target_header = "comment_only_candidates.h";
    spec.strict_source_observation = true;

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec,
            "tools/checker/shim/tilexr/checker/comment_only_candidates_trace_shim.h",
            &output);

    ExpectTrue(status.ok(), "strict manifest ignores comments and strings status");
    ExpectContains(output, "\"manual_review_candidates\":[]",
                   "strict manifest no candidates from comments or strings");
    ExpectContains(output, "\"symbol\":\"CpGM2GMPingPong\"",
                   "strict manifest still observes real gm2gm call");
    ExpectContains(output,
                   "{\"kind\":\"COPY\",\"source_file\":\"" + source_path +
                       "\",\"source_line\":5}",
                   "strict manifest keeps real hook coverage line");
}

void TestSourceObservationUsesSymbolBoundaries() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/boundary_candidates.h";
    WriteFile(source_path,
              "int DataCopyCounter = 0;\n"
              "DataCopyPad(dst, src, params, pad);\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "boundary_candidates";
    spec.source_file = source_path;
    spec.target_header = "boundary_candidates.h";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceAdapterManifest(
            spec, "tools/checker/shim/tilexr/checker/boundary_candidates_trace_shim.h",
            &output);

    ExpectTrue(status.ok(), "symbol boundary manifest status");
    ExpectContains(output, "\"manual_review_candidates\"",
                   "symbol boundary manual review section");
    ExpectContains(output, "\"symbol\":\"DataCopyPad\"",
                   "symbol boundary data copy pad candidate");
    ExpectContains(output, "\"lines\":[2]",
                   "symbol boundary data copy pad line");
    ExpectNotContains(output, "\"symbol\":\"DataCopy\"",
                      "symbol boundary should not match DataCopy prefix");
    ExpectNotContains(output, "DataCopyCounter",
                      "symbol boundary should not record variable name");
}

void TestGeneratedTraceAdapterCompilesAndAuditsMetadata() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/fake_generated_collective.h";
    const std::string shim_path = dir + "/fake_generated_trace_shim.h";
    const std::string probe_path = dir + "/fake_generated_probe.cpp";
    const std::string probe_bin = dir + "/fake_generated_probe";

    WriteFile(source_path,
              "#pragma once\n"
              "class FakeGeneratedCollective {\n"
              "public:\n"
              "    void Process() { AscendC::PipeBarrier<PIPE_ALL>(); }\n"
              "};\n");

    tilexr::checker::TraceAdapterSpec spec;
    spec.adapter_name = "fake_generated";
    spec.source_file = source_path;
    spec.target_header = "fake_generated_collective.h";

    const tilexr::checker::CheckerStatus write_status =
        tilexr::checker::WriteTraceAdapterFile(spec, shim_path);
    ExpectTrue(write_status.ok(), "write generated compile probe shim");

    WriteFile(probe_path,
              "#include \"fake_generated_trace_shim.h\"\n"
              "#include <type_traits>\n"
              "int main() {\n"
              "    static_assert(std::is_class<FakeGeneratedCollective>::value,\n"
              "                  \"fake target header must remain visible\");\n"
              "    static_assert(tilexr::checker::trace_adapter_fake_generated::kKnownHookCount == 1,\n"
              "                  \"generated metadata should record one hook\");\n"
              "    const tilexr::checker::TraceAdapterMetadata &metadata =\n"
              "        tilexr::checker::trace_adapter_fake_generated::Metadata();\n"
              "    const char *reason = nullptr;\n"
              "    if (!tilexr::checker::trace_adapter_fake_generated::Audit(&reason)) return 1;\n"
              "    if (reason != nullptr) return 2;\n"
              "    if (metadata.known_hook_count != 1) return 3;\n"
              "    if (metadata.known_hooks == nullptr) return 4;\n"
              "    if (metadata.known_hooks[0].line_count != 1) return 5;\n"
              "    if (metadata.manual_review_candidate_count != 0) return 6;\n"
              "    return 0;\n"
              "}\n");

    const std::string compile_cmd =
        "c++ -std=c++17 -I" + dir +
        " -I" + SourcePath("tools/checker/shim") +
        " -I" + SourcePath("tools/checker/include") +
        " -I" + SourcePath("src/include") +
        " -I" + SourcePath("src/collectives/kernels") +
        " " + probe_path + " -o " + probe_bin + " && " + probe_bin;
    const int rc = std::system(compile_cmd.c_str());
    ExpectTrue(rc == 0, "generated trace adapter compile probe command");
}

void TestWriteTraceAdapterManifestRejectsProductionSourcePath() {
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteTraceAdapterManifestFile(
            MakeSpec(),
            "tools/checker/shim/tilexr/checker/alltoall_demo_trace_shim.h",
            "src/ep/alltoall_demo_trace_manifest.json");

    ExpectTrue(!status.ok(), "adapter manifest rejects production output");
    ExpectContains(status.message, "checker output path",
                   "adapter manifest rejection message");
}

void TestWriteTraceAdapterManifestFile() {
    const std::string dir = MakeTempDir();
    const std::string path = dir + "/alltoall_demo_trace_manifest.json";
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteTraceAdapterManifestFile(
            MakeSpec(), dir + "/alltoall_demo_trace_shim.h", path);

    ExpectTrue(status.ok(), "write trace adapter manifest status");
    const std::string output = ReadFile(path);
    ExpectContains(output, "\"adapter_name\":\"alltoall_demo\"",
                   "written manifest adapter name");
    ExpectContains(output, "\"runner_integration\"",
                   "written manifest runner integration");
}

void TestRenderTraceRunnerSkeletonDocumentsManualFillPoints() {
    tilexr::checker::TraceRunnerSkeletonSpec skeleton_spec;
    skeleton_spec.function_name = "RunAllToAllDemoTrace";
    skeleton_spec.shim_include = "tilexr/checker/alltoall_demo_trace_shim.h";
    skeleton_spec.materializer_name = "MaterializeAllToAllDemo";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceRunnerSkeleton(skeleton_spec, &output);

    ExpectTrue(status.ok(), "render trace runner skeleton status");
    ExpectContains(output, "CheckerStatus RunAllToAllDemoTrace",
                   "runner skeleton function");
    ExpectContains(output, "#include \"tilexr/checker/alltoall_demo_trace_shim.h\"",
                   "runner skeleton shim include");
    ExpectContains(output, "RegisterCollectiveTraceRanges",
                   "runner skeleton registers ranges");
    ExpectContains(output, "InstallCollectiveTracePeerMems",
                   "runner skeleton installs peer mems");
    ExpectContains(output, "TraceRuntime::SetCurrent(&runtime)",
                   "runner skeleton sets runtime");
    ExpectContains(output, "MaterializeAllToAllDemo",
                   "runner skeleton materializer placeholder");
    ExpectContains(output, "TODO(checker-runner)",
                   "runner skeleton manual fill marker");
    ExpectContains(output, "RunnerIntegrationChecklist",
                   "runner skeleton integration checklist");
    ExpectContains(output, "source_preservation_checked",
                   "runner skeleton source preservation checklist item");
    ExpectContains(output, "block_schedule_defined",
                   "runner skeleton block schedule checklist item");
    ExpectContains(output, "operator_init_process_wired",
                   "runner skeleton op init/process checklist item");
    ExpectContains(output, "oracle_materializer_reviewed",
                   "runner skeleton materializer checklist item");
    ExpectContains(output, "CollectiveExecutor",
                   "runner skeleton executor wiring note");
    ExpectNotContains(output, "allreduce_big_data.h",
                      "runner skeleton should not hard-code existing algorithm");
}

void TestRenderTraceHeaderProbeSkeletonAuditsGeneratedShim() {
    tilexr::checker::TraceHeaderProbeSpec probe_spec;
    probe_spec.adapter_name = "alltoall_demo";
    probe_spec.shim_include = "tilexr/checker/alltoall_demo_trace_shim.h";
    probe_spec.expected_source_file = "src/collectives/kernels/alltoall_demo.h";
    probe_spec.expected_target_header = "alltoall_demo.h";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceHeaderProbeSkeleton(probe_spec, &output);

    ExpectTrue(status.ok(), "render trace header probe skeleton status");
    ExpectContains(output, "#include \"tilexr/checker/alltoall_demo_trace_shim.h\"",
                   "probe skeleton shim include");
    ExpectContains(output, "trace_adapter_alltoall_demo::Metadata()",
                   "probe skeleton metadata call");
    ExpectContains(output, "trace_adapter_alltoall_demo::Audit(&reason)",
                   "probe skeleton audit call");
    ExpectContains(output, "std::strcmp(metadata.adapter_name, \"alltoall_demo\") != 0",
                   "probe skeleton checks adapter name");
    ExpectContains(output,
                   "std::strcmp(metadata.source_file, \"src/collectives/kernels/alltoall_demo.h\") != 0",
                   "probe skeleton checks source file");
    ExpectContains(output,
                   "std::strcmp(metadata.target_header, \"alltoall_demo.h\") != 0",
                   "probe skeleton checks target header");
    ExpectContains(output,
                   "std::strcmp(metadata.include_guard, \"TILEXR_CHECKER_ALLTOALL_DEMO_TRACE_SHIM_H\") != 0",
                   "probe skeleton checks include guard");
    ExpectContains(output, "kKnownHookCount",
                   "probe skeleton known hook count");
    ExpectContains(output, "kManualReviewCandidateCount",
                   "probe skeleton manual review count");
    ExpectContains(output, "return 0;",
                   "probe skeleton main success");
}

void TestWriteTraceHeaderProbeSkeletonRejectsProductionSourcePath() {
    tilexr::checker::TraceHeaderProbeSpec probe_spec;
    probe_spec.adapter_name = "alltoall_demo";
    probe_spec.shim_include = "tilexr/checker/alltoall_demo_trace_shim.h";

    const tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteTraceHeaderProbeSkeletonFile(
            probe_spec, "src/collectives/kernels/alltoall_demo_trace_probe.cpp");

    ExpectTrue(!status.ok(), "probe skeleton rejects production output");
    ExpectContains(status.message, "checker output path",
                   "probe skeleton rejection message");
}

void TestWriteTraceHeaderProbeSkeletonFile() {
    const std::string dir = MakeTempDir();
    const std::string path = dir + "/alltoall_demo_trace_probe.cpp";
    tilexr::checker::TraceHeaderProbeSpec probe_spec;
    probe_spec.adapter_name = "alltoall_demo";
    probe_spec.shim_include = "tilexr/checker/alltoall_demo_trace_shim.h";

    const tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteTraceHeaderProbeSkeletonFile(probe_spec, path);

    ExpectTrue(status.ok(), "write trace header probe status");
    const std::string output = ReadFile(path);
    ExpectContains(output, "trace_adapter_alltoall_demo",
                   "written probe skeleton adapter namespace");
}

void TestGeneratedTraceHeaderProbeCompilesAndAuditsExactMetadata() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/fake_generated_collective.h";
    const std::string shim_path = dir + "/fake_generated_trace_shim.h";
    const std::string probe_path = dir + "/fake_generated_trace_probe.cpp";
    const std::string probe_bin = dir + "/fake_generated_trace_probe";

    WriteFile(source_path,
              "#pragma once\n"
              "class FakeGeneratedCollectiveProbe {\n"
              "public:\n"
              "    void Process() { AscendC::PipeBarrier<PIPE_ALL>(); }\n"
              "};\n");

    tilexr::checker::TraceAdapterSpec adapter_spec;
    adapter_spec.adapter_name = "fake_generated_probe";
    adapter_spec.source_file = source_path;
    adapter_spec.target_header = "fake_generated_collective.h";

    const tilexr::checker::CheckerStatus shim_status =
        tilexr::checker::WriteTraceAdapterFile(adapter_spec, shim_path);
    ExpectTrue(shim_status.ok(), "write exact metadata probe shim");

    tilexr::checker::TraceHeaderProbeSpec probe_spec;
    probe_spec.adapter_name = "fake_generated_probe";
    probe_spec.shim_include = "fake_generated_trace_shim.h";
    probe_spec.expected_source_file = source_path;
    probe_spec.expected_target_header = "fake_generated_collective.h";

    const tilexr::checker::CheckerStatus probe_status =
        tilexr::checker::WriteTraceHeaderProbeSkeletonFile(probe_spec, probe_path);
    ExpectTrue(probe_status.ok(), "write exact metadata generated probe");

    const std::string probe = ReadFile(probe_path);
    ExpectContains(probe, "std::strcmp(metadata.adapter_name, \"fake_generated_probe\") != 0",
                   "generated probe exact adapter check");
    ExpectContains(probe, "std::strcmp(metadata.source_file, \"" + source_path + "\") != 0",
                   "generated probe exact source check");
    ExpectContains(probe, "std::strcmp(metadata.target_header, \"fake_generated_collective.h\") != 0",
                   "generated probe exact target check");

    const std::string compile_cmd =
        "c++ -std=c++17 -I" + dir +
        " -I" + SourcePath("tools/checker/shim") +
        " -I" + SourcePath("tools/checker/include") +
        " -I" + SourcePath("src/include") +
        " -I" + SourcePath("src/collectives/kernels") +
        " " + probe_path + " -o " + probe_bin + " && " + probe_bin;
    const int rc = std::system(compile_cmd.c_str());
    ExpectTrue(rc == 0, "generated exact metadata probe compile command");
}

void TestWriteTraceRunnerSkeletonRejectsProductionSourcePath() {
    tilexr::checker::TraceRunnerSkeletonSpec skeleton_spec;
    skeleton_spec.function_name = "RunAllToAllDemoTrace";
    skeleton_spec.shim_include = "tilexr/checker/alltoall_demo_trace_shim.h";
    skeleton_spec.materializer_name = "MaterializeAllToAllDemo";

    const tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteTraceRunnerSkeletonFile(
            skeleton_spec, "src/collectives/kernels/alltoall_demo_trace_runner.cpp");

    ExpectTrue(!status.ok(), "runner skeleton rejects production output");
    ExpectContains(status.message, "checker output path",
                   "runner skeleton rejection message");
}

void TestWriteTraceRunnerSkeletonFile() {
    const std::string dir = MakeTempDir();
    const std::string path = dir + "/alltoall_demo_trace_runner.cpp";
    tilexr::checker::TraceRunnerSkeletonSpec skeleton_spec;
    skeleton_spec.function_name = "RunAllToAllDemoTrace";
    skeleton_spec.shim_include = "tilexr/checker/alltoall_demo_trace_shim.h";
    skeleton_spec.materializer_name = "MaterializeAllToAllDemo";

    const tilexr::checker::CheckerStatus status =
        tilexr::checker::WriteTraceRunnerSkeletonFile(skeleton_spec, path);

    ExpectTrue(status.ok(), "write trace runner skeleton status");
    const std::string output = ReadFile(path);
    ExpectContains(output, "RunAllToAllDemoTrace",
                   "written runner skeleton function");
    ExpectContains(output, "TODO(checker-runner)",
                   "written runner skeleton manual fill marker");
}

void TestRenderTraceOnboardingPlanDocumentsExecutorWork() {
    tilexr::checker::TraceRunnerSkeletonSpec runner_spec;
    runner_spec.function_name = "RunAllToAllDemoTrace";
    runner_spec.shim_include = "tilexr/checker/alltoall_demo_trace_shim.h";
    runner_spec.materializer_name = "MaterializeAllToAllDemo";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceOnboardingPlan(
            MakeSpec(), runner_spec,
            "tools/checker/shim/tilexr/checker/alltoall_demo_trace_shim.h",
            "/tmp/alltoall_demo_trace_manifest.json",
            "/tmp/alltoall_demo_trace_runner.cpp", &output);

    ExpectTrue(status.ok(), "render trace onboarding plan status");
    ExpectContains(output, "alltoall_demo",
                   "onboarding plan adapter name");
    ExpectContains(output, "CollectiveExecutor",
                   "onboarding plan executor wiring");
    ExpectContains(output, "tools/checker/CMakeLists.txt",
                   "onboarding plan cmake wiring");
    ExpectContains(output, "header probe",
                   "onboarding plan header probe");
    ExpectContains(output, "probe_test",
                   "onboarding plan installed manifest probe");
    ExpectContains(output, "test_tilexr_checker",
                   "onboarding plan test wiring");
    ExpectContains(output, "RunAllToAllDemoTrace",
                   "onboarding plan runner function");
    ExpectContains(output, "MaterializeAllToAllDemo",
                   "onboarding plan materializer");
    ExpectContains(output, "--verify-trace-bundle",
                   "onboarding plan generated bundle verifier command");
    ExpectContains(output, "--verify-installed-trace",
                   "onboarding plan installed verifier command");
    ExpectContains(output, "tools/checker/installed_traces/<adapter>_trace_manifest.json",
                   "onboarding plan installed manifest path");
    ExpectContains(output, "AlgorithmId",
                   "onboarding plan algorithm enum wiring");
    ExpectContains(output, "ParseAlgorithm",
                   "onboarding plan cli algorithm parser wiring");
    ExpectContains(output, "git diff -- src/collectives src/ep",
                   "onboarding plan source preservation check");
    ExpectContains(output, "Do not edit production sources",
                   "onboarding plan production source guard");
}

void TestRenderTraceOnboardingPlanIncludesManualReviewActions() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/raw_copy_collective.h";
    WriteFile(source_path,
              "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n"
              "DataCopy(dst, src, 32);\n");

    tilexr::checker::TraceAdapterSpec adapter_spec;
    adapter_spec.adapter_name = "raw_copy_collective";
    adapter_spec.source_file = source_path;
    adapter_spec.target_header = "raw_copy_collective.h";

    tilexr::checker::TraceRunnerSkeletonSpec runner_spec;
    runner_spec.function_name = "RunRawCopyCollectiveTrace";
    runner_spec.shim_include = "tilexr/checker/raw_copy_collective_trace_shim.h";
    runner_spec.materializer_name = "MaterializeRawCopyCollective";

    std::string output;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::RenderTraceOnboardingPlan(
            adapter_spec, runner_spec,
            "tools/checker/shim/tilexr/checker/raw_copy_collective_trace_shim.h",
            "/tmp/raw_copy_collective_trace_manifest.json",
            "/tmp/raw_copy_collective_trace_runner.cpp", &output);

    ExpectTrue(status.ok(), "render trace onboarding manual review status");
    ExpectContains(output, "## Manual Review Required",
                   "onboarding manual review section");
    ExpectContains(output, "DataCopy",
                   "onboarding manual review symbol");
    ExpectContains(output, "line 2",
                   "onboarding manual review source line");
    ExpectContains(output, "pending",
                   "onboarding manual review pending status");
    ExpectContains(output, "add a trace wrapper or explicit unsupported gate for DataCopy",
                   "onboarding manual review action");
}

}  // namespace

int main() {
    TestNormalizeTraceAdapterSpecInfersCollectiveNameAndTargetHeader();
    TestRenderTraceAdapterIsThinAndSourcePreserving();
    TestWriteTraceAdapterRejectsProductionSourcePath();
    TestWriteTraceAdapterFile();
    TestRenderTraceAdapterManifestDocumentsIntegrationPlan();
    TestRenderTraceAdapterManifestIsBalancedJsonObject();
    TestRenderTraceAdapterManifestReportsObservedSourceHooks();
    TestRenderTraceAdapterManifestIncludesSemiAutomaticTraceStubPlan();
    TestRenderTraceAdapterManifestMarksNoManualReviewWhenSourceIsCovered();
    TestRenderTraceAdapterEmitsObservedHookMetadata();
    TestStrictTraceAdapterManifestRejectsManualReviewCandidates();
    TestTraceAdapterManifestRequiresCoverageDecisionWhenNoHooksObserved();
    TestStrictTraceAdapterManifestRejectsMissingHookCoverage();
    TestTraceAdapterManifestRequiresSourceUnavailableDecision();
    TestAnalyzeTraceSourceReportsCoverageDecisionAction();
    TestAnalyzeTraceSourceReportsSourceUnavailableAction();
    TestStrictTraceAdapterManifestRejectsUnavailableSource();
    TestSourceObservationIgnoresCommentsAndStrings();
    TestSourceObservationUsesSymbolBoundaries();
    TestGeneratedTraceAdapterCompilesAndAuditsMetadata();
    TestWriteTraceAdapterManifestRejectsProductionSourcePath();
    TestWriteTraceAdapterManifestFile();
    TestRenderTraceRunnerSkeletonDocumentsManualFillPoints();
    TestRenderTraceHeaderProbeSkeletonAuditsGeneratedShim();
    TestWriteTraceHeaderProbeSkeletonRejectsProductionSourcePath();
    TestWriteTraceHeaderProbeSkeletonFile();
    TestGeneratedTraceHeaderProbeCompilesAndAuditsExactMetadata();
    TestWriteTraceRunnerSkeletonRejectsProductionSourcePath();
    TestWriteTraceRunnerSkeletonFile();
    TestRenderTraceOnboardingPlanDocumentsExecutorWork();
    TestRenderTraceOnboardingPlanIncludesManualReviewActions();
    return g_failures == 0 ? 0 : 1;
}
