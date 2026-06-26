#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "tilexr/checker/trace_bundle_verifier.h"
#include "tilexr/checker/trace_adapter_generator.h"

namespace {

int g_failures = 0;

void ExpectTrue(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << "\n";
        ++g_failures;
    }
}

void ExpectEqInt(int actual, int expected, const char *message) {
    if (actual != expected) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
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

std::string MakeTempDir() {
    char dir_template[] = "/tmp/tilexr-checker-bundle-XXXXXX";
    char *created = mkdtemp(dir_template);
    if (created == nullptr) {
        std::cerr << "mkdtemp failed\n";
        ++g_failures;
        return std::string();
    }
    return std::string(created);
}

void WriteFile(const std::string &path, const std::string &content) {
    std::ofstream output(path.c_str(), std::ios::out | std::ios::trunc);
    output << content;
    if (!output) {
        std::cerr << "write failed: " << path << "\n";
        ++g_failures;
    }
}

void MakeDir(const std::string &path) {
    if (mkdir(path.c_str(), 0700) != 0) {
        std::cerr << "mkdir failed: " << path << "\n";
        ++g_failures;
    }
}

void WriteIncompleteBundle(const std::string &dir) {
    WriteFile(dir + "/demo_trace_shim.h",
              "#define TILEXR_CHECKER_TRACE_SOURCE_FILE \"src/collectives/kernels/demo.h\"\n"
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_demo {\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "}}}\n");
    WriteFile(dir + "/demo_trace_manifest.json",
              "{\"source_preservation\":{},\"runner_integration\":{\"checklist\":["
              "\"block_schedule_defined\",\"operator_init_process_wired\","
              "\"oracle_materializer_reviewed\"]}}\n");
    WriteFile(dir + "/demo_trace_runner.cpp",
              "struct RunnerIntegrationChecklist {\n"
              "  bool block_schedule_defined = false;\n"
              "  bool operator_init_process_wired = false;\n"
              "  bool oracle_materializer_reviewed = false;\n"
              "};\n");
    WriteFile(dir + "/demo_trace_probe.cpp",
              "trace_adapter_demo::Metadata();\n"
              "trace_adapter_demo::Audit(&reason);\n");
    WriteFile(dir + "/demo_trace_onboarding.md",
              "Do not edit production sources.\nCollectiveExecutor\n");
}

void TestVerifyTraceBundleReportsMissingAdapterMetadata() {
    const std::string dir = MakeTempDir();
    WriteFile(dir + "/demo_trace_shim.h",
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n");
    WriteFile(dir + "/demo_trace_manifest.json",
              "{\"source_preservation\":{},\"runner_integration\":{\"checklist\":[]}}\n");
    WriteFile(dir + "/demo_trace_runner.cpp",
              "struct RunnerIntegrationChecklist {\n"
              "  bool block_schedule_defined = true;\n"
              "  bool operator_init_process_wired = true;\n"
              "  bool oracle_materializer_reviewed = true;\n"
              "};\n");
    WriteFile(dir + "/demo_trace_probe.cpp",
              "trace_adapter_demo::Metadata();\n"
              "trace_adapter_demo::Audit(&reason);\n");
    WriteFile(dir + "/demo_trace_onboarding.md",
              "Do not edit production sources.\nCollectiveExecutor\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo", &result);

    ExpectTrue(!status.ok(), "verify bundle missing metadata status");
    ExpectTrue(!result.complete, "verify bundle missing metadata complete flag");
    ExpectContains(result.summary, "shim:TraceAdapterMetadata",
                   "verify bundle missing metadata summary");
    ExpectContains(result.summary, "shim:Metadata",
                   "verify bundle missing metadata function summary");
    ExpectContains(result.summary, "shim:Audit",
                   "verify bundle missing audit function summary");
}

void TestVerifyTraceBundleReportsUncheckedRunnerItems() {
    const std::string dir = MakeTempDir();
    WriteIncompleteBundle(dir);

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo", &result);

    ExpectTrue(!status.ok(), "verify incomplete bundle status");
    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "verify incomplete bundle status code");
    ExpectTrue(!result.complete, "verify incomplete bundle complete flag");
    ExpectEqInt(static_cast<int>(result.missing_artifacts.size()), 0,
                "verify incomplete bundle missing artifact count");
    ExpectEqInt(static_cast<int>(result.unchecked_items.size()), 3,
                "verify incomplete bundle unchecked item count");
    ExpectContains(result.summary, "block_schedule_defined",
                   "verify incomplete bundle summary block schedule");
    ExpectContains(result.summary, "operator_init_process_wired",
                   "verify incomplete bundle summary operator wiring");
    ExpectContains(result.summary, "oracle_materializer_reviewed",
                   "verify incomplete bundle summary materializer");
}

void TestVerifyTraceBundleRequiresManualReviewActionsHandled() {
    const std::string dir = MakeTempDir();
    WriteFile(dir + "/demo_trace_shim.h",
              "#define TILEXR_CHECKER_TRACE_SOURCE_FILE \"src/collectives/kernels/demo.h\"\n"
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_demo {\n"
              "struct TraceAdapterMetadata;\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "}}}\n");
    WriteFile(dir + "/demo_trace_manifest.json",
              "{\n"
              "  \"metadata_api\": {},\n"
              "  \"source_preservation\": {},\n"
              "  \"runner_integration\": {\"checklist\": []},\n"
              "  \"manual_review_required\": true,\n"
              "  \"required_actions\": [\n"
              "    {\"symbol\": \"DataCopy\", \"line\": 7,\n"
              "     \"action\": \"add a trace wrapper or explicit unsupported gate for DataCopy\",\n"
              "     \"status\": \"pending\"}\n"
              "  ]\n"
              "}\n");
    WriteFile(dir + "/demo_trace_runner.cpp",
              "struct RunnerIntegrationChecklist {\n"
              "  bool block_schedule_defined = true;\n"
              "  bool operator_init_process_wired = true;\n"
              "  bool oracle_materializer_reviewed = true;\n"
              "};\n");
    WriteFile(dir + "/demo_trace_probe.cpp",
              "trace_adapter_demo::Metadata();\n"
              "trace_adapter_demo::Audit(&reason);\n");
    WriteFile(dir + "/demo_trace_onboarding.md",
              "Do not edit production sources.\nCollectiveExecutor\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo", &result);

    ExpectTrue(!status.ok(), "verify bundle pending manual review status");
    ExpectTrue(!result.complete, "verify bundle pending manual review complete flag");
    ExpectContains(result.summary, "manual_review:DataCopy:7",
                   "verify bundle pending manual review summary");
}

void TestVerifyTraceBundleRequiresCoverageDecisionHandled() {
    const std::string dir = MakeTempDir();
    WriteFile(dir + "/demo_trace_shim.h",
              "#define TILEXR_CHECKER_TRACE_SOURCE_FILE \"src/collectives/kernels/demo.h\"\n"
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_demo {\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "}}}\n");
    WriteFile(dir + "/demo_trace_manifest.json",
              "{\n"
              "  \"metadata_api\": {},\n"
              "  \"source_preservation\": {},\n"
              "  \"runner_integration\": {\"checklist\": []},\n"
              "  \"manual_review_required\": true,\n"
              "  \"required_actions\": [\n"
              "    {\"symbol\": \"coverage_decision\", \"line\": 0,\n"
              "     \"action\": \"add a traced runner/materializer coverage decision or explicit unsupported gate\",\n"
              "     \"status\": \"pending\"}\n"
              "  ]\n"
              "}\n");
    WriteFile(dir + "/demo_trace_runner.cpp",
              "struct RunnerIntegrationChecklist {\n"
              "  bool block_schedule_defined = true;\n"
              "  bool operator_init_process_wired = true;\n"
              "  bool oracle_materializer_reviewed = true;\n"
              "};\n");
    WriteFile(dir + "/demo_trace_probe.cpp",
              "trace_adapter_demo::Metadata();\n"
              "trace_adapter_demo::Audit(&reason);\n");
    WriteFile(dir + "/demo_trace_onboarding.md",
              "Do not edit production sources.\nCollectiveExecutor\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo", &result);

    ExpectTrue(!status.ok(), "verify bundle pending coverage decision status");
    ExpectTrue(!result.complete, "verify bundle pending coverage decision complete flag");
    ExpectContains(result.summary, "manual_review:coverage_decision:0",
                   "verify bundle pending coverage decision summary");
}

void TestVerifyTraceBundleRequiresSemiAutomaticTraceForGeneratedManifest() {
    const std::string dir = MakeTempDir();
    WriteFile(dir + "/demo_trace_shim.h",
              "#define TILEXR_CHECKER_TRACE_SOURCE_FILE \"src/collectives/kernels/demo.h\"\n"
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_demo {\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "}}}\n");
    WriteFile(dir + "/demo_trace_manifest.json",
              "{\n"
              "  \"metadata_api\": {},\n"
              "  \"generated_outputs\": {},\n"
              "  \"source_preservation\": {},\n"
              "  \"runner_integration\": {\"checklist\": []},\n"
              "  \"required_actions\": []\n"
              "}\n");
    WriteFile(dir + "/demo_trace_runner.cpp",
              "struct RunnerIntegrationChecklist {\n"
              "  bool block_schedule_defined = true;\n"
              "  bool operator_init_process_wired = true;\n"
              "  bool oracle_materializer_reviewed = true;\n"
              "};\n");
    WriteFile(dir + "/demo_trace_probe.cpp",
              "trace_adapter_demo::Metadata();\n"
              "trace_adapter_demo::Audit(&reason);\n");
    WriteFile(dir + "/demo_trace_onboarding.md",
              "Do not edit production sources.\nCollectiveExecutor\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo", &result);

    ExpectTrue(!status.ok(), "verify bundle missing semi automatic trace status");
    ExpectTrue(!result.complete,
               "verify bundle missing semi automatic trace complete flag");
    ExpectContains(result.summary, "semi_automatic_trace",
                   "verify bundle missing semi automatic trace summary");
}

void TestVerifyTraceBundleReportsMissingArtifacts() {
    const std::string dir = MakeTempDir();
    WriteFile(dir + "/demo_trace_manifest.json", "{}\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo", &result);

    ExpectTrue(!status.ok(), "verify missing bundle status");
    ExpectEqInt(static_cast<int>(result.missing_artifacts.size()), 4,
                "verify missing bundle artifact count");
    ExpectContains(result.summary, "demo_trace_shim.h",
                   "verify missing bundle summary shim");
    ExpectContains(result.summary, "demo_trace_runner.cpp",
                   "verify missing bundle summary runner");
    ExpectContains(result.summary, "demo_trace_probe.cpp",
                   "verify missing bundle summary probe");
    ExpectContains(result.summary, "demo_trace_onboarding.md",
                   "verify missing bundle summary onboarding");
}

void TestVerifyTraceBundleRequiresProbeMetadataAuditCalls() {
    const std::string dir = MakeTempDir();
    WriteFile(dir + "/demo_trace_shim.h",
              "#define TILEXR_CHECKER_TRACE_SOURCE_FILE \"src/collectives/kernels/demo.h\"\n"
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_demo {\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "}}}\n");
    WriteFile(dir + "/demo_trace_manifest.json",
              "{\"metadata_api\":{},\"source_preservation\":{},"
              "\"runner_integration\":{\"checklist\":[]}}\n");
    WriteFile(dir + "/demo_trace_runner.cpp",
              "struct RunnerIntegrationChecklist {\n"
              "  bool block_schedule_defined = true;\n"
              "  bool operator_init_process_wired = true;\n"
              "  bool oracle_materializer_reviewed = true;\n"
              "};\n");
    WriteFile(dir + "/demo_trace_probe.cpp",
              "int main() { return 0; }\n");
    WriteFile(dir + "/demo_trace_onboarding.md",
              "Do not edit production sources.\nCollectiveExecutor\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo", &result);

    ExpectTrue(!status.ok(), "verify bundle missing probe calls status");
    ExpectTrue(!result.complete, "verify bundle missing probe calls complete flag");
    ExpectContains(result.summary, "probe:Metadata",
                   "verify bundle missing probe metadata summary");
    ExpectContains(result.summary, "probe:Audit",
                   "verify bundle missing probe audit summary");
}

void TestVerifyTraceBundleUsesSanitizedAdapterNamespaceForProbe() {
    const std::string dir = MakeTempDir();
    WriteFile(dir + "/demo-adapter_trace_shim.h",
              "#define TILEXR_CHECKER_TRACE_SOURCE_FILE \"src/collectives/kernels/demo.h\"\n"
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_demo_adapter {\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "}}}\n");
    WriteFile(dir + "/demo-adapter_trace_manifest.json",
              "{\"metadata_api\":{},\"source_preservation\":{},"
              "\"runner_integration\":{\"checklist\":[]},"
              "\"required_actions\":[]}\n");
    WriteFile(dir + "/demo-adapter_trace_runner.cpp",
              "struct RunnerIntegrationChecklist {\n"
              "  bool block_schedule_defined = true;\n"
              "  bool operator_init_process_wired = true;\n"
              "  bool oracle_materializer_reviewed = true;\n"
              "};\n");
    WriteFile(dir + "/demo-adapter_trace_probe.cpp",
              "trace_adapter_demo_adapter::Metadata();\n"
              "trace_adapter_demo_adapter::Audit(&reason);\n");
    WriteFile(dir + "/demo-adapter_trace_onboarding.md",
              "Do not edit production sources.\nCollectiveExecutor\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo-adapter", &result);

    ExpectTrue(status.ok(), "verify bundle sanitized namespace status");
    ExpectTrue(result.complete, "verify bundle sanitized namespace complete flag");
}

void TestVerifyTraceBundleWithRepoRootCompilesGeneratedProbe() {
    const std::string dir = MakeTempDir();
    const std::string source_path = dir + "/demo_trace_target.h";
    WriteFile(source_path,
              "#pragma once\n"
              "class DemoTraceBundleTarget {\n"
              "public:\n"
              "    void Process() { AscendC::PipeBarrier<PIPE_ALL>(); }\n"
              "};\n");

    tilexr::checker::TraceAdapterSpec adapter_spec;
    adapter_spec.adapter_name = "demo";
    adapter_spec.source_file = source_path;
    adapter_spec.target_header = "demo_trace_target.h";

    tilexr::checker::TraceHeaderProbeSpec probe_spec;
    probe_spec.adapter_name = "demo";
    probe_spec.shim_include = "demo_trace_shim.h";
    probe_spec.expected_source_file = source_path;
    probe_spec.expected_target_header = "demo_trace_target.h";

    const tilexr::checker::CheckerStatus shim_status =
        tilexr::checker::WriteTraceAdapterFile(adapter_spec, dir + "/demo_trace_shim.h");
    ExpectTrue(shim_status.ok(), "verify bundle compile probe write shim");
    const tilexr::checker::CheckerStatus manifest_status =
        tilexr::checker::WriteTraceAdapterManifestFile(
            adapter_spec, dir + "/demo_trace_shim.h",
            dir + "/demo_trace_manifest.json");
    ExpectTrue(manifest_status.ok(), "verify bundle compile probe write manifest");
    const tilexr::checker::CheckerStatus probe_status =
        tilexr::checker::WriteTraceHeaderProbeSkeletonFile(
            probe_spec, dir + "/demo_trace_probe.cpp");
    ExpectTrue(probe_status.ok(), "verify bundle compile probe write probe");
    WriteFile(dir + "/demo_trace_runner.cpp",
              "struct RunnerIntegrationChecklist {\n"
              "  bool block_schedule_defined = true;\n"
              "  bool operator_init_process_wired = true;\n"
              "  bool oracle_materializer_reviewed = true;\n"
              "};\n");
    WriteFile(dir + "/demo_trace_onboarding.md",
              "Do not edit production sources.\nCollectiveExecutor\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo", TILEXR_SOURCE_ROOT, &result);

    ExpectTrue(status.ok(), "verify bundle compile probe status");
    ExpectTrue(result.complete, "verify bundle compile probe complete flag");
    ExpectTrue(result.probe_compile_attempted,
               "verify bundle compile probe attempted flag");
    ExpectTrue(result.probe_compile_passed,
               "verify bundle compile probe passed flag");
    ExpectContains(result.summary, "probe_compile: PASS",
                   "verify bundle compile probe summary");
}

void TestVerifyTraceBundleWithRepoRootReportsProbeCompileFailure() {
    const std::string dir = MakeTempDir();
    WriteFile(dir + "/demo_trace_shim.h",
              "#include \"missing_target_header.h\"\n"
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_demo {\n"
              "static const int kKnownHookCount = 0;\n"
              "static const int kManualReviewCandidateCount = 0;\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "}}}\n");
    WriteFile(dir + "/demo_trace_manifest.json",
              "{\"metadata_api\":{},\"source_preservation\":{},"
              "\"runner_integration\":{\"checklist\":[]},"
              "\"required_actions\":[]}\n");
    WriteFile(dir + "/demo_trace_runner.cpp",
              "struct RunnerIntegrationChecklist {\n"
              "  bool block_schedule_defined = true;\n"
              "  bool operator_init_process_wired = true;\n"
              "  bool oracle_materializer_reviewed = true;\n"
              "};\n");
    WriteFile(dir + "/demo_trace_probe.cpp",
              "#include \"demo_trace_shim.h\"\n"
              "int main() {\n"
              "  const char *reason = nullptr;\n"
              "  tilexr::checker::trace_adapter_demo::Metadata();\n"
              "  tilexr::checker::trace_adapter_demo::Audit(&reason);\n"
              "  return 0;\n"
              "}\n");
    WriteFile(dir + "/demo_trace_onboarding.md",
              "Do not edit production sources.\nCollectiveExecutor\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyTraceBundle(dir, "demo", TILEXR_SOURCE_ROOT, &result);

    ExpectTrue(!status.ok(), "verify bundle compile failure status");
    ExpectTrue(!result.complete, "verify bundle compile failure complete flag");
    ExpectTrue(result.probe_compile_attempted,
               "verify bundle compile failure attempted flag");
    ExpectTrue(!result.probe_compile_passed,
               "verify bundle compile failure passed flag");
    ExpectContains(result.summary, "probe_compile: FAIL",
                   "verify bundle compile failure summary status");
    ExpectContains(result.summary, "probe_compile:",
                   "verify bundle compile failure summary log path");
}

void TestVerifyInstalledAlgorithmReportsCompleteAllReduceBigData() {
    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyInstalledTraceAlgorithm(TILEXR_SOURCE_ROOT,
                                                       "allreduce_big_data", &result);

    ExpectTrue(status.ok(), "verify installed allreduce big data status");
    ExpectTrue(result.complete, "verify installed allreduce big data complete");
    ExpectEqInt(static_cast<int>(result.missing_artifacts.size()), 0,
                "verify installed allreduce big data missing artifact count");
    ExpectEqInt(static_cast<int>(result.unchecked_items.size()), 0,
                "verify installed allreduce big data unchecked item count");
    ExpectTrue(result.runtime_smoke_passed, "verify installed allreduce big data smoke flag");
    ExpectTrue(result.probe_compile_attempted,
               "verify installed allreduce big data probe compile attempted");
    ExpectTrue(result.probe_compile_passed,
               "verify installed allreduce big data probe compile passed");
    ExpectTrue(result.runtime_smoke_event_count > 0,
               "verify installed allreduce big data smoke event count");
    ExpectContains(result.summary, "installed algorithm verification: complete",
                   "verify installed allreduce big data summary");
    ExpectContains(result.summary,
                   "manifest: tools/checker/installed_traces/"
                   "allreduce_big_data_trace_manifest.json",
                   "verify installed allreduce big data manifest summary");
    ExpectContains(result.summary, "RunAllReduceBigDataTrace",
                   "verify installed allreduce big data runner summary");
    ExpectContains(result.summary, "CollectiveExecutor",
                   "verify installed allreduce big data executor summary");
    ExpectContains(result.summary, "runtime_smoke: PASS",
                   "verify installed allreduce big data smoke summary");
    ExpectContains(result.summary, "probe_compile: PASS",
                   "verify installed allreduce big data probe compile summary");
    ExpectContains(result.summary, "runtime_smoke_cases: 2",
                   "verify installed allreduce big data smoke case count summary");
    ExpectContains(result.summary, "materializer: allreduce_big_data_sum_int32",
                   "verify installed allreduce big data materializer summary");
    ExpectContains(result.summary, "schedule: allreduce_big_data_block_major",
                   "verify installed allreduce big data schedule summary");
    ExpectContains(result.summary, "algorithm: allreduce_big_data",
                   "verify installed allreduce big data algorithm summary");
}

void TestVerifyInstalledAlgorithmReportsCompleteAllGatherHierarchyDoubleRing() {
    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyInstalledTraceAlgorithm(
            TILEXR_SOURCE_ROOT, "allgather_hierarchy_double_ring", &result);

    ExpectTrue(status.ok(), "verify installed hdb status");
    ExpectTrue(result.complete, "verify installed hdb complete");
    ExpectEqInt(static_cast<int>(result.missing_artifacts.size()), 0,
                "verify installed hdb missing artifact count");
    ExpectTrue(result.runtime_smoke_passed, "verify installed hdb smoke flag");
    ExpectTrue(result.probe_compile_attempted,
               "verify installed hdb probe compile attempted");
    ExpectTrue(result.probe_compile_passed,
               "verify installed hdb probe compile passed");
    ExpectTrue(result.runtime_smoke_event_count > 0,
               "verify installed hdb smoke event count");
    ExpectContains(result.summary, "RunAllGatherHierarchyDoubleRingTrace",
                   "verify installed hdb runner summary");
    ExpectContains(result.summary,
                   "manifest: tools/checker/installed_traces/"
                   "allgather_hierarchy_double_ring_trace_manifest.json",
                   "verify installed hdb manifest summary");
    ExpectContains(result.summary, "runtime_smoke: PASS",
                   "verify installed hdb smoke summary");
    ExpectContains(result.summary, "probe_compile: PASS",
                   "verify installed hdb probe compile summary");
    ExpectContains(result.summary, "runtime_smoke_cases: 2",
                   "verify installed hdb smoke case count summary");
    ExpectContains(result.summary, "materializer: allgather_hierarchy_double_ring_int32",
                   "verify installed hdb materializer summary");
    ExpectContains(result.summary, "schedule: allgather_hierarchy_double_ring_stage_major",
                   "verify installed hdb schedule summary");
    ExpectContains(result.summary, "algorithm: allgather_hierarchy_double_ring",
                   "verify installed hdb algorithm summary");
}

void TestVerifyInstalledAlgorithmReportsMissingManifestForUnknownAdapter() {
    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyInstalledTraceAlgorithm(TILEXR_SOURCE_ROOT,
                                                       "missing_trace_adapter", &result);

    ExpectTrue(!status.ok(), "verify missing installed manifest status");
    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "verify missing installed manifest status code");
    ExpectTrue(!result.complete, "verify missing installed manifest complete flag");
    ExpectContains(result.summary, "installed algorithm verification: incomplete",
                   "verify missing installed manifest summary status");
    ExpectContains(result.summary,
                   "tools/checker/installed_traces/"
                   "missing_trace_adapter_trace_manifest.json",
                   "verify missing installed manifest path summary");
}

void TestVerifyInstalledAlgorithmReportsManifestFieldProblems() {
    const std::string repo = MakeTempDir();
    MakeDir(repo + "/tools");
    MakeDir(repo + "/tools/checker");
    MakeDir(repo + "/tools/checker/installed_traces");
    WriteFile(repo + "/tools/checker/installed_traces/demo_trace_manifest.json",
              "{\n"
              "  \"adapter_name\": \"demo\",\n"
              "  \"runner_function\": \"RunDemoTrace\"\n"
              "}\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyInstalledTraceAlgorithm(repo, "demo", &result);

    ExpectTrue(!status.ok(), "verify incomplete installed manifest status");
    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "verify incomplete installed manifest status code");
    ExpectTrue(!result.complete, "verify incomplete installed manifest complete flag");
    ExpectContains(result.summary, "installed algorithm verification: incomplete",
                   "verify incomplete installed manifest summary status");
    ExpectContains(result.summary, "manifest: tools/checker/installed_traces/demo_trace_manifest.json",
                   "verify incomplete installed manifest path summary");
    ExpectContains(result.summary, "manifest:shim_path",
                   "verify incomplete installed manifest field summary");
    ExpectContains(result.summary, "manifest:algorithm",
                   "verify incomplete installed manifest algorithm summary");
}

void TestVerifyInstalledAlgorithmStillRequiresManifestEventCoverageWithShimMetadata() {
    const std::string repo = MakeTempDir();
    MakeDir(repo + "/tools");
    MakeDir(repo + "/tools/checker");
    MakeDir(repo + "/tools/checker/installed_traces");
    MakeDir(repo + "/tools/checker/shim");
    MakeDir(repo + "/tools/checker/shim/tilexr");
    MakeDir(repo + "/tools/checker/shim/tilexr/checker");
    MakeDir(repo + "/tools/checker/src");
    MakeDir(repo + "/tools/checker/include");
    MakeDir(repo + "/tools/checker/include/tilexr");
    MakeDir(repo + "/tools/checker/include/tilexr/checker");
    MakeDir(repo + "/tests");
    MakeDir(repo + "/tests/checker");
    MakeDir(repo + "/tests/checker/unit");

    WriteFile(repo + "/tools/checker/installed_traces/allreduce_big_data_trace_manifest.json",
              "{\n"
              "  \"adapter_name\": \"allreduce_big_data\",\n"
              "  \"source_file\": \"src/collectives/kernels/allreduce_big_data.h\",\n"
              "  \"shim_path\": \"tools/checker/shim/tilexr/checker/arbd.h\",\n"
              "  \"runner_source\": \"tools/checker/src/arbd_runner.cpp\",\n"
              "  \"runner_function\": \"RunAllReduceBigDataTrace\",\n"
              "  \"probe_test\": \"tests/checker/unit/arbd_probe.cpp\",\n"
              "  \"probe_cmake_entry\": \"unit/arbd_probe.cpp\",\n"
              "  \"algorithm_enum\": \"AlgorithmId::kAllReduceBigData\",\n"
              "  \"materializer_name\": \"allreduce_big_data_sum_int32\",\n"
              "  \"schedule_name\": \"allreduce_big_data_block_major\",\n"
              "  \"schedule_function\": \"GetAllReduceBigDataScheduleSpec\",\n"
              "  \"op\": \"allreduce\",\n"
              "  \"algorithm\": \"allreduce_big_data\",\n"
              "  \"rank_size\": 4,\n"
              "  \"count\": 524288,\n"
              "  \"server_count\": 2\n"
              "}\n");
    WriteFile(repo + "/tools/checker/shim/tilexr/checker/arbd.h",
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_allreduce_big_data {\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "static const char *kSource = \"src/collectives/kernels/allreduce_big_data.h\";\n"
              "}}}\n");
    WriteFile(repo + "/tools/checker/include/tilexr/checker/collective_trace_runner.h",
              "CheckerStatus RunAllReduceBigDataTrace();\n");
    WriteFile(repo + "/tools/checker/src/arbd_runner.cpp",
              "RunAllReduceBigDataTrace();\n"
              "runtime.ApplyMaterializer(\"allreduce_big_data_sum_int32\");\n"
              "GetAllReduceBigDataScheduleSpec(test_case);\n");
    WriteFile(repo + "/tools/checker/src/executor.cpp",
              "AlgorithmId::kAllReduceBigData\n");
    WriteFile(repo + "/tests/checker/unit/arbd_probe.cpp",
              "int main() { static_assert(true, \"probe\"); return 0; }\n");
    WriteFile(repo + "/tests/checker/CMakeLists.txt",
              "unit/arbd_probe.cpp\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyInstalledTraceAlgorithm(repo, "allreduce_big_data", &result);

    ExpectTrue(!status.ok(), "verify installed requires manifest event coverage status");
    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "verify installed requires manifest event coverage status code");
    ExpectTrue(!result.complete,
               "verify installed requires manifest event coverage complete flag");
    ExpectContains(result.summary, "manifest:required_event_coverage",
                   "verify installed requires manifest event coverage summary");
}

void TestVerifyInstalledAlgorithmRequiresShimMetadata() {
    const std::string repo = MakeTempDir();
    MakeDir(repo + "/tools");
    MakeDir(repo + "/tools/checker");
    MakeDir(repo + "/tools/checker/installed_traces");
    MakeDir(repo + "/tools/checker/shim");
    MakeDir(repo + "/tools/checker/shim/tilexr");
    MakeDir(repo + "/tools/checker/shim/tilexr/checker");
    MakeDir(repo + "/tools/checker/src");
    MakeDir(repo + "/tools/checker/include");
    MakeDir(repo + "/tools/checker/include/tilexr");
    MakeDir(repo + "/tools/checker/include/tilexr/checker");
    MakeDir(repo + "/tests");
    MakeDir(repo + "/tests/checker");
    MakeDir(repo + "/tests/checker/unit");

    WriteFile(repo + "/tools/checker/installed_traces/allreduce_big_data_trace_manifest.json",
              "{\n"
              "  \"adapter_name\": \"allreduce_big_data\",\n"
              "  \"source_file\": \"src/collectives/kernels/allreduce_big_data.h\",\n"
              "  \"shim_path\": \"tools/checker/shim/tilexr/checker/arbd.h\",\n"
              "  \"runner_source\": \"tools/checker/src/arbd_runner.cpp\",\n"
              "  \"runner_function\": \"RunAllReduceBigDataTrace\",\n"
              "  \"probe_test\": \"tests/checker/unit/arbd_probe.cpp\",\n"
              "  \"probe_cmake_entry\": \"unit/arbd_probe.cpp\",\n"
              "  \"algorithm_enum\": \"AlgorithmId::kAllReduceBigData\",\n"
              "  \"materializer_name\": \"allreduce_big_data_sum_int32\",\n"
              "  \"schedule_name\": \"allreduce_big_data_block_major\",\n"
              "  \"schedule_function\": \"GetAllReduceBigDataScheduleSpec\",\n"
              "  \"op\": \"allreduce\",\n"
              "  \"algorithm\": \"allreduce_big_data\",\n"
              "  \"rank_size\": 4,\n"
              "  \"count\": 524288,\n"
              "  \"server_count\": 2,\n"
              "  \"required_event_coverage\": [\n"
              "    {\"kind\": \"COPY\", \"source_file\": \"src/collectives/kernels/"
              "allreduce_big_data.h\", \"source_line\": 156}\n"
              "  ]\n"
              "}\n");
    WriteFile(repo + "/tools/checker/shim/tilexr/checker/arbd.h",
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n");
    WriteFile(repo + "/tools/checker/include/tilexr/checker/collective_trace_runner.h",
              "CheckerStatus RunAllReduceBigDataTrace();\n");
    WriteFile(repo + "/tools/checker/src/arbd_runner.cpp",
              "RunAllReduceBigDataTrace();\n"
              "runtime.ApplyMaterializer(\"allreduce_big_data_sum_int32\");\n"
              "GetAllReduceBigDataScheduleSpec(test_case);\n");
    WriteFile(repo + "/tools/checker/src/executor.cpp",
              "AlgorithmId::kAllReduceBigData\n");
    WriteFile(repo + "/tests/checker/unit/arbd_probe.cpp",
              "int main() { static_assert(true, \"probe\"); return 0; }\n");
    WriteFile(repo + "/tests/checker/CMakeLists.txt",
              "unit/arbd_probe.cpp\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyInstalledTraceAlgorithm(repo, "allreduce_big_data", &result);

    ExpectTrue(!status.ok(), "verify installed missing shim metadata status");
    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "verify installed missing shim metadata status code");
    ExpectTrue(!result.complete, "verify installed missing shim metadata complete flag");
    ExpectContains(result.summary, "shim:TraceAdapterMetadata",
                   "verify installed missing shim metadata summary");
    ExpectContains(result.summary, "shim:Metadata",
                   "verify installed missing shim metadata function summary");
    ExpectContains(result.summary, "shim:Audit",
                   "verify installed missing shim audit function summary");
}

void TestVerifyInstalledAlgorithmReportsMissingRequiredEventCoverage() {
    const std::string repo = MakeTempDir();
    MakeDir(repo + "/src");
    MakeDir(repo + "/src/collectives");
    MakeDir(repo + "/src/collectives/kernels");
    MakeDir(repo + "/tools");
    MakeDir(repo + "/tools/checker");
    MakeDir(repo + "/tools/checker/installed_traces");
    MakeDir(repo + "/tools/checker/shim");
    MakeDir(repo + "/tools/checker/shim/tilexr");
    MakeDir(repo + "/tools/checker/shim/tilexr/checker");
    MakeDir(repo + "/tools/checker/src");
    MakeDir(repo + "/tools/checker/include");
    MakeDir(repo + "/tools/checker/include/tilexr");
    MakeDir(repo + "/tools/checker/include/tilexr/checker");
    MakeDir(repo + "/tests");
    MakeDir(repo + "/tests/checker");
    MakeDir(repo + "/tests/checker/unit");

    WriteFile(repo + "/src/collectives/kernels/allreduce_big_data.h",
              "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n");
    WriteFile(repo + "/tools/checker/installed_traces/allreduce_big_data_trace_manifest.json",
              "{\n"
              "  \"adapter_name\": \"allreduce_big_data\",\n"
              "  \"source_file\": \"src/collectives/kernels/allreduce_big_data.h\",\n"
              "  \"shim_path\": \"tools/checker/shim/tilexr/checker/arbd.h\",\n"
              "  \"runner_source\": \"tools/checker/src/arbd_runner.cpp\",\n"
              "  \"runner_function\": \"RunAllReduceBigDataTrace\",\n"
              "  \"probe_test\": \"tests/checker/unit/arbd_probe.cpp\",\n"
              "  \"probe_cmake_entry\": \"unit/arbd_probe.cpp\",\n"
              "  \"algorithm_enum\": \"AlgorithmId::kAllReduceBigData\",\n"
              "  \"materializer_name\": \"allreduce_big_data_sum_int32\",\n"
              "  \"schedule_name\": \"allreduce_big_data_block_major\",\n"
              "  \"schedule_function\": \"GetAllReduceBigDataScheduleSpec\",\n"
              "  \"op\": \"allreduce\",\n"
              "  \"algorithm\": \"allreduce_big_data\",\n"
              "  \"rank_size\": 4,\n"
              "  \"count\": 524288,\n"
              "  \"server_count\": 2,\n"
              "  \"required_event_coverage\": [\n"
              "    {\"kind\": \"COPY\", \"source_file\": \"src/collectives/kernels/"
              "allreduce_big_data.h\", \"source_line\": 9999}\n"
              "  ]\n"
              "}\n");
    WriteFile(repo + "/tools/checker/shim/tilexr/checker/arbd.h",
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_allreduce_big_data {\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "static const char *kSource = \"src/collectives/kernels/allreduce_big_data.h\";\n"
              "static const int kKnownHook0Lines[] = {1};\n"
              "{\"CpGM2GMPingPong\", kKnownHook0Lines, 1};\n"
              "}}}\n");
    WriteFile(repo + "/tools/checker/include/tilexr/checker/collective_trace_runner.h",
              "CheckerStatus RunAllReduceBigDataTrace();\n");
    WriteFile(repo + "/tools/checker/src/arbd_runner.cpp",
              "RunAllReduceBigDataTrace();\n"
              "runtime.ApplyMaterializer(\"allreduce_big_data_sum_int32\");\n"
              "GetAllReduceBigDataScheduleSpec(test_case);\n");
    WriteFile(repo + "/tools/checker/src/executor.cpp",
              "AlgorithmId::kAllReduceBigData\n");
    WriteFile(repo + "/tests/checker/unit/arbd_probe.cpp",
              "int main() { static_assert(true, \"probe\"); return 0; }\n");
    WriteFile(repo + "/tests/checker/CMakeLists.txt",
              "unit/arbd_probe.cpp\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyInstalledTraceAlgorithm(repo, "allreduce_big_data", &result);

    ExpectTrue(!status.ok(), "verify installed missing event coverage status");
    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "verify installed missing event coverage status code");
    ExpectTrue(!result.complete, "verify installed missing event coverage complete flag");
    ExpectContains(result.summary,
                   "event_coverage:COPY:src/collectives/kernels/allreduce_big_data.h:9999",
                   "verify installed missing event coverage summary");
}

void TestVerifyInstalledAlgorithmRejectsSourceHookDriftNotInShimMetadata() {
    const std::string repo = MakeTempDir();
    MakeDir(repo + "/src");
    MakeDir(repo + "/src/collectives");
    MakeDir(repo + "/src/collectives/kernels");
    MakeDir(repo + "/tools");
    MakeDir(repo + "/tools/checker");
    MakeDir(repo + "/tools/checker/installed_traces");
    MakeDir(repo + "/tools/checker/shim");
    MakeDir(repo + "/tools/checker/shim/tilexr");
    MakeDir(repo + "/tools/checker/shim/tilexr/checker");
    MakeDir(repo + "/tools/checker/src");
    MakeDir(repo + "/tools/checker/include");
    MakeDir(repo + "/tools/checker/include/tilexr");
    MakeDir(repo + "/tools/checker/include/tilexr/checker");
    MakeDir(repo + "/tests");
    MakeDir(repo + "/tests/checker");
    MakeDir(repo + "/tests/checker/unit");

    WriteFile(repo + "/src/collectives/kernels/demo.h",
              "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n");
    WriteFile(repo + "/tools/checker/installed_traces/demo_trace_manifest.json",
              "{\n"
              "  \"adapter_name\": \"demo\",\n"
              "  \"source_file\": \"src/collectives/kernels/demo.h\",\n"
              "  \"shim_path\": \"tools/checker/shim/tilexr/checker/demo.h\",\n"
              "  \"runner_source\": \"tools/checker/src/demo_runner.cpp\",\n"
              "  \"runner_function\": \"RunDemoTrace\",\n"
              "  \"probe_test\": \"tests/checker/unit/demo_probe.cpp\",\n"
              "  \"probe_cmake_entry\": \"unit/demo_probe.cpp\",\n"
              "  \"algorithm_enum\": \"AlgorithmId::kAllReduceBigData\",\n"
              "  \"materializer_name\": \"allreduce_big_data_sum_int32\",\n"
              "  \"schedule_name\": \"allreduce_big_data_block_major\",\n"
              "  \"schedule_function\": \"GetAllReduceBigDataScheduleSpec\",\n"
              "  \"op\": \"allreduce\",\n"
              "  \"algorithm\": \"allreduce_big_data\",\n"
              "  \"rank_size\": 4,\n"
              "  \"count\": 524288,\n"
              "  \"server_count\": 2,\n"
              "  \"required_event_coverage\": [\n"
              "    {\"kind\": \"COPY\", \"source_file\": \"src/collectives/kernels/"
              "demo.h\", \"source_line\": 1}\n"
              "  ]\n"
              "}\n");
    WriteFile(repo + "/tools/checker/shim/tilexr/checker/demo.h",
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_demo {\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "static const char *kSource = \"src/collectives/kernels/demo.h\";\n"
              "static const int kKnownHook0Lines[] = {99};\n"
              "}}}\n");
    WriteFile(repo + "/tools/checker/include/tilexr/checker/collective_trace_runner.h",
              "CheckerStatus RunDemoTrace();\n");
    WriteFile(repo + "/tools/checker/src/demo_runner.cpp",
              "RunDemoTrace();\n"
              "runtime.ApplyMaterializer(\"allreduce_big_data_sum_int32\");\n"
              "GetAllReduceBigDataScheduleSpec(test_case);\n");
    WriteFile(repo + "/tools/checker/src/executor.cpp",
              "AlgorithmId::kAllReduceBigData\n");
    WriteFile(repo + "/tests/checker/unit/demo_probe.cpp",
              "int main() { static_assert(true, \"probe\"); return 0; }\n");
    WriteFile(repo + "/tests/checker/CMakeLists.txt",
              "unit/demo_probe.cpp\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyInstalledTraceAlgorithm(repo, "demo", &result);

    ExpectTrue(!status.ok(), "verify installed source hook drift status");
    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "verify installed source hook drift status code");
    ExpectTrue(!result.complete, "verify installed source hook drift complete flag");
    ExpectContains(result.summary, "source_scan:CpGM2GMPingPong:1",
                   "verify installed source hook drift summary");
}

void TestVerifyInstalledEpDispatchUsesEpSmokeCaseShape() {
    const std::string repo = MakeTempDir();
    MakeDir(repo + "/src");
    MakeDir(repo + "/src/ep");
    MakeDir(repo + "/src/ep/kernels");
    MakeDir(repo + "/tools");
    MakeDir(repo + "/tools/checker");
    MakeDir(repo + "/tools/checker/installed_traces");
    MakeDir(repo + "/tools/checker/shim");
    MakeDir(repo + "/tools/checker/shim/tilexr");
    MakeDir(repo + "/tools/checker/shim/tilexr/checker");
    MakeDir(repo + "/tools/checker/src");
    MakeDir(repo + "/tools/checker/include");
    MakeDir(repo + "/tools/checker/include/tilexr");
    MakeDir(repo + "/tools/checker/include/tilexr/checker");
    MakeDir(repo + "/tests");
    MakeDir(repo + "/tests/checker");
    MakeDir(repo + "/tests/checker/unit");

    WriteFile(repo + "/src/ep/kernels/tilexr_ep_dispatch_kernel.cpp",
              "Collectives::CpGM2GMPingPong(bytes, input, output, op);\n");
    WriteFile(repo + "/tools/checker/installed_traces/ep_dispatch_trace_manifest.json",
              "{\n"
              "  \"adapter_name\": \"ep_dispatch\",\n"
              "  \"source_file\": \"src/ep/kernels/tilexr_ep_dispatch_kernel.cpp\",\n"
              "  \"shim_path\": \"tools/checker/shim/tilexr/checker/ep_dispatch.h\",\n"
              "  \"runner_source\": \"tools/checker/src/ep_dispatch_runner.cpp\",\n"
              "  \"runner_function\": \"RunEpDispatchTrace\",\n"
              "  \"probe_test\": \"tests/checker/unit/ep_dispatch_probe.cpp\",\n"
              "  \"probe_cmake_entry\": \"unit/ep_dispatch_probe.cpp\",\n"
              "  \"algorithm_enum\": \"CollectiveOp::kEpDispatch\",\n"
              "  \"materializer_name\": \"ep_dispatch_fp16\",\n"
              "  \"schedule_name\": \"ep_dispatch_rank_major\",\n"
              "  \"schedule_function\": \"GetEpDispatchScheduleSpec\",\n"
              "  \"op\": \"ep_dispatch\",\n"
              "  \"algorithm\": \"default\",\n"
              "  \"rank_size\": 2,\n"
              "  \"server_count\": 2,\n"
              "  \"datatype\": \"fp16\",\n"
              "  \"smoke_cases\": [\n"
              "    {\"rank_size\": 2, \"server_count\": 2, \"bs\": 3, \"h\": 4,\n"
              "     \"top_k\": 2, \"moe_expert_num\": 4, \"datatype\": \"fp16\"}\n"
              "  ],\n"
              "  \"required_event_coverage\": [\n"
              "    {\"kind\": \"COPY\", \"source_file\": \"src/ep/kernels/"
              "tilexr_ep_dispatch_kernel.cpp\", \"source_line\": 296}\n"
              "  ]\n"
              "}\n");
    WriteFile(repo + "/tools/checker/shim/tilexr/checker/ep_dispatch.h",
              "#include \"tilexr/checker/collective_trace_adapter.h\"\n"
              "namespace tilexr { namespace checker { namespace trace_adapter_ep_dispatch {\n"
              "inline const TraceAdapterMetadata &Metadata();\n"
              "inline bool Audit(const char **reason);\n"
              "static const char *kSource = \"src/ep/kernels/tilexr_ep_dispatch_kernel.cpp\";\n"
              "static const int kKnownHook0Lines[] = {1};\n"
              "{\"CpGM2GMPingPong\", kKnownHook0Lines, 1};\n"
              "}}}\n");
    WriteFile(repo + "/tools/checker/include/tilexr/checker/collective_trace_runner.h",
              "CheckerStatus RunEpDispatchTrace();\n");
    WriteFile(repo + "/tools/checker/src/ep_dispatch_runner.cpp",
              "RunEpDispatchTrace();\n"
              "runtime.ApplyMaterializer(\"ep_dispatch_fp16\");\n"
              "GetEpDispatchScheduleSpec(test_case);\n");
    WriteFile(repo + "/tools/checker/src/executor.cpp",
              "CollectiveOp::kEpDispatch\n");
    WriteFile(repo + "/tests/checker/unit/ep_dispatch_probe.cpp",
              "int main() { static_assert(true, \"probe\"); return 0; }\n");
    WriteFile(repo + "/tests/checker/CMakeLists.txt",
              "unit/ep_dispatch_probe.cpp\n");

    tilexr::checker::TraceBundleVerificationResult result;
    const tilexr::checker::CheckerStatus status =
        tilexr::checker::VerifyInstalledTraceAlgorithm(repo, "ep_dispatch", &result);

    ExpectTrue(status.ok(), "verify installed ep dispatch smoke status");
    ExpectTrue(result.complete, "verify installed ep dispatch complete flag");
    ExpectTrue(result.runtime_smoke_passed, "verify installed ep dispatch smoke flag");
    ExpectEqInt(static_cast<int>(result.runtime_smoke_case_count), 1,
                "verify installed ep dispatch smoke case count");
    ExpectTrue(result.runtime_smoke_event_count > 0,
               "verify installed ep dispatch smoke event count");
    ExpectContains(result.summary, "runtime_smoke: PASS",
                   "verify installed ep dispatch smoke summary");
}

}  // namespace

int main() {
    TestVerifyTraceBundleReportsMissingAdapterMetadata();
    TestVerifyTraceBundleReportsUncheckedRunnerItems();
    TestVerifyTraceBundleRequiresManualReviewActionsHandled();
    TestVerifyTraceBundleRequiresCoverageDecisionHandled();
    TestVerifyTraceBundleRequiresSemiAutomaticTraceForGeneratedManifest();
    TestVerifyTraceBundleReportsMissingArtifacts();
    TestVerifyTraceBundleRequiresProbeMetadataAuditCalls();
    TestVerifyTraceBundleUsesSanitizedAdapterNamespaceForProbe();
    TestVerifyTraceBundleWithRepoRootCompilesGeneratedProbe();
    TestVerifyTraceBundleWithRepoRootReportsProbeCompileFailure();
    TestVerifyInstalledAlgorithmReportsCompleteAllReduceBigData();
    TestVerifyInstalledAlgorithmReportsCompleteAllGatherHierarchyDoubleRing();
    TestVerifyInstalledAlgorithmReportsMissingManifestForUnknownAdapter();
    TestVerifyInstalledAlgorithmReportsManifestFieldProblems();
    TestVerifyInstalledAlgorithmStillRequiresManifestEventCoverageWithShimMetadata();
    TestVerifyInstalledAlgorithmRequiresShimMetadata();
    TestVerifyInstalledAlgorithmReportsMissingRequiredEventCoverage();
    TestVerifyInstalledAlgorithmRejectsSourceHookDriftNotInShimMetadata();
    TestVerifyInstalledEpDispatchUsesEpSmokeCaseShape();
    return g_failures == 0 ? 0 : 1;
}
