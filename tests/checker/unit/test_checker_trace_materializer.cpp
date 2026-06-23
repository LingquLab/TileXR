#include <cstdint>
#include <iostream>
#include <string>

#include "tilexr/checker/trace_runtime.h"
#include "tilexr/checker/world.h"

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

void ExpectEqI32(int32_t actual, int32_t expected, const char *message) {
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

tilexr::checker::CheckerCase MakeAllReduceCase() {
    tilexr::checker::CheckerCase test_case;
    test_case.op = tilexr::checker::CollectiveOp::kAllReduce;
    test_case.algorithm = tilexr::checker::AlgorithmId::kAllReduceBigData;
    test_case.rank_size = 2;
    test_case.count = 4;
    test_case.data_type = TileXR::TILEXR_DATA_TYPE_INT32;
    test_case.reduce_op = TileXR::TILEXR_REDUCE_SUM;
    test_case.scheduler = tilexr::checker::SchedulerMode::kRoundRobin;
    test_case.magic = 0x1234;
    return test_case;
}

tilexr::checker::RankWorld MakeWorld(const tilexr::checker::CheckerCase &test_case) {
    const size_t bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    return tilexr::checker::RankWorld::Create(test_case.rank_size, bytes, bytes, bytes);
}

void SeedInputs(tilexr::checker::RankWorld *world,
                const tilexr::checker::CheckerCase &test_case) {
    for (int rank = 0; rank < test_case.rank_size; ++rank) {
        for (int64_t i = 0; i < test_case.count; ++i) {
            world->UserInput(rank).WriteInt32(
                static_cast<size_t>(i), static_cast<int32_t>(rank * 100 + i));
        }
    }
}

void AddOutputWriteCoverage(tilexr::checker::RankWorld *world,
                            const tilexr::checker::CheckerCase &test_case) {
    const size_t bytes = static_cast<size_t>(test_case.count) * sizeof(int32_t);
    for (int rank = 0; rank < test_case.rank_size; ++rank) {
        tilexr::checker::Event event;
        event.kind = tilexr::checker::EventKind::kWrite;
        event.rank = rank;
        event.peer_rank = rank;
        event.server = world->ServerOfRank(rank);
        event.peer_server = world->ServerOfRank(rank);
        event.buffer_role = tilexr::checker::BufferRole::kUserOutput;
        event.offset = 0;
        event.bytes = bytes;
        event.source_file = "src/collectives/kernels/allreduce_big_data.h";
        event.source_line = 207;
        event.detail = "test production output write coverage";
        world->events().Add(event);
    }
}

void TestRegisteredMaterializerDescribesAndAppliesAllReduce() {
    tilexr::checker::CheckerCase test_case = MakeAllReduceCase();
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    SeedInputs(&world, test_case);
    tilexr::checker::TraceRuntime runtime(&world, test_case);

    const tilexr::checker::TraceMaterializer *materializer =
        runtime.FindMaterializer("allreduce_big_data_sum_int32");
    ExpectTrue(materializer != nullptr, "allreduce materializer is registered");
    if (materializer == nullptr) {
        return;
    }
    ExpectEqInt(static_cast<int>(materializer->op),
                static_cast<int>(tilexr::checker::CollectiveOp::kAllReduce),
                "allreduce materializer op");
    ExpectEqInt(static_cast<int>(materializer->data_type),
                static_cast<int>(TileXR::TILEXR_DATA_TYPE_INT32),
                "allreduce materializer dtype");
    ExpectEqInt(static_cast<int>(materializer->reduce_op),
                static_cast<int>(TileXR::TILEXR_REDUCE_SUM),
                "allreduce materializer reduce op");
    ExpectTrue(materializer->output_span_bytes ==
                   static_cast<size_t>(test_case.count) * sizeof(int32_t),
               "allreduce materializer output span");

    tilexr::checker::CheckerStatus status =
        runtime.ApplyMaterializer("allreduce_big_data_sum_int32");
    ExpectEqInt(static_cast<int>(status.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "allreduce materializer requires output write coverage");
    ExpectContains(status.message, "user-output write coverage",
                   "allreduce materializer missing coverage message");

    AddOutputWriteCoverage(&world, test_case);
    status = runtime.ApplyMaterializer("allreduce_big_data_sum_int32");
    ExpectTrue(status.ok(), "allreduce apply materializer status");

    int32_t value = 0;
    world.UserOutput(0).ReadInt32(3, &value);
    ExpectEqI32(value, 106, "allreduce rank0 materialized output");
    world.UserOutput(1).ReadInt32(3, &value);
    ExpectEqI32(value, 106, "allreduce rank1 materialized output");
}

void TestMaterializerApplyIsIdempotentAndUnknownIsUnsupported() {
    tilexr::checker::CheckerCase test_case = MakeAllReduceCase();
    tilexr::checker::RankWorld world = MakeWorld(test_case);
    SeedInputs(&world, test_case);
    AddOutputWriteCoverage(&world, test_case);
    tilexr::checker::TraceRuntime runtime(&world, test_case);

    tilexr::checker::CheckerStatus first =
        runtime.ApplyMaterializer("allreduce_big_data_sum_int32");
    tilexr::checker::CheckerStatus second =
        runtime.ApplyMaterializer("allreduce_big_data_sum_int32");
    ExpectTrue(first.ok(), "first materializer apply status");
    ExpectTrue(second.ok(), "second materializer apply status");

    tilexr::checker::CheckerStatus unknown =
        runtime.ApplyMaterializer("not_registered");
    ExpectEqInt(static_cast<int>(unknown.code),
                static_cast<int>(tilexr::checker::CheckerStatusCode::kUnsupported),
                "unknown materializer status");
    ExpectContains(unknown.message, "not_registered",
                   "unknown materializer message");
}

}  // namespace

int main() {
    TestRegisteredMaterializerDescribesAndAppliesAllReduce();
    TestMaterializerApplyIsIdempotentAndUnknownIsUnsupported();
    return g_failures == 0 ? 0 : 1;
}
