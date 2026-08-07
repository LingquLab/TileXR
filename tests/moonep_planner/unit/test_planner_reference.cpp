#include <iostream>
#include <string>

#include "planner_reference.h"

int main()
{
    const int rankSize = 4;
    const int64_t s = 17;
    const int64_t k = 4;
    const int64_t experts = 32;
    const int64_t b = 4;
    const int64_t tokenPadding = 4;
    bool sawEmptyGroup = false;
    for (const std::string pattern : {"balanced", "biased", "all_local", "all_remote", "duplicate"}) {
        const std::vector<int32_t> routing =
            TileXRMoonEpTest::MakeRouting(pattern, rankSize, s, k, experts, 1234);
        for (int rank = 0; rank < rankSize; ++rank) {
            TileXRMoonEpTest::ReferencePlan plan;
            std::string error;
            if (!TileXRMoonEpTest::BuildReferencePlan(
                    rank, rankSize, s, k, experts, b, tokenPadding,
                    routing, &plan, &error)) {
                std::cerr << pattern << " rank=" << rank << " failed: " << error << std::endl;
                return 1;
            }
            if (plan.dst.size() != static_cast<size_t>(s * k) ||
                plan.cuSeqlens.size() != static_cast<size_t>(experts + b) ||
                plan.expertsToCopy.size() != static_cast<size_t>(rankSize * b) ||
                plan.zeroFillRanges.size() != static_cast<size_t>((experts + b) * 2) ||
                plan.remoteStats.size() != 2) {
                std::cerr << pattern << " output size mismatch" << std::endl;
                return 1;
            }
            const int64_t expectedNvS = s * k + (tokenPadding - 1) * 2 * (experts / rankSize);
            if (plan.nvS != expectedNvS || plan.cuSeqlens.back() > expectedNvS) {
                std::cerr << pattern << " rank=" << rank << " padded capacity mismatch" << std::endl;
                return 1;
            }
            int32_t previousEnd = 0;
            for (int32_t end : plan.cuSeqlens) {
                if (end < previousEnd || end > plan.nvS) {
                    std::cerr << pattern << " rank=" << rank
                              << " invalid cu_seqlens" << std::endl;
                    return 1;
                }
                sawEmptyGroup = sawEmptyGroup || end == previousEnd;
                previousEnd = end;
            }
            for (int32_t encoded : plan.dst) {
                const int64_t raw = encoded < 0 ? -static_cast<int64_t>(encoded) - 1 : encoded;
                if (raw < 0 || raw >= rankSize * plan.nvS) {
                    std::cerr << pattern << " rank=" << rank
                              << " dst encoding out of range" << std::endl;
                    return 1;
                }
            }
        }
    }
    const std::vector<int32_t> routing =
        TileXRMoonEpTest::MakeRouting("balanced", rankSize, s, k, experts, 77);
    TileXRMoonEpTest::ReferencePlan smallB;
    std::string error;
    if (!TileXRMoonEpTest::BuildReferencePlan(
            0, rankSize, s, k, experts, 2, 1, routing, &smallB, &error) ||
        smallB.expertsToCopy.size() != static_cast<size_t>(rankSize * 2)) {
        std::cerr << "small-B inference plan failed: " << error << std::endl;
        return 1;
    }
    if (!sawEmptyGroup) {
        std::cerr << "empty-group repeated-end case was not exercised" << std::endl;
        return 1;
    }
    return 0;
}
