#include "planner_reference.h"

#include <algorithm>
#include <limits>

namespace TileXRMoonEpTest {
namespace {

int64_t AllocIndex(int64_t expert, int32_t dest, int32_t rankSize)
{
    return expert * rankSize + dest;
}

uint32_t Hash(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

} // namespace

std::vector<int32_t> MakeRouting(const std::string &pattern, int32_t rankSize,
    int64_t s, int64_t k, int64_t expertCount, uint32_t seed)
{
    const int64_t n = s * k;
    const int64_t expertsPerRank = expertCount / rankSize;
    std::vector<int32_t> allTopk(static_cast<size_t>(rankSize * n));
    for (int32_t rank = 0; rank < rankSize; ++rank) {
        for (int64_t token = 0; token < s; ++token) {
            int32_t duplicateBase = static_cast<int32_t>(
                Hash(seed + static_cast<uint32_t>(rank * s + token)) % expertCount);
            for (int64_t topk = 0; topk < k; ++topk) {
                const int64_t route = token * k + topk;
                int32_t expert = 0;
                if (pattern == "all_local") {
                    expert = static_cast<int32_t>(rank * expertsPerRank + route % expertsPerRank);
                } else if (pattern == "all_remote") {
                    const int32_t owner = (rank + 1) % rankSize;
                    expert = static_cast<int32_t>(owner * expertsPerRank + route % expertsPerRank);
                } else if (pattern == "duplicate") {
                    expert = topk < k / 2 ? duplicateBase : static_cast<int32_t>(
                        Hash(seed + static_cast<uint32_t>(rank * n + route)) % expertCount);
                } else if (pattern == "biased") {
                    if ((route + rank) % 5 != 0) {
                        expert = static_cast<int32_t>(route % expertsPerRank);
                    } else {
                        expert = static_cast<int32_t>(
                            Hash(seed + static_cast<uint32_t>(rank * n + route)) % expertCount);
                    }
                } else {
                    expert = static_cast<int32_t>((rank * n + route) % expertCount);
                }
                allTopk[static_cast<size_t>(rank * n + route)] = expert;
            }
        }
    }
    return allTopk;
}

bool BuildReferencePlan(int32_t rank, int32_t rankSize, int64_t s, int64_t k,
    int64_t expertCount, const std::vector<int32_t> &allTopk,
    ReferencePlan *plan, std::string *error)
{
    if (plan == nullptr || error == nullptr || rank < 0 || rank >= rankSize ||
        rankSize <= 0 || s <= 0 || k <= 0 || expertCount <= 0 ||
        expertCount % rankSize != 0 ||
        s > std::numeric_limits<int64_t>::max() / k) {
        return false;
    }
    error->clear();
    const int64_t n = s * k;
    const int64_t b = expertCount / rankSize;
    const int64_t nvs = n;
    const uint64_t encodedCapacity =
        static_cast<uint64_t>(rankSize) * static_cast<uint64_t>(nvs);
    if (nvs > std::numeric_limits<int32_t>::max() ||
        encodedCapacity > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1U) {
        *error = "encoded destination range exceeds int32";
        return false;
    }
    if (static_cast<int64_t>(allTopk.size()) != rankSize * n) {
        *error = "allTopk size mismatch";
        return false;
    }

    std::vector<int32_t> tpe(static_cast<size_t>(rankSize * expertCount), 0);
    for (int32_t source = 0; source < rankSize; ++source) {
        for (int64_t route = 0; route < n; ++route) {
            const int32_t expert = allTopk[static_cast<size_t>(source * n + route)];
            if (expert < 0 || expert >= expertCount) {
                *error = "expert id out of range";
                return false;
            }
            ++tpe[static_cast<size_t>(source * expertCount + expert)];
        }
    }

    std::vector<int32_t> tpePrefix = tpe;
    std::vector<int32_t> expertCountGlobal(static_cast<size_t>(expertCount), 0);
    for (int64_t expert = 0; expert < expertCount; ++expert) {
        int32_t cumulative = 0;
        for (int32_t source = 0; source < rankSize; ++source) {
            cumulative += tpe[static_cast<size_t>(source * expertCount + expert)];
            tpePrefix[static_cast<size_t>(source * expertCount + expert)] = cumulative;
        }
        expertCountGlobal[static_cast<size_t>(expert)] = cumulative;
    }

    std::vector<int32_t> balance(static_cast<size_t>(rankSize), 0);
    for (int32_t home = 0; home < rankSize; ++home) {
        int32_t total = 0;
        for (int64_t localExpert = 0; localExpert < b; ++localExpert) {
            total += expertCountGlobal[static_cast<size_t>(home * b + localExpert)];
        }
        balance[static_cast<size_t>(home)] = total - static_cast<int32_t>(n);
    }

    std::vector<int32_t> z(static_cast<size_t>(rankSize * rankSize), 0);
    while (true) {
        int32_t surplusRank = 0;
        int32_t deficitRank = 0;
        for (int32_t r = 1; r < rankSize; ++r) {
            if (balance[static_cast<size_t>(r)] > balance[static_cast<size_t>(surplusRank)]) {
                surplusRank = r;
            }
            if (balance[static_cast<size_t>(r)] < balance[static_cast<size_t>(deficitRank)]) {
                deficitRank = r;
            }
        }
        if (balance[static_cast<size_t>(surplusRank)] <= 0 ||
            balance[static_cast<size_t>(deficitRank)] >= 0) {
            break;
        }
        const int32_t move = -balance[static_cast<size_t>(deficitRank)];
        z[static_cast<size_t>(surplusRank * rankSize + deficitRank)] = move;
        balance[static_cast<size_t>(surplusRank)] -= move;
        balance[static_cast<size_t>(deficitRank)] = 0;
    }

    std::vector<int32_t> alloc(static_cast<size_t>(expertCount * rankSize), 0);
    for (int64_t expert = 0; expert < expertCount; ++expert) {
        alloc[static_cast<size_t>(AllocIndex(expert, static_cast<int32_t>(expert / b), rankSize))] =
            expertCountGlobal[static_cast<size_t>(expert)];
    }
    for (int32_t home = 0; home < rankSize; ++home) {
        std::vector<int32_t> remaining(static_cast<size_t>(b));
        std::vector<int32_t> quotas(static_cast<size_t>(rankSize));
        for (int64_t localExpert = 0; localExpert < b; ++localExpert) {
            remaining[static_cast<size_t>(localExpert)] =
                expertCountGlobal[static_cast<size_t>(home * b + localExpert)];
        }
        for (int32_t dest = 0; dest < rankSize; ++dest) {
            quotas[static_cast<size_t>(dest)] = z[static_cast<size_t>(home * rankSize + dest)];
        }
        while (true) {
            const int32_t dest = static_cast<int32_t>(
                std::max_element(quotas.begin(), quotas.end()) - quotas.begin());
            if (quotas[static_cast<size_t>(dest)] <= 0) {
                break;
            }
            const int64_t localExpert = std::max_element(remaining.begin(), remaining.end()) - remaining.begin();
            if (remaining[static_cast<size_t>(localExpert)] <= 0) {
                *error = "allocation exhausted before quota";
                return false;
            }
            const int32_t take = std::min(
                remaining[static_cast<size_t>(localExpert)], quotas[static_cast<size_t>(dest)]);
            const int64_t expert = home * b + localExpert;
            alloc[static_cast<size_t>(AllocIndex(expert, dest, rankSize))] += take;
            alloc[static_cast<size_t>(AllocIndex(expert, home, rankSize))] -= take;
            remaining[static_cast<size_t>(localExpert)] -= take;
            quotas[static_cast<size_t>(dest)] -= take;
        }
    }

    std::vector<int32_t> expertOffsets(static_cast<size_t>(rankSize * expertCount), 0);
    std::vector<int32_t> cuAll(static_cast<size_t>(rankSize * (expertCount + b)), 0);
    std::vector<int32_t> expertsToCopy(static_cast<size_t>(rankSize * b), -1);
    std::vector<int32_t> remoteStatsAll(static_cast<size_t>(rankSize * 2), 0);

    for (int32_t dest = 0; dest < rankSize; ++dest) {
        std::vector<int32_t> remote;
        const int64_t localBegin = dest * b;
        const int64_t localEnd = localBegin + b;
        for (int64_t expert = 0; expert < expertCount; ++expert) {
            if ((expert < localBegin || expert >= localEnd) &&
                alloc[static_cast<size_t>(AllocIndex(expert, dest, rankSize))] > 0) {
                remote.push_back(static_cast<int32_t>(expert));
            }
        }
        std::sort(remote.begin(), remote.end(), [&](int32_t lhs, int32_t rhs) {
            const int32_t lhsCount = alloc[static_cast<size_t>(AllocIndex(lhs, dest, rankSize))];
            const int32_t rhsCount = alloc[static_cast<size_t>(AllocIndex(rhs, dest, rankSize))];
            return lhsCount != rhsCount ? lhsCount > rhsCount : lhs > rhs;
        });
        remoteStatsAll[static_cast<size_t>(dest * 2)] = static_cast<int32_t>(remote.size());
        std::vector<uint8_t> selected(static_cast<size_t>(expertCount), 0);
        for (int64_t slot = 0; slot < b && slot < static_cast<int64_t>(remote.size()); ++slot) {
            const int32_t expert = remote[static_cast<size_t>(slot)];
            expertsToCopy[static_cast<size_t>(dest * b + slot)] = expert;
            selected[static_cast<size_t>(expert)] = 1;
            ++remoteStatsAll[static_cast<size_t>((expert / b) * 2 + 1)];
        }

        int32_t start = 0;
        for (int64_t group = 0; group < expertCount + b; ++group) {
            int32_t count = 0;
            int32_t expert = -1;
            if (group < expertCount) {
                if (selected[static_cast<size_t>(group)] == 0) {
                    expert = static_cast<int32_t>(group);
                    count = alloc[static_cast<size_t>(AllocIndex(group, dest, rankSize))];
                }
            } else {
                expert = expertsToCopy[static_cast<size_t>(dest * b + group - expertCount)];
                if (expert >= 0) {
                    count = alloc[static_cast<size_t>(AllocIndex(expert, dest, rankSize))];
                }
            }
            if (count > 0) {
                expertOffsets[static_cast<size_t>(dest * expertCount + expert)] = start;
            }
            start += count;
            cuAll[static_cast<size_t>(dest * (expertCount + b) + group)] = start;
        }
        if (start != nvs) {
            *error = "compact layout does not fill NvS";
            return false;
        }
    }

    std::vector<int32_t> allocPrefix = alloc;
    for (int64_t expert = 0; expert < expertCount; ++expert) {
        int32_t cumulative = 0;
        for (int32_t dest = 0; dest < rankSize; ++dest) {
            cumulative += alloc[static_cast<size_t>(AllocIndex(expert, dest, rankSize))];
            allocPrefix[static_cast<size_t>(AllocIndex(expert, dest, rankSize))] = cumulative;
        }
    }

    std::vector<int32_t> localCount(static_cast<size_t>(expertCount), 0);
    std::vector<int32_t> dst(static_cast<size_t>(n));
    for (int64_t token = 0; token < s; ++token) {
        uint64_t seenLow = 0;
        uint64_t seenHigh = 0;
        for (int64_t topk = 0; topk < k; ++topk) {
            const int64_t route = token * k + topk;
            const int32_t expert = allTopk[static_cast<size_t>(rank * n + route)];
            const int32_t occurrence = localCount[static_cast<size_t>(expert)]++;
            const int32_t sourcePrefix = rank == 0 ? 0 :
                tpePrefix[static_cast<size_t>((rank - 1) * expertCount + expert)];
            const int32_t globalRank = sourcePrefix + occurrence;
            int32_t dest = 0;
            while (dest < rankSize &&
                allocPrefix[static_cast<size_t>(AllocIndex(expert, dest, rankSize))] <= globalRank) {
                ++dest;
            }
            if (dest >= rankSize) {
                *error = "destination not found";
                return false;
            }
            const int32_t previous = dest == 0 ? 0 :
                allocPrefix[static_cast<size_t>(AllocIndex(expert, dest - 1, rankSize))];
            const int32_t raw = static_cast<int32_t>(dest * nvs +
                expertOffsets[static_cast<size_t>(dest * expertCount + expert)] + globalRank - previous);
            bool duplicate = false;
            if (dest < 64) {
                const uint64_t bit = static_cast<uint64_t>(1) << dest;
                duplicate = (seenLow & bit) != 0;
                seenLow |= bit;
            } else {
                const uint64_t bit = static_cast<uint64_t>(1) << (dest - 64);
                duplicate = (seenHigh & bit) != 0;
                seenHigh |= bit;
            }
            dst[static_cast<size_t>(route)] = duplicate ? -raw - 1 : raw;
        }
    }

    plan->dispatchedCapacity = nvs;
    plan->dst = std::move(dst);
    plan->expertsToCopy = std::move(expertsToCopy);
    plan->cuSeqlens.assign(
        cuAll.begin() + rank * (expertCount + b), cuAll.begin() + (rank + 1) * (expertCount + b));
    plan->remoteStats.assign(
        remoteStatsAll.begin() + rank * 2, remoteStatsAll.begin() + (rank + 1) * 2);
    return true;
}

} // namespace TileXRMoonEpTest
