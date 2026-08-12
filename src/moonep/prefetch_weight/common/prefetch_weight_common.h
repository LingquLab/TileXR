#ifndef TILEXR_MOONEP_PREFETCH_WEIGHT_COMMON_H
#define TILEXR_MOONEP_PREFETCH_WEIGHT_COMMON_H

#include <cstdint>

namespace TileXRMoonEp {
namespace Kernel {

constexpr uint32_t kPrefetchWeightStatusSuccess = 4000;
constexpr uint32_t kPrefetchWeightStatusInvalidRuntime = 4001;
constexpr uint32_t kPrefetchWeightStatusInvalidExpert = 4002;
constexpr uint32_t kPrefetchWeightStatusLocalExpert = 4003;
constexpr uint32_t kPrefetchWeightStatusCqErrorBase = 4100;
constexpr uint32_t kPrefetchWeightStatusSubmitErrorBase = 4200;

constexpr uint32_t kPrefetchWeightUdmaDebugMagic = 0x50524644U;

struct PrefetchWeightUdmaDebugRecord {
    uint32_t magic;
    uint32_t worker;
    uint32_t queue;
    int32_t peer;
    uint64_t queueId;
    uint32_t initialTarget;
    uint32_t targetAfterPost[3];
    uint32_t postCount;
    uint32_t quietTarget;
    uint32_t cqTailAtQuiet;
    uint32_t completionCount;
    uint32_t sqHeadAtQuiet;
    uint32_t sqTailAtQuiet;
    uint32_t pollStatus;
    uint32_t cqTailAfter;
    uint32_t sqTailAfter;
    uint32_t wqeCountAfter;
};

} // namespace Kernel
} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PREFETCH_WEIGHT_COMMON_H
