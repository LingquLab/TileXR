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
constexpr uint32_t kPrefetchWeightStatusCqInvalidUnsupported = 4301;
constexpr uint32_t kPrefetchWeightStatusCqInvalidQueue = 4302;
constexpr uint32_t kPrefetchWeightStatusCqInvalidInfo = 4303;
constexpr uint32_t kPrefetchWeightStatusCqInvalidContext = 4304;
constexpr uint32_t kPrefetchWeightStatusCqInvalidCompletionDepth = 4305;
constexpr uint32_t kPrefetchWeightStatusCqInvalidSqOutstanding = 4306;
constexpr uint32_t kPrefetchWeightStatusCqInvalidCompletedBb = 4307;
constexpr uint32_t kPrefetchWeightStatusCqInvalidUnknown = 4308;

} // namespace Kernel
} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PREFETCH_WEIGHT_COMMON_H
