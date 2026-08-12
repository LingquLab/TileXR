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

} // namespace Kernel
} // namespace TileXRMoonEp

#endif // TILEXR_MOONEP_PREFETCH_WEIGHT_COMMON_H
