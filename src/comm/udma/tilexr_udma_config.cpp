/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_config.h"

#include <cctype>
#include <cstdlib>
#include <limits>

namespace TileXR {
namespace {

void SetError(std::string* error, const std::string& message)
{
    if (error != nullptr) {
        *error = message;
    }
}

std::string Trim(const std::string& value)
{
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

bool ParsePositiveUint32(const std::string& text, uint32_t& value)
{
    if (text.empty()) {
        return false;
    }
    uint32_t parsed = 0;
    for (char ch : text) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        const uint32_t digit = static_cast<uint32_t>(ch - '0');
        if (parsed > (std::numeric_limits<uint32_t>::max() - digit) / 10U) {
            return false;
        }
        parsed = parsed * 10U + digit;
    }
    if (parsed == 0) {
        return false;
    }
    value = parsed;
    return true;
}

bool IsKnownSelector(uint32_t selector)
{
    return selector == static_cast<uint32_t>(UDMAQpRouteSelector::TOPOLOGY) ||
        selector == static_cast<uint32_t>(UDMAQpRouteSelector::PORT_COUNT);
}

bool RuleValid(const UDMAQpRouteWireRule& rule)
{
    if (!IsKnownSelector(rule.selectorKind)) {
        return false;
    }
    if (rule.selectorKind == static_cast<uint32_t>(UDMAQpRouteSelector::TOPOLOGY)) {
        return rule.selectorValue == 0;
    }
    return rule.selectorValue != 0;
}

bool RuleEmpty(const UDMAQpRouteWireRule& rule)
{
    return rule.selectorKind == 0 && rule.selectorValue == 0;
}

} // namespace

UDMAQpConfigParseStatus ParseUDMAQpRouteSpec(
    const char* routeSpec, UDMAQpConfig& config, std::string* error)
{
    config = {};
    if (error != nullptr) {
        error->clear();
    }

    const std::string normalized = Trim(routeSpec == nullptr ? std::string() : std::string(routeSpec));
    if (normalized.empty()) {
        return UDMAQpConfigParseStatus::SUCCESS;
    }

    config.explicitConfig = true;
    size_t begin = 0;
    while (begin <= normalized.size()) {
        const size_t comma = normalized.find(',', begin);
        const size_t end = comma == std::string::npos ? normalized.size() : comma;
        const std::string ruleText = Trim(normalized.substr(begin, end - begin));
        if (ruleText.empty()) {
            SetError(error, "UDMA QP route specification contains an empty rule");
            config = {};
            return UDMAQpConfigParseStatus::INVALID;
        }
        if (config.routes.size() == TILEXR_UDMA_MAX_QP_COUNT) {
            SetError(error, "UDMA QP route specification requests more than eight QPs");
            config = {};
            return UDMAQpConfigParseStatus::INVALID;
        }

        UDMAQpRouteRule rule {};
        if (ruleText == "topology") {
            rule.selector = UDMAQpRouteSelector::TOPOLOGY;
        } else {
            constexpr char prefix[] = "port_count:";
            if (ruleText.compare(0, sizeof(prefix) - 1, prefix) != 0) {
                SetError(error, "unknown UDMA QP route selector: " + ruleText);
                config = {};
                return UDMAQpConfigParseStatus::INVALID;
            }
            uint32_t portCount = 0;
            if (!ParsePositiveUint32(ruleText.substr(sizeof(prefix) - 1), portCount)) {
                SetError(error, "UDMA QP port_count must be a positive decimal uint32");
                config = {};
                return UDMAQpConfigParseStatus::INVALID;
            }
            rule.selector = UDMAQpRouteSelector::PORT_COUNT;
            rule.value = portCount;
        }
        config.routes.push_back(rule);

        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return UDMAQpConfigParseStatus::SUCCESS;
}

UDMAQpConfigParseStatus LoadUDMAQpConfigFromEnv(UDMAQpConfig& config, std::string* error)
{
    return ParseUDMAQpRouteSpec(std::getenv(TILEXR_UDMA_QP_ROUTE_SPEC_ENV), config, error);
}

uint32_t UDMAQpConfigQpCount(const UDMAQpConfig& config)
{
    if (!config.explicitConfig) {
        return 1;
    }
    if (config.routes.empty() || config.routes.size() > TILEXR_UDMA_MAX_QP_COUNT) {
        return 0;
    }
    return static_cast<uint32_t>(config.routes.size());
}

UDMAQpConfigWireDescriptor BuildUDMAQpConfigWireDescriptor(
    const UDMAQpConfig& config, UDMAQpConfigParseStatus parseStatus)
{
    UDMAQpConfigWireDescriptor descriptor {};
    descriptor.version = TILEXR_UDMA_QP_CONFIG_WIRE_VERSION;
    descriptor.parseStatus = static_cast<uint32_t>(parseStatus);
    if (parseStatus != UDMAQpConfigParseStatus::SUCCESS) {
        descriptor.qpCount = 0;
        return descriptor;
    }

    descriptor.qpCount = UDMAQpConfigQpCount(config);
    if (!config.explicitConfig || descriptor.qpCount == 0) {
        return descriptor;
    }
    for (uint32_t i = 0; i < descriptor.qpCount; ++i) {
        descriptor.routeRules[i].selectorKind = static_cast<uint32_t>(config.routes[i].selector);
        descriptor.routeRules[i].selectorValue = config.routes[i].value;
    }
    return descriptor;
}

bool ValidateUDMAQpConfigWireDescriptor(
    const UDMAQpConfigWireDescriptor& descriptor, std::string* error)
{
    if (error != nullptr) {
        error->clear();
    }
    if (descriptor.version != TILEXR_UDMA_QP_CONFIG_WIRE_VERSION) {
        SetError(error, "unsupported UDMA QP configuration wire version");
        return false;
    }

    const uint32_t success = static_cast<uint32_t>(UDMAQpConfigParseStatus::SUCCESS);
    const uint32_t invalid = static_cast<uint32_t>(UDMAQpConfigParseStatus::INVALID);
    if (descriptor.parseStatus != success && descriptor.parseStatus != invalid) {
        SetError(error, "invalid UDMA QP configuration parse status");
        return false;
    }
    if (descriptor.parseStatus == invalid) {
        if (descriptor.qpCount != 0) {
            SetError(error, "failed UDMA QP configuration must report zero QPs");
            return false;
        }
        for (const auto& rule : descriptor.routeRules) {
            if (!RuleEmpty(rule)) {
                SetError(error, "failed UDMA QP configuration contains route data");
                return false;
            }
        }
        return true;
    }

    if (descriptor.qpCount == 0 || descriptor.qpCount > TILEXR_UDMA_MAX_QP_COUNT) {
        SetError(error, "UDMA QP configuration wire count is out of range");
        return false;
    }
    const bool legacy = RuleEmpty(descriptor.routeRules[0]);
    if (legacy && descriptor.qpCount != 1) {
        SetError(error, "legacy UDMA QP configuration must use one QP");
        return false;
    }
    for (uint32_t i = 0; i < TILEXR_UDMA_MAX_QP_COUNT; ++i) {
        if (i < descriptor.qpCount) {
            if (legacy ? !RuleEmpty(descriptor.routeRules[i]) : !RuleValid(descriptor.routeRules[i])) {
                SetError(error, "invalid active UDMA QP route rule");
                return false;
            }
        } else if (!RuleEmpty(descriptor.routeRules[i])) {
            SetError(error, "inactive UDMA QP route rule is not zero");
            return false;
        }
    }
    return true;
}

bool UDMAQpConfigWireDescriptorsEqual(
    const UDMAQpConfigWireDescriptor& lhs, const UDMAQpConfigWireDescriptor& rhs)
{
    if (lhs.version != rhs.version || lhs.parseStatus != rhs.parseStatus || lhs.qpCount != rhs.qpCount) {
        return false;
    }
    for (uint32_t i = 0; i < TILEXR_UDMA_MAX_QP_COUNT; ++i) {
        if (lhs.routeRules[i].selectorKind != rhs.routeRules[i].selectorKind ||
            lhs.routeRules[i].selectorValue != rhs.routeRules[i].selectorValue) {
            return false;
        }
    }
    return true;
}

bool UDMAQpConfigFromWireDescriptor(
    const UDMAQpConfigWireDescriptor& descriptor, UDMAQpConfig& config, std::string* error)
{
    config = {};
    if (!ValidateUDMAQpConfigWireDescriptor(descriptor, error) ||
        descriptor.parseStatus != static_cast<uint32_t>(UDMAQpConfigParseStatus::SUCCESS)) {
        return false;
    }
    if (descriptor.routeRules[0].selectorKind == 0) {
        return true;
    }
    config.explicitConfig = true;
    for (uint32_t i = 0; i < descriptor.qpCount; ++i) {
        UDMAQpRouteRule rule {};
        rule.selector = static_cast<UDMAQpRouteSelector>(descriptor.routeRules[i].selectorKind);
        rule.value = descriptor.routeRules[i].selectorValue;
        config.routes.push_back(rule);
    }
    return true;
}

} // namespace TileXR
