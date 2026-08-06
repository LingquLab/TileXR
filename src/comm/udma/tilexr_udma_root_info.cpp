/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#include "udma/tilexr_udma_root_info.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <utility>

namespace TileXR {
namespace {

enum class JsonType {
    NIL,
    BOOL,
    UINT,
    STRING,
    ARRAY,
    OBJECT,
};

struct JsonValue {
    JsonType type = JsonType::NIL;
    bool boolValue = false;
    uint64_t uintValue = 0;
    std::string stringValue;
    std::vector<JsonValue> arrayValue;
    std::map<std::string, JsonValue> objectValue;
};

void SetError(std::string* error, const std::string& message)
{
    if (error != nullptr) {
        *error = message;
    }
}

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    bool Parse(JsonValue& value, std::string* error)
    {
        SkipWhitespace();
        if (!ParseValue(value)) {
            SetError(error, error_ + " at byte " + std::to_string(pos_));
            return false;
        }
        SkipWhitespace();
        if (pos_ != text_.size()) {
            SetError(error, "unexpected trailing JSON data at byte " + std::to_string(pos_));
            return false;
        }
        return true;
    }

private:
    bool ParseValue(JsonValue& value)
    {
        SkipWhitespace();
        if (pos_ >= text_.size()) {
            return Fail("unexpected end of JSON input");
        }
        switch (text_[pos_]) {
            case '{': return ParseObject(value);
            case '[': return ParseArray(value);
            case '"':
                value.type = JsonType::STRING;
                return ParseString(value.stringValue);
            case 't':
                value.type = JsonType::BOOL;
                value.boolValue = true;
                return ConsumeLiteral("true");
            case 'f':
                value.type = JsonType::BOOL;
                value.boolValue = false;
                return ConsumeLiteral("false");
            case 'n':
                value.type = JsonType::NIL;
                return ConsumeLiteral("null");
            default:
                if (text_[pos_] >= '0' && text_[pos_] <= '9') {
                    value.type = JsonType::UINT;
                    return ParseUint(value.uintValue);
                }
                return Fail("unsupported JSON value");
        }
    }

    bool ParseObject(JsonValue& value)
    {
        value = {};
        value.type = JsonType::OBJECT;
        ++pos_;
        SkipWhitespace();
        if (Consume('}')) {
            return true;
        }
        while (true) {
            std::string key;
            if (!ParseString(key)) {
                return false;
            }
            SkipWhitespace();
            if (!Consume(':')) {
                return Fail("expected ':' after JSON object key");
            }
            JsonValue member;
            if (!ParseValue(member)) {
                return false;
            }
            if (!value.objectValue.emplace(key, std::move(member)).second) {
                return Fail("duplicate JSON object key");
            }
            SkipWhitespace();
            if (Consume('}')) {
                return true;
            }
            if (!Consume(',')) {
                return Fail("expected ',' or '}' in JSON object");
            }
            SkipWhitespace();
        }
    }

    bool ParseArray(JsonValue& value)
    {
        value = {};
        value.type = JsonType::ARRAY;
        ++pos_;
        SkipWhitespace();
        if (Consume(']')) {
            return true;
        }
        while (true) {
            JsonValue element;
            if (!ParseValue(element)) {
                return false;
            }
            value.arrayValue.push_back(std::move(element));
            SkipWhitespace();
            if (Consume(']')) {
                return true;
            }
            if (!Consume(',')) {
                return Fail("expected ',' or ']' in JSON array");
            }
            SkipWhitespace();
        }
    }

    bool ParseString(std::string& value)
    {
        value.clear();
        if (!Consume('"')) {
            return Fail("expected JSON string");
        }
        while (pos_ < text_.size()) {
            const unsigned char ch = static_cast<unsigned char>(text_[pos_++]);
            if (ch == '"') {
                return true;
            }
            if (ch < 0x20) {
                return Fail("unescaped control character in JSON string");
            }
            if (ch != '\\') {
                value.push_back(static_cast<char>(ch));
                continue;
            }
            if (pos_ >= text_.size()) {
                return Fail("truncated JSON string escape");
            }
            const char escaped = text_[pos_++];
            switch (escaped) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'u': {
                    uint32_t codePoint = 0;
                    if (!ParseHexQuad(codePoint)) {
                        return false;
                    }
                    if (codePoint >= 0xD800 && codePoint <= 0xDBFF) {
                        if (pos_ + 2 > text_.size() || text_[pos_] != '\\' || text_[pos_ + 1] != 'u') {
                            return Fail("missing low surrogate in JSON string");
                        }
                        pos_ += 2;
                        uint32_t low = 0;
                        if (!ParseHexQuad(low) || low < 0xDC00 || low > 0xDFFF) {
                            return Fail("invalid low surrogate in JSON string");
                        }
                        codePoint = 0x10000 + ((codePoint - 0xD800) << 10) + (low - 0xDC00);
                    } else if (codePoint >= 0xDC00 && codePoint <= 0xDFFF) {
                        return Fail("unexpected low surrogate in JSON string");
                    }
                    AppendUtf8(codePoint, value);
                    break;
                }
                default: return Fail("unsupported JSON string escape");
            }
        }
        return Fail("unterminated JSON string");
    }

    bool ParseHexQuad(uint32_t& value)
    {
        if (pos_ + 4 > text_.size()) {
            return Fail("truncated JSON unicode escape");
        }
        value = 0;
        for (size_t i = 0; i < 4; ++i) {
            const char ch = text_[pos_++];
            uint32_t digit = 0;
            if (ch >= '0' && ch <= '9') {
                digit = static_cast<uint32_t>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                digit = static_cast<uint32_t>(ch - 'a' + 10);
            } else if (ch >= 'A' && ch <= 'F') {
                digit = static_cast<uint32_t>(ch - 'A' + 10);
            } else {
                return Fail("invalid JSON unicode escape");
            }
            value = (value << 4) | digit;
        }
        return true;
    }

    static void AppendUtf8(uint32_t codePoint, std::string& value)
    {
        if (codePoint <= 0x7F) {
            value.push_back(static_cast<char>(codePoint));
        } else if (codePoint <= 0x7FF) {
            value.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
            value.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        } else if (codePoint <= 0xFFFF) {
            value.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
            value.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
            value.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        } else {
            value.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
            value.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
            value.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
            value.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        }
    }

    bool ParseUint(uint64_t& value)
    {
        const size_t begin = pos_;
        if (text_[pos_] == '0') {
            ++pos_;
            if (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                return Fail("leading zero in JSON number");
            }
            value = 0;
            return true;
        }
        value = 0;
        while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
            const uint64_t digit = static_cast<uint64_t>(text_[pos_] - '0');
            if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10U) {
                return Fail("JSON unsigned integer overflow");
            }
            value = value * 10U + digit;
            ++pos_;
        }
        return pos_ > begin;
    }

    bool ConsumeLiteral(const char* literal)
    {
        const size_t length = std::strlen(literal);
        if (text_.compare(pos_, length, literal) != 0) {
            return Fail("invalid JSON literal");
        }
        pos_ += length;
        return true;
    }

    bool Consume(char expected)
    {
        if (pos_ < text_.size() && text_[pos_] == expected) {
            ++pos_;
            return true;
        }
        return false;
    }

    void SkipWhitespace()
    {
        while (pos_ < text_.size()) {
            const char ch = text_[pos_];
            if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
                break;
            }
            ++pos_;
        }
    }

    bool Fail(const std::string& message)
    {
        if (error_.empty()) {
            error_ = message;
        }
        return false;
    }

    const std::string& text_;
    size_t pos_ = 0;
    std::string error_;
};

const JsonValue* FindMember(const JsonValue& object, const std::string& key)
{
    if (object.type != JsonType::OBJECT) {
        return nullptr;
    }
    const auto it = object.objectValue.find(key);
    return it == object.objectValue.end() ? nullptr : &it->second;
}

const JsonValue* FindFirstMemberRecursive(const JsonValue& value, const std::string& key)
{
    if (value.type == JsonType::OBJECT) {
        const JsonValue* direct = FindMember(value, key);
        if (direct != nullptr) {
            return direct;
        }
        for (const auto& member : value.objectValue) {
            const JsonValue* nested = FindFirstMemberRecursive(member.second, key);
            if (nested != nullptr) {
                return nested;
            }
        }
    } else if (value.type == JsonType::ARRAY) {
        for (const auto& element : value.arrayValue) {
            const JsonValue* nested = FindFirstMemberRecursive(element, key);
            if (nested != nullptr) {
                return nested;
            }
        }
    }
    return nullptr;
}

void CollectObjectsWithMember(
    const JsonValue& value, const std::string& key, std::vector<const JsonValue*>& objects)
{
    if (value.type == JsonType::OBJECT) {
        if (FindMember(value, key) != nullptr) {
            objects.push_back(&value);
        }
        for (const auto& member : value.objectValue) {
            CollectObjectsWithMember(member.second, key, objects);
        }
    } else if (value.type == JsonType::ARRAY) {
        for (const auto& element : value.arrayValue) {
            CollectObjectsWithMember(element, key, objects);
        }
    }
}

bool JsonUint32(const JsonValue* value, uint32_t& result)
{
    if (value == nullptr) {
        return false;
    }
    uint64_t parsed = 0;
    if (value->type == JsonType::UINT) {
        parsed = value->uintValue;
    } else if (value->type == JsonType::STRING) {
        if (value->stringValue.empty()) {
            return false;
        }
        for (char ch : value->stringValue) {
            if (ch < '0' || ch > '9') {
                return false;
            }
            const uint64_t digit = static_cast<uint64_t>(ch - '0');
            if (parsed > (std::numeric_limits<uint32_t>::max() - digit) / 10U) {
                return false;
            }
            parsed = parsed * 10U + digit;
        }
    } else {
        return false;
    }
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    result = static_cast<uint32_t>(parsed);
    return true;
}

bool JsonStringArray(const JsonValue* value, std::vector<std::string>& result)
{
    result.clear();
    if (value == nullptr || value->type != JsonType::ARRAY) {
        return false;
    }
    for (const auto& element : value->arrayValue) {
        if (element.type != JsonType::STRING) {
            return false;
        }
        result.push_back(element.stringValue);
    }
    return true;
}

std::string ReadTextFile(const std::string& path)
{
    std::ifstream input(path.c_str(), std::ios::binary);
    if (!input.is_open()) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

bool ParseJsonDocument(const std::string& json, JsonValue& document, std::string* error)
{
    if (error != nullptr) {
        error->clear();
    }
    if (json.empty()) {
        SetError(error, "JSON input is empty");
        return false;
    }
    JsonParser parser(json);
    if (!parser.Parse(document, error)) {
        return false;
    }
    if (document.type != JsonType::OBJECT) {
        SetError(error, "JSON document root must be an object");
        return false;
    }
    return true;
}

} // namespace

bool ParseUDMAEidHex(const std::string& text, HccpEid& eid)
{
    if (text.size() != sizeof(eid.raw) * 2) {
        return false;
    }
    HccpEid parsed {};
    for (size_t i = 0; i < sizeof(parsed.raw); ++i) {
        uint32_t value = 0;
        for (size_t half = 0; half < 2; ++half) {
            const char ch = text[i * 2 + half];
            uint32_t digit = 0;
            if (ch >= '0' && ch <= '9') {
                digit = static_cast<uint32_t>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                digit = static_cast<uint32_t>(ch - 'a' + 10);
            } else if (ch >= 'A' && ch <= 'F') {
                digit = static_cast<uint32_t>(ch - 'A' + 10);
            } else {
                return false;
            }
            value = value * 16U + digit;
        }
        parsed.raw[i] = static_cast<uint8_t>(value);
    }
    eid = parsed;
    return true;
}

bool ParseUDMARootInfoJson(const std::string& json, UDMARootInfo& rootInfo, std::string* error)
{
    rootInfo = {};
    JsonValue document;
    if (!ParseJsonDocument(json, document, error)) {
        return false;
    }

    const JsonValue* topoPath = FindFirstMemberRecursive(document, "topo_file_path");
    if (topoPath == nullptr || topoPath->type != JsonType::STRING || topoPath->stringValue.empty()) {
        SetError(error, "RootInfo requires a non-empty topo_file_path string");
        return false;
    }
    rootInfo.topoPath = topoPath->stringValue;

    std::vector<const JsonValue*> rankObjects;
    CollectObjectsWithMember(document, "device_id", rankObjects);
    if (rankObjects.empty()) {
        SetError(error, "RootInfo contains no device_id entries");
        return false;
    }

    uint32_t firstEidCount = 0;
    uint32_t minimumDeviceId = std::numeric_limits<uint32_t>::max();
    for (const JsonValue* rankObject : rankObjects) {
        uint32_t deviceId = 0;
        uint32_t localId = 0;
        if (!JsonUint32(FindMember(*rankObject, "device_id"), deviceId) ||
            !JsonUint32(FindMember(*rankObject, "local_id"), localId)) {
            SetError(error, "RootInfo device entry requires uint32 device_id and local_id");
            rootInfo = {};
            return false;
        }
        if (!rootInfo.deviceToLocalId.emplace(deviceId, localId).second ||
            rootInfo.eidByLocalId.count(localId) != 0) {
            SetError(error, "RootInfo contains duplicate device_id or local_id");
            rootInfo = {};
            return false;
        }
        minimumDeviceId = std::min(minimumDeviceId, deviceId);

        std::vector<const JsonValue*> addressObjects;
        CollectObjectsWithMember(*rankObject, "addr", addressObjects);
        uint32_t eidIndex = 0;
        for (const JsonValue* addressObject : addressObjects) {
            const JsonValue* address = FindMember(*addressObject, "addr");
            const JsonValue* portsValue = FindMember(*addressObject, "ports");
            std::vector<std::string> ports;
            HccpEid eid {};
            if (address == nullptr || address->type != JsonType::STRING ||
                !ParseUDMAEidHex(address->stringValue, eid) || !JsonStringArray(portsValue, ports)) {
                SetError(error, "RootInfo EID entry requires a 32-digit hex addr and string ports array");
                rootInfo = {};
                return false;
            }
            rootInfo.eidByLocalId[localId][eidIndex] = eid;
            rootInfo.portCountByEidByLocalId[localId][eidIndex] = static_cast<uint32_t>(ports.size());
            for (const std::string& port : ports) {
                if (port.empty() || !rootInfo.portToEidByLocalId[localId].emplace(port, eidIndex).second) {
                    SetError(error, "RootInfo contains an empty or duplicate local port");
                    rootInfo = {};
                    return false;
                }
            }
            ++eidIndex;
        }
        if (eidIndex == 0) {
            SetError(error, "RootInfo device entry contains no EIDs");
            rootInfo = {};
            return false;
        }
        if (firstEidCount == 0) {
            firstEidCount = eidIndex;
        } else if (eidIndex != firstEidCount) {
            SetError(error, "RootInfo devices have inconsistent EID counts");
            rootInfo = {};
            return false;
        }
    }

    rootInfo.deviceIdOffset = minimumDeviceId;
    rootInfo.eidCount = firstEidCount;
    return true;
}

bool LoadUDMARootInfoFromPath(const std::string& path, UDMARootInfo& rootInfo, std::string* error)
{
    if (path.empty()) {
        rootInfo = {};
        SetError(error, "RootInfo path is empty");
        return false;
    }
    const std::string json = ReadTextFile(path);
    if (json.empty()) {
        rootInfo = {};
        SetError(error, "failed to read RootInfo file: " + path);
        return false;
    }
    return ParseUDMARootInfoJson(json, rootInfo, error);
}

bool LoadUDMARootInfo(UDMARootInfo& rootInfo, std::string* error)
{
    const char* overridePath = std::getenv(TILEXR_UDMA_ROOTINFO_PATH_ENV);
    const std::string path = overridePath == nullptr || overridePath[0] == '\0'
        ? TILEXR_UDMA_DEFAULT_ROOTINFO_PATH : overridePath;
    return LoadUDMARootInfoFromPath(path, rootInfo, error);
}

bool ParseUDMATopologyJson(
    const std::string& json, std::vector<UDMATopologyEdge>& edges, std::string* error)
{
    edges.clear();
    JsonValue document;
    if (!ParseJsonDocument(json, document, error)) {
        return false;
    }

    std::vector<const JsonValue*> edgeObjects;
    CollectObjectsWithMember(document, "local_a", edgeObjects);
    for (const JsonValue* edgeObject : edgeObjects) {
        const JsonValue* localB = FindMember(*edgeObject, "local_b");
        const JsonValue* localBPorts = FindMember(*edgeObject, "local_b_ports");
        if (localB == nullptr && localBPorts == nullptr) {
            continue;
        }
        UDMATopologyEdge edge {};
        if (!JsonUint32(FindMember(*edgeObject, "local_a"), edge.localA) ||
            !JsonUint32(localB, edge.localB) ||
            !JsonStringArray(FindMember(*edgeObject, "local_a_ports"), edge.localAPorts) ||
            !JsonStringArray(localBPorts, edge.localBPorts) ||
            edge.localAPorts.empty() || edge.localBPorts.empty()) {
            edges.clear();
            SetError(error, "topology edge requires local IDs and non-empty port arrays");
            return false;
        }
        edges.push_back(std::move(edge));
    }
    if (edges.empty()) {
        SetError(error, "topology contains no edges");
        return false;
    }
    return true;
}

bool LoadUDMATopologyFromPath(
    const std::string& path, std::vector<UDMATopologyEdge>& edges, std::string* error)
{
    const std::string json = ReadTextFile(path);
    if (json.empty()) {
        edges.clear();
        SetError(error, "failed to read topology file: " + path);
        return false;
    }
    return ParseUDMATopologyJson(json, edges, error);
}

bool ResolveUDMALocalId(const UDMARootInfo& rootInfo, uint32_t deviceId, uint32_t& localId)
{
    if (deviceId <= std::numeric_limits<uint32_t>::max() - rootInfo.deviceIdOffset) {
        const auto offsetIt = rootInfo.deviceToLocalId.find(deviceId + rootInfo.deviceIdOffset);
        if (offsetIt != rootInfo.deviceToLocalId.end()) {
            localId = offsetIt->second;
            return true;
        }
    }
    const auto directIt = rootInfo.deviceToLocalId.find(deviceId);
    if (directIt == rootInfo.deviceToLocalId.end()) {
        return false;
    }
    localId = directIt->second;
    return true;
}

bool ResolveUDMATopologyEid(
    const UDMARootInfo& rootInfo, const std::vector<UDMATopologyEdge>& edges,
    uint32_t localId, uint32_t peerLocalId, uint32_t& eidIndex)
{
    std::string localPort;
    for (const auto& edge : edges) {
        if (edge.localA == localId && edge.localB == peerLocalId) {
            localPort = edge.localAPorts[0];
            break;
        }
        if (edge.localB == localId && edge.localA == peerLocalId) {
            localPort = edge.localBPorts[0];
            break;
        }
    }
    const auto localIt = rootInfo.portToEidByLocalId.find(localId);
    if (localPort.empty() || localIt == rootInfo.portToEidByLocalId.end()) {
        return false;
    }
    const auto portIt = localIt->second.find(localPort);
    if (portIt == localIt->second.end()) {
        return false;
    }
    const auto eidIt = rootInfo.eidByLocalId.find(localId);
    if (eidIt == rootInfo.eidByLocalId.end() || eidIt->second.count(portIt->second) == 0) {
        return false;
    }
    eidIndex = portIt->second;
    return true;
}

bool ResolveUDMAPortCountEid(
    const UDMARootInfo& rootInfo, uint32_t localId, uint32_t portCount, uint32_t& eidIndex)
{
    if (portCount == 0) {
        return false;
    }
    const auto localIt = rootInfo.portCountByEidByLocalId.find(localId);
    const auto eidIt = rootInfo.eidByLocalId.find(localId);
    if (localIt == rootInfo.portCountByEidByLocalId.end() || eidIt == rootInfo.eidByLocalId.end()) {
        return false;
    }
    for (const auto& entry : localIt->second) {
        if (entry.second == portCount && eidIt->second.count(entry.first) != 0) {
            eidIndex = entry.first;
            return true;
        }
    }
    return false;
}

bool ResolveUDMAAggregateEid(
    const UDMARootInfo& rootInfo, uint32_t localId, uint32_t& eidIndex)
{
    const auto localIt = rootInfo.portCountByEidByLocalId.find(localId);
    const auto eidIt = rootInfo.eidByLocalId.find(localId);
    if (localIt == rootInfo.portCountByEidByLocalId.end() ||
        eidIt == rootInfo.eidByLocalId.end()) {
        return false;
    }
    for (const auto& entry : localIt->second) {
        if (entry.second > 1U && eidIt->second.count(entry.first) != 0U) {
            eidIndex = entry.first;
            return true;
        }
    }
    return false;
}

} // namespace TileXR
