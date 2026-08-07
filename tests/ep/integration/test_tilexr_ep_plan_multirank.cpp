#include <acl/acl.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "comm_args.h"
#include "ep_plan_downstream.h"
#include "ep_plan_layout.h"
#include "ep_plan_peer_mailbox.h"
#include "ep_plan_reference.h"
#include "ep_plan_types.h"
#include "tilexr_api.h"
#include "tilexr_ep_plan.h"
#include "tilexr_types.h"

namespace {
constexpr int64_t kS = 8;
constexpr int64_t kTopK = 2;
constexpr int64_t kRoutes = kS * kTopK;
constexpr int64_t kPrefetchSlots = 2;
constexpr int64_t kNvS = 96;
constexpr uint64_t kEpochBase = 0x504C414E0000ULL;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
constexpr size_t kMetaWorkspaceAlignment = TileXREp::Plan::kPlanWorkspaceAlignment;
constexpr int32_t kSentinel = 0x5A5A5A5A;
constexpr int kControlTimeoutSeconds = 240;
constexpr size_t kPeerVisibilityDebugOffset = 5000ULL * 4ULL * sizeof(int64_t);
constexpr int32_t kPlannerReadyEventId = 4096;
constexpr int32_t kPlannerBarrierPhaseData = 1;
constexpr size_t kSyncUnitBytes = 4 * sizeof(int64_t);
constexpr size_t kPlannerBarrierBaseBytes =
    static_cast<size_t>(kPlannerReadyEventId) * kSyncUnitBytes;
constexpr size_t kPlannerBarrierStrideBytes = 512;
constexpr size_t kBarrierDumpWords = 4;

enum ControlStage : int32_t {
    COMM_READY = 1,
    MEMORY_READY = 2,
    LEGACY_PLAN_DONE = 3,
    METADATA_PLAN_DONE = 4,
    VALIDATION_DONE = 5,
};
struct HostPort { std::string host; int port = 0; };
struct ControlRecord {
    int32_t stage = 0;
    int32_t rank = -1;
    int32_t rankSize = 0;
    int32_t success = 0;
    int32_t status = PLAN_ERROR_INTERNAL_INVARIANT;
    int32_t actualDstCount = 0;
    uint64_t requestedEpoch = 0;
    uint64_t committedEpoch = 0;
    uint64_t canonicalDigest = 0;
    uint64_t localOutputDigest = 0;
    uint64_t globalDigest = 0;
    int32_t actualDst[kRoutes] = {};
};
struct ControlAck {
    int32_t success = 0;
    int32_t reserved = 0;
    uint64_t globalDigest = 0;
};

bool ValidateGlobalTokenRemap(const std::vector<ControlRecord> &records,
    int rankSize, uint64_t *globalDigest);

bool CheckAcl(const std::string &op, aclError result)
{
    if (result == ACL_SUCCESS) return true;
    std::cerr << op << " failed with ACL error " << result;
    const char *recent = aclGetRecentErrMsg();
    if (recent != nullptr && recent[0] != '\0') std::cerr << ": " << recent;
    std::cerr << std::endl;
    return false;
}

bool CheckTileXR(const std::string &op, int result)
{
    if (result == TileXR::TILEXR_SUCCESS) return true;
    std::cerr << op << " failed with TileXR error " << result << std::endl;
    return false;
}

int32_t PlannerBarrierEventId(int32_t phase, int32_t sourceRank, int32_t rankSize)
{
    return kPlannerReadyEventId + (phase - 1) * rankSize + sourceRank;
}

void DumpPlannerBarrierSlots(int rank, int rankSize, const TileXR::CommArgs *commArgs, int32_t phase)
{
    if (commArgs == nullptr) return;
    std::cerr << "rank " << rank << " host barrier slots phase=" << phase << " [";
    bool first = true;
    for (int targetRank = 0; targetRank < rankSize; ++targetRank) {
        for (int sourceRank = 0; sourceRank < rankSize; ++sourceRank) {
            const int32_t eventId = PlannerBarrierEventId(phase, sourceRank, rankSize);
            const size_t slotIndex = static_cast<size_t>((phase - 1) * rankSize + sourceRank);
            const size_t byteOffset = kPlannerBarrierBaseBytes +
                slotIndex * kPlannerBarrierStrideBytes;
            int64_t words[kBarrierDumpWords] = {};
            const aclError result = aclrtMemcpy(words, sizeof(words),
                commArgs->peerMems[targetRank] + byteOffset, sizeof(words),
                ACL_MEMCPY_DEVICE_TO_HOST);
            std::cerr << (first ? "" : ",") << "{target=" << targetRank
                      << ",source=" << sourceRank << ",event=" << eventId
                      << ",offset=0x" << std::hex << byteOffset
                      << ",rc=" << std::dec << result << ",words=[";
            for (size_t word = 0; word < kBarrierDumpWords; ++word) {
                std::cerr << (word == 0 ? "" : ",") << "0x" << std::hex
                          << static_cast<uint64_t>(words[word]);
            }
            std::cerr << std::dec << "]}";
            first = false;
        }
    }
    std::cerr << "]" << std::endl;
}

void DumpPlannerPeerStatuses(int rank, int rankSize, int64_t expertNum,
    const TileXR::CommArgs *commArgs)
{
    if (commArgs == nullptr) return;
    const TileXREp::Plan::PlanPeerMailboxLayout layout =
        TileXREp::Plan::BuildPlanPeerMailboxLayout(rankSize, expertNum);
    std::cerr << "rank " << rank << " host peer mailbox statuses rowBytes=0x"
              << std::hex << layout.rowBytes << std::dec << " [";
    bool first = true;
    for (int ownerRank = 0; ownerRank < rankSize; ++ownerRank) {
        for (int sourceRank = 0; sourceRank < rankSize; ++sourceRank) {
            const uint64_t statusOffset = TileXR::IPC_DATA_OFFSET +
                TileXREp::Plan::PlanPeerMailboxRowOffset(layout, sourceRank) + layout.status;
            int32_t words[TileXREp::Plan::kPlanStatusWords] = {};
            const aclError result = aclrtMemcpy(words, sizeof(words),
                commArgs->peerMems[ownerRank] + statusOffset, sizeof(words),
                ACL_MEMCPY_DEVICE_TO_HOST);
            std::cerr << (first ? "" : ",") << "{owner=" << ownerRank
                      << ",source=" << sourceRank << ",offset=0x" << std::hex
                      << statusOffset << std::dec << ",rc=" << result << ",words=[";
            for (int word = 0; word < TileXREp::Plan::kPlanStatusWords; ++word) {
                std::cerr << (word == 0 ? "" : ",") << words[word];
            }
            std::cerr << "]}";
            first = false;
        }
    }
    std::cerr << "]" << std::endl;
}

int GetEnvInt(const char *name, int fallback)
{
    const char *value = std::getenv(name);
    return value == nullptr || value[0] == '\0' ? fallback : std::atoi(value);
}

bool ParseHostPort(const std::string &value, HostPort *endpoint)
{
    if (endpoint == nullptr) return false;
    const size_t pos = value.rfind(':');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= value.size()) return false;
    const int port = std::atoi(value.substr(pos + 1).c_str());
    if (port <= 0 || port > 65535) return false;
    endpoint->host = value.substr(0, pos);
    endpoint->port = port;
    return true;
}

HostPort GetControlEndpoint()
{
    HostPort endpoint {"127.0.0.1", 19277};
    const char *control = std::getenv("TILEXR_PLAN_CONTROL_ADDR");
    if (control != nullptr && ParseHostPort(control, &endpoint)) return endpoint;
    const char *commId = std::getenv("TILEXR_COMM_ID");
    HostPort comm {};
    if (commId != nullptr && ParseHostPort(commId, &comm)) {
        endpoint.host = comm.host;
        endpoint.port = comm.port + 131;
        if (endpoint.port > 65535) endpoint.port = comm.port - 131;
    }
    return endpoint;
}

bool SetSocketTimeout(int fd)
{
    timeval timeout {};
    timeout.tv_sec = kControlTimeoutSeconds;
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0;
}

bool SendAll(int fd, const void *data, size_t bytes)
{
    const char *cursor = static_cast<const char *>(data);
    size_t sent = 0;
    while (sent < bytes) {
        const ssize_t result = send(fd, cursor + sent, bytes - sent, 0);
        if (result <= 0) return false;
        sent += static_cast<size_t>(result);
    }
    return true;
}

bool RecvAll(int fd, void *data, size_t bytes)
{
    char *cursor = static_cast<char *>(data);
    size_t received = 0;
    while (received < bytes) {
        const ssize_t result = recv(fd, cursor + received, bytes - received, MSG_WAITALL);
        if (result <= 0) return false;
        received += static_cast<size_t>(result);
    }
    return true;
}

int OpenControlServer(const HostPort &endpoint, int rankSize)
{
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int reuse = 1;
    (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    (void)SetSocketTimeout(fd);
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(static_cast<uint16_t>(endpoint.port));
    if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(fd, rankSize) != 0) {
        std::cerr << "control listen failed on " << endpoint.port << ": " << std::strerror(errno) << std::endl;
        close(fd);
        return -1;
    }
    return fd;
}

int ConnectControl(const HostPort &endpoint)
{
    sockaddr_in address {};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(endpoint.port));
    if (inet_pton(AF_INET, endpoint.host.c_str(), &address.sin_addr) != 1) return -1;
    for (int attempt = 0; attempt < kControlTimeoutSeconds * 10; ++attempt) {
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        (void)SetSocketTimeout(fd);
        if (connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0) return fd;
        close(fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return -1;
}

bool ControlExchange(int rank, int rankSize, ControlStage stage, const ControlRecord &local,
    std::vector<ControlRecord> *allRecords, uint64_t *globalDigest = nullptr)
{
    const HostPort endpoint = GetControlEndpoint();
    if (rank == 0) {
        const int listenFd = OpenControlServer(endpoint, rankSize);
        if (listenFd < 0) return false;
        std::vector<ControlRecord> records(static_cast<size_t>(rankSize));
        std::vector<int> clients;
        std::set<int32_t> seen;
        records[0] = local;
        seen.insert(0);
        bool ok = local.stage == stage && local.rank == 0 && local.rankSize == rankSize;
        for (int peer = 1; peer < rankSize; ++peer) {
            const int fd = accept(listenFd, nullptr, nullptr);
            if (fd < 0) { ok = false; break; }
            (void)SetSocketTimeout(fd);
            ControlRecord record {};
            if (!RecvAll(fd, &record, sizeof(record)) || record.stage != stage ||
                record.rank < 1 || record.rank >= rankSize || record.rankSize != rankSize ||
                !seen.insert(record.rank).second) {
                close(fd); ok = false; break;
            }
            records[static_cast<size_t>(record.rank)] = record;
            clients.push_back(fd);
        }
        if (seen.size() != static_cast<size_t>(rankSize)) ok = false;
        uint64_t gatheredDigest = 0;
        if (ok && stage == VALIDATION_DONE) {
            const uint64_t canonical = records[0].canonicalDigest;
            const uint64_t requested = records[0].requestedEpoch;
            const uint64_t committed = records[0].committedEpoch;
            for (const ControlRecord &record : records) {
                if (record.success != 1 || record.status != PLAN_OK || requested == 0 ||
                    requested != committed || record.requestedEpoch != requested ||
                    record.committedEpoch != committed || record.canonicalDigest != canonical) ok = false;
            }
            if (ok) ok = ValidateGlobalTokenRemap(records, rankSize, &gatheredDigest);
            for (ControlRecord &record : records) record.globalDigest = gatheredDigest;
        } else if (ok) {
            for (const ControlRecord &record : records) if (record.success != 1) ok = false;
        }
        ControlAck ack {};
        ack.success = ok ? 1 : 0;
        ack.globalDigest = gatheredDigest;
        for (int fd : clients) { (void)SendAll(fd, &ack, sizeof(ack)); close(fd); }
        close(listenFd);
        if (allRecords != nullptr) *allRecords = records;
        if (globalDigest != nullptr) *globalDigest = gatheredDigest;
        return ok;
    }
    const int fd = ConnectControl(endpoint);
    if (fd < 0) return false;
    ControlAck ack {};
    const bool ok = SendAll(fd, &local, sizeof(local)) && RecvAll(fd, &ack, sizeof(ack));
    close(fd);
    if (globalDigest != nullptr) *globalDigest = ack.globalDigest;
    return ok && ack.success == 1;
}

class DeviceAllocation {
public:
    ~DeviceAllocation() { if (pointer_ != nullptr) (void)aclrtFree(pointer_); }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    DeviceAllocation() = default;
    bool Allocate(size_t bytes, const std::string &name) {
        bytes_ = bytes;
        return bytes != 0 && CheckAcl("aclrtMalloc(" + name + ")",
            aclrtMalloc(&pointer_, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    bool Zero(const std::string &name) {
        return CheckAcl("aclrtMemset(" + name + ")", aclrtMemset(pointer_, bytes_, 0, bytes_));
    }
    void *data() const { return pointer_; }
    GM_ADDR gm() const { return reinterpret_cast<GM_ADDR>(pointer_); }
    size_t bytes() const { return bytes_; }
private:
    void *pointer_ = nullptr;
    size_t bytes_ = 0;
};

bool AllocateZero(DeviceAllocation *allocation, size_t bytes, const std::string &name)
{
    return allocation->Allocate(bytes, name) && allocation->Zero(name);
}

template <typename T>
bool CopyH2D(DeviceAllocation *destination, const std::vector<T> &source, const std::string &name)
{
    const size_t bytes = source.size() * sizeof(T);
    return bytes <= destination->bytes() && CheckAcl("aclrtMemcpy H2D(" + name + ")",
        aclrtMemcpy(destination->data(), destination->bytes(), source.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename T>
bool CopyD2H(std::vector<T> *destination, const DeviceAllocation &source, const std::string &name)
{
    const size_t bytes = destination->size() * sizeof(T);
    return bytes <= source.bytes() && CheckAcl("aclrtMemcpy D2H(" + name + ")",
        aclrtMemcpy(destination->data(), bytes, source.data(), bytes, ACL_MEMCPY_DEVICE_TO_HOST));
}

bool CopyBytesD2H(std::vector<uint8_t> *destination, const DeviceAllocation &source,
    size_t bytes, const std::string &name)
{
    destination->assign(bytes, 0);
    return bytes <= source.bytes() && CheckAcl("aclrtMemcpy D2H(" + name + ")",
        aclrtMemcpy(destination->data(), bytes, source.data(), bytes, ACL_MEMCPY_DEVICE_TO_HOST));
}

template <typename T>
std::vector<T> Slice(const std::vector<T> &values, size_t offset, size_t count)
{
    if (offset > values.size() || count > values.size() - offset) return {};
    return std::vector<T>(values.begin() + static_cast<std::ptrdiff_t>(offset),
        values.begin() + static_cast<std::ptrdiff_t>(offset + count));
}

template <typename T>
bool CompareVector(const std::string &name, const std::vector<T> &actual,
    const std::vector<T> &expected, int rank)
{
    if (actual.size() != expected.size()) {
        std::cerr << "rank " << rank << " " << name << " size expected " << expected.size()
                  << " got " << actual.size() << std::endl;
        return false;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::cerr << "rank " << rank << " " << name << "[" << index << "] expected "
                      << expected[index] << " got " << actual[index] << std::endl;
            return false;
        }
    }
    return true;
}

uint64_t HashBytes(uint64_t hash, const void *data, size_t bytes)
{
    const uint8_t *cursor = static_cast<const uint8_t *>(data);
    for (size_t index = 0; index < bytes; ++index) {
        hash ^= cursor[index];
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
uint64_t HashVector(uint64_t hash, const std::vector<T> &values)
{
    const uint64_t count = static_cast<uint64_t>(values.size());
    hash = HashBytes(hash, &count, sizeof(count));
    if (!values.empty()) hash = HashBytes(hash, values.data(), values.size() * sizeof(T));
    return hash;
}

bool ValidateGlobalTokenRemap(const std::vector<ControlRecord> &records,
    int rankSize, uint64_t *globalDigest)
{
    if (records.size() != static_cast<size_t>(rankSize)) return false;
    std::vector<uint64_t> tokenRemap(static_cast<size_t>(rankSize * kNvS),
        TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID);
    std::vector<uint64_t> routeIds(static_cast<size_t>(rankSize * kRoutes),
        TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID);
    std::vector<int32_t> routeDstRanks(static_cast<size_t>(rankSize * kRoutes), -1);
    std::vector<int32_t> routeSlots(static_cast<size_t>(rankSize * kRoutes), -1);

    for (int32_t srcRank = 0; srcRank < rankSize; ++srcRank) {
        const ControlRecord &record = records[static_cast<size_t>(srcRank)];
        if (record.rank != srcRank || record.actualDstCount != kRoutes) {
            std::cerr << "rank 0 gathered invalid dst count from rank " << srcRank
                      << ": " << record.actualDstCount << std::endl;
            return false;
        }
        for (int32_t token = 0; token < kS; ++token) {
            for (int32_t topKId = 0; topKId < kTopK; ++topKId) {
                const size_t localRoute = static_cast<size_t>(token * kTopK + topKId);
                const size_t routeIndex = static_cast<size_t>(srcRank * kRoutes) + localRoute;
                TileXREp::Plan::MoonEPRouteDescriptor route {};
                const TileXRMoonEPPlanStatus buildStatus = TileXREp::Plan::BuildMoonEPRouteDescriptor(
                    srcRank, token, topKId, record.actualDst[localRoute], rankSize, kS, kTopK,
                    kNvS, &route);
                if (buildStatus != PLAN_OK) {
                    std::cerr << "rank 0 failed to build actual route descriptor src=" << srcRank
                              << " token=" << token << " topk=" << topKId
                              << " status=" << buildStatus << std::endl;
                    return false;
                }

                int32_t decodedRank = -1;
                int32_t decodedToken = -1;
                int32_t decodedTopK = -1;
                if (TileXREp::Plan::DecodeMoonEPGlobalTokenId(route.globalTokenId,
                    rankSize, kS, kTopK, &decodedRank, &decodedToken, &decodedTopK) != PLAN_OK ||
                    decodedRank != srcRank || decodedToken != token || decodedTopK != topKId) {
                    std::cerr << "rank 0 globalTokenId decode mismatch src=" << srcRank
                              << " token=" << token << " topk=" << topKId << std::endl;
                    return false;
                }

                uint64_t &recvSlot = tokenRemap[static_cast<size_t>(route.dstRank * kNvS + route.recvSlot)];
                if (recvSlot != TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID) {
                    std::cerr << "rank 0 tokenRemap recvSlot collision dst=" << route.dstRank
                              << " slot=" << route.recvSlot << " prior_id=" << recvSlot
                              << " new_id=" << route.globalTokenId << std::endl;
                    return false;
                }
                recvSlot = route.globalTokenId;
                routeIds[routeIndex] = route.globalTokenId;
                routeDstRanks[routeIndex] = route.dstRank;
                routeSlots[routeIndex] = route.recvSlot;

                for (int32_t priorTopK = 0; priorTopK < topKId; ++priorTopK) {
                    const size_t priorIndex = static_cast<size_t>(srcRank * kRoutes +
                        token * kTopK + priorTopK);
                    if (routeIds[priorIndex] == route.globalTokenId ||
                        (routeDstRanks[priorIndex] == route.dstRank &&
                         routeSlots[priorIndex] == route.recvSlot)) {
                        std::cerr << "rank 0 duplicate route did not receive an independent slot/id src="
                                  << srcRank << " token=" << token << " topk=" << topKId << std::endl;
                        return false;
                    }
                }
            }
        }
    }

    size_t populated = 0;
    for (uint64_t globalTokenId : tokenRemap) {
        if (globalTokenId == TILEXR_MOONEP_INVALID_GLOBAL_TOKEN_ID) continue;
        ++populated;
        int32_t decodedRank = -1;
        int32_t decodedToken = -1;
        int32_t decodedTopK = -1;
        if (TileXREp::Plan::DecodeMoonEPGlobalTokenId(globalTokenId,
            rankSize, kS, kTopK, &decodedRank, &decodedToken, &decodedTopK) != PLAN_OK) {
            std::cerr << "rank 0 failed to decode reconstructed tokenRemap id "
                      << globalTokenId << std::endl;
            return false;
        }
    }
    if (populated != static_cast<size_t>(rankSize * kRoutes)) {
        std::cerr << "rank 0 reconstructed tokenRemap populated " << populated
                  << " routes, expected " << rankSize * kRoutes << std::endl;
        return false;
    }

    uint64_t hash = kFnvOffset;
    for (const ControlRecord &record : records) {
        hash = HashBytes(hash, &record.localOutputDigest, sizeof(record.localOutputDigest));
    }
    hash = HashVector(hash, tokenRemap);
    hash = HashVector(hash, routeIds);
    if (globalDigest != nullptr) *globalDigest = hash;
    return true;
}

uint64_t BuildTopologyHash(const std::vector<int32_t> &globalRankIds)
{
    uint64_t hash = kFnvOffset;
    for (int32_t globalRankId : globalRankIds) {
        hash ^= static_cast<uint32_t>(globalRankId);
        hash *= kFnvPrime;
    }
    return hash;
}

class AlignedDeviceAllocation {
public:
    ~AlignedDeviceAllocation() { if (raw_ != nullptr) (void)aclrtFree(raw_); }
    AlignedDeviceAllocation(const AlignedDeviceAllocation &) = delete;
    AlignedDeviceAllocation &operator=(const AlignedDeviceAllocation &) = delete;
    AlignedDeviceAllocation() = default;
    bool Allocate(size_t bytes, size_t alignment, const std::string &name) {
        if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
        requestedBytes_ = bytes;
        allocationBytes_ = bytes + alignment - 1;
        if (!CheckAcl("aclrtMalloc(" + name + ")",
            aclrtMalloc(&raw_, allocationBytes_, ACL_MEM_MALLOC_HUGE_FIRST))) return false;
        const uintptr_t address = reinterpret_cast<uintptr_t>(raw_);
        const uintptr_t aligned = (address + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
        aligned_ = reinterpret_cast<void *>(aligned);
        usableBytes_ = allocationBytes_ - static_cast<size_t>(aligned - address);
        return usableBytes_ >= requestedBytes_;
    }
    bool Zero(const std::string &name) {
        return CheckAcl("aclrtMemset(" + name + ")",
            aclrtMemset(aligned_, usableBytes_, 0, usableBytes_));
    }
    void *data() const { return aligned_; }
    GM_ADDR gm() const { return reinterpret_cast<GM_ADDR>(aligned_); }
    size_t requestedBytes() const { return requestedBytes_; }
    size_t usableBytes() const { return usableBytes_; }
private:
    void *raw_ = nullptr;
    void *aligned_ = nullptr;
    size_t requestedBytes_ = 0;
    size_t allocationBytes_ = 0;
    size_t usableBytes_ = 0;
};

TileXREp::Plan::ReferenceInput BuildInput(int rankSize)
{
    TileXREp::Plan::ReferenceInput input {};
    input.rankSize = rankSize;
    input.s = kS;
    input.topK = kTopK;
    input.expertNum = static_cast<int64_t>(rankSize) * 2;
    input.config.prefetchSlots = kPrefetchSlots;
    input.config.rankTokenCapacity = kRoutes;
    input.config.nvS = kNvS;
    input.config.tokenPadding = 1;
    input.config.tokenRouteLimitPerPair = 0;
    input.config.cardsPerServer = 8;
    input.config.cardsPerCabinet = 64;
    input.config.crossCandidateCount = 3;
    input.config.reserved = 0;
    if (rankSize == 2) {
        input.globalRankIds = {0, 8};
    } else if (rankSize == 8) {
        input.globalRankIds = {0, 1, 8, 9, 16, 17, 24, 25};
    } else {
        input.globalRankIds.resize(static_cast<size_t>(rankSize));
        for (int rank = 0; rank < rankSize; ++rank) input.globalRankIds[static_cast<size_t>(rank)] = rank;
    }

    std::vector<int32_t> remaining(static_cast<size_t>(input.expertNum), 0);
    input.topkExperts.assign(static_cast<size_t>(rankSize * kRoutes), -1);
    input.tokensPerExpert.assign(static_cast<size_t>(rankSize * input.expertNum), 0);
    for (int rank = 0; rank < rankSize; ++rank) {
        const int serverKey = input.globalRankIds[static_cast<size_t>(rank)] / 8;
        const int rankLoad = (serverKey % 2 == 0) ? 18 : 14;
        remaining[static_cast<size_t>(rank * 2)] = rankLoad / 2;
        remaining[static_cast<size_t>(rank * 2 + 1)] = rankLoad - rankLoad / 2;
    }
    for (int rank = 0; rank < rankSize; ++rank) {
        for (int route = 0; route < kRoutes; ++route) {
            int best = -1;
            for (int probe = 0; probe < input.expertNum; ++probe) {
                const int expert = (probe + rank * 7 + route * 3) % static_cast<int>(input.expertNum);
                if (remaining[static_cast<size_t>(expert)] > 0 &&
                    (best < 0 || remaining[static_cast<size_t>(expert)] > remaining[static_cast<size_t>(best)])) {
                    best = expert;
                }
            }
            if (best < 0) return TileXREp::Plan::ReferenceInput {};
            input.topkExperts[static_cast<size_t>(rank * kRoutes + route)] = best;
            ++input.tokensPerExpert[static_cast<size_t>(rank * input.expertNum + best)];
            --remaining[static_cast<size_t>(best)];
        }
    }
    return input;
}

std::vector<int32_t> BuildExpectedAffinity(const std::vector<int32_t> &globalRankIds)
{
    const int rankSize = static_cast<int>(globalRankIds.size());
    std::vector<int32_t> result(static_cast<size_t>(rankSize * rankSize));
    for (int dst = 0; dst < rankSize; ++dst) {
        std::vector<int32_t> row(static_cast<size_t>(rankSize));
        for (int src = 0; src < rankSize; ++src) row[static_cast<size_t>(src)] = src;
        std::sort(row.begin(), row.end(), [&](int32_t lhs, int32_t rhs) {
            const uint32_t dstId = static_cast<uint32_t>(globalRankIds[static_cast<size_t>(dst)]);
            const uint32_t lhsId = static_cast<uint32_t>(globalRankIds[static_cast<size_t>(lhs)]);
            const uint32_t rhsId = static_cast<uint32_t>(globalRankIds[static_cast<size_t>(rhs)]);
            const uint32_t lhsDistance = lhsId ^ dstId;
            const uint32_t rhsDistance = rhsId ^ dstId;
            if (lhsDistance != rhsDistance) return lhsDistance < rhsDistance;
            if (lhsId != rhsId) return lhsId < rhsId;
            return lhs < rhs;
        });
        std::copy(row.begin(), row.end(), result.begin() + static_cast<std::ptrdiff_t>(dst * rankSize));
    }
    return result;
}

template <typename T>
T LoadMeta(const std::vector<uint8_t> &bytes, uint64_t offset)
{
    T value {};
    if (offset <= bytes.size() && sizeof(T) <= bytes.size() - static_cast<size_t>(offset)) {
        std::memcpy(&value, bytes.data() + static_cast<size_t>(offset), sizeof(T));
    }
    return value;
}

bool ValidateDownstreamMetadata(int rank, const TileXREp::Plan::ReferenceInput &input,
    const TileXREp::Plan::ReferenceOutput &reference, const std::vector<int32_t> &dupGroups,
    const std::vector<int32_t> &dupLoffs, const std::vector<int32_t> &dupCounts)
{
    if (!std::all_of(dupGroups.begin(), dupGroups.end(), [](int32_t value) { return value == kSentinel; }) ||
        !std::all_of(dupLoffs.begin(), dupLoffs.end(), [](int32_t value) { return value == kSentinel; }) ||
        !std::all_of(dupCounts.begin(), dupCounts.end(), [](int32_t value) { return value == kSentinel; })) {
        std::cerr << "rank " << rank << " Plan unexpectedly modified downstream duplicate arrays" << std::endl;
        return false;
    }

    std::vector<TileXREp::Plan::MoonEPReceivedRoute> records;
    for (int srcRank = 0; srcRank < input.rankSize; ++srcRank) {
        for (int token = 0; token < input.s; ++token) {
            for (int topKId = 0; topKId < input.topK; ++topKId) {
                const size_t index = static_cast<size_t>((srcRank * input.s + token) * input.topK + topKId);
                TileXREp::Plan::MoonEPRouteTarget target {};
                if (TileXREp::Plan::DecodeMoonEPDst(reference.dst[index], input.config.nvS,
                    input.rankSize, &target) != PLAN_OK) {
                    std::cerr << "rank " << rank << " reference dst decode failed at " << index << std::endl;
                    return false;
                }
                if (target.dstRank == rank) {
                    records.push_back({srcRank, token, topKId, target.recvSlot, target.sendHidden});
                }
            }
        }
    }
    std::vector<int32_t> builtGroups(static_cast<size_t>(input.config.nvS * 3), -1);
    std::vector<int32_t> builtLoffs(static_cast<size_t>(input.config.nvS), -1);
    std::vector<int32_t> builtCounts(2, 0);
    const TileXRMoonEPPlanStatus status = TileXREp::Plan::BuildMoonEPDuplicateMetadata(
        records.empty() ? nullptr : records.data(), static_cast<int64_t>(records.size()),
        input.rankSize, input.s, input.topK, input.config.nvS,
        builtGroups.data(), builtLoffs.data(), builtCounts.data());
    if (status != PLAN_OK || builtCounts[0] < 0 || builtCounts[1] < 0 ||
        builtCounts[0] > input.config.nvS || builtCounts[1] > input.config.nvS) {
        std::cerr << "rank " << rank << " BuildMoonEPDuplicateMetadata failed with status "
                  << status << " counts={" << builtCounts[0] << "," << builtCounts[1] << "}" << std::endl;
        return false;
    }
    return true;
}

bool ValidateMetaWorkspace(int rank, const TileXREp::Plan::ReferenceInput &input,
    const TileXREp::Plan::ReferenceOutput &reference, uint64_t epoch,
    const TileXREp::Plan::PlanWorkspaceLayout &layout, const std::vector<uint8_t> &meta,
    uint64_t *requestedEpoch, uint64_t *committedEpoch)
{
    if (meta.size() != layout.registeredMeta.totalBytes) {
        std::cerr << "rank " << rank << " registered meta size mismatch" << std::endl;
        return false;
    }
    const uint64_t topologyHash = BuildTopologyHash(input.globalRankIds);
    for (int peer = 0; peer < input.rankSize; ++peer) {
        const uint64_t offset = layout.registeredMeta.planCallHeaders.offset +
            static_cast<uint64_t>(peer) * TileXREp::Plan::kPlanHeaderStrideBytes;
        const TileXREp::Plan::PlanCallHeader header = LoadMeta<TileXREp::Plan::PlanCallHeader>(meta, offset);
        if (header.abiVersion != TileXREp::Plan::kPlanAbiVersion ||
            header.headerBytes != static_cast<int32_t>(sizeof(TileXREp::Plan::PlanCallHeader)) ||
            header.rankSize != input.rankSize || header.s != input.s || header.k != input.topK ||
            header.expertNum != input.expertNum || header.prefetchSlots != input.config.prefetchSlots ||
            header.rankTokenCapacity != input.config.rankTokenCapacity || header.nvS != input.config.nvS ||
            header.tokenPadding != input.config.tokenPadding ||
            header.tokenRouteLimitPerPair != input.config.tokenRouteLimitPerPair ||
            header.cardsPerServer != input.config.cardsPerServer ||
            header.cardsPerCabinet != input.config.cardsPerCabinet ||
            header.crossCandidateCount != input.config.crossCandidateCount ||
            header.epoch != epoch || header.topologyHash != topologyHash) {
            std::cerr << "rank " << rank << " registered plan header mismatch for peer " << peer << std::endl;
            return false;
        }
    }

    std::vector<int32_t> metaTpe(input.tokensPerExpert.size(), 0);
    std::memcpy(metaTpe.data(), meta.data() + static_cast<size_t>(layout.registeredMeta.tpe.offset),
        metaTpe.size() * sizeof(int32_t));
    if (!CompareVector("registered tpe", metaTpe, input.tokensPerExpert, rank)) return false;

    std::vector<int32_t> metaGlobalIds(input.globalRankIds.size(), 0);
    std::memcpy(metaGlobalIds.data(),
        meta.data() + static_cast<size_t>(layout.registeredMeta.globalRankIds.offset),
        metaGlobalIds.size() * sizeof(int32_t));
    if (!CompareVector("registered globalRankIds", metaGlobalIds, input.globalRankIds, rank)) return false;

    const TileXREp::Plan::PlanEpochState state =
        LoadMeta<TileXREp::Plan::PlanEpochState>(meta, layout.registeredMeta.epochState.offset);
    if (requestedEpoch != nullptr) *requestedEpoch = state.requestedEpoch;
    if (committedEpoch != nullptr) *committedEpoch = state.committedEpoch;
    if (state.requestedEpoch != epoch || state.committedEpoch != epoch ||
        state.topologyHash != topologyHash ||
        (state.reserved & TileXREp::Plan::kPlanAffinityCacheValid) == 0) {
        std::cerr << "rank " << rank << " epoch state mismatch requested=" << state.requestedEpoch
                  << " committed=" << state.committedEpoch << " topology=0x" << std::hex
                  << state.topologyHash << " flags=0x" << state.reserved << std::dec << std::endl;
        return false;
    }

    const std::vector<int32_t> expectedAffinity = BuildExpectedAffinity(input.globalRankIds);
    std::vector<int32_t> actualAffinity(expectedAffinity.size(), 0);
    std::memcpy(actualAffinity.data(),
        meta.data() + static_cast<size_t>(layout.registeredMeta.affinityOrder.offset),
        actualAffinity.size() * sizeof(int32_t));
    if (!CompareVector("registered affinityOrder", actualAffinity, expectedAffinity, rank)) return false;

    for (int peer = 0; peer < input.rankSize; ++peer) {
        const uint64_t base = layout.registeredMeta.localStatusByRank.offset +
            static_cast<uint64_t>(peer) * TileXREp::Plan::kPlanStatusStrideBytes;
        for (int word = 0; word < TileXREp::Plan::kPlanStatusWords; ++word) {
            const int32_t actual = LoadMeta<int32_t>(meta, base + static_cast<uint64_t>(word) * sizeof(int32_t));
            const int32_t expected = reference.statusByRank[static_cast<size_t>(
                peer * TileXREp::Plan::kPlanStatusWords + word)];
            if (actual != expected) {
                std::cerr << "rank " << rank << " registered status peer=" << peer << " word=" << word
                          << " expected " << expected << " got " << actual << std::endl;
                return false;
            }
        }
    }

    return true;
}

uint64_t BuildCanonicalDigest(const TileXREp::Plan::ReferenceInput &input,
    const TileXREp::Plan::ReferenceOutput &reference, const std::vector<uint8_t> &registeredMeta)
{
    uint64_t hash = kFnvOffset;
    hash = HashVector(hash, input.globalRankIds);
    hash = HashVector(hash, input.topkExperts);
    hash = HashVector(hash, input.tokensPerExpert);
    hash = HashVector(hash, reference.dst);
    hash = HashVector(hash, reference.cuSeqlens);
    hash = HashVector(hash, reference.expertsToCopy);
    hash = HashVector(hash, reference.expertTargets);
    hash = HashVector(hash, reference.remoteStats);
    hash = HashVector(hash, reference.statusByRank);
    hash = HashVector(hash, registeredMeta);
    return hash;
}

bool RunValidation(int rank, int rankSize, TileXRCommPtr comm, aclrtStream stream,
    ControlRecord *finalRecord)
{
    TileXREp::Plan::ReferenceInput input = BuildInput(rankSize);
    TileXREp::Plan::ReferenceOutput reference {};
    std::string invariantError;
    bool prepared = input.rankSize == rankSize && input.expertNum == static_cast<int64_t>(rankSize) * 2 &&
        TileXREp::Plan::BuildReferencePlan(input, &reference) == PLAN_OK &&
        reference.finalStatus == PLAN_OK &&
        TileXREp::Plan::CheckReferencePlanInvariants(input, reference, &invariantError);
    if (!prepared) {
        std::cerr << "rank " << rank << " CPU reference preparation failed: " << invariantError << std::endl;
    }

    uint64_t localWorkspaceBytes = 0;
    uint64_t registeredMetaBytes = 0;
    TileXREp::Plan::PlanWorkspaceLayout layout {};
    if (prepared) {
        prepared = CheckTileXR("TileXRMoeEpPlanV2GetWorkspaceSize",
            TileXRMoeEpPlanV2GetWorkspaceSize(rankSize, input.s, input.topK, input.expertNum,
                &input.config, &localWorkspaceBytes, &registeredMetaBytes)) &&
            CheckTileXR("BuildPlanWorkspaceLayout", TileXREp::Plan::BuildPlanWorkspaceLayout(
                rankSize, input.s, input.topK, input.expertNum, input.config, &layout)) &&
            localWorkspaceBytes == layout.local.totalBytes &&
            registeredMetaBytes == layout.registeredMeta.totalBytes;
    }

    DeviceAllocation topk;
    DeviceAllocation tpe;
    DeviceAllocation globalRankIds;
    DeviceAllocation legacyDst;
    DeviceAllocation legacyCuSeqlens;
    DeviceAllocation legacyExpertsToCopy;
    DeviceAllocation legacyRemoteStats;
    DeviceAllocation legacyDupGroups;
    DeviceAllocation legacyDupLoffs;
    DeviceAllocation legacyDupCounts;
    DeviceAllocation legacyStatus;
    DeviceAllocation metadataDst;
    DeviceAllocation metadataCuSeqlens;
    DeviceAllocation metadataRemoteExperts;
    DeviceAllocation metadataExpertTargets;
    DeviceAllocation metadataRemoteStats;
    DeviceAllocation metadataDupGroups;
    DeviceAllocation metadataDupLoffs;
    DeviceAllocation metadataDupCounts;
    DeviceAllocation metadataStatus;
    DeviceAllocation localWorkspace;
    AlignedDeviceAllocation registeredMeta;

    const size_t dstCount = static_cast<size_t>(kRoutes);
    const size_t groupCount = static_cast<size_t>(input.expertNum + input.config.prefetchSlots);
    const size_t remoteExpertsCount = static_cast<size_t>(rankSize * input.config.prefetchSlots);
    const size_t expertsPerRank = rankSize > 0 ? static_cast<size_t>(input.expertNum / rankSize) : 0;
    const size_t expertTargetWords = static_cast<size_t>((rankSize + 63) / 64);
    const size_t expertTargetsCount = expertsPerRank * expertTargetWords;
    const std::vector<int32_t> localTopk = prepared ?
        Slice(input.topkExperts, static_cast<size_t>(rank * kRoutes), dstCount) : std::vector<int32_t> {};
    const std::vector<int32_t> localTpe = prepared ?
        Slice(input.tokensPerExpert, static_cast<size_t>(rank * input.expertNum),
            static_cast<size_t>(input.expertNum)) : std::vector<int32_t> {};
    const std::vector<int32_t> sentinelGroups(static_cast<size_t>(kNvS * 3), kSentinel);
    const std::vector<int32_t> sentinelLoffs(static_cast<size_t>(kNvS), kSentinel);
    const std::vector<int32_t> sentinelCounts(2, kSentinel);

    if (prepared) {
        prepared = AllocateZero(&topk, localTopk.size() * sizeof(int32_t), "topk") &&
            AllocateZero(&tpe, localTpe.size() * sizeof(int32_t), "tokensPerExpert") &&
            AllocateZero(&globalRankIds, input.globalRankIds.size() * sizeof(int32_t), "globalRankIds") &&
            AllocateZero(&legacyDst, dstCount * sizeof(int32_t), "legacyDst") &&
            AllocateZero(&legacyCuSeqlens, groupCount * sizeof(int32_t), "legacyCuSeqlens") &&
            AllocateZero(&legacyExpertsToCopy, static_cast<size_t>(kPrefetchSlots) * sizeof(int32_t),
                "legacyExpertsToCopy") &&
            AllocateZero(&legacyRemoteStats, 2 * sizeof(int32_t), "legacyRemoteStats") &&
            AllocateZero(&legacyDupGroups, sentinelGroups.size() * sizeof(int32_t), "legacyDupGroups") &&
            AllocateZero(&legacyDupLoffs, sentinelLoffs.size() * sizeof(int32_t), "legacyDupLoffs") &&
            AllocateZero(&legacyDupCounts, sentinelCounts.size() * sizeof(int32_t), "legacyDupCounts") &&
            AllocateZero(&legacyStatus, TileXREp::Plan::kPlanStatusWords * sizeof(int32_t), "legacyStatus") &&
            AllocateZero(&metadataDst, dstCount * sizeof(int32_t), "metadataDst") &&
            AllocateZero(&metadataCuSeqlens, groupCount * sizeof(int32_t), "metadataCuSeqlens") &&
            AllocateZero(&metadataRemoteExperts, remoteExpertsCount * sizeof(int32_t),
                "metadataRemoteExperts") &&
            AllocateZero(&metadataExpertTargets, expertTargetsCount * sizeof(uint64_t),
                "metadataExpertTargets") &&
            AllocateZero(&metadataRemoteStats, 2 * sizeof(int32_t), "metadataRemoteStats") &&
            AllocateZero(&metadataDupGroups, sentinelGroups.size() * sizeof(int32_t), "metadataDupGroups") &&
            AllocateZero(&metadataDupLoffs, sentinelLoffs.size() * sizeof(int32_t), "metadataDupLoffs") &&
            AllocateZero(&metadataDupCounts, sentinelCounts.size() * sizeof(int32_t), "metadataDupCounts") &&
            AllocateZero(&metadataStatus, TileXREp::Plan::kPlanStatusWords * sizeof(int32_t), "metadataStatus") &&
            AllocateZero(&localWorkspace, static_cast<size_t>(localWorkspaceBytes), "localWorkspace") &&
            registeredMeta.Allocate(static_cast<size_t>(registeredMetaBytes),
                kMetaWorkspaceAlignment, "metaWorkspace") && registeredMeta.Zero("metaWorkspace");
    }
    if (prepared) {
        prepared = CopyH2D(&topk, localTopk, "topk") &&
            CopyH2D(&tpe, localTpe, "tokensPerExpert") &&
            CopyH2D(&globalRankIds, input.globalRankIds, "globalRankIds") &&
            CopyH2D(&legacyDupGroups, sentinelGroups, "legacyDupGroups sentinel") &&
            CopyH2D(&legacyDupLoffs, sentinelLoffs, "legacyDupLoffs sentinel") &&
            CopyH2D(&legacyDupCounts, sentinelCounts, "legacyDupCounts sentinel") &&
            CopyH2D(&metadataDupGroups, sentinelGroups, "metadataDupGroups sentinel") &&
            CopyH2D(&metadataDupLoffs, sentinelLoffs, "metadataDupLoffs sentinel") &&
            CopyH2D(&metadataDupCounts, sentinelCounts, "metadataDupCounts sentinel");
    }

    TileXR::CommArgs *commArgs = nullptr;
    if (prepared) {
        prepared = CheckTileXR("TileXRGetCommArgsHost", TileXRGetCommArgsHost(comm, commArgs)) &&
            commArgs != nullptr && commArgs->rank == rank && commArgs->rankSize == rankSize &&
            commArgs->localRankSize > 0 && commArgs->localRankSize <= rankSize;
        for (int peer = 0; prepared && peer < rankSize; ++peer) {
            prepared = commArgs->peerMems[peer] != nullptr;
            if (!prepared) {
                std::cerr << "rank " << rank << " missing peer memory window for peer "
                          << peer << std::endl;
            }
        }
    }

    ControlRecord memoryRecord {};
    memoryRecord.stage = MEMORY_READY;
    memoryRecord.rank = rank;
    memoryRecord.rankSize = rankSize;
    memoryRecord.success = prepared ? 1 : 0;
    if (std::getenv("TILEXR_PLAN_DEBUG") != nullptr) {
        std::cerr << "rank " << rank << " stage MEMORY_READY enter prepared=" << prepared << std::endl;
    }
    if (!ControlExchange(rank, rankSize, MEMORY_READY, memoryRecord, nullptr)) prepared = false;
    if (std::getenv("TILEXR_PLAN_DEBUG") != nullptr) {
        std::cerr << "rank " << rank << " stage MEMORY_READY leave prepared=" << prepared << std::endl;
    }

    const uint64_t legacyEpoch = kEpochBase + static_cast<uint64_t>(rankSize);
    const uint64_t metadataEpoch = legacyEpoch + 1;
    TileXRMoonEPPlanDesc legacyPlan {};
    legacyPlan.dst = reinterpret_cast<int32_t *>(legacyDst.data());
    legacyPlan.cuSeqlens = reinterpret_cast<int32_t *>(legacyCuSeqlens.data());
    legacyPlan.expertsToCopy = reinterpret_cast<int32_t *>(legacyExpertsToCopy.data());
    legacyPlan.remoteStats = reinterpret_cast<int32_t *>(legacyRemoteStats.data());
    legacyPlan.dupGroups = reinterpret_cast<int32_t *>(legacyDupGroups.data());
    legacyPlan.dupLoffs = reinterpret_cast<int32_t *>(legacyDupLoffs.data());
    legacyPlan.dupCounts = reinterpret_cast<int32_t *>(legacyDupCounts.data());
    legacyPlan.status = reinterpret_cast<int32_t *>(legacyStatus.data());
    legacyPlan.s = input.s;
    legacyPlan.k = input.topK;
    legacyPlan.r = rankSize;
    legacyPlan.e = input.expertNum;
    legacyPlan.b = input.config.prefetchSlots;
    legacyPlan.cap = input.config.rankTokenCapacity;
    legacyPlan.nvS = input.config.nvS;
    legacyPlan.tokenPadding = input.config.tokenPadding;
    legacyPlan.epoch = legacyEpoch;

    int legacyLaunchResult = TileXR::TILEXR_ERROR_INTERNAL;
    bool legacyPlanDone = false;
    if (prepared) {
        if (std::getenv("TILEXR_PLAN_DEBUG") != nullptr) {
            std::cerr << "rank " << rank << " stage LEGACY_PLAN_LAUNCH enter" << std::endl;
        }
        legacyLaunchResult = TileXRMoeEpPlanV2(reinterpret_cast<const int32_t *>(topk.data()),
            reinterpret_cast<const int32_t *>(tpe.data()),
            reinterpret_cast<const int32_t *>(globalRankIds.data()), comm,
            input.s, input.topK, input.expertNum, &input.config, &legacyPlan,
            localWorkspace.data(), localWorkspaceBytes, registeredMeta.data(), registeredMetaBytes, stream);
        if (CheckTileXR("TileXRMoeEpPlanV2(legacy)", legacyLaunchResult)) {
            legacyPlanDone = CheckAcl("aclrtSynchronizeStream(legacy plan)",
                aclrtSynchronizeStream(stream));
        }
    }

    ControlRecord legacyPlanRecord {};
    legacyPlanRecord.stage = LEGACY_PLAN_DONE;
    legacyPlanRecord.rank = rank;
    legacyPlanRecord.rankSize = rankSize;
    legacyPlanRecord.success = legacyPlanDone ? 1 : 0;
    if (!ControlExchange(rank, rankSize, LEGACY_PLAN_DONE, legacyPlanRecord, nullptr)) {
        legacyPlanDone = false;
    }

    TileXRMoonEPPlanMetadataV2 metadata {};
    metadata.structSize = sizeof(metadata);
    metadata.abiVersion = TILEXR_MOONEP_PLAN_METADATA_V2_ABI_VERSION;
    metadata.dst = reinterpret_cast<int32_t *>(metadataDst.data());
    metadata.dstCount = dstCount;
    metadata.cuSeqlens = reinterpret_cast<int32_t *>(metadataCuSeqlens.data());
    metadata.cuSeqlensCount = groupCount;
    metadata.remoteExperts = reinterpret_cast<int32_t *>(metadataRemoteExperts.data());
    metadata.remoteExpertsCount = remoteExpertsCount;
    metadata.expertTargets = reinterpret_cast<uint64_t *>(metadataExpertTargets.data());
    metadata.expertTargetsCount = expertTargetsCount;
    metadata.remoteStats = reinterpret_cast<int32_t *>(metadataRemoteStats.data());
    metadata.remoteStatsCount = 2;
    metadata.status = reinterpret_cast<int32_t *>(metadataStatus.data());
    metadata.statusCount = TileXREp::Plan::kPlanStatusWords;
    metadata.dupGroups = reinterpret_cast<int32_t *>(metadataDupGroups.data());
    metadata.dupGroupsCount = sentinelGroups.size();
    metadata.dupLoffs = reinterpret_cast<int32_t *>(metadataDupLoffs.data());
    metadata.dupLoffsCount = sentinelLoffs.size();
    metadata.dupCounts = reinterpret_cast<int32_t *>(metadataDupCounts.data());
    metadata.dupCountsCount = sentinelCounts.size();
    metadata.s = input.s;
    metadata.k = input.topK;
    metadata.r = rankSize;
    metadata.e = input.expertNum;
    metadata.b = input.config.prefetchSlots;
    metadata.nvS = input.config.nvS;
    metadata.epoch = metadataEpoch;

    int metadataLaunchResult = TileXR::TILEXR_ERROR_INTERNAL;
    bool metadataPlanDone = false;
    if (legacyPlanDone) {
        if (std::getenv("TILEXR_PLAN_DEBUG") != nullptr) {
            std::cerr << "rank " << rank << " stage METADATA_PLAN_LAUNCH enter" << std::endl;
        }
        metadataLaunchResult = TileXRMoeEpPlanV2WithMetadata(
            reinterpret_cast<const int32_t *>(topk.data()),
            reinterpret_cast<const int32_t *>(tpe.data()),
            reinterpret_cast<const int32_t *>(globalRankIds.data()), comm,
            input.s, input.topK, input.expertNum, &input.config, &metadata,
            localWorkspace.data(), localWorkspaceBytes, registeredMeta.data(), registeredMetaBytes, stream);
        if (CheckTileXR("TileXRMoeEpPlanV2WithMetadata", metadataLaunchResult)) {
            metadataPlanDone = CheckAcl("aclrtSynchronizeStream(metadata plan)",
                aclrtSynchronizeStream(stream));
        }
    }

    ControlRecord metadataPlanRecord {};
    metadataPlanRecord.stage = METADATA_PLAN_DONE;
    metadataPlanRecord.rank = rank;
    metadataPlanRecord.rankSize = rankSize;
    metadataPlanRecord.success = metadataPlanDone ? 1 : 0;
    if (!ControlExchange(rank, rankSize, METADATA_PLAN_DONE, metadataPlanRecord, nullptr)) {
        metadataPlanDone = false;
    }
    if (std::getenv("TILEXR_PLAN_DEBUG") != nullptr) {
        DumpPlannerBarrierSlots(rank, rankSize, commArgs, kPlannerBarrierPhaseData);
        DumpPlannerBarrierSlots(rank, rankSize, commArgs, 2);
        DumpPlannerBarrierSlots(rank, rankSize, commArgs, 3);
        DumpPlannerPeerStatuses(rank, rankSize, input.expertNum, commArgs);
    }

    std::vector<int32_t> legacyActualDst(dstCount, 0);
    std::vector<int32_t> legacyActualCu(groupCount, 0);
    std::vector<int32_t> legacyActualExperts(static_cast<size_t>(kPrefetchSlots), 0);
    std::vector<int32_t> legacyActualRemote(2, 0);
    std::vector<int32_t> legacyActualGroups(sentinelGroups.size(), 0);
    std::vector<int32_t> legacyActualLoffs(sentinelLoffs.size(), 0);
    std::vector<int32_t> legacyActualCounts(sentinelCounts.size(), 0);
    std::vector<int32_t> legacyActualStatus(TileXREp::Plan::kPlanStatusWords, 0);
    std::vector<int32_t> actualDst(dstCount, 0);
    std::vector<int32_t> actualCu(groupCount, 0);
    std::vector<int32_t> actualRemoteExperts(remoteExpertsCount, 0);
    std::vector<uint64_t> actualExpertTargets(expertTargetsCount, 0);
    std::vector<int32_t> actualRemote(2, 0);
    std::vector<int32_t> actualGroups(sentinelGroups.size(), 0);
    std::vector<int32_t> actualLoffs(sentinelLoffs.size(), 0);
    std::vector<int32_t> actualCounts(sentinelCounts.size(), 0);
    std::vector<int32_t> actualStatus(TileXREp::Plan::kPlanStatusWords, 0);
    std::vector<uint8_t> actualMeta(static_cast<size_t>(registeredMetaBytes), 0);

    bool legacyRead = legacyPlanDone &&
        CopyD2H(&legacyActualDst, legacyDst, "legacy dst") &&
        CopyD2H(&legacyActualCu, legacyCuSeqlens, "legacy cuSeqlens") &&
        CopyD2H(&legacyActualExperts, legacyExpertsToCopy, "legacy expertsToCopy") &&
        CopyD2H(&legacyActualRemote, legacyRemoteStats, "legacy remoteStats") &&
        CopyD2H(&legacyActualGroups, legacyDupGroups, "legacy dupGroups") &&
        CopyD2H(&legacyActualLoffs, legacyDupLoffs, "legacy dupLoffs") &&
        CopyD2H(&legacyActualCounts, legacyDupCounts, "legacy dupCounts") &&
        CopyD2H(&legacyActualStatus, legacyStatus, "legacy status");
    bool metadataRead = metadataPlanDone &&
        CopyD2H(&actualDst, metadataDst, "metadata dst") &&
        CopyD2H(&actualCu, metadataCuSeqlens, "metadata cuSeqlens") &&
        CopyD2H(&actualRemoteExperts, metadataRemoteExperts, "metadata remoteExperts") &&
        CopyD2H(&actualExpertTargets, metadataExpertTargets, "metadata expertTargets") &&
        CopyD2H(&actualRemote, metadataRemoteStats, "metadata remoteStats") &&
        CopyD2H(&actualGroups, metadataDupGroups, "metadata dupGroups") &&
        CopyD2H(&actualLoffs, metadataDupLoffs, "metadata dupLoffs") &&
        CopyD2H(&actualCounts, metadataDupCounts, "metadata dupCounts") &&
        CopyD2H(&actualStatus, metadataStatus, "metadata status") &&
        CheckAcl("aclrtMemcpy D2H(registeredMeta)", aclrtMemcpy(actualMeta.data(), actualMeta.size(),
            registeredMeta.data(), registeredMetaBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    bool valid = legacyRead && metadataRead;

    uint64_t requestedEpoch = 0;
    uint64_t committedEpoch = 0;
    if (valid) {
        const std::vector<int32_t> expectedDst = Slice(reference.dst,
            static_cast<size_t>(rank * kRoutes), dstCount);
        const std::vector<int32_t> expectedCu = Slice(reference.cuSeqlens,
            static_cast<size_t>(rank) * groupCount, groupCount);
        const std::vector<int32_t> expectedLegacyExperts = Slice(reference.expertsToCopy,
            static_cast<size_t>(rank * kPrefetchSlots), static_cast<size_t>(kPrefetchSlots));
        const std::vector<int32_t> expectedRemote = Slice(reference.remoteStats,
            static_cast<size_t>(rank * 2), 2);
        const std::vector<int32_t> expectedStatus = Slice(reference.statusByRank,
            static_cast<size_t>(rank * TileXREp::Plan::kPlanStatusWords),
            static_cast<size_t>(TileXREp::Plan::kPlanStatusWords));
        const std::vector<uint64_t> expectedExpertTargets = Slice(reference.expertTargets,
            static_cast<size_t>(rank) * expertTargetsCount, expertTargetsCount);
        const std::vector<int32_t> metadataLegacySlice = Slice(actualRemoteExperts,
            static_cast<size_t>(rank * kPrefetchSlots), static_cast<size_t>(kPrefetchSlots));
        std::vector<uint64_t> rebuiltExpertTargets(expertTargetsCount, 0);
        const TileXRMoonEPPlanStatus rebuildStatus = TileXREp::Plan::BuildMoonEPExpertTargets(
            actualRemoteExperts.data(), rankSize, input.expertNum, input.config.prefetchSlots,
            rank, rebuiltExpertTargets.data(), rebuiltExpertTargets.size());

        valid = CompareVector("legacy dst", legacyActualDst, expectedDst, rank) &&
            CompareVector("legacy cuSeqlens", legacyActualCu, expectedCu, rank) &&
            CompareVector("legacy expertsToCopy", legacyActualExperts, expectedLegacyExperts, rank) &&
            CompareVector("legacy remoteStats", legacyActualRemote, expectedRemote, rank) &&
            CompareVector("legacy status", legacyActualStatus, expectedStatus, rank) &&
            legacyActualStatus[0] == PLAN_OK && legacyActualStatus[2] > 0 &&
            ValidateDownstreamMetadata(rank, input, reference,
                legacyActualGroups, legacyActualLoffs, legacyActualCounts) &&
            CompareVector("metadata dst", actualDst, expectedDst, rank) &&
            CompareVector("metadata cuSeqlens", actualCu, expectedCu, rank) &&
            CompareVector("metadata remoteExperts", actualRemoteExperts, reference.expertsToCopy, rank) &&
            CompareVector("metadata expertTargets", actualExpertTargets, expectedExpertTargets, rank) &&
            CompareVector("metadata remoteExperts compatibility slice",
                metadataLegacySlice, legacyActualExperts, rank) &&
            rebuildStatus == PLAN_OK &&
            CompareVector("metadata rebuilt expertTargets",
                actualExpertTargets, rebuiltExpertTargets, rank) &&
            CompareVector("metadata remoteStats", actualRemote, expectedRemote, rank) &&
            CompareVector("metadata status", actualStatus, expectedStatus, rank) &&
            actualStatus[0] == PLAN_OK && actualStatus[2] > 0 &&
            ValidateDownstreamMetadata(rank, input, reference,
                actualGroups, actualLoffs, actualCounts) &&
            ValidateMetaWorkspace(rank, input, reference, metadataEpoch, layout, actualMeta,
                &requestedEpoch, &committedEpoch);
        if (rebuildStatus != PLAN_OK) {
            std::cerr << "rank " << rank << " BuildMoonEPExpertTargets(actual remoteExperts) failed with status "
                      << rebuildStatus << std::endl;
        }
        if (legacyActualStatus.size() > 2 && legacyActualStatus[2] <= 0) {
            std::cerr << "rank " << rank << " legacy validation did not exercise an inter-server round"
                      << std::endl;
        }
        if (actualStatus.size() > 2 && actualStatus[2] <= 0) {
            std::cerr << "rank " << rank << " metadata validation did not exercise an inter-server round"
                      << std::endl;
        }
    }

    uint64_t localDigest = kFnvOffset;
    localDigest = HashVector(localDigest, legacyActualDst);
    localDigest = HashVector(localDigest, legacyActualCu);
    localDigest = HashVector(localDigest, legacyActualExperts);
    localDigest = HashVector(localDigest, legacyActualRemote);
    localDigest = HashVector(localDigest, legacyActualGroups);
    localDigest = HashVector(localDigest, legacyActualLoffs);
    localDigest = HashVector(localDigest, legacyActualCounts);
    localDigest = HashVector(localDigest, legacyActualStatus);
    localDigest = HashVector(localDigest, actualDst);
    localDigest = HashVector(localDigest, actualCu);
    localDigest = HashVector(localDigest, actualRemoteExperts);
    localDigest = HashVector(localDigest, actualExpertTargets);
    localDigest = HashVector(localDigest, actualRemote);
    localDigest = HashVector(localDigest, actualGroups);
    localDigest = HashVector(localDigest, actualLoffs);
    localDigest = HashVector(localDigest, actualCounts);
    localDigest = HashVector(localDigest, actualStatus);
    localDigest = HashVector(localDigest, actualMeta);
    localDigest = HashBytes(localDigest, &legacyEpoch, sizeof(legacyEpoch));
    localDigest = HashBytes(localDigest, &metadataEpoch, sizeof(metadataEpoch));
    const uint64_t canonicalDigest = actualMeta.empty() ? 0 :
        BuildCanonicalDigest(input, reference, actualMeta);

    ControlRecord validationRecord {};
    validationRecord.stage = VALIDATION_DONE;
    validationRecord.rank = rank;
    validationRecord.rankSize = rankSize;
    validationRecord.success = valid ? 1 : 0;
    validationRecord.status = actualStatus.empty() ? PLAN_ERROR_INTERNAL_INVARIANT : actualStatus[0];
    validationRecord.actualDstCount = actualDst.size() == static_cast<size_t>(kRoutes) ?
        static_cast<int32_t>(kRoutes) : 0;
    if (validationRecord.actualDstCount == kRoutes) {
        std::copy(actualDst.begin(), actualDst.end(), validationRecord.actualDst);
    }
    validationRecord.requestedEpoch = requestedEpoch;
    validationRecord.committedEpoch = committedEpoch;
    validationRecord.canonicalDigest = canonicalDigest;
    validationRecord.localOutputDigest = localDigest;
    std::vector<ControlRecord> allRecords;
    uint64_t globalDigest = 0;
    const bool globallyValid = ControlExchange(rank, rankSize, VALIDATION_DONE,
        validationRecord, rank == 0 ? &allRecords : nullptr, &globalDigest);
    validationRecord.globalDigest = globalDigest;
    valid = valid && globallyValid;

    if (rank == 0 && globallyValid) {
        std::cout << "ALL_RANKS_PASS" << std::endl;
        std::cout << "PLAN_GLOBAL_SUMMARY rankSize=" << rankSize
                  << " requested_epoch=" << requestedEpoch
                  << " committed_epoch=" << committedEpoch
                  << " canonical_digest=0x" << std::hex << canonicalDigest
                  << " global_digest=0x" << globalDigest << std::dec << std::endl;
        for (const ControlRecord &record : allRecords) {
            std::cout << "PLAN_RANK_SUMMARY rank=" << record.rank
                      << " status=" << record.status
                      << " local_output_digest=0x" << std::hex << record.localOutputDigest
                      << " canonical_digest=0x" << record.canonicalDigest
                      << " global_digest=0x" << record.globalDigest << std::dec << std::endl;
        }
    }

    if (finalRecord != nullptr) *finalRecord = validationRecord;
    return valid;
}

} // namespace

int main(int argc, char **argv)
{
    const int rankSize = argc > 1 ? std::atoi(argv[1]) : GetEnvInt("RANK_SIZE", 8);
    const int rank = argc > 2 ? std::atoi(argv[2]) : GetEnvInt("RANK", 0);
    const int deviceId = argc > 3 ? std::atoi(argv[3]) : GetEnvInt("DEVICE_ID", rank % 8);
    if ((rankSize != 2 && rankSize != 8 && rankSize != 32 && rankSize != 128) || rank < 0 || rank >= rankSize ||
        deviceId < 0 || deviceId >= 8) {
        std::cerr << "usage: test_tilexr_ep_plan_multirank <2|8|32|128> <rank> <device 0..7>" << std::endl;
        return 2;
    }

    bool aclReady = false;
    aclrtStream stream = nullptr;
    TileXRCommPtr comm = nullptr;
    bool ok = CheckAcl("aclInit", aclInit(nullptr));
    aclReady = ok;
    if (ok) ok = CheckAcl("aclrtSetDevice", aclrtSetDevice(deviceId));
    if (ok) ok = CheckAcl("aclrtCreateStream", aclrtCreateStream(&stream));
    if (ok) ok = CheckTileXR("TileXRCommInitRankLocal", TileXRCommInitRankLocal(rankSize, rank, &comm));

    TileXR::CommArgs *commArgs = nullptr;
    const bool debugEnabled = std::getenv("TILEXR_PLAN_DEBUG") != nullptr;
    bool debugMarkerWritten = false;
    if (ok) {
        ok = CheckTileXR("TileXRGetCommArgsHost", TileXRGetCommArgsHost(comm, commArgs)) &&
            commArgs != nullptr && commArgs->rank == rank && commArgs->rankSize == rankSize &&
            commArgs->localRankSize > 0 && commArgs->localRankSize <= rankSize;
        for (int peer = 0; ok && peer < rankSize; ++peer) {
            ok = commArgs->peerMems[peer] != nullptr;
        }
        if (!ok) std::cerr << "rank " << rank << " did not initialize all peer memory windows" << std::endl;
    }
    if (ok && debugEnabled) {
        GM_ADDR deviceCommArgs = nullptr;
        TileXR::CommArgs deviceSnapshot {};
        const int getDevResult = TileXRGetCommArgsDev(comm, deviceCommArgs);
        const aclError copyArgsResult = getDevResult == TileXR::TILEXR_SUCCESS && deviceCommArgs != nullptr ?
            aclrtMemcpy(&deviceSnapshot, sizeof(deviceSnapshot), deviceCommArgs, sizeof(deviceSnapshot),
                ACL_MEMCPY_DEVICE_TO_HOST) : ACL_ERROR_INVALID_PARAM;
        std::cerr << "rank " << rank << " comm args debug get_dev=" << getDevResult
                  << " copy_dev=" << copyArgsResult << " host_peers=[";
        for (int peer = 0; peer < rankSize; ++peer) {
            std::cerr << (peer == 0 ? "" : ",") << static_cast<void *>(commArgs->peerMems[peer]);
        }
        std::cerr << "] device_peers=[";
        for (int peer = 0; peer < rankSize; ++peer) {
            std::cerr << (peer == 0 ? "" : ",") << static_cast<void *>(deviceSnapshot.peerMems[peer]);
        }
        std::cerr << "]" << std::endl;

        const uint64_t marker[4] = {
            0x5458525045455200ULL | static_cast<uint64_t>(rank),
            0x1111000000000000ULL | static_cast<uint64_t>(rank),
            0x2222000000000000ULL | static_cast<uint64_t>(rank),
            0x3333000000000000ULL | static_cast<uint64_t>(rank)
        };
        const aclError markerResult = aclrtMemcpy(
            commArgs->peerMems[rank] + kPeerVisibilityDebugOffset, sizeof(marker),
            marker, sizeof(marker), ACL_MEMCPY_HOST_TO_DEVICE);
        debugMarkerWritten = markerResult == ACL_SUCCESS;
        std::cerr << "rank " << rank << " peer visibility marker write=" << markerResult
                  << " value=0x" << std::hex << marker[0] << std::dec << std::endl;
    }

    ControlRecord commRecord {};
    commRecord.stage = COMM_READY;
    commRecord.rank = rank;
    commRecord.rankSize = rankSize;
    commRecord.success = ok ? 1 : 0;
    if (comm != nullptr && !ControlExchange(rank, rankSize, COMM_READY, commRecord, nullptr)) ok = false;
    if (ok && debugEnabled && debugMarkerWritten) {
        std::cerr << "rank " << rank << " peer visibility reads=[";
        for (int peer = 0; peer < rankSize; ++peer) {
            uint64_t marker[4] = {};
            const aclError readResult = aclrtMemcpy(marker, sizeof(marker),
                commArgs->peerMems[peer] + kPeerVisibilityDebugOffset, sizeof(marker),
                ACL_MEMCPY_DEVICE_TO_HOST);
            std::cerr << (peer == 0 ? "" : ",") << "{peer=" << peer
                      << ",rc=" << readResult << ",value=0x" << std::hex << marker[0] << std::dec << "}";
        }
        std::cerr << "]" << std::endl;
    }

    ControlRecord finalRecord {};
    if (ok) ok = RunValidation(rank, rankSize, comm, stream, &finalRecord);

    if (comm != nullptr) (void)TileXRCommDestroy(comm);
    if (stream != nullptr) (void)aclrtDestroyStream(stream);
    if (aclReady) (void)aclFinalize();

    if (ok) {
        std::cout << "PLAN_VALIDATION_PASS rank=" << rank << " rankSize=" << rankSize
                  << " device=" << deviceId << " status=" << finalRecord.status
                  << " requested_epoch=" << finalRecord.requestedEpoch
                  << " committed_epoch=" << finalRecord.committedEpoch
                  << " local_output_digest=0x" << std::hex << finalRecord.localOutputDigest
                  << " canonical_digest=0x" << finalRecord.canonicalDigest
                  << " global_digest=0x" << finalRecord.globalDigest << std::dec << std::endl;
        return 0;
    }
    std::cerr << "PLAN_VALIDATION_FAIL rank=" << rank << " rankSize=" << rankSize
              << " device=" << deviceId << std::endl;
    return 1;
}
