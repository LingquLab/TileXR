/*
 * Copyright (c) 2024-2026 TileXR Project
 * Licensed under the Apache License, Version 2.0
 */

#ifndef TILEXR_HCCP_DEFS_H
#define TILEXR_HCCP_DEFS_H

#include <arpa/inet.h>
#include <cstdint>
#include <map>
#include <net/if.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

namespace TileXR {

constexpr int32_t TILEXR_DEV_EID_INFO_MAX_NAME = 64;
constexpr int32_t TILEXR_DEV_QP_KEY_SIZE = 64;
constexpr int32_t TILEXR_CUSTOM_CHAN_DATA_MAX_SIZE = 2048;
constexpr int32_t TILEXR_MAX_INTERFACE_NUM = 8;
constexpr uint32_t TILEXR_UDMA_TOKEN_VALUE = 0;
constexpr int32_t TILEXR_UDMA_MEM_KEY_SIZE = 128;
constexpr int32_t TILEXR_URMA_TOKEN_PLAIN_TEXT = 1;
constexpr uint32_t TILEXR_UDMA_RQ_DEPTH_DEFAULT = 256;

enum HccpNetworkMode {
    NETWORK_ONLINE = 0,
    NETWORK_OFFLINE = 1,
    NETWORK_PEER_ONLINE = 2,
};

enum HccpNotifyType {
    HCCP_NOTIFY_NONE = 0,
};

enum DrvHdcServiceType : int {
    HDC_SERVICE_TYPE_RDMA = 6,
    HDC_SERVICE_TYPE_RDMA_V2 = 18,
};

struct RaInitConfig {
    unsigned int phyId;
    HccpNetworkMode nicPosition;
    DrvHdcServiceType hdcType;
    bool enableHdcAsync;
};

enum SubProcType {
    TSD_SUB_PROC_HCCP = 0,
};

struct ProcEnvParam {
    const char* envName;
    uint64_t nameLen;
    const char* envValue;
    uint64_t valueLen;
};

struct ProcExtParam {
    const char* paramInfo;
    uint64_t paramLen;
};

struct ProcOpenArgs {
    SubProcType procType;
    ProcEnvParam* envParaList;
    uint64_t envCnt;
    const char* filePath;
    uint64_t pathLen;
    ProcExtParam* extParamList;
    uint64_t extParamCnt;
    pid_t* subPid;
};

struct RaInfo {
    HccpNetworkMode mode;
    unsigned int phyId;
};

union HccpEid {
    uint8_t raw[16];
    struct {
        uint64_t reserved;
        uint32_t prefix;
        uint32_t addr;
    } in4;
    struct {
        uint64_t subnetPrefix;
        uint64_t interfaceId;
    } in6;
};

struct DevEidInfo {
    char name[TILEXR_DEV_EID_INFO_MAX_NAME];
    uint32_t type;
    uint32_t eidIndex;
    HccpEid eid;
    uint32_t dieId;
    uint32_t chipId;
    uint32_t funcId;
    uint32_t resv;
};

struct CtxInitCfg {
    HccpNetworkMode mode;
    union {
        struct {
            bool disabledLiteThread;
        } rdma;
    };
};

struct CtxInitAttr {
    unsigned int phyId;
    union {
        struct {
            HccpNotifyType notifyType;
            int family;
            uint8_t localIp[16];
        } rdma;
        struct {
            uint32_t eidIndex;
            HccpEid eid;
        } ub;
    };
    uint32_t resv[16];
};

struct MemKey {
    uint8_t value[TILEXR_UDMA_MEM_KEY_SIZE];
    uint8_t size;
};

struct DevNotifyInfo {
    uint64_t va;
    uint64_t size;
    MemKey key;
    uint32_t resv[4];
};

struct DevBaseAttrT {
    uint32_t sqMaxDepth;
    uint32_t rqMaxDepth;
    uint32_t sqMaxSge;
    uint32_t rqMaxSge;
    union {
        struct {
            DevNotifyInfo globalNotifyInfo;
        } rdma;
        struct {
            uint32_t maxJfsInlineLen;
            uint32_t maxJfsRsge;
            uint32_t dieId;
            uint32_t chipId;
            uint32_t funcId;
        } ub;
    } devInfo;
    uint32_t resv[16];
};

union DataPlaneCstmFlag {
    struct {
        uint32_t poolCqCstm : 1;
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

struct ChanInfoT {
    struct {
        DataPlaneCstmFlag dataPlaneFlag;
    } in;
    struct {
        int fd;
    } out;
};

enum JfcMode {
    JFC_MODE_NORMAL = 0,
    JFC_MODE_USER_CTL_NORMAL = 3,
};

union JfcFlag {
    struct {
        uint32_t lockFree : 1;
        uint32_t jfcInline : 1;
        uint32_t reserved : 30;
    } bs;
    uint32_t value;
};

struct CqCreateAttr {
    void* chanHandle;
    uint32_t depth;
    union {
        struct {
            uint64_t cqContext;
            uint32_t mode;
            uint32_t compVector;
        } rdma;
        struct {
            uint64_t userCtx;
            JfcMode mode;
            uint32_t ceqn;
            JfcFlag flag;
            struct {
                bool valid;
                uint32_t cqeFlag;
            } ccuExCfg;
        } ub;
    };
};

struct CqCreateInfo {
    uint64_t va;
    uint32_t id;
    uint32_t cqeSize;
    uint64_t bufAddr;
    uint64_t swdbAddr;
};

struct CqInfoT {
    CqCreateAttr in;
    CqCreateInfo out;
};

enum JettyMode {
    JETTY_MODE_URMA_NORMAL = 0,
    JETTY_MODE_USER_CTL_NORMAL = 3,
};

enum TransportModeT {
    CONN_RM = 1,
    CONN_RC = 2,
};

union JettyFlag {
    struct {
        uint32_t shareJfr : 1;
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

union JfsFlag {
    struct {
        uint32_t lockFree : 1;
        uint32_t errorSuspend : 1;
        uint32_t outorderComp : 1;
        uint32_t orderType : 8;
        uint32_t multiPath : 1;
        uint32_t reserved : 20;
    } bs;
    uint32_t value;
};

struct JettyQueCfgEx {
    uint32_t buffSize;
    uint64_t buffVa;
};

union CstmJfsFlag {
    struct {
        uint32_t sqCstm : 1;
        uint32_t dbCstm : 1;
        uint32_t dbCtlCstm : 1;
        uint32_t reserved : 29;
    } bs;
    uint32_t value;
};

struct QpCreateAttr {
    void* scqHandle;
    void* rcqHandle;
    void* srqHandle;
    uint32_t sqDepth;
    uint32_t rqDepth;
    TransportModeT transportMode;
    union {
        struct {
            uint32_t mode;
            uint32_t udpSport;
            uint8_t trafficClass;
            uint8_t sl;
            uint8_t timeout;
            uint8_t rnrRetry;
            uint8_t retryCnt;
        } rdm;
        struct {
            JettyMode mode;
            uint32_t jettyId;
            JettyFlag flag;
            JfsFlag jfsFlag;
            void* tokenIdHandle;
            uint32_t tokenValue;
            uint8_t priority;
            uint8_t rnrRetry;
            uint8_t errTimeout;
            union {
                struct {
                    JettyQueCfgEx sq;
                    bool piType;
                    CstmJfsFlag cstmFlag;
                    uint32_t sqebbNum;
                } extMode;
                struct {
                    bool lockFlag;
                    uint32_t sqeBufIdx;
                } taCacheMode;
            };
        } ub;
    };
    uint32_t resv[16];
};

struct QpKeyT {
    uint8_t value[TILEXR_DEV_QP_KEY_SIZE];
    uint8_t size;
};

struct QpCreateInfo {
    QpKeyT key;
    union {
        struct {
            uint32_t qpn;
        } rdma;
        struct {
            uint32_t uasid;
            uint32_t id;
            uint64_t sqBuffVa;
            uint64_t wqebbSize;
            uint64_t dbAddr;
            uint32_t dbTokenId;
            uint64_t ciAddr;
        } ub;
    };
    uint64_t va;
    uint32_t resv[16U];
};

struct HccpTokenId {
    uint32_t tokenId;
};

enum TokenPolicy : uint32_t {
    TOKEN_POLICY_NONE = 0,
    TOKEN_POLICY_PLAIN_TEXT = 1,
};

union ImportJettyFlag {
    struct {
        uint32_t tokenPolicy : 3;
        uint32_t orderType : 8;
        uint32_t shareTp : 1;
        uint32_t reserved : 20;
    } bs;
    uint32_t value;
};

enum JettyGrpPolicy : uint32_t {
    JETTY_GRP_POLICY_RR = 0,
};

enum TargetType {
    TARGET_TYPE_JFR = 0,
    TARGET_TYPE_JETTY = 1,
};

enum JettyImportMode {
    JETTY_IMPORT_MODE_NORMAL = 0,
};

struct JettyImportExpCfg {
    uint64_t tpHandle;
    uint64_t peerTpHandle;
    uint64_t tag;
    uint32_t txPsn;
    uint32_t rxPsn;
    uint32_t rsv[16];
};

struct QpImportAttr {
    QpKeyT key;
    union {
        struct {
            JettyImportMode mode;
            uint32_t tokenValue;
            JettyGrpPolicy policy;
            TargetType type;
            ImportJettyFlag flag;
            JettyImportExpCfg expImportCfg;
            uint32_t tpType;
        } ub;
    };
    uint32_t resv[7];
};

struct QpImportInfo {
    union {
        struct {
            uint64_t tjettyHandle;
            uint32_t tpn;
        } ub;
    };
    uint32_t resv[8];
};

struct QpImportInfoT {
    QpImportAttr in;
    QpImportInfo out;
};

enum MemSegTokenPolicy {
    MEM_SEG_TOKEN_NONE = 0,
    MEM_SEG_TOKEN_PLAIN_TEXT = 1,
};

struct MemInfo {
    uint64_t addr;
    uint64_t size;
};

union RegSegFlag {
    struct {
        uint32_t tokenPolicy : 3;
        uint32_t cacheable : 1;
        uint32_t dsva : 1;
        uint32_t access : 6;
        uint32_t nonPin : 1;
        uint32_t userIova : 1;
        uint32_t tokenIdValid : 1;
        uint32_t reserved : 18;
    } bs;
    uint32_t value;
};

struct MemRegAttr {
    MemInfo mem;
    union {
        struct {
            int access;
        } rdma;
        struct {
            RegSegFlag flags;
            uint32_t tokenValue;
            void* tokenIdHandle;
        } ub;
    };
    uint32_t resv[8];
};

struct MemRegInfo {
    MemKey key;
    union {
        struct {
            uint32_t lkey;
        } rdma;
        struct {
            uint32_t tokenId;
            uint64_t targetSegHandle;
        } ub;
    };
    uint32_t resv[8U];
};

struct MrRegInfoT {
    MemRegAttr in;
    MemRegInfo out;
};

union ImportSegFlag {
    struct {
        uint32_t cacheable : 1;
        uint32_t access : 6;
        uint32_t mapping : 1;
        uint32_t reserved : 24;
    } bs;
    uint32_t value;
};

struct MemImportAttr {
    MemKey key;
    union {
        struct {
            ImportSegFlag flags;
            uint64_t mappingAddr;
            uint32_t tokenValue;
        } ub;
    };
    uint32_t resv[4];
};

struct MemImportInfo {
    union {
        struct {
            uint32_t key;
        } rdma;
        struct {
            uint64_t targetSegHandle;
        } ub;
    };
    uint32_t resv[4];
};

struct MrImportInfoT {
    MemImportAttr in;
    MemImportInfo out;
};

enum MemSegAccessFlags {
    MEM_SEG_ACCESS_LOCAL_ONLY = 1,
    MEM_SEG_ACCESS_READ = (1 << 1),
    MEM_SEG_ACCESS_WRITE = (1 << 2),
    MEM_SEG_ACCESS_ATOMIC = (1 << 3),
    MEM_SEG_ACCESS_DEFAULT = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC,
};

struct RegMemResultInfo {
    uint32_t reserved{0};
    uint64_t address{0};
    uint64_t size{0};
    void* lmemHandle{nullptr};
    MemKey key{{0}, 0};
    uint32_t tokenId{0};
    uint32_t tokenValue{0};
    uint64_t targetSegHandle{0};
    void* tokenIdHandle{nullptr};
    uint32_t cacheable{0};
    int32_t access{0};
};

using MemoryRegionMap = std::map<uint64_t, std::map<uint32_t, RegMemResultInfo>, std::greater<uint64_t>>;

} // namespace TileXR

#endif // TILEXR_HCCP_DEFS_H
