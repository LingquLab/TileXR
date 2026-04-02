# 模块：hcomm/topo（拓扑发现与交换）

## 1. 模块作用

负责集群拓扑信息的采集、解析、交换和维护。具体包括：从 rank table（JSON 配置）解析 rank/服务器/设备映射关系、在 init 阶段通过 agent/server/dispatcher 模式在所有 rank 间交换拓扑信息、以及为算法层提供 `HcclTopoInfo` 结构和拓扑匹配（`TopoMatcher`）能力。

---

## 2. 目录结构

```
src/framework/common/src/topo/          # 约 30 个文件
├── topoinfo_parse.cc/.h                # rank table 解析（JSON → 内存结构）
├── topoinfo_detect.cc/.h               # 本地设备拓扑探测（链路类型检测）
├── topoinfo_exchange_agent.cc/.h       # 拓扑交换 Agent（每个 rank 的交换客户端）
├── topoinfo_exchange_server.cc/.h      # 拓扑交换 Server（协调者，通常 rank 0）
├── topoinfo_exchange_dispatcher.cc/.h  # 拓扑交换 Dispatcher（调度 agent/server）
├── topoinfo_ranktable_xxxx.cc/.h       # 各格式 rank table 解析实现（多文件）
└── （其他拓扑辅助文件）
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `topoinfo_parse.cc` | 解析 rank table JSON：rank → 服务器 → 设备 → NIC IP 的映射；验证 4P/8P 等合法组合 |
| `topoinfo_detect.cc` | 本地探测：通过 HCCP/PCIe 拓扑查询确定相邻设备的链路类型（HCCS/PIX/PIB/PHB） |
| `topoinfo_exchange_agent.cc` | 每 rank 一个 agent，收集本地拓扑信息并发送给 server，接收全局汇总结果 |
| `topoinfo_exchange_server.cc` | rank 0（或指定 rank）作为 server，汇总所有 agent 上报的拓扑，广播全局视图 |
| `topoinfo_exchange_dispatcher.cc` | 管理 agent/server 的创建和生命周期，协调交换流程 |

---

## 4. 核心函数 / 类 / 接口

### `HcclTopoInfo` 结构体（`topo_matcher.h`）

```cpp
struct HcclTopoInfo {
    u32  userRank;
    u32  userRankSize;
    u32  devicePhyId;
    s32  deviceLogicId;
    std::vector<u32> nicList;
    bool isSingleMeshAggregation;
    u32  deviceNumPerAggregation;  // 每模块设备数
    u32  superPodNum;              // 超节点总数
    DevType  deviceType;
    TopoType topoType;
    bool is310P3Common;
    u32  serverNum;
    u32  meshAggregationRankSize;
    u32  moduleNum;
    bool useSuperPodMode;
    bool isDiffDeviceModule;
    bool isDiffDeviceType;
    bool isARSDoubleRing;          // 默认 true
    std::unordered_map<u32, bool>  isUsedRdmaMap;
    std::unordered_map<u32, u32>   pairLinkCounter; // 片内链路类型计数
    std::vector<std::vector<std::vector<std::vector<u32>>>> CommPlaneSubGroupVector;
};
```

### `HcclExternalEnable` 结构体（`topo_matcher.h`）

```cpp
using HcclExternalEnable = struct HcclExternalEnableDef {
    u32  enableFfts;          // FFTS 开关（默认 1）
    u32  deterministic;       // 确定性计算（默认 0）
    u32  intraRoceSwitch;     // 片内 RoCE 开关（默认 0）
    u32  dumpDebug;           // 调试 dump（默认 0）
    u32  interHccsDisable;    // 禁用跨机 HCCS（默认 0）
    bool aivMode;             // AIV 执行模式（默认 false）
    bool aicpuUnfold;         // AICPU 展开模式（默认 false）
    bool isOnlyAiv;           // 仅 AIV 模式（默认 false）
    s32  execTimeOut;         // 执行超时
    std::map<HcclCMDType, std::vector<HcclAlgoType>> algoConfig; // 每算子类型的算法配置
};
```

### `TopoMatcher`（`topo_matcher.h` + `.cc`）

```cpp
class TopoMatcher {
    // 构造：注入通信平面、bridge rank、拓扑/算法/特性开关
    explicit TopoMatcher(
        const std::vector<std::vector<std::vector<u32>>> CommPlaneRanks,
        const std::vector<bool> isBridgeVector,
        HcclTopoInfo& topoInfo, HcclAlgoInfo& algoInfo,
        HcclExternalEnable& externalEnable,
        std::vector<std::vector<std::vector<u32>>>& serverAndsuperPodToRank);

    // 计算通信平面（核心接口）
    HcclResult CalcCommPlaneInfo(const std::string& tag,
        const CommParaInfo& commParaInfo,
        std::vector<SingleSubCommTransport>& commTransport,
        TransportMemType inputMemType, TransportMemType outputMemType);

    // 特性开关读取/设置
    u32  GetExternalInputHcclEnableFfts();
    u32  GetExternalInputIntraRoceSwitch();
    bool GetAivModeConfig() const;
    bool GetAicpuUnfoldConfig() const;
    HcclResult SetDeterministicConfig(u8 deterministic);
    HcclResult SetAlgoConfig(const std::map<HcclCMDType, std::vector<HcclAlgoType>>&);

    // 超节点/服务器级 rank 分组
    HcclResult GetLocalSuperPodRankSize(u32 userRank, u32& devNum, u32& rankIdx);
    HcclResult GetLocalServerRankSize(u32 userRank, u32& devNum, u32& rankIdx);
    u32  GetSubRootUserRank(u32 userRank, u32 rootUserRank);
    u32  GetSubRootWithSuperPod(u32 userRank, u32 rootUserRank);

    // 拓扑拓展/AHC
    void GetAHCAlgOption(std::map<AHCConcOpType, TemplateType>&);
    void SetAHCAlgOption(std::map<AHCConcOpType, TemplateType>&);
    bool CheckSdmaWithRohTopo(const std::vector<u32>& nicList, std::vector<u32>& topoList);
};
```

### `TopoInfoParse`（`topoinfo_parse.cc`）

```cpp
class TopoInfoRanktableParser {  // 基类
    virtual HcclResult Init();
    virtual HcclResult GetClusterInfo(RankTable_t&);
    HcclResult GetRanktableVersion(std::string&);

    // rank table 版本常量
    // HCCL_CLUSTER_VERSION   = "1.0"  标准 HCCL 集群
    // HETEROG_CLUSTER_VERSION = "1.1" 异构集群（CPU+NPU rank 混合）
    // SUPERPOD_CLUSTER_VERSION = "1.2" 超节点集群
};
```

### `HcclBasicRankInfo`（`topoinfo_exchange_agent.h`）

```cpp
// 拓扑交换前每个 rank 上报的基本信息
using HcclBasicRankInfo = struct HcclBasicRankInfoDef {
    HcclIpAddress             hostIP;
    u32                       hostPort       {HCCL_INVALID_PORT};
    u32                       rank           {0};
    u32                       rankSize       {0};
    NICDeployment             nicDeploy      {NICDeployment::NIC_DEPLOYMENT_DEVICE};
    DevType                   deviceType     {DevType::DEV_TYPE_910};
    s32                       deviceLogicID  {0};
    u32                       devicePhysicID {0};
    std::vector<HcclIpAddress> deviceIP;
    std::vector<HcclIpAddress> backupDeviceIP;
    u32                       deviceNicPort  {HCCL_INVALID_PORT};
    u32                       deviceVnicPort {HCCL_INVALID_PORT};
    u32                       superDeviceId  {INVALID_UINT};
    std::string               superPodId;
    TlsStatus                 tlsStatus;
};
```

### 拓扑交换 Dispatcher（`topoinfo_exchange_dispatcher.h`）

```cpp
// epoll 驱动的异步广播
class TopoInfoExchangeDispather {
    // 常量
    static constexpr u32 DEFAULT_THREAD_NUM      = 1;
    static constexpr u32 MAX_THREAD_NUM          = 4;   // 最多 4 个 worker 线程
    static constexpr s32 RANK_CAPACITY_PER_THREAD = 512; // 每线程处理 rank 数
    static constexpr s32 EPOLL_TIMEOUT_MS         = 100;
    static constexpr s32 LAST_EPOLL_TIMEOUT_MS    = 5;

    // FdContext：epoll 事件关联的 socket + 发送状态
    struct FdContext {
        std::shared_ptr<HcclSocket> socket;
        SendState txState;
    };

    // SendState：单次非阻塞发送进度跟踪
    struct SendState {
        u32    rankId;
        size_t headerLen;    // 头部总长度
        size_t headerSended; // 已发送头部字节
        size_t bodyLen;      // 载荷总长度
        size_t bodySended;   // 已发送载荷字节
        void*  data;
        bool IsOk(); // bodyLen≠0 && 头部/载荷均已发完
    };

    // 广播接口
    HcclResult BroadcastRankTable(connectSockets, clusterInfo, failedAgentIdList);
    HcclResult BroadcastGroupLeaderInfo(connectSockets, leaderInfo);
};
```

### 拓扑交换 Agent（`topoinfo_exchange_agent.h`）

```cpp
class TopoInfoExchangeAgent {
    // 完整初始化并获取全局拓扑
    HcclResult Setup();
    // 拆除连接
    HcclResult Teardown();
    // 获取拓扑结果
    HcclResult GetClusterTopoInfo(RankTable_t& clusterInfo);
    // 获取分组 Leader
    HcclResult GetGroupLeader(HcclRankHandle& rankHandle);
};
```

---

## 5. 数据流向

```
HcclCommunicator::Init()
  │
  ├── TopoInfoParse::Parse(rankTableContent)
  │       JSON rank table → std::vector<RankInfo>
  │
  ├── TopoInfoDetect::Detect()
  │       本地 HCCP/PCIe 查询 → LinkTypeInServer（HCCS/PIX/...）
  │
  ├── TopoInfoExchangeDispatcher::Exchange()
  │       ┌────────────────────────────────┐
  │       │  所有 rank 的 agent 并发上报    │
  │       │  → rank 0 server 汇总          │
  │       │  → 广播全局 HcclTopoInfo[]     │
  │       └────────────────────────────────┘
  │
  └── TopoMatcher 初始化（持有全局拓扑视图）
        └── 为 CommFactory / hcclImpl 提供通信平面计算
```

---

## 6. 关键业务逻辑

### Rank Table 格式兼容
`topoinfo_ranktable_*.cc` 系列文件处理不同版本的 rank table 格式，通过工厂模式根据 JSON 中的 `version` 字段选择对应解析器：
- **`"1.0"`**（`HCCL_CLUSTER_VERSION`）：标准 HCCL 集群
- **`"1.1"`**（`HETEROG_CLUSTER_VERSION`）：异构集群（CPU rank 与 NPU rank 混合）
- **`"1.2"`**（`SUPERPOD_CLUSTER_VERSION`）：超节点（SuperPod）集群

合法性校验通过 `JsonUniqueInfoType` 枚举（DEVICE_IP/SERVER_ID/ETH_IP/SUPER_POD_ID 等 8 类）确保唯一性。

### 链路类型探测
`TopoInfoDetect` 通过 HCCP 驱动接口枚举物理链路（HCCS/PCIe），结合设备 ID 映射确定 `LinkTypeInServer`。这个结果直接影响算法选择（HCCS → ring allreduce，PCIe → RDMA allreduce）。

### 4P/8P 有效组合校验
`topoinfo_parse.cc` 中使用 `HCCL_AISERVER_VAILD_4P_RANKS`（hardcoded set）存储 AIServer 上 4P rank 的合法组合，用于验证用户提供的 rank table 在硬件上是否可行。

### 多层级通信平面计算
`TopoMatcher::CalcCommPlaneInfo()` 将全局 rank 按服务器边界划分为 level0（片内）、level1（跨机同超节点）、level2（超节点间）三层，为 `CommFactory` 提供各层的 rank 分组。内部通过 `serverAndsuperPodToRank_` 二维向量：`[0]` 存放每个超节点内的服务器 rank 列表，`[1]` 存放每个超节点的 rank 列表。

### Dispatcher epoll 广播机制
`TopoInfoExchangeDispather` 使用 1~4 个 worker 线程（每线程最多处理 512 个 rank）通过 epoll 驱动非阻塞发送。每个连接的 socket 被封装为 `FdContext`（含 `SendState` 进度跟踪），所有发送完成后用原子计数器 `sendDoneCount_` 确认广播完毕。

---

## 7. 开发注意事项

- 拓扑交换（Exchange）必须在所有 rank 上同步调用，任何 rank 缺席都会导致 server 等待超时。
- `TopoInfoDetect` 依赖 HCCP 驱动，在纯 CPU 环境或 mock 环境下需要提供 stub 实现。
- Rank table 解析错误应尽早报告（在 `Parse()` 阶段），不要延迟到通信时才发现。
- `HcclTopoInfo` 中的 `nicIp` 字段在单机场景下可能为空（不需要 NIC），使用前需检查。

---

## 8. 未来可扩展点

- **动态拓扑**：当前拓扑在 init 时确定且不变，可扩展为支持运行时设备热插拔的动态拓扑更新。
- **自动 rank table 生成**：根据 `TopoInfoDetect` 的物理探测结果自动生成 rank table，免去用户手工配置。
- **多云拓扑**：扩展 `HcclTopoInfo` 支持跨数据中心（WAN）拓扑，增加网络延迟/带宽元数据。
