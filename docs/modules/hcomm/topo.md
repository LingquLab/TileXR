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
    u32 userRank;
    u32 userRankSize;
    u32 devicePhyId;
    u32 serverIdx;
    u32 superPodIdx;
    std::string serverId;
    std::vector<HcclIpAddress> nicIp;         // 设备 NIC IP 列表
    std::vector<HcclIpAddress> backupNicIp;
    LinkTypeInServer deviceToNextRankLinkType; // 与相邻 rank 的链路类型
    // ... 更多字段
};
```

### `TopoMatcher`（`topo_matcher.h` + `.cc`）

```cpp
class TopoMatcher {
    // 计算各层通信平面（level0/level1/level2）的 rank 分组
    HcclResult CalcCommPlaneInfo(const std::string& tag,
        const CommParaInfo& commParaInfo,
        std::vector<std::vector<std::vector<RankInfo>>>& commPlanes);

    // 查询两个 rank 之间的链路类型
    LinkTypeInServer GetLinkTypeByRank(u32 rankA, u32 rankB) const;

    // 是否是 bridge rank（多环拓扑中的桥接节点）
    bool IsBridgeRank(u32 rank, u32 ringIdx) const;

    // SuperPod 场景下的 rank 分组
    HcclResult CalcSuperPodCommPlane(...);
};
```

### `TopoInfoParse`（`topoinfo_parse.cc`）

```cpp
class TopoInfoParse {
    // 解析 rank table（支持 JSON v1/v2 等多种格式）
    HcclResult Parse(const std::string& rankTableContent,
        std::vector<RankInfo>& rankInfoList);

    // 校验 rank table 合法性（rank 数量、IP 格式、设备 ID 唯一性等）
    HcclResult Validate(const std::vector<RankInfo>& rankInfoList);
};
```

### 拓扑交换流程（Exchange）

```cpp
// Dispatcher 协调 agent 和 server
class TopoInfoExchangeDispatcher {
    HcclResult Exchange(std::vector<HcclTopoInfo>& localInfos,
        std::vector<HcclTopoInfo>& globalInfos);
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
`topoinfo_ranktable_*.cc` 系列文件处理不同版本的 rank table 格式（v1/v2/legacy/cluster），通过工厂模式根据 JSON 中的 `version` 字段选择对应解析器。

### 链路类型探测
`TopoInfoDetect` 通过 HCCP 驱动接口枚举物理链路（HCCS/PCIe），结合设备 ID 映射确定 `LinkTypeInServer`。这个结果直接影响算法选择（HCCS → ring allreduce，PCIe → RDMA allreduce）。

### 4P/8P 有效组合校验
`topoinfo_parse.cc` 中内嵌了 AIServer 上 4P rank 的合法组合表（`VALID_4P_RANK_COMBINATIONS`），用于验证用户提供的 rank table 在硬件上是否可行。

### 多层级通信平面计算
`TopoMatcher::CalcCommPlaneInfo()` 将全局 rank 按服务器边界划分为 level0（片内）、level1（跨机同超节点）、level2（超节点间）三层，为 `CommFactory` 提供各层的 rank 分组。

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
