# 模块：hcomm/framework-cluster（集群维护）

## 1. 模块作用

负责分布式训练集群的健康监控、故障检测和自动恢复。包含三个子系统：
- **Heartbeat**：周期性心跳广播，检测 rank 存活状态
- **Anomaly Detection**：连接异常检测（CQE 错误、链路断开）
- **Operator Retry**：算子级重试机制，在可恢复故障时自动重执行失败的集合通信算子

---

## 2. 目录结构

```
src/framework/cluster_maintenance/
├── detect/
│   └── detect_connect_anomalies/
│       ├── detect_connect_anomalies.cc    # 连接异常检测（28KB）
│       └── detect_connect_anomalies.h     # 检测器接口（6.7KB）
├── health/
│   └── heartbeat/
│       ├── heartbeat.cc                   # 心跳实现（91KB，最大文件）
│       ├── heartbeat.h                    # 心跳接口（18KB）
│       ├── reference_map.h                # 引用计数 map
│       └── ring_buffer.h                  # 环形 buffer（心跳数据存储）
└── recovery/
    └── operator_retry/
        ├── opretry_agent.cc/.h            # 重试 Agent（58KB）
        ├── opretry_base.cc/.h             # 重试基类（37KB）
        ├── opretry_server.cc/.h           # 重试 Server（55KB）
        ├── opretry_connection.cc/.h       # 重试连接管理（25KB）
        └── opretry_manager.cc/.h          # 重试管理器（15KB）
```

---

## 3. 核心文件说明

| 文件 | 说明 |
|------|------|
| `heartbeat/heartbeat.cc` | 最大文件（91KB）：心跳广播线程、状态机（OK/LOST/STUCK/INCONSISTENT）、超时检测、CQE 错误上报 |
| `heartbeat/heartbeat.h` | `HeartBeatStatus` 枚举、`UIDType`（512 字节 ID）、常量：`BROADCAST_INTERVAL=50ms`、`STUCK_INTERVAL=300000ms` |
| `detect/detect_connect_anomalies.cc` | 监听 ibverbs CQE 错误事件，检测链路异常，触发心跳状态更新 |
| `recovery/opretry_agent.cc` | 重试 Agent：处理各种重试状态（RUNNING/RETRY_FAIL/RESP_AICPU_ERR/RESP_STREAM_STOPED 等），通过 `CreateOpRetryAgentByState()` 工厂函数创建 |
| `recovery/opretry_server.cc` | 重试 Server：协调所有 rank 的重试决策，确保全局一致性 |
| `recovery/opretry_manager.cc` | 重试管理器：对外接口，决定是否触发重试、管理重试次数上限 |

---

## 4. 核心函数 / 类 / 接口

### Heartbeat

```cpp
// 心跳状态枚举
enum class HeartBeatStatus {
    OK,
    LOST,                  // 心跳丢失（rank 可能宕机）
    NOTIFY,                // 收到异常通知
    CQE_ERR,               // ibverbs CQE 错误
    OPRETRY_NOT_SUPPORT,   // 不支持重试
    STUCK,                 // 算子卡住（超过 STUCK_INTERVAL=5min）
    INCONSISTENT,          // rank 间状态不一致
};

class Heartbeat {
    // 启动心跳广播线程（间隔 BROADCAST_INTERVAL=50ms）
    HcclResult Start(u32 rank, u32 rankSize);

    // 查询指定 rank 的心跳状态
    HeartBeatStatus GetStatus(u32 rank) const;

    // 注册状态变化回调
    void RegisterStatusCallback(HeartBeatStatusCallback cb);

    // 512 字节的唯一 ID（用于区分不同通信域的心跳）
    struct UIDType { char id[512]; };
};
```

### 连接异常检测

```cpp
class DetectConnectionAnomalies {
    // 常量
    static constexpr u32 DEST_MAX_LEN         = 128;
    static constexpr u32 MAX_WHITE_LIST_ENTRY  = 16;  // 白名单最大条目数
    static constexpr u32 ACCEPT_TIME_OF_USLEEP = 100000;  // 100ms
    static constexpr u32 CLIENT_TIME_OF_USLEEP = 1500000; // 1.5s 客户端轮询间隔

    // 单例访问（每逻辑设备一个实例）
    static DetectConnectionAnomalies& GetInstance(s32 deviceLogicID);

    // 初始化（RegisterRanksToDca 中调用）
    void Init(std::vector<RankInfo>& rankInfos, bool isNeedNic);

    // 加入待检测的 IP 对（NIC 或 VNIC）
    void AddIpQueue(RankInfo& localRankInfo, RankInfo& remoteRankInfo,
        NicType nicType, s32 deviceLogicId);

    // 发起异常检测
    HcclResult Detect();

    // 释放资源
    void Deinit();
};

// 内部检测流程：
// DetectMonitor() 线程每 10s 广播一次
//   CreateServers() → 监听本端 socket
//   CreateClients() → 对每个 remoteRank 建立连接
//   processWhiteList() → 验证连接来源合法性
//   GetStatus() → 轮询 socket 状态
//   ProcessDetectionResults() → 报告 PrintDetectInfo() 错误信息
```

### Operator Retry

```cpp
// 重试 Agent 工厂（根据当前状态创建对应 Agent）
std::shared_ptr<OpRetryAgent> CreateOpRetryAgentByState(
    OpRetryState state,
    std::shared_ptr<OpRetryContext> ctx);

// 重试状态枚举
enum class OpRetryState {
    AGENT_RUNNING,
    AGENT_RETRY_FAIL,
    RESP_AICPU_ERR,
    RESP_STREAM_STOPED,
    // ...
};

class OpRetryManager {
    // 判断当前错误是否可重试
    bool IsRetryable(HcclResult errorCode) const;

    // 触发算子重试（协调所有 rank）
    HcclResult TriggerRetry(const std::string& tag, u32 retryCount);

    // 最大重试次数（默认 3 次）
    static constexpr u32 MAX_RETRY_COUNT = 3;
};
```

---

## 5. 数据流向

```
正常运行
  Heartbeat 线程（每 50ms）
    └── 广播心跳包（含 UIDType + 状态）
          └── 所有 rank 接收并更新 HeartBeatStatus

异常检测
  DetectConnectAnomalies
    └── 监听 ibverbs CQE 错误事件
          └── OnAnomalyDetected() → 更新 HeartBeatStatus = CQE_ERR

故障恢复
  算子执行失败
    └── OpRetryManager::IsRetryable()
          ├── 可重试 → OpRetryServer 协调所有 rank
          │     └── CreateOpRetryAgentByState() → 各 rank 执行重试
          └── 不可重试 → 上报错误，终止训练
```

---

## 6. 关键业务逻辑

### 心跳状态机
心跳状态按严重程度递增：`OK → LOST → STUCK → INCONSISTENT`。一旦进入 `STUCK`（超过 5 分钟无响应），触发 `OpRetryManager` 尝试恢复；若恢复失败，上报 `INCONSISTENT` 并终止。

### CQE 错误处理
ibverbs CQE（Completion Queue Entry）错误表示 RDMA 传输失败。`DetectConnectAnomalies` 监听 ibverbs 异步事件通道，检测到 CQE 错误后立即更新心跳状态为 `CQE_ERR`，触发重试流程。

### 重试一致性保证
`OpRetryServer` 确保所有 rank 在同一时刻进入重试状态（通过 barrier 同步），避免部分 rank 重试、部分 rank 继续执行导致的死锁。

### 环形 Buffer（ring_buffer.h）
心跳数据存储在环形 buffer 中，固定大小，新数据覆盖旧数据。用于保存最近 N 次心跳的历史记录，供故障诊断使用。

---

## 7. 开发注意事项

- `heartbeat.cc` 是 91KB 的大文件，修改时注意状态机的完整性，不要破坏状态转换逻辑。
- `STUCK_INTERVAL=300000ms`（5 分钟）是硬编码常量，生产环境中可能需要根据集群规模调整。
- `OpRetryManager` 的重试次数上限（`MAX_RETRY_COUNT=3`）超过后会直接终止，不会无限重试。
- `DetectConnectAnomalies` 依赖 ibverbs 异步事件通道，在纯 TCP 模式下不工作，需要有 fallback 机制。
- 重试期间所有 rank 必须同步进入重试状态，任何 rank 的超时都会导致整体重试失败。

---

## 8. 未来可扩展点

- **可配置超时**：将 `BROADCAST_INTERVAL`、`STUCK_INTERVAL` 改为运行时可配置参数（环境变量或配置文件）。
- **细粒度重试**：当前重试粒度是整个算子，可细化到 ring 中的单个 step，减少重试开销。
- **故障预测**：基于心跳历史数据（ring_buffer）做趋势分析，在故障发生前预警。
- **多级重试策略**：增加重试策略插件接口，支持指数退避、随机延迟等不同策略。
