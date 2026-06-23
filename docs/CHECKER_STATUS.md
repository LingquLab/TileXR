# Checker Status

本文档整理当前 `checker` 的已实现能力、验证范围和仍未完成的工作。

## 定位

`checker` 的目标是在没有昇腾硬件环境的情况下，对集合通信和 EP 类算子做离线功能与时序正确性检查。

当前路线不是直接执行 AscendC AICore 指令级实现，而是：

1. 尽量保持生产算子源码不变。
2. 通过 checker 侧头文件、宏和 shim 编译生产头文件或 source-aligned 模型。
3. 抽象出通信、同步、pipe、窗口读写等事件。
4. 用软件 oracle、时序诊断和 manifest coverage gate 检查结果。
5. 输出可快速定位源码的报告。

## 已实现能力

### 工程隔离

- checker 开发在独立 worktree 中进行。
- 生产算子源码保持 checker-free。
- 已验证生产源码零 diff，覆盖路径包括：
  - `src/ep`
  - `src/collectives/kernels/91093/allgather_hierarchy_double_ring.h`
  - `src/collectives/kernels/allreduce_big_data.h`
  - `src/collectives/kernels/collectives.h`
  - `src/collectives/kernels/allreduce_quant.h`

### 基础 checker 执行

- 支持多 rank 软件模型。
- 支持基础 collective case：
  - `allgather`
  - `allreduce`
- 支持基础 correctness oracle。
- 支持 ordering diagnostics。
- 支持 server topology 字段：
  - `server`
  - `peer_server`
  - same-server / cross-server 报告。

### 已接入的生产/生产对齐算法

- `allreduce_big_data`
  - 生产 header trace。
  - 算法选择条件检查。
  - 输出 oracle。
  - 源码定位报告。
- `allgather_hierarchy_double_ring`
  - 生产 header trace。
  - 算法选择条件检查。
  - 输出 oracle。
  - 源码定位报告。
- `ep_dispatch`
  - source-aligned CPU oracle。
  - dispatch payload、assist、recv count、expert token nums 检查。
  - registered communication window 写事件。
- `ep_combine`
  - source-aligned behavior oracle。
  - combine 侧 registered communication window 读事件。
  - combine-side `READ_BEFORE_WRITE` 定位能力。

### 半自动 trace 接入

已支持以下 CLI：

- `--analyze-trace-source`
- `--event-trace-template-output`
- `--generate-trace-adapter`
- `--scaffold-trace-bundle`
- `--verify-trace-bundle`
- `--verify-installed-trace`
- `--verify-all-installed-traces`
- `--list-installed-traces`
- `--list-capabilities`

source analysis 当前能输出：

- checker-only header shim 形态。
- auto hook candidates。
- manual review actions。
- 可直接交给 `--validate-event-trace` 的 JSONL 事件模板。

### 头文件、宏和 shim 打桩

已支持自动或半自动抽象的事件源：

- `CpGM2GMPingPong`
  - 映射为 copy/read/write 类事件。
- `SetSyncFlag`
  - 映射为 `FLAG_STORE`。
- `WaitSyncFlag`
  - 映射为 `FLAG_WAIT`。
- `AscendC::SetFlag`
  - 映射为 `PIPE_SET`。
- `AscendC::WaitFlag`
  - 映射为 `PIPE_WAIT`。
- `AscendC::PipeBarrier`
  - 映射为 `PIPE_BARRIER`。
- raw `AscendC::DataCopy(dst, src, bytes)`
  - 在注册 range 内映射为 `READ` + `COPY` 或 `WRITE`。
  - 地址未注册时输出 unsupported diagnostic。
- contiguous/no-padding `AscendC::DataCopyPad`
  - 对 `blockCount * blockLen` 连续搬运建模。
  - stride/padding 等复杂语义输出 unsupported diagnostic，不静默通过。

### 抽象事件模型

checker 使用 JSONL 抽象事件模型。

每条事件包含：

- `id`
- `kind`
- `rank`
- `peer_rank`
- `server`
- `peer_server`
- `core`
- `pipe`
- `event_id`
- `buffer_role`
- `slot`
- `magic`
- `offset`
- `bytes`
- `allow_future_producer`
- `source_file`
- `source_line`
- `detail`

已支持：

- checker 执行后输出 `events.jsonl`。
- `--validate-event-trace` 直接校验外部、手写或生成的 JSONL 事件。
- `--event-trace-template-output` 从 source analysis 生成可校验事件模板。
- 保留输入事件 ID，便于从报告反查外部生成器或手写模型。

### Manifest coverage gate

installed trace manifest 支持 `required_event_coverage`。

`--validate-event-trace --adapter-name <name>` 会读取：

```text
tools/checker/installed_traces/<name>_trace_manifest.json
```

并检查 trace 是否覆盖 manifest 中固定的 `{kind, source_file, source_line}`。

如果缺失，会输出：

- `EVENT_COVERAGE_GAP`
- 缺失的 event kind。
- 缺失的源码路径和行号。
- 标准 `summary.txt`、`findings.json`、`checker_report.json`。

这个能力用于防止半自动 trace 漏掉关键源码事件但仍误报通过。

### 报告与问题定位

每次 checker run 输出：

- `summary.txt`
- `findings.json`
- `events.jsonl`
- `checker_report.json`

报告中已包含：

- top finding。
- rank / peer rank。
- server / peer server。
- core / pipe / event id。
- buffer role。
- slot。
- offset / bytes。
- source file / source line。
- source excerpt。
- event context。
- timeline。
- next action。

已支持的主要 finding 包括：

- `READ_BEFORE_COPY`
- `READ_BEFORE_WRITE`
- `FLAG_NO_PRODUCER`
- `FLAG_STALE_MAGIC`
- `DEADLOCK`
- `PIPE_WAIT_NO_PRODUCER`
- `DIRECT_PEER_USER_BUFFER`
- `OUTPUT_MISMATCH`
- `UNSUPPORTED_API`
- `EVENT_COVERAGE_GAP`

## 已验证内容

当前已通过：

- 干净构建 `tilexr_checker_all`。
- 干净构建 `tilexr_checker`。
- checker 测试：17/17 passed。
- `ep_combine` smoke：PASS。
- source analysis 生成 event template。
- event template 通过 `--validate-event-trace`。
- manifest coverage 缺失时触发 `EVENT_COVERAGE_GAP`。
- `--list-capabilities`。
- `--verify-all-installed-traces`。
- `git diff --check`。
- 生产算子源码零 diff。

## 当前边界

当前 checker 验证的是：

- 算法选择条件。
- 抽象通信事件。
- 抽象同步事件。
- 抽象 AICore pipe 事件。
- 输出 oracle。
- installed trace coverage。
- 源码定位报告。

当前 checker 还不是：

- AscendC AICore 指令级解释器。
- 真实硬件执行器。
- 真实 NPU latency 模型。
- 完整链路带宽/拥塞模拟器。
- 完整 C++ AST/预处理器分析器。

## 未完成内容

### 指令级与性能时序

- 还没有执行 AscendC AICore 指令级实现。
- 还没有真实 pipe latency 模型。
- 还没有 bank conflict、queue depth、DMA latency 等硬件细节模型。
- 还没有 link-level 性能 oracle。

### 多机链路模型

当前事件里有 `server` / `peer_server`，能区分 same-server 和 cross-server。

尚未实现：

- 多机链路带宽模型。
- 链路占用和竞争检查。
- link-level deadlock / livelock 检查。
- 多链路拓扑调度模拟。
- 并发 copy 对同一链路的冲突诊断。

### 更完整的 AscendC API 打桩

已支持一部分 `DataCopy` / `DataCopyPad`。

尚未完整支持：

- strided `DataCopyPad` 的精确语义。
- padding `DataCopyPad` 的精确语义。
- burst 间隔和多维搬运语义。
- `SetAtomic*` oracle。
- vector/local tensor 计算语义。
- 更复杂的 LocalTensor / GlobalTensor 行为建模。

### 自动生成能力

当前 source scan 是轻量词法扫描。

尚未实现：

- 完整 C++ AST 分析。
- 预处理后路径分析。
- 宏别名自动展开。
- 模板转发路径识别。
- 条件编译分支覆盖分析。
- 自动生成完整可执行 runner。

新生产算子接入仍需要人工确认：

- block schedule。
- operator `Init` / `Process` wiring。
- trace range 注册。
- peer memory 安装与恢复。
- oracle materializer。
- installed manifest。

### EP 后续

已支持 `ep_dispatch` 和 `ep_combine` 的 source-aligned 行为级检查。

尚未完成：

- `ep_combine` installed trace manifest。
- 更多 EP 算子的统一接入规范。
- EP 多机窗口生命周期完整模型。
- EP window reuse / stale header 更强诊断。
- EP 路由、反路由和跨 server 调度的更完整时序模型。

### 外部 trace 生态

已有 JSONL 输入和模板生成。

尚未完成：

- 外部 instrumentation 工具链。
- 从编译产物自动抽取 trace 的流程。
- trace schema versioning。
- trace 压缩和大规模多 rank trace 处理。
- trace diff 工具。
- trace minimizer。

## 推荐下一步

优先级建议：

1. 补 link-level timing model 的最小版本，先能报告 cross-server link contention 为 unsupported/inconclusive，再逐步加带宽模型。
2. 为 `ep_combine` 增加 installed trace manifest，让 EP combine 进入 `--verify-all-installed-traces`。
3. 增强 `DataCopyPad` strided/padded 语义，减少 manual review 面积。
4. 为 `SetAtomic*` 增加 explicit unsupported gate 或 op-specific oracle。
5. 把 source scan 从纯词法提升到预处理感知，至少识别常见宏包装和模板 helper 转发。
