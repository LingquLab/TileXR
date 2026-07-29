# A5 Dispatch 可达性裁剪设计

## 目标

精简 `reference/moe_distribute_dispatch_v2_full_mesh_3510_simplified.h`，仅保留
A5 `Init()` 和 `Process()` 流程可能使用的代码，同时保证所有受支持路径的行为不变。

## 范围

- 以 `Init()` 和 `Process()` 作为调用图根节点。
- 除下文明确固定的特性输入外，所有模板实例和运行时分支均视为可能生效。
- 删除不可达成员函数的类内声明和类外定义。
- 删除函数裁剪后没有引用的类成员变量。
- 删除函数和成员裁剪后没有引用的文件级常量。
- 保持现有 `#include` 指令不变。
- 保留公开类型别名、构造函数以及 `Init()`、`Process()` 的受支持行为。

## 固定特性输入

简化后的 A5 参考代码采用以下固定输入：

- `zeroComputeExpertNum` 永远为 `0`。
- `isPerformance` 永远为 `false`。
- `hasElasticInfo` 永远为 `false`。

代码不再读取这些 tiling 字段，也不再保存对应类成员。依赖这些输入的条件将折叠到
固定分支，由此变为不可达的特性代码将被删除。

为了保持调用接口兼容，`Init()` 声明和定义中的 `elasticInfo`、`performanceInfo`
形参继续保留，但不再绑定 GlobalTensor，也不会在函数体中使用。

## 函数候选

应用固定特性输入前，当前词法调用图包含 49 个类外成员函数定义。从 `Init()` 和
`Process()` 出发的传递可达闭包包含 42 个定义。以下 7 个函数已经不可达：

- `AllToAllDispatchA3`
- `CalcBSTokenRange`
- `CalExpertSendNum`
- `SendBSExpertLoop`
- `SendToMoeExpertByBS`
- `SetExpertTokenNums`
- `SplitExpertNumToCore`

这些函数的类内声明和类外定义都将删除。

折叠三个固定特性输入后，以下 6 个函数也会变为不可达：

- `InitElasticInfo`
- `CalAndSendCntByExp`
- `RecordRankCommDuration`
- `GenerateGatherMaskTensor`
- `MaskZeroComputeExpert`
- `ZeroComputeExpertMaskCal`

分支折叠完成后会重新计算调用图。只有确认不属于 `Init()`、`Process()` 传递可达
闭包的函数才会继续删除。

## 分支折叠

实现时进行以下等价替换：

- Mask Buffer 的分配条件简化为 `isTokenMaskFlag_ || isExpertMaskFlag_`，删除
  zero 专家 Mask 初始化和计算。
- `CalCumSum()` 直接调用 `CalAndSendCntByRank()`。
- Rank 地址计算直接使用非扩缩容场景的 Rank ID，删除所有
  `isScalingDownFlag_` 重映射分支。
- LocalWindow 源数据索引条件简化为 `if (!isShareExpertRankFlag_)`。
- 删除性能 Buffer、计时调用和性能输出拷贝。
- `Init()` 保留兼容形参，但不创建弹性信息或性能信息 GlobalTensor。

## 成员变量候选

删除原有不可达函数后，以下 18 个私有成员不再有引用：

- `axisHExpandXAlignSize_`
- `cleanStatusTensor_`
- `cumSumTime1Tensor_`
- `cumSumTime2Tensor_`
- `cumSumTimes_`
- `cumSumUB_`
- `dealRankPerCore_`
- `delLastExpertId_`
- `flagPadOffset_`
- `gatherTmpTensor_`
- `maskSizePerExpert_`
- `remainderExpertNum_`
- `sharedTmpBufTensor_`
- `statusSumOutTensor_`
- `syncOnCoreTensor_`
- `tempTime1Tensor_`
- `tempTime2Tensor_`
- `tokenNumToExpertTensor_`

编辑后会重新进行引用分析。只有声明成为唯一剩余引用时，成员才会被删除。

固定特性输入还会使以下 12 个成员失去用途：

- `zeroComputeExpertNum_`
- `hasElasticInfoFlag_`
- `isScalingDownFlag_`
- `isPerformanceFlag_`
- `elasticInfoGMTensor_`
- `elasticInfoTensor_`
- `elasticInfoBuf_`
- `performanceInfoGMTensor_`
- `performanceInfoTensor_`
- `performanceFlagTensor_`
- `performanceInfoBuf_`
- `performanceFlagBuf_`

如果共享 Mask 或临时 Buffer 仍被 Token Mask、Expert Mask 或正常 Dispatch 路径
使用，则继续保留。

## 文件级常量候选

以下 4 个常量在可达代码中没有引用：

- `AIV_STATE_SIZE`
- `MIN_ACTIVE_BS_FOR_BS_MODE`
- `SFFVALUE_SIZE`
- `SYNC_OFFSET`

删除性能打点后，`DURATION_OFFSET` 也会失去引用。

只有在编辑后的引用扫描确认常量仅剩声明时，才会将其删除。

## 编辑方法

删除函数定义时使用大括号配平后的精确范围，不通过下一个函数签名推断删除边界，
避免误删相邻的模板声明和注释。函数声明、成员变量和常量使用局部补丁删除。

可达函数体只允许进行上文明确列出的固定输入分支折叠。其他可达代码不进行格式化
或重写。

## 验证标准

处理结果必须满足以下检查：

1. `Init()` 和 `Process()` 仍然存在，其可达调用闭包不存在缺失的类成员函数定义。
2. 每个保留的类外函数定义都有对应的类内声明。
3. 原有 7 个不可达函数和新增 6 个特性函数均不存在声明、定义或调用点。
4. `zeroComputeExpertNum_`、`hasElasticInfoFlag_`、`isScalingDownFlag_`、
   `isPerformanceFlag_` 不再出现。
5. `elasticInfo`、`performanceInfo` 仅作为 `Init()` 兼容形参存在，不在函数体中使用。
6. 已删除的成员变量和常量不存在残留引用。
7. 可达函数体中使用的成员标识符仍有对应类成员声明。
8. 预处理指令和大括号保持配平。
9. 文件保持无 BOM 的 UTF-8 编码和 LF 换行。

该参考头文件不参与当前 Windows 构建，并依赖外部 Ascend C 头文件，因此本阶段采用
结构化验证，不执行本地编译。
