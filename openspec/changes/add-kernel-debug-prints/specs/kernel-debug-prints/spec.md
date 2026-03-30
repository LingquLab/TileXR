## ADDED Requirements

### Requirement: Print basic comm context info
The kernel SHALL print `rankId`, `rankNum`, and `winSize` from `HcclA2CombineOpParam` during `Init()` when `GetBlockIdx() == 0`.

#### Scenario: Kernel launch with valid context
- **WHEN** `AllGatherMatmulFullMesh::Init()` is called on block 0
- **THEN** system prints `[cwh] rankId[X] rankNum[Y] winSize[Z]` where X, Y, Z are the actual values from `winContext_`

### Requirement: Print P2P window addresses
The kernel SHALL print `windowsIn[i]` and `windowsOut[i]` for each rank `i` from 0 to `rankNum-1`.

#### Scenario: Two-rank single-server setup
- **WHEN** `rankNum == 2` and both ranks have P2P windows mapped
- **THEN** system prints two lines: `[cwh] windowsIn[0]=0x... windowsOut[0]=0x...` and `[cwh] windowsIn[1]=0x... windowsOut[1]=0x...` with non-zero addresses

#### Scenario: P2P window not mapped
- **WHEN** `windowsIn[peer]` is zero for a peer rank
- **THEN** system prints `[cwh] windowsIn[peer]=0x0 windowsOut[peer]=0x...` indicating missing P2P mapping

### Requirement: Print ibverbs data pointer
The kernel SHALL print the `data` pointer and `dataSize` field from `HcclA2CombineOpParam`.

#### Scenario: ibverbs transport enabled
- **WHEN** `SetCommResource` has filled `commLevel0Rdma` data via `SetDevIbverbsData()`
- **THEN** system prints `[cwh] data[0xNNNN...] dataSize[Z]` where `data` is non-null and `dataSize > 0`

#### Scenario: ibverbs transport not enabled
- **WHEN** `commLevel0Rdma` was not created or `SetDevIbverbsData()` was not called
- **THEN** system prints `[cwh] data[0x0] dataSize[0]` or `data[(nil)]`

### Requirement: Output searchable via plog
All debug prints SHALL use the `[cwh]` prefix to enable filtering via `plog_grep.sh cwh`.

#### Scenario: User searches plog after test run
- **WHEN** user runs `bash plog_grep.sh cwh` after `ops_only_run.sh`
- **THEN** all kernel debug prints appear in the output with `[cwh]` prefix
