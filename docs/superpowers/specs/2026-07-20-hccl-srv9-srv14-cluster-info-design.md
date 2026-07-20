# HCCL srv_9 + srv_14 Cluster Info Design

## Goal

Validate HCCL AllToAll between `srv_9` (`141.61.49.223`) and `srv_14`
(`141.61.49.192`) using configuration generated from the machines' actual
HCCL root information. Validate two ranks first, then the full physical 2x8
configuration.

## Constraints

- Remote writes are limited to `/home/h30059441` and `/tmp`.
- `/etc/hccl_rootinfo.json` and the installed CANN files are read-only inputs.
- The test copy must use the same CANN installation as the TileXR UDMA test.
- Do not invent physical topology edges that cannot be derived from an
  authoritative source.

## Configuration

Generate cluster-info files by combining rank entries from each host's
`/etc/hccl_rootinfo.json`:

- `ranktable_2x1.json`: device 0 from each server, assigned ranks 0 and 1.
- `ranktable_2x8.json`: all eight devices from `srv_9` as ranks 0 through 7,
  followed by all eight devices from `srv_14` as ranks 8 through 15.

Preserve each device's actual level list, EIDs, ports, and network instance
identifiers. The local hardware topology remains the vendor-provided
`/usr/local/Ascend/driver/topo/950/atlas_950_1.json`; no synthetic inter-server
edge graph is generated.

## Test Integration

Modify only the HCCL test copy under `/home/h30059441`. Add an explicit
cluster-info initialization path so the test consumes the generated ranktable
instead of relying on `HcclGetRootInfo` and MPI root-info exchange. Keep the
original path available for comparison.

Resolve and record the CANN environment used by the deployed TileXR UDMA test.
Build and run the HCCL test against that same installation. Use
`data0.3001` as `HCCL_SOCKET_IFNAME` on both hosts.

## Validation Sequence

1. Validate generated JSON structure and rank uniqueness without running NPU
   work.
2. Build the modified test copy on both hosts.
3. Run 2x1 on device 0 of each host with a small payload to validate channel
   creation and correctness.
4. Run 2x1 with 128 MiB input/output per rank.
5. Only if both 2x1 runs pass, run physical 2x8 with 128 MiB input/output per
   rank, five warmups, and fifty measured iterations.

Every run has an external timeout. A failure stops progression and preserves
the command, logs, generated configuration, and exit status under
`/home/h30059441` for diagnosis.

## Success Criteria

- The test log proves that cluster-info initialization consumed the intended
  generated file.
- All participating ranks initialize and complete without channel timeout.
- Data verification passes.
- The 2x8 run reports timing only after the 2x1 connectivity checks pass.
