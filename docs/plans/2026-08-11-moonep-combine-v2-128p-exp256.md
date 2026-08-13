# MoonEP Combine V2 128P ExpNum=256 Execution Plan

## Goal

Run the Combine V2 single-operator benchmark on cabinets 0 and 2 with 128
ranks, `BS=8192`, `K=16`, `H=3584`, BF16, and 256 experts. Report the
average and maximum of the 128 per-rank 80-iteration means, with equivalent
bandwidth for both times.

## Scope

- Add a runtime expert-count option to the C++ benchmark and Bash launchers.
- Accept the operator-supported world sizes through 128 ranks.
- Compile on cabinet 2 CPU1 (`141.61.52.35`).
- Flat-sync runtime artifacts from that server to all 16 cabinet 0/2 hosts.
- Launch ranks through direct SSH without MPI.

The expert count is benchmark metadata and a sharding contract. It is not an
argument to the Combine V2 operator and does not change the bandwidth payload,
which remains `BS * K * H * sizeof(BF16)` per rank.

## Implementation

1. Update `tests/moonep_combine_v2/demo/tilexr_moonep_combine_v2_hardware_probe.cpp`
   to parse `--experts`, validate `experts % world_size == 0`, and report the
   configured value.
2. Update `tools/moonep/run_combine_v2_perf_multihost.sh` to accept and forward
   `--experts`, validate 128P, and emit the configured value in final results.
3. Update `tools/moonep/run_combine_v2_perf_cluster.sh` and the performance
   guide with the same interface and semantics.
4. Use the cabinet mapping from `D:\3_codex\512P环境信息.txt` to construct the
   16-host, 128-rank hostfile in cabinet 0 then cabinet 2 order.
5. Sync source with Mutagen, build and stage the non-MPI runtime on
   `141.61.52.35`, then flat-sync artifacts to every host.
6. Run `warmup=20`, `iterations=80`, and preserve the full per-rank logs.

## Verification

- Source/build checks pass and the benchmark has no MPI dependency.
- Runtime SHA256 verification passes on all 16 hosts.
- NPU preflight follows the 15-second retry and 120-second maximum wait rule.
- The log contains 128 passing rank records and 10,240 timed samples.
- Final output contains `experts=256`, `avg_ms`, `avg_alg_bw_GBps`, `max_ms`,
  and `max_alg_bw_GBps`.
- No benchmark processes remain after completion.
