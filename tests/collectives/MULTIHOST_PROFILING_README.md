# Standalone Collectives Multi-Host Profiling README

This README is a runbook for humans and AI agents. It explains how to run TileXR standalone collective profiling on two hosts, generate `trace.json`, and verify that the trace is useful in `https://ui.perfetto.dev`.

## Goal

Use `tilexr_collective_perf` through `run_collective_perf_multihost.sh` to profile standalone collective kernels across hosts.

Supported profiling targets:

- `allgather`
- `allreduce`
- `reducescatter`
- `alltoall`
- `broadcast`
- `profile-probe`

Expected root-level outputs in each profile directory:

- `trace.json`: Perfetto-compatible trace. Upload this to `https://ui.perfetto.dev`.
- `perfetto_trace.json`: Same Perfetto-compatible content, kept for backward compatibility.
- `trace_index.json`: Structured report input and diagnostics.
- `report.html`: Human-readable timeline and rank summary.
- `analysis.md`: Text summary.
- `ai_prompt.md`: Optional AI analysis prompt when `--profile-ai-prompt 1` is used.
- `multihost_rank<N>.log`: Rank launch logs.
- `plog/`: CANN/runtime logs.

Checked-in example output:

- `tests/collectives/examples/multihost-profiling/fine-allgather-big/trace.json`: real aggregate `trace.json` from a two-host `allgather` profiling run. Use it to inspect the expected Perfetto schema without rerunning the benchmark.

## Mental Model

The helper script starts one rank per peer over SSH. Rank 0 also acts as the socket bootstrap server through `TILEXR_COMM_ID`.

Each rank writes local launch traces under:

```text
<profile_dir>/rank<N>/launch<M>/trace.json
```

After all ranks finish, the helper copies rank traces back to the launcher host and writes an aggregate root trace:

```text
<profile_dir>/trace.json
```

Perfetto event names include launch, rank, host, and stage:

```text
launch0/rank0@141.62.24.62/kernel_total
launch0/rank1@141.62.24.70/flag_poll_wait
```

The script measures remote wall-clock offsets and passes `TILEXR_PROFILE_CLOCK_OFFSET_NS` to each rank. This compensates for host time skew, including cases where one machine has an incorrect system clock. Kernel stages are still derived from device-side timestamps per rank; use the aggregated timeline to compare stage shape and relative rank behavior, not as proof of globally synchronized device cycles.

## Required Environment

Assumptions used in examples:

```bash
REPO=/home/l00929943/TileXR
BUILD=$REPO/build-profile-950
PROF=$REPO/run/prof/collectives-2host
PEERS='0,root@141.62.24.62,141.62.24.62,0;1,root@141.62.24.70,141.62.24.70,0'
```

Peer format:

```text
rank,ssh_target,host_ip,device_id;rank,ssh_target,host_ip,device_id
```

Important environment variables:

```bash
export TILEXR_MULTIHOST_REMOTE_REPO_DIR="$REPO"
export TILEXR_MULTIHOST_PEERS="$PEERS"
export TILEXR_COMM_ID='141.62.24.62:10067'
export TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=300
```

`TILEXR_COMM_ID` must use a host/port reachable by all ranks. Avoid concurrent multi-host runs with the same port.

## Build On Every Host

Run this on every peer host before profiling:

```bash
cd "$REPO"
source /root/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pt311 2>/dev/null || true
source /usr/local/Ascend/cann/set_env.sh

cmake -S . -B "$BUILD" \
  -DTILEXR_BUILD_COLLECTIVES=ON \
  -DTILEXR_BUILD_TESTS=ON \
  -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON \
  -DBUILD_TESTING=OFF

cmake --build "$BUILD" --target \
  test_tilexr_collectives_kernel_ownership \
  test_tilexr_collectives_tools_sources \
  tilexr_collective_perf -j8
```

If `cmake` is not in PATH on a remote host, use the full path that exists in that environment, for example:

```bash
/root/anaconda3/envs/pt311/bin/cmake --build "$BUILD" --target tilexr_collective_perf -j8
```

Run guard tests on every host:

```bash
"$BUILD/tests/collectives/test_tilexr_collectives_kernel_ownership"
"$BUILD/tests/collectives/test_tilexr_collectives_tools_sources"
python3 "$REPO/tests/collectives/unit/test_collective_profile_report.py"
```

## Run One Profile Case

Launch from rank 0 host:

```bash
cd "$REPO/tests/collectives"

export TILEXR_MULTIHOST_REMOTE_REPO_DIR="$REPO"
export TILEXR_MULTIHOST_PEERS="$PEERS"
export TILEXR_COMM_ID='141.62.24.62:10067'
export TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=300

bash ./run_collective_perf_multihost.sh "$PROF/allgather-16m" "$BUILD/tests/collectives" \
  --op allgather \
  --min-bytes 16777216 --max-bytes 16777216 \
  --datatype int32 --check 0 \
  --warmup-iters 1 --iters 2 \
  --profile 1 --profile-sample-every 1 --profile-ai-prompt 1
```

When this command is embedded in a larger SSH heredoc script, add `</dev/null` to prevent the child script from consuming the parent script input:

```bash
bash ./run_collective_perf_multihost.sh "$PROF/allgather-16m" "$BUILD/tests/collectives" \
  --op allgather --min-bytes 16777216 --max-bytes 16777216 \
  --datatype int32 --check 0 --warmup-iters 1 --iters 2 \
  --profile 1 --profile-sample-every 1 --profile-ai-prompt 1 </dev/null
```

## Run All Standard Collective Cases

Use one profile directory per op. Multi-host runs should be serial because `TILEXR_COMM_ID` owns a fixed socket port.

```bash
cd "$REPO/tests/collectives"

run_case() {
  local name="$1"
  local op="$2"
  local bytes="$3"
  bash ./run_collective_perf_multihost.sh "$PROF/$name" "$BUILD/tests/collectives" \
    --op "$op" \
    --min-bytes "$bytes" --max-bytes "$bytes" \
    --datatype int32 --check 0 \
    --warmup-iters 1 --iters 2 \
    --profile 1 --profile-sample-every 1 --profile-ai-prompt 1 </dev/null
}

run_case fine-allgather-big allgather 16777216
run_case coarse-allreduce-big allreduce 16777216
run_case coarse-reducescatter-big reducescatter 16777216
run_case coarse-alltoall-big alltoall 16777216
run_case coarse-broadcast-128m broadcast 134217728
```

Optional smoke case:

```bash
bash ./run_collective_perf_multihost.sh "$PROF/fine-profile-probe" "$BUILD/tests/collectives" \
  --op profile-probe \
  --min-bytes 1048576 --max-bytes 1048576 \
  --datatype int8 --check 0 \
  --warmup-iters 1 --iters 2 \
  --profile 1 --profile-sample-every 1 --profile-ai-prompt 1 </dev/null
```

## Expected Stage Names

A useful aggregate trace should contain these stage suffixes:

```text
kernel_total
chunk_total
local_input_to_ipc
flag_poll_wait
peer_ipc_to_output
chunk_barrier
post_sync
window
```

Some stages may be absent for a specific rank role if a kernel path does not execute that stage on that rank. For the current two-host big-data collective smoke runs, all standard stage suffixes are expected in the aggregate trace.

## Verify A Profile Directory

Run this on the host that owns the aggregated profile directory:

```bash
python3 - "$PROF/allgather-16m" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "kernel_total",
    "chunk_total",
    "local_input_to_ipc",
    "flag_poll_wait",
    "peer_ipc_to_output",
    "chunk_barrier",
    "post_sync",
}

trace = json.loads((root / "trace.json").read_text(encoding="utf-8"))
names = {
    str(event.get("name", "")).rsplit("/", 1)[-1]
    for event in trace.get("traceEvents", [])
    if isinstance(event, dict)
}
missing = sorted(expected - names)

diagnostics = []
for rank_trace in root.glob("rank*/launch*/trace.json"):
    data = json.loads(rank_trace.read_text(encoding="utf-8"))
    diagnostics.extend(data.get("diagnostics", []))

print("trace:", root / "trace.json")
print("stages:", ",".join(sorted(names)))
print("missing:", ",".join(missing) if missing else "none")
print("diagnostics:", len(diagnostics))
raise SystemExit(1 if missing or diagnostics else 0)
PY
```

Expected result:

```text
missing: none
diagnostics: 0
```

Also check rank logs:

```bash
tail -n 30 "$PROF/allgather-16m"/multihost_rank*.log
```

Expected rank 0 perf line contains:

```text
errors
0
```

## View In Perfetto

Open:

```text
<profile_dir>/trace.json
```

Upload it to:

```text
https://ui.perfetto.dev
```

Useful searches:

```text
kernel_total
flag_poll_wait
rank0@
rank1@
launch0/
```

Interpretation hints:

- Compare `kernel_total` across ranks first.
- If one rank is slower, search for that rank label and inspect stage durations.
- Large `flag_poll_wait` often means a peer arrived late or synchronization dominated.
- Large `local_input_to_ipc` or `peer_ipc_to_output` points at data movement stage cost.
- Use `window` events to align each sampled launch.

## Pull Results To A Local Machine

Example from Windows PowerShell:

```powershell
scp -r root@141.62.24.62:/home/l00929943/TileXR/run/prof/collectives-2host D:\l00929943\TileXR\run\prof\
```

Open:

```text
D:\l00929943\TileXR\run\prof\collectives-2host\<case>\trace.json
D:\l00929943\TileXR\run\prof\collectives-2host\<case>\report.html
```

## Regenerate Aggregate Reports

If `rank*/launch*/trace.json` already exists:

```bash
python3 "$REPO/tests/collectives/tilexr_collective_profile_report.py" "$PROF/allgather-16m" \
  --warmup-iters 1 \
  --iters 2 \
  --profile-sample-every 1 \
  --emit-ai-prompt
```

This rewrites:

```text
trace.json
perfetto_trace.json
trace_index.json
analysis.md
report.html
ai_prompt.md
```

## Find Logs

The helper sets per-rank CANN log directories under the profile output:

```bash
find "$PROF" -name "plog"
```

Common non-fatal noise after successful runs:

- UDMA topology route fallback to EID 0.
- HCCP/RA deinit or unimport messages during process teardown.
- Driver share-log cleanup messages.

Treat these as residual runtime cleanup noise when rank logs show `errors=0`, `trace.json` exists, stage names are present, and diagnostics are zero.

## AI Agent Checklist

Use this checklist before saying the profiling run succeeded:

- Confirm the code was built with `-DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON` on every host.
- Confirm `TILEXR_MULTIHOST_PEERS` lists every rank exactly once.
- Confirm `TILEXR_COMM_ID` uses an unused port reachable from every host.
- Run multi-host cases serially, not concurrently.
- Confirm each case has root-level `trace.json`.
- Parse `trace.json` and verify expected stage suffixes.
- Parse every `rank*/launch*/trace.json` and verify diagnostics count is zero.
- Check `multihost_rank0.log` for the op summary and `errors=0`.
- If the profile was launched through a parent SSH heredoc, use `</dev/null` on each nested `run_collective_perf_multihost.sh` call.
- If a machine has a wrong system clock, rely on the helper clock offset path; do not manually edit trace timestamps.
- If a run fails, inspect `multihost_rank*.log` first, then `find "$PWD" -name "plog"`.

## Quick Troubleshooting

`cmake: command not found`

Use the full conda cmake path, for example `/root/anaconda3/envs/pt311/bin/cmake`.

`Address already in use` or socket bind failure

Change `TILEXR_COMM_ID` to a free port or wait for the previous rank 0 process to exit.

Run hangs

Check SSH connectivity in both directions, peer order, and whether a nested script consumed stdin. Use `</dev/null` for nested script calls.

Trace only shows `kernel_total`

Rebuild from a clean build directory or ensure CMake tracks included CCE/header dependencies. On hosts with incorrect future mtimes, old CCE objects can otherwise survive.

No root-level `trace.json`

Check that all ranks finished successfully and that the report helper ran. If rank launch traces exist, regenerate the aggregate report with `tilexr_collective_profile_report.py`.
