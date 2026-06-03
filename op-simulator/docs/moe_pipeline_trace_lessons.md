# MoE op-simulator Pipeline Trace Lessons

This note records the reusable lessons from the MoE combine comparison around
`ProcessMoeExpertsLoop() -> ProcessMoeExpert()`. The concrete case compared
`combine-pr-0601.h` with `moe_distribute_combine_v2_final.h`, but the workflow
is useful for other AscendC pipeline changes too.

## Scope Control

Keep the simulator question narrow before reading trace data.

- Isolate the function under test when possible. For this case, loop-only
  simulation skipped `WaitDispatch` flag polling so the trace focused on
  `ProcessMoeExpertsLoop()` and `ProcessMoeExpert()`.
- Record the modeled topology explicitly. `32P` and `64P` are card counts here,
  not total expert counts. Total experts stay fixed at 128, so the experts per
  rank are 4 and 2 respectively.
- Add cheap validation metadata to the runner. The rank histogram in this case
  caught the EP modeling error quickly by reporting `nonZeroRanks` and
  `expertsPerRank`.
- Compare only rows with successful simulator status. Interrupted or timeout
  rows can still be useful for debugging, but should not enter speedup math.

## Trace Reading Checklist

For `cannsim record` output, the useful file is normally:

```text
<case>/report/trace_core0.json
```

Use structured JSON parsing instead of text search; large traces can make plain
grep-style scans slow. At minimum, collect:

- Total event count and `ph == "X"` event count.
- Trace span: `max(ts + dur) - min(ts)` over complete events.
- Pipeline category duration, such as `RVECEX`, `RVECLD`, `RVECST`, `MTE2`,
  `MTE3`, `SCALAR`, and `PUSHQ`.
- Barrier count. In these traces, `PipeBarrier` appears as instant `BAR` events,
  with the pipe in `args.extra_info`, for example `PIPE:ALL`.
- Per-category gaps. A reduced total span with nearly unchanged compute
  category duration usually means the change removed bubbles or serialization
  rather than reducing arithmetic work.

`op-simulator/scripts/summarize_matrix.py` implements this checklist for a
matrix summary TSV and produces both JSON and Markdown.

## Pipe Wait Reasoning

The key code pattern was:

```cpp
PipeBarrier<PIPE_ALL>();
moeSumQueue_.FreeTensor<XType>(tmpUb);
```

The narrower dependency is:

```cpp
SyncFunc<AscendC::HardEvent::V_MTE2>();
moeSumQueue_.FreeTensor<XType>(tmpUb);
```

This fix is applied in the ops-transformer MoE combine kernel at
`mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.h` and in
the matching arch35 host-KFC implementation.

The reason is lifetime, not only performance. After `DeQue`, `tmpUb` is still
read by vector work such as cast, multiply, and add. If the queue tensor is
freed before vector has finished reading it, a later `AllocTensor` and MTE2
copy can reuse the same UB storage and overwrite data that vector still needs.
`PIPE_ALL` prevents this but stalls every pipe. `V_MTE2` protects the relevant
reuse path before allowing MTE2 to write into that UB storage again.

When replacing barriers:

- Identify the producer pipe and consumer pipe for the actual buffer lifetime.
- Wait for the consumer that still reads the buffer before freeing or reusing
  the buffer.
- Prefer the narrowest hard event that protects the data dependency.
- Keep correctness tests or simulator cases that can expose early reuse; pure
  timing traces do not prove correctness.

## Interpreting The MoE Result

In the loop-only simulator runs, the optimized version did not materially reduce
the main vector compute categories. The observed speedup came from removing
repeated `PIPE_ALL` barriers at the tail of `ProcessMoeExpert`.

The barrier count pattern was:

- Baseline: roughly `BS * TOP_K + 1` `BAR` events.
- Final: one remaining `BAR` event in the loop-only trace.

The span reduction tracked the reduction in pipeline gaps, while `RVECEX`,
`RVECLD`, and `RVECST` were nearly unchanged. That is the signature of removing
pipeline bubbles rather than making each vector operation cheaper.

## Small-BS Reality Check

Small batch real-hardware results can show no improvement or slight regression
even when loop-only simulator traces improve.

Common causes:

- Fixed end-to-end overhead dominates at small BS.
- Real execution includes `WaitDispatch`, flag polling and clearing,
  communication window state, and scheduler effects that the loop-only harness
  intentionally skipped.
- Small BS has less pipeline depth, so a broad barrier may not sit on the
  critical path often enough to matter.
- If the optimized file contains changes beyond the tail barrier, those changes
  may add fixed overhead that is visible at BS32.
- Replacing `PIPE_ALL` with a narrow `SyncFunc` can preserve correctness but
  still has a cost; measure it as a separate variant.

For a clean small-BS diagnosis, run a four-way comparison:

1. Original `PIPE_ALL`.
2. Remove only the tail `PIPE_ALL`.
3. Replace only the tail `PIPE_ALL` with `SyncFunc<V_MTE2>`.
4. Full optimized file.

This separates the benefit of removing broad serialization from unrelated
changes in the optimized implementation.

## Artifact Handling

For long remote simulator runs, keep the artifacts self-describing:

- Save one directory per case with `record.log` and `report/trace_core0.json`.
- Maintain a single `matrix_summary.tsv` with topology, variant, status,
  timing, trace path, and case directory.
- Copy results locally before summarizing if remote paths will not be stable.
- When transferring source files through environments that block suffixes, stage
  the file without its suffix first, then rename it on the remote side.

This keeps the trace comparison reproducible even after the temporary simulator
workspace is removed.
