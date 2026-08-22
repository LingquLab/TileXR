# MoonEP Model Replay Cache Design

## Goal

Add an opt-in `scripts/run_moonep.sh` mode that captures the communication-relevant
inputs and performance of a real TileXR MindSpeed model, replays one model iteration
as the ordered MoonEP operator cascade, and compares model and replay performance on
rank 0. The mode supports arbitrary validated `S/K/H/EP/R` shapes and can check in
a compact, privacy-sanitized route/performance bundle instead of raw captures.

## Interface

The existing case interface remains unchanged. The new path is selected with
`--model-replay` and accepts:

- `--s`, `--k`, `--hidden-size`, and `--ep`;
- the existing `--rank-size` as `R`;
- `--model-replay-from cache|meta|model` to select the first allowed source;
- `--model-replay-cache-dir` to override the ignored runtime cache root;
- the existing warmup and iteration controls for replay timing;
- an explicit profiler switch passed through to the model runner.

All five shape values must be positive integers and `EP == R` for the current model
runner. If a value is missing
and stdin is a terminal, the script prompts for it. A non-interactive invocation with
any missing value exits before environment setup or device launch. Existing numbered
cases and the six MoonEP method signatures consumed by `test_npu_e2e.py` are not
changed.

The default `cache` source falls through from runtime cache to compatible checked-in
meta and then model capture. `meta` bypasses runtime cache and falls through to model;
`model` bypasses both persisted sources and captures immediately. This single selector
is the only force/refresh interface.

## Captured Contract

For each rank and each of the ten forward planning calls, the adapter records:

- exact TopK indices as compressed little-endian int32 plus SHA256;
- `tokens_per_expert`, `remote_stats`, and `experts_to_copy`;
- rank, logical call, source call, capture ID, and shape;
- payload tensor shape, dtype, layout, stride, and deterministic regeneration seed.

Full hidden, route-weight, projection, and gradient payload values are regenerated.
They do not determine route ownership or communication volume, while persisting them
at production shapes would make the cache several GB. The manifest states this
boundary explicitly so a metadata-only cache cannot be mistaken for an exact numeric
model checkpoint.

One model invocation enables route and performance capture together. Model performance
artifacts contain stage, backend, kernel version, algorithm bytes, latency, and
applicable algorithm bandwidth, with all 55 operator records for every rank.
Profiler-on results are labeled and are not mixed with profiler-off throughput
evidence.

## Cache Identity And Layout

The cache key is the SHA256 of canonical JSON containing:

- cache and capture schema versions;
- `S/K/H/EP/R`, expert count, token padding, forward call count, and the 55-call model
  operator-order contract;
- dtype/layout and payload-regeneration contract;
- TileXR commit plus relevant runner and adapter hashes;
- model stack identity and model configuration hash;
- backend and kernel versions;
- CANN, driver, firmware, SoC, topology, host count, and rank-to-device mapping;
- profiler/capture contract versions.

Artifacts live below the ignored runtime path:

```text
run/moonep/model_replay_cache/<cache-key>/
  generation-<time>-<capture-id>/
    manifest.json
    model/performance.json
    model/rank_<rank>/...
    replay/route_replay.json
    replay/rank_<rank>/...
    complete
```

Reusable source bundles live below `tools/moonep/model_replay_meta`, or the root set by
`TILEXR_MOONEP_MODEL_REPLAY_META_ROOT`:

```text
tools/moonep/model_replay_meta/<case-id>/
  meta.json
  routes.u8.zst
  performance.json
```

`meta.json` globally deduplicates all rank/call TopK payloads by the SHA256 of the
original little-endian int32 bytes. Expert IDs are range-checked, concatenated as
uint8, and compressed with standard zstd level 3. Each of the ten calls still has an
ordered record that references a `route_id`; runtime materialization expands those
references back into the schema-1 `route_replay.json` consumed by existing code. E
greater than 256 is rejected rather than truncated.

Capture writes to a unique staging directory below the cache-key directory. Each rank
record is written by temporary-file replacement. The controller validates every
expected rank and call, shape, capture ID, checksum, and plan field before writing the
manifest and completion marker, then publishes an immutable generation by atomic
directory rename. A `model` source run never removes a usable runtime generation;
readers select the newest valid generation. Readers reject incomplete, stale,
mismatched, corrupt, or schema-incompatible entries.

Meta publication also validates all files in a staging directory before a directory
swap. A compatible meta hit creates a complete runtime generation under `run/`; replay
consumers never read checked-in meta directly. Performance captured on an environment
whose full provenance differs from the current environment is labeled `checked-in
reference`, not a direct baseline. Host addresses, usernames, credentials, absolute
paths, PIDs, raw captures, and profiler directories are never included in a bundle.
On Linux, replacement uses an atomic directory exchange when the filesystem supports
`renameat2(RENAME_EXCHANGE)`; unsupported platforms retain the validated rollback path.
The loader rejects missing or unavailable CANN/driver/firmware/SoC/code provenance and
profiler/stage-barrier mismatches before classifying a performance baseline.

Checked-in data validation opens every bundle through the production loader. The
repository currently carries the 4K/K8/H7168 EP8/R8 and EP16/R16 shapes. Their compact
route tables contain 80/160 rank-call references and 40/80 unique payloads respectively;
the R16 bundle occupies 2,092,469 bytes. Round-trip acceptance compares all runtime
route semantics exactly while validating original-capture and materialized-meta
provenance independently, because a new generation must identify its actual source.

## Replay And Comparison

Replay executes the MindSpeed iteration order used by the validated model-flow oracle:
five initial forward layers, five recompute forward layers in reverse order, then five
backward dispatch/combine/reduce sequences. Planning output and saved plans are reused
at the same boundaries as the model. Payload tensors are generated deterministically
from cached descriptors and seeds.

All ranks write structured stage samples. Existing global report aggregation is
extended for ordered model-flow samples. Rank 0 prints model and replay values side by
side in model order, including stage, backend/native status, kernel version, algorithm
bytes, human-readable latency, applicable bandwidth, and rank median/min/max. It also
prints cache key, hit/meta-hit/miss-captured status, provenance, and profiler state.

## Failure Handling And Compatibility

- Capture/replay is disabled unless explicitly requested and adds no normal-model hot
  path work.
- Adapter capture is environment-controlled and does not alter return tuples, keyword
  arguments, async events, zero-copy aliases, or public plan fields.
- A failed capture leaves only a staging directory and cannot poison a prior cache.
- Aggregation requires all ranks. Missing or duplicate ranks and non-finite timings are
  errors, not partial success.
- No kernel or runtime change is included unless a hardware reproduction proves it is
  required for these shapes.

## Validation

Local tests cover CLI prompting and non-interactive errors, all three source starts,
cache key determinism, global route deduplication and exact round-trip, zstd corruption,
route-reference and uint8 bounds, atomic publication, all-rank completeness, 55-call
order, aggregation, and unchanged `test_npu_e2e.py` calls.

Hardware validation covers cache miss, cache hit, forced model capture, meta-only
materialization without a model launch, model capture,
cascade replay, and comparison for:

- `4096/8/7168/EP8/R8` and `4096/8/7168/EP16/R16`.

The `8192/16/3584` shapes are not covered by this checked-in bundle validation; the
known shared-QP/CQ frontier evidence remains documented separately and must not be
reported as a passing replay case. Validated performance runs keep the framework NPU
profiler and Dispatch/Combine stage barrier enabled on both model and replay. Kernel
compile-time DFX/profiling and runtime trace/dump diagnostics remain disabled.
