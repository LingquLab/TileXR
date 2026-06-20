# TileXR Collective Simulation Design

## Context

TileXR currently has three related but separate surfaces:

- `src/collectives` implements standalone collective APIs and kernels.
- `tests/collectives` provides correctness, performance, profiling, CSV output, and HTML reports for real hardware runs.
- `op-simulator` supports operator and AICore simulation, not topology-level collective network simulation.

The next algorithm work needs a no-hardware tool for choosing collective communication algorithms across different
topologies. The first target topology is a 1D-Clos cluster:

- 8 devices per server.
- Intra-server devices are connected by full-mesh P2P links.
- Intra-server P2P bandwidth approaches 50 GB/s but depends on message size.
- Each device has an uplink to the Clos network with bandwidth approaching 300 GB/s, also message-size dependent.
- The Clos network has two switch layers.
- 64P can be configured as non-blocking.
- 128P and larger configurations are expected to be oversubscribed.
- The initial oversubscription mode is 2:1.
- Many-to-one traffic can create congestion and reduce effective bandwidth.
- The evaluation range should eventually reach 1024P.

This design adds an independent offline collective simulation and visualization tool. It should reuse the existing
collective message semantics as reference behavior, but it should not be embedded into the real-hardware performance
runner or the AICore operator simulator.

TileXR collective communication has an additional data-placement constraint: communication operations act on TileXR
communication buffers, not arbitrary user tensors. DataCopy paths must have at least one endpoint in a communication
buffer, and UDMA paths follow the same constraint. Algorithm simulations therefore need to model local copy-in and
copy-out stages explicitly instead of treating a user input buffer as a direct network endpoint.

## Goals

- Model collective algorithms as a generic scheduling DAG, not as hard-coded AllGather, AllReduce, or ReduceScatter
  implementations.
- Use AllGather as the first example algorithm and validation scenario.
- Provide static DAG validation and small-scale semantic correctness checks before performance simulation.
- Support a parameterized 1D-Clos topology model suitable for 64P, 128P, and future 1024P studies.
- Include message-size dependent bandwidth and latency curves.
- Model communication-buffer placement constraints and the local copies needed to move data between user buffers and
  communication buffers.
- Support default configured curves and imported measured data for calibration.
- Run offline simulations that predict relative performance, bottlenecks, and algorithm ranking.
- Generate visual reports for algorithm selection, congestion diagnosis, and rank-level scheduling timelines.
- Keep the tool independent from production collective kernels and real-hardware measurement scripts.

## Non-Goals

- The first release does not implement a browser-based algorithm editor.
- The first release does not replace `tests/collectives/tilexr_collective_perf`.
- The first release does not promise accurate end-to-end wall-clock prediction without calibration.
- The first release does not modify existing collective kernels or host launch paths.
- The first release does not require explicit switch-port-level routing to be implemented, though the schemas should
  leave room for it.
- The first release does not model every AICore, SDMA, UDMA, or memory-system detail.

## Selected Approach

Add an independent offline simulator under a new tool directory such as:

```text
tools/collective_sim/
```

The simulator consumes a standard algorithm DAG, topology configuration, calibration data, and case definition. It
then validates correctness, runs a discrete-event simulation, and writes machine-readable results plus a static HTML
report.

The first complete example is AllGather on simplified 1D-Clos. AllGather is an example of the generic DAG model, not a
special case in the simulator core.

## Architecture

The tool is organized into these logical modules:

- `dsl/`: Python DSL helpers for generating algorithm DAG JSON. This is a producer only; the simulator core consumes
  standard JSON.
- `schema/`: JSON/YAML schemas for algorithms, topology, calibration data, case files, and results.
- `validator/`: Static DAG checks plus semantic execution on small symbolic inputs.
- `topology/`: Topology models. The first model is simplified 1D-Clos with shared bottleneck pools. The schema also
  reserves fields for explicit switch, port, link, and route models.
- `simulator/`: Discrete-event engine for dependencies, resource contention, transmission timing, queueing, and
  bottleneck attribution.
- `report/`: Static report generation for HTML, CSV, and JSON artifacts.
- `cli.py`: Command-line entry point for validation, single-case runs, sweeps, and report regeneration.

Data flow:

```text
Python DSL -> algorithm.json -> validator -> simulator -> result.json -> report.html
topology.yaml + calibration.yaml/csv + case.yaml --------^
```

This approach prioritizes reproducible algorithm descriptions and batch comparison. A later web app can reuse the same
algorithm and result JSON files.

## Algorithm DAG

The standard DAG JSON is the execution contract. It should be stable enough for Python DSL, YAML generators, future UI
tools, and tests to produce the same representation.

Core concepts:

- `buffers`: Logical data blocks owned by ranks. Buffers have roles such as `user_input`, `user_output`, and
  `comm_buffer`. For AllGather, symbolic blocks can be represented as `rank0.chunk0`, `rank1.chunk0`, and so on.
- `ops`: Scheduling nodes such as `send`, `recv`, `copy`, `reduce`, `wait`, and `barrier`. A future `compute` node can
  model local work.
- `deps`: Explicit dependencies between ops.
- `resources`: Optional resource hints or constraints such as intra-server P2P, cross-server uplink, Clos pool, local
  copy engine, or local reduce engine.
- `metadata`: Algorithm name, collective type, rank count, chunking strategy, rank mapping, and sweep parameters.

The simulator should not infer algorithm correctness from a collective type alone. It executes the DAG and records the
resulting data movement and resource usage.

## Communication Buffer Model

The DAG must represent TileXR communication buffers explicitly. Cross-rank communication cannot read directly from an
arbitrary user input buffer or write directly to an arbitrary user output buffer.

Buffer roles:

- `user_input`: User-visible source tensor or logical input block.
- `user_output`: User-visible destination tensor or logical output block.
- `comm_buffer`: TileXR communication buffer region, matching the role of communicator peer memory in runtime paths.
- `registered_comm_buffer`: Optional future role for memory made visible to UDMA-style paths when the implementation
  distinguishes it from the default IPC communication buffer.

Operation constraints:

- A cross-rank `send`, `recv`, `put`, or `get` op must use a communication buffer endpoint.
- A DataCopy-mode local or peer copy must have at least one endpoint in a communication buffer.
- A UDMA-mode transfer must also have a communication-buffer endpoint in this simulator model.
- Copy-in from `user_input` to `comm_buffer` and copy-out from `comm_buffer` to `user_output` are explicit `copy` ops
  with their own timing and dependencies.
- The AllGather example should start by copying each rank's local input chunk into a rank-local communication buffer,
  perform cross-rank movement between communication buffers, then copy gathered chunks to the user output view when
  the algorithm requires a separate output placement.

This model prevents the simulator from underestimating latency by skipping local movement. It also lets algorithm
variants compare different staging choices, such as copying once into a reusable communication buffer versus repeatedly
copying smaller chunks.

## Correctness Validation

Correctness is a first-class gate. Performance simulation should run only after validation passes by default.

Static validation checks:

- DAG is acyclic.
- All op ids, dependency ids, rank ids, buffer ids, and resource ids resolve.
- `send` and `recv` operations are compatible when the algorithm uses paired receive semantics.
- Reads happen after a buffer is defined.
- Writes do not silently overwrite live data unless the op explicitly allows it.
- Communication ops obey the communication-buffer endpoint constraint.
- Local copy-in and copy-out requirements are explicit when data crosses between user buffers and communication buffers.
- No duplicate op id or ambiguous buffer ownership exists.
- Rank count and topology rank count agree.
- Resource references are valid for the selected topology.
- Obvious dependency patterns that can deadlock are rejected or reported as warnings when they cannot be proven.

Semantic validation checks:

- The validator runs the DAG on small symbolic inputs without bandwidth timing.
- For AllGather, every rank starts with its own symbolic chunk and must end with every rank's chunk exactly once.
- Missing chunks, duplicated chunks, incorrect sources, and dependency ordering bugs are reported with the responsible
  op, rank, and buffer where possible.
- Small exhaustive cases should be supported first, for example 4 ranks and 1 to 2 chunks per rank.

Default CLI behavior should validate first:

```text
collective-sim validate algorithm.json --topology topo.yaml
collective-sim run case.yaml --validate
```

A `--skip-validation` flag may exist for low-level debugging, but it is not the default path.

## 1D-Clos Topology Model

The first implemented topology model is simplified 1D-Clos. It should capture the behavior needed for algorithm
ranking before exact switch routing is known.

Model requirements:

- `ranks_per_server` is configurable and defaults to 8.
- Intra-server traffic uses full-mesh P2P links.
- P2P link bandwidth is selected from a message-size curve with an upper asymptote of 50 GB/s.
- Cross-server traffic uses device uplinks and one or more Clos bottleneck pools.
- Uplink bandwidth is selected from a message-size curve with an upper asymptote of 300 GB/s.
- Configurations up to 64P can be marked non-blocking.
- Configurations from 128P onward can enable 2:1 oversubscription.
- Many-to-one congestion is represented by destination-oriented shared pools and fair sharing among active flows.
- Queueing delay and effective bandwidth are recorded per flow.

Example topology:

```yaml
topology:
  type: one_d_clos
  rank_count: 128
  ranks_per_server: 8
  intra_server:
    kind: full_mesh
    bandwidth_curve: p2p_50g
  uplink:
    bandwidth_curve: uplink_300g
  clos:
    mode: simplified_pool
    non_blocking_until_ranks: 64
    oversubscription:
      enabled_from_ranks: 128
      ratio: "2:1"
    congestion:
      model: destination_pool_fair_share
```

The topology schema should also reserve an explicit model for later:

- `nodes`: Devices and switches.
- `links`: Directed or bidirectional links with capacity curves.
- `routes`: Rank-pair or server-pair paths.
- `capacity_groups`: Shared bottleneck resources such as oversubscribed switch groups.

The first implementation does not need to compute routes through explicit Clos switches.

## Performance Model

The simulator is a discrete-event model. It advances through DAG dependencies and resource availability.

Transmission event timing:

```text
duration = startup_latency + message_bytes / effective_bandwidth
```

Effective bandwidth is derived from:

- Configured or calibrated message-size curve.
- Link or pool resource type.
- Current competing flows on the same resource.
- Oversubscription factor.
- Congestion policy, initially fair-share.

Resource selection:

- Same-server rank pairs use intra-server P2P.
- Cross-server rank pairs use source uplink, Clos bottleneck pool, and destination-side resources as represented by
  the simplified topology model.
- Local `copy` and `reduce` ops use configurable local throughput and latency parameters.
- SDMA-backed local movement defaults to an 800 GB/s bandwidth curve until measured calibration data overrides it.
- DataCopy-mode and UDMA-mode transfer ops are rejected by validation if neither endpoint is a communication buffer.

The result should preserve enough event detail for bottleneck explanation:

- op id
- rank
- source rank and destination rank when applicable
- message bytes
- start time and finish time
- wait time
- transfer time
- selected resources
- effective bandwidth
- bottleneck resource
- data source classification: `estimate`, `calibrated_estimate`, or `measured`

The first release should optimize for explainable relative ranking and bottleneck diagnosis. Absolute prediction error
is expected until calibration data exists.

## Calibration

Calibration supports two modes:

- Default parameterized curves in YAML.
- Imported measured CSV that can override or fit curves for a specific link or operation family.

Default curves should describe startup latency and bandwidth as a function of message bytes. The curve can be
piecewise-linear or table-based in the first release.

Example curve idea:

```yaml
calibration:
  curves:
    p2p_50g:
      kind: table
      points:
        - {bytes: 1024, gbps: 5}
        - {bytes: 1048576, gbps: 35}
        - {bytes: 67108864, gbps: 50}
      startup_latency_us: 2.0
    uplink_300g:
      kind: table
      points:
        - {bytes: 1024, gbps: 10}
        - {bytes: 1048576, gbps: 120}
        - {bytes: 67108864, gbps: 300}
      startup_latency_us: 3.0
    sdma_800g:
      kind: table
      points:
        - {bytes: 1024, gbps: 40}
        - {bytes: 1048576, gbps: 500}
        - {bytes: 67108864, gbps: 800}
      startup_latency_us: 1.0
```

Imported measured data can later come from microbenchmarks or from `tilexr_collective_perf` CSV outputs. Reports must
label whether a plotted number is estimated, calibrated, or measured.

## CLI and File Formats

Initial commands:

```text
collective-sim validate algorithm.json --topology topo.yaml
collective-sim run case.yaml --out run/clos-allgather
collective-sim sweep sweep.yaml --out run/clos-sweep
collective-sim report result.json --out report.html
```

Initial file types:

- `algorithm.json`: Standard DAG produced by Python DSL or other generators.
- `topology.yaml`: Simplified 1D-Clos topology or future explicit topology.
- `calibration.yaml`: Default latency and bandwidth curves.
- `calibration.csv`: Optional measured data import.
- `case.yaml`: A single run definition binding algorithm, topology, calibration, rank count, and message sizes.
- `sweep.yaml`: Batch definition for multiple algorithms, rank sizes, message sizes, and topology parameters.
- `result.json`: Full validation and simulation output.
- `summary.csv`: Flat summary for external analysis.
- `report.html`: Static visual report.

## Reports and Visualization

The first release generates offline static reports. It does not run a dashboard server.

Default output for a run:

```text
result.json
summary.csv
report.html
```

Primary view: algorithm selection dashboard.

- Compare predicted latency, algorithm bandwidth, bus bandwidth, and utilization across algorithms.
- Sweep over message size and rank size.
- Show best algorithm per case and the dominant reason other algorithms lose.
- Keep `estimate`, `calibrated_estimate`, and `measured` values visually distinct.

Drilldown view: congestion and resource usage.

- P2P link utilization.
- Uplink utilization.
- Clos pool utilization.
- Queueing delay.
- Oversubscription impact.
- Many-to-one destination hot spots.

Drilldown view: rank scheduling timeline.

- One row per rank.
- Bars for `send`, `recv`, `copy`, `reduce`, and `wait`.
- Critical path highlighting.
- Dependency wait attribution.
- Bottleneck resource annotation.

The report should be useful without external services, similar in deployment style to the existing collectives
profiling HTML artifacts.

## AllGather Example

The first example directory should be small and reproducible, for example:

```text
tools/collective_sim/examples/allgather_1d_clos/
```

Example contents:

- Python DSL source for at least one AllGather algorithm.
- Generated `algorithm.json`.
- `topology_64p.yaml` for a non-blocking configuration.
- `topology_128p_2to1.yaml` for an oversubscribed configuration.
- `calibration.yaml` with default P2P, uplink, and SDMA curves.
- `case.yaml` for a single run.
- `sweep.yaml` for message-size and rank-size comparison.
- Expected validator pass/fail fixtures.

The AllGather semantic validator should prove that every rank receives all symbolic chunks for small cases before any
performance result is trusted. The example should include the local copy-in and copy-out stages required by the
communication-buffer model.

## Tests

Schema tests:

- Reject malformed DAG, topology, calibration, case, and sweep files.
- Produce actionable error messages for invalid rank ids, missing resources, and unsupported topology modes.

Validator tests:

- A valid AllGather DAG passes static and semantic validation.
- Missing receive, missing dependency, illegal rank, repeated write, undefined buffer, and cyclic dependency fixtures
  fail with clear diagnostics.
- Communication ops that read from a user buffer and write directly to a remote user buffer fail validation.
- DataCopy or UDMA ops without a communication-buffer endpoint fail validation.
- Small AllGather symbolic cases cover at least 4 ranks and multiple chunk counts.

Simulator tests:

- Single-link transfer duration follows the configured curve.
- SDMA-backed local copy duration follows the default 800 GB/s curve.
- Multiple simultaneous flows on one resource use fair-share behavior.
- Many-to-one congestion increases queueing or lowers effective bandwidth.
- 64P non-blocking and 128P 2:1 oversubscription produce different bottleneck attribution.
- Cross-server traffic uses uplink and Clos resources; same-server traffic uses P2P.

Report tests:

- `result.json` can generate `summary.csv` and `report.html`.
- Summary rows include latency, bandwidth, bottleneck, validation status, rank count, message bytes, algorithm id, and
  data source classification.
- HTML contains the dashboard, congestion drilldown, and timeline drilldown sections.

Regression fixtures:

- Keep a minimal AllGather 1D-Clos example in source control.
- Use deterministic inputs and deterministic scheduling tie-breaks so result snapshots are stable.

## Integration Boundaries

The simulator should read existing TileXR conventions but remain isolated:

- It may use the collective message-size semantics documented and tested under `tests/collectives`.
- It should not call real ACL, CANN, or NPU runtime APIs.
- It should not depend on compiled `libtile-comm.so` or `libtilexr-collectives.so`.
- It should not be implemented inside `op-simulator`, because that directory is for operator/AICore simulation.
- It should not require hardware to run tests.

Future integration points:

- Import `tilexr_collective_perf` CSV as measured comparison data.
- Export a compact case description that can later drive real hardware validation.
- Use real profile reports to refine local copy, reduce, SDMA, or UDMA timing models.

## Open Extension Points

These are intentionally deferred but accounted for in the design:

- Explicit two-layer Clos switch and route modeling.
- Additional collective semantics beyond AllGather, including AllReduce, ReduceScatter, Broadcast, AllToAll, and custom
  collective DAGs.
- Local compute, reduce, and memory-copy engine contention.
- Routing policies beyond fair-share destination pools.
- Calibration fitting and confidence intervals.
- Interactive browser-based topology and DAG editor.
- Hardware validation orchestration against `ssh blue` or another NPU host.

## Acceptance Criteria

The design is considered implemented when the first tool version can:

- Generate or load an AllGather DAG.
- Represent user buffers, communication buffers, and local copy-in/copy-out stages explicitly in the DAG.
- Validate the AllGather DAG statically and semantically on small symbolic cases.
- Reject communication DAGs that violate the communication-buffer endpoint constraint.
- Load simplified 1D-Clos topology and calibration files.
- Simulate at least one non-blocking 64P case and one 128P 2:1 oversubscribed case.
- Include SDMA-backed local copy timing with an 800 GB/s default curve.
- Produce `result.json`, `summary.csv`, and `report.html`.
- Show algorithm selection summary, congestion bottlenecks, and rank timeline drilldown in the report.
- Run its unit and example tests without Ascend hardware.
