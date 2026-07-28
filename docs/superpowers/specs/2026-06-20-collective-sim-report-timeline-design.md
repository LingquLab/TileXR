# Collective Simulator Interactive Report Design

## Context

The current `report.html` expands every simulated event into the `Congestion Drilldown` and `Rank Timeline` tables.
For the 64P plus 128P AllGather sweep this creates a 61 MB HTML file, which is too large to open comfortably and too
dense to inspect.

## Goals

- Keep the main `report.html` small enough to act as an algorithm-selection index.
- Let users choose a specific card/rank from the main report.
- Generate one HTML file per rank under `rank_reports/`.
- Render each rank file as an interactive timeline instead of a full static event table.
- Include both local events and network resources involving the selected rank.
- Keep all reports static, offline, and dependency-free.

## Selected Approach

Use a report bundle:

- `report.html` contains algorithm summary, top bottleneck resources, and a rank selector that opens
  `rank_reports/rank_XXX.html`.
- `rank_reports/rank_XXX.html` embeds only the events relevant to that rank and draws an SVG timeline with JavaScript.
- The CLI `run`, `sweep`, and `report` commands all write the same bundle shape.

An event is relevant to rank `N` when `event.rank == N` or one of its resource ids mentions that rank as an endpoint,
for example `sdma:rankN`, `uplink:src:N`, `uplink:dst:N`, or `p2p:s0:N->M` / `p2p:s0:M->N`.

## Timeline Interaction

Each rank report provides:

- A result selector for algorithm/message-size cases.
- Op-type checkboxes for `copy`, `send`, `recv`, `put`, and `get`.
- A zoom slider that changes the horizontal scale.
- An SVG timeline where each event is a bar from `start_us` to `end_us`.
- A detail table for the events currently selected by the controls.

## Testing

The report tests must prove:

- The main report has rank navigation and no expanded all-event drilldown table.
- Per-rank files are written for every rank in the results.
- Rank files contain the timeline controls and only a compact rank-scoped event payload.
- CLI commands generate `rank_reports/rank_000.html`.
