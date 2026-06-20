# Collective Simulator Interactive Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace expanded all-event HTML tables with a small report index plus generated per-rank interactive timeline files.

**Architecture:** Extend `report.py` with bundle writers. The main report renders summaries and rank links. Per-rank reports embed filtered JSON and draw a static offline SVG timeline with inline JavaScript. The CLI calls the bundle writer from `run`, `sweep`, and `report`.

**Tech Stack:** Python 3 stdlib, unittest, static HTML/CSS/JS, JSON serialization.

## Global Constraints

- Do not add frontend package dependencies.
- Do not require a web server; reports must open directly from disk.
- Do not embed the full event list in the main `report.html`.
- Keep `write_html_report` available for existing callers.
- Unit tests must run without Ascend hardware.

---

## Task 1: Failing Report Tests

**Files:**
- Modify: `tests/collective_sim/test_report_cli.py`

**Interfaces:**
- Consumes existing `write_html_report`.
- Produces tests for `write_report_bundle`, `write_html_report_from_plain`, and CLI output.

- [ ] Add a test that calls `write_report_bundle([result], out)` and expects `out/report.html`, `out/rank_reports/rank_000.html`, and `out/rank_reports/rank_001.html`.
- [ ] Assert the main report contains `rankSelect` and `rank_reports/rank_000.html`.
- [ ] Assert the main report does not contain an expanded op id such as `send_r0_to_r1`.
- [ ] Assert `rank_000.html` contains `timelineSvg`, `zoomRange`, `resultSelect`, and `timeline-data`.
- [ ] Run `PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_report_cli -v` and verify the new tests fail because `write_report_bundle` is missing.

## Task 2: Report Bundle Implementation

**Files:**
- Modify: `tools/collective_sim/tilexr_collective_sim/report.py`

**Interfaces:**
- Produces `write_report_bundle(results: Sequence[SimulationResult], out_dir: Path) -> None`.
- Produces `write_report_bundle_from_plain(result_data: Any, out_dir: Path) -> None`.
- Preserves `write_html_report(results, path)` and `write_html_report_from_plain(result_data, path)`.

- [ ] Add helpers to collect rank ids from `rank_count`.
- [ ] Add helpers to decide whether an event is relevant to a rank by event rank or resource endpoint.
- [ ] Render `report.html` as a small index with algorithm summary, bottleneck summary, and rank selector.
- [ ] Render `rank_reports/rank_XXX.html` with embedded rank-scoped JSON and timeline JS.
- [ ] Run `PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_report_cli -v` and verify it passes.

## Task 3: CLI and Documentation

**Files:**
- Modify: `tools/collective_sim/tilexr_collective_sim/cli.py`
- Modify: `tools/collective_sim/README.md`

**Interfaces:**
- CLI `run`, `sweep`, and `report` write report bundles.

- [ ] Replace CLI calls to `write_html_report` with `write_report_bundle`.
- [ ] Replace CLI `report` regeneration with `write_report_bundle_from_plain`.
- [ ] Document `rank_reports/rank_XXX.html` in the README artifact list.
- [ ] Run the report CLI test and full collective simulator test suite.

## Task 4: Sample Regeneration and Verification

**Files:**
- Generated: `run/collective_sim/allgather_sweep/report.html`
- Generated: `run/collective_sim/allgather_sweep/rank_reports/rank_000.html`

**Interfaces:**
- Consumes the existing AllGather sweep fixture.

- [ ] Run the sweep command to regenerate the sample report bundle.
- [ ] Check the main report size is much smaller than the previous 61 MB file.
- [ ] Confirm a rank report exists and contains the interactive timeline controls.
- [ ] Review `git diff` and commit the implementation.
