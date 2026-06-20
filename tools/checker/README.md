# TileXR Checker

This directory holds the opt-in no-NPU checker surface for TileXR.

Scope:
- Build only when `TILEXR_BUILD_CHECKER=ON`.
- No NPU runtime dependency is expected for the checker build surface itself.
- Initial MVP coverage is source-guard smoke testing and checker-core scaffolding.
- Planned MVP operator coverage for later tasks is AllGather and AllReduce SUM on INT32.
- Report files for this task live under `.superpowers/sdd/`.
