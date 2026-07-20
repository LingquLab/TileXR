# Full-Mesh Isolated Task Profiling Design

## Goal

Measure the execution cost of each full-mesh task without synchronization
waits so the 2 ms full-pipeline latency can be decomposed into independent
copy, UDMA, control, and receive costs.

## Interface

`TILEXR_DEMO_BIGDATA_ISOLATED_TASK` selects one diagnostic task. Zero keeps
the existing full pipeline unchanged.

| Value | Task | Active cores |
|---:|---|---|
| 1 | self-copy | 0..15 |
| 2 | peer-copy and copyDone store | 0..15 |
| 3 | copy-ready stores without copyDone waits | 0 and 15 |
| 4 | remote payload put and quiet | 16 |
| 5 | primary segmentDone store | 16 |
| 6 | ready signal and quiet | 16 |
| 7 | secondary zero-payload segmentDone store | 17 |
| 8 | local payload put-signal and quiet | 18 |
| 9 | output-copy | 19..34 |
| 10 | recvDone store | 19..34 |
| 11 | direct registered-memory ACK store | 34 |

## Execution Rules

- Physical 2x8 and the existing 16:0 split remain unchanged.
- No isolated task calls `BigDataWaitTokenMte` or `BigDataWaitCopyReady`.
- UDMA tasks retain separate `data-put` and `quiet` spans.
- Task 6 retains the signal API's required quiet.
- Task 11 writes ACK directly to the remote registered control slot.
- Every launch still uses the 35-core exit barrier so inactive cores exit
  coherently.
- Isolated mode always skips output and CQ-result validation.
- Full-mesh raw tracing remains the source of task timing.

## Validation

Build with `/home/pkg/b101/cann`, run layout and converter tests, then run all
eleven task values on physical 2x8 with repeat50. Require all ranks to finish,
all traces to contain no wait phases, and summarize iteration49 per task.
