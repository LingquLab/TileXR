# CCU AlltoAll Loop Reuse Design

## Goal

Validate ten consecutive two-rank bidirectional CCU AlltoAll submissions while reusing one communicator, one installed mission, and the same QP, jetty, CKE, XN, and registered-memory resources.

## Round Protocol

The smoke probe prepares the long AlltoAll mission once. For each loop index from zero through nine, both ranks:

1. Fill the source buffer with a rank-and-loop-specific pattern.
2. Fill the destination buffer with a sentinel value.
3. Set SQE argument zero to a rank-and-loop-specific 64-bit marker.
4. Enter a loop-specific host ready gate using `phase=loopIndex`.
5. Submit the same prepared task and synchronize its stream.
6. Read the peer marker XN and require the exact marker for the current peer and loop.
7. Read the complete destination buffer and compare it with the current peer pattern.
8. Enter a loop-specific done gate with the local validation result.

Any failure terminates the loop and reports the loop index, mission context, XN/CKE state, marker value, and mismatch details.

## Device Marker

PreSync carries three variables on the same channel:

- peer receive address with CKE mask `0x2`;
- peer receive token with CKE mask `0x4`;
- current SQE loop marker with CKE mask `0x1`.

The receiver waits for mask `0x7`. The marker is loaded with `LoadSqeArgsToX`, so the installed instruction sequence remains unchanged while each submission supplies a distinct value. The marker uses a fixed magic prefix plus rank and loop index, allowing host readback to reject a signal from another loop.

The CKE bits remain presence flags and are cleared by the existing wait instruction. The loop-specific XN marker supplies the generation identity that the existing CKE protocol lacks.

## Configuration

`TILEXR_CCU_ALLTOALL_LOOP_COUNT` controls repeated submissions and defaults to one. Values outside `1..1024` fail parameter validation. The runner forwards the variable to both ranks.

## Verification

- Unit tests verify the marker instruction ordering, mask `0x7`, instruction capacity, default loop count, runner forwarding, and loop-specific phase gates.
- The remote hardware test uses NPU 6 and 7 with loop count ten.
- Success requires ten loop results per rank (twenty total), exact peer markers, and zero mismatches in every loop.
