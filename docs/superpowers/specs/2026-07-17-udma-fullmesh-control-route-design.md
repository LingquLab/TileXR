# UDMA Full-Mesh Control Route Design

## Goal

Fix the physical 2x8 full-mesh AllToAll path so payload completion signals do
not fall back to generic multi-route UDMA operations that can select a failing
route.

## Scope

- Keep the existing 12:4 payload split between cores 16 and 17.
- Bind the primary segment to the maximum-weight QP and the secondary segment
  to the minimum-weight QP.
- Send each segment payload and wait for quiet on its selected QP.
- Publish the final ready signal on the primary segment's selected QP.
- Publish receive ACKs by writing the peer's registered control slot directly,
  without a generic UDMA put or quiet.
- Do not import the XY pipeline, 16:0 split, tracing, or scheduling changes.

## Data Flow

Core 16 waits for shards 0 through 11, sends the primary payload on the
maximum-weight QP, and records local segment completion. Core 17 waits for
shards 12 through 15, sends the secondary payload on the minimum-weight QP,
and records local segment completion. Core 16 waits for both local completion
tokens and publishes ready on the same maximum-weight QP used for its payload.

After all receive-copy shards complete, the receiver writes the ACK token to
the sender's registered ACK control slot. The sender continues to wait on its
local registered ACK slot, so the synchronization contract is unchanged.

## Verification

1. Add source-structure tests that isolate the full-mesh send worker and check
   explicit QP payload, ready, and quiet operations.
2. Check that the full-mesh receive worker no longer calls the generic UDMA ACK
   helper and instead resolves the remote registered ACK slot.
3. Run the UDMA layout unit test locally and on both remote hosts.
4. Deploy only a committed Git bundle.
5. Run physical 2x8, 128 MiB per rank, repeat1, profile stage 8, with full
   output validation. Run repeat50 only if all 16 ranks pass.
