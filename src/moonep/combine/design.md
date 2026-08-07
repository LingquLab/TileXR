# MoonEP Combine Operator Design

Combine is the inverse same-host direct-launch stage for the saved V1 Plan:

```text
hiddenNvsh        BF16 [NvS,H]
routeWeightsNvs   optional FP32 [NvS]
hiddenSh          BF16 [S,H]
routeWeightsSk    optional FP32 [S,K]
```

Host validation matches Plan dimensions and requires paired optional weight
descriptors. The peer-window layout uses the same 32-byte-strided hidden chunks as
Dispatch. Each chunk publishes local input, pre-reduces saved duplicate groups in
FP32, waits for all peers, gathers primary routes, and completes a drained barrier.

For each token, the kernel decodes all K `dst` entries. Negative entries are skipped
after their BF16 rows have been accumulated into the saved primary row. Primary rows
are fetched from the owning peer, accumulated in FP32, and converted once to BF16.
Optional route weights are gathered from every recovered offset without arithmetic and
are therefore bit-exact.

The kernel is a pure AICore ELF registered explicitly and launched only with
`rtKernelLaunchWithFlagV2`. Calls are asynchronous on the caller stream and use fresh,
bounded magic-tagged synchronization. Success status is 3000; failure ranges are in
`moonep_peer_window.h`.

Host tests cover paired descriptors, checked chunk arithmetic, exact launch arguments,
and saved metadata. Real-device validation includes fresh/saved Dispatch and forward/
backward Combine on four physical NPU 4-7 at `S=64 K=4 E=32 H=512`.
