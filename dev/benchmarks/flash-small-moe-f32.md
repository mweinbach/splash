This private small-window candidate preserves the current target verifier's
F32 coefficients and exact contiguous expert-QMV reduction order. It never
converts expert operands to BF16 or uses the large-row mixed-INT8 alternative.

Gate/up retain lane K indices `block*512 + lane*16 + j`, two U32 packs in
original pack/j order, SG4/C2 and an independent F32 accumulator/simd_sum per
route and column. Down retains `block*256 + lane*8 + j`, SG2/C4; only lanes
0..15 participate in its final K640 block. Every projection rounds to BF16 at
the original boundary. The compiled BF16 sigmoid/SiLU/up multiplication and
canonical routed combine remain unchanged.

Group1 fuses gate/up/SwiGLU without a job-emitter dispatch. Group2/4 use a tiny
GPU stable membership list to reuse each decoded F32 coefficient across at
most two/four original routes. Each route keeps its own accumulation and
original canonical output slot. Valid duplicates and invalid I64 IDs remain
supported. The grouped emitter is included in complete-chain timing; no CPU
expert counts or large expert buffers are introduced.

The oracle runs three graphs: untouched production, an original-QMV copy with
F32 sum taps, and the candidate. The tapped reference must reproduce every
production BF16 byte before its F32 sums are trusted. Candidate gate/up,
activation, down and combine bytes must match production, and every projection
F32 sum must match the reference bit-for-bit. It also checks independent
diagnostics, stable grouped job records, output/job/count canaries and finite
positive timing. F32 taps are disabled for timed candidate commands.

All 43 dependencies and baseline AIRs were copied from fresh
`build/flash-default-v6`; `sizeof(CommandTiming)==200` and object/source hashes
are recorded in `build/flash-small-moe-f32/private-freeze.json`. Metal 4.1 and
CPU order/coefficient/grouping checks passed without GPU execution. Root must
qualify actual GPU exactness and performance before any promotion.

```sh
FLASH_SMALL_MOE_LAYER=0 FLASH_SMALL_MOE_ROWS=16 \
FLASH_SMALL_MOE_GROUP=1 FLASH_SMALL_MOE_PAIRS=4 \
build/flash-small-moe-f32/flash-small-moe-f32-oracle \
  build/flash-small-moe-f32/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/small-f32-moe-r16-gr1.json
```

Vary `GROUP=2/4`, `ROWS=4/8/16`, `LAYER=47`, or
`ERROR=invalid_ids/nonfinite`. `FLASH_SMALL_MOE_IDS` accepts exact-extent I64
fixtures, for example
`build/flash-small-moe-f32/fixtures/sample0-layer00-decode-r8.ids.i64le.bin`.
The fixture manifest includes 24 source-matched captures from layers 0/47,
R4/8 decode, diverse prefill windows and clearly labeled R16 boundary proxies.
Both decode captures generated the same counting stream, so reuse statistics
are workload-specific. Captured-ID screens use synthetic hidden activations
and do not establish broad quality. No original weights, production files or
defaults are modified by this candidate.

Root's GPU screens passed all BF16/F32-sum checks but rejected the v1 routes
for speed: Group1 spread measured 0.816 ms control versus 0.877 ms fused;
Group4 actual R16 layer0 IDs measured 0.865 ms versus 2.540 ms. These routes remain
disabled. Both actual layers have 33 unique experts among 160 routes, yielding
52/53 Group4 jobs. That still gives 4,160/4,240 active gate groups and
16,640/16,960 down groups, far above 80 cores. Fewer jobs alone therefore does
not establish global GPU starvation.

Accessible pipeline metadata reports zero static TGM and 1024 maximum threads
for control and candidates. Neither value reveals actual register occupancy
or thread-local spills. Static LLVM contains local arrays in both routes; this
is evidence of compiler representation, not proof of hardware spilling.
Source grouping has 16 F32 accumulators per lane, runtime row predicates inside
MAC loops, and down activation reload expressions across all four columns.
Unconditional inactive-slot reductions add 30–32.5% relative to 160 real routes.
These costs and hardware cache reuse can outweigh fewer coefficient decodes.

One bounded v2 follow-up is Group2 with explicit activation preloads shared
across columns/projections, constant loops marked for unrolling and uniform
inactive-row reduction skips. It retains all original F32/BF16 math, with the
same exact qualification checks. Its fresh private executable is
`build/flash-small-moe-f32/flash-small-moe-f32-preload-v2-oracle`, paired with
`preload-v2.metallib`; use the existing environment names and
`FLASH_SMALL_MOE_IDS=build/release/flash/v6-real-verify-r16-layer0-ids.i64`,
`ROWS=16`, `LAYER=0`, `PAIRS=4`. Group2 is the sole supported mode. Compilation
and CPU checks passed, with v1 artifacts unchanged. Root subsequently measured Group2 at 0.894 ms control versus 1.268 ms
candidate (1.42 times slower), with every BF16/F32 sum bit still exact. The
preload improves the earlier Group4 contrast in separate matched screens but
also remains disabled. No further small-window experiments are active this round;
actual hardware register/spill instrumentation is needed before another design. CPU findings and provenance are recorded alongside the binaries.
