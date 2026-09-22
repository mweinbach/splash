# Separate policy proposal: range fallback by chunk

This is a design proposal only. Sealed v6 and its guard decisions remain
unchanged. Root's CPU decay predictor found 15/48 layer-0 heads with at least
one T32 nonzero-segment range risk. That predictor is not a measured replay
count and does not include cancellation, norm or nonfinite checks. Actual GPU
capture and whole-model telemetry must establish whether this work is useful.

The cheapest credible first alternative is **range-only native chunks**, with
cancellation/norm/nonfinite retaining v6's full-head, full-row restore/replay.
Preparation stores the existing range predicate per (chunk, head), rather than
using it to exclude the head from every matrix chunk. Each value tile sees the
same precomputed range decision, so no cross-workgroup synchronization is
needed for range selection. Unsafe chunks run native recurrence from the
current incoming F32 chunk state; safe chunks retain the WY matrix path.

With one existing 256-thread SG8 workgroup owning 32 value rows, the native
branch can process four sets of eight rows with the captured native recurrence
sequence. Every value row still uses one 32-lane SIMD group and the original
four-key-elements-per-lane sum, SIMD reduction, F32 state updates, BF16 beta,
and BF16 output conversion. The chunk begins at a pointer offset into mixed,
alpha/beta and output; a bounded chunk-local parameter uses rows<=32 and the
same recurrent stride. Exact input/output/tape pointer bounds and native
source-level reduction equivalence require a new proof. Reusing a native
512-thread/V16 helper directly in a 256-thread workgroup would be incorrect.

For a range chunk, state must be untouched by WY before the native branch.
For a safe chunk, candidate cancellation is known before the state update.
If cancellation/norm/nonfinite is detected, retain sticky head reasons and
complete the existing v6 end-of-call restore/replay from the original whole
head snapshot. That preserves native reference equivalence for those heads
without inventing an inter-workgroup barrier. Telemetry must distinguish
range-native chunk rows from full-row replay work.

Extending **all reasons** to per-chunk whole-head native replay is substantially
harder: v6's cancellation/norm decision may originate in any of four value-tile
workgroups. All four must see the final decision before they commit a chunk.
An atomic spin barrier in a GPU kernel can deadlock when workgroups are not
simultaneously resident and is not an acceptable design. Options are a separate
decision/commit dispatch per chunk (up to 128 added dispatches at R2K), or one
workgroup owning all 128 value rows with a separately qualified matrix layout.
Neither has measured performance or a compiled SDK/resource proof here.

The alternative's incoming state can already contain rounding from previous
safe WY chunks. Native replay from that state is exact relative to the same
incoming F32 state, but does not prove equality to a whole native prefix or
universal F64 history accuracy. The unchanged raw F64 tensor gates must remain
reported. This is a new numerical policy and requires its own identity,
range/cancellation source proof, exact local native and tape/continuation proof,
then actual-capture and full-model semantic/MTP verification. The old v6 exact
comparison cannot be reused as if the policy had not changed.

No implementation or performance claim is authorized by this document. Wait
for Root's actual GPU fallback evidence before building the alternative.
