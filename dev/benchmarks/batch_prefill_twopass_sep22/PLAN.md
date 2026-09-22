# Current batch packed-V QSA transfer — plan only

Root authorized CPU-only preparation of this plan using a full separate batch
arena. GPU/model/capture execution remains exclusively Root-owned after gates.
Parent: `build/batch-prefill-restored-teacher-clock-sep22-worker-v1`.
Its actual library is `1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf`.

The candidate calls the existing qualified `prefill4k::addTwoPassQSA` with
`packedV=true` separately for each real fresh batch lane. It transfers the
existing singleton QSA arithmetic without changing singleton bodies, other
batch projections, original trained MTP, scalar teacher310 MB, or decode/Verify.
It changes the old batch attention floating tree and requires fresh numerical
and task qualification. Old batch output-byte parity is not inherited.

## Measured budget and decision

Current warm medians are command timing; prefill numerator is all real prompt
tokens, 8192 for B4 and 4096 for B2, divided by the summed native prefill command
wall time including teacher commands. They are not per-lane rates or a common
TTFT interval. Decode uses actual emitted post-first tokens over the common
native first-emission-to-last-DONE span; B4 MTP148.643 and B2 MTP97.110 are
aggregate rates (individual medians37.182/48.563).

| Mode | Width | Prefill tok/s | Command wall ms | GPU ms | Saving to >4000 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Standard | 4 | 3838.258 | 2134.302 | 2096.709 | >86.302 |
| MTP3 | 4 | 3769.218 | 2173.395 | 2126.268 | >125.395 |
| Standard | 2 | 3693.224 | 1109.058 | 1078.731 | >85.058 |
| MTP3 | 2 | 3606.914 | 1135.597 | 1096.078 | >111.597 |

Preparation+encoding is only1.2–1.4 ms. Scheduled callback25–29 ms is callback
arrival, not proof of GPU start or rewiring. Entire teacher command wall budget
is36.765 ms B4 /24.292 ms B2; removing it cannot close the MTP gap.

The complete old singleton QSA component measured inclusive6.871 ms versus
packed-V two-pass3.32665 ms. Conditional transfer of3.54435 ms per attention
layer ×12 ×width suggests170.129 ms B4 /85.064 ms B2. Applied directly to current
MTP command medians this predicts4089/3899 tok/s. These are hypotheses, not
service measurements: current batch producer values and geometry must be
captured and measured. B4 has a credible budget; B2 may need a separately
qualified exact per-lane HC inject/norm fusion (~12.5 ms extrapolation), which
alone does not reliably close its remaining26.5 ms. Do not combine changes
before isolating the QSA gain.

## Minimal source and allocation plan

Add one private default-off strict0/1 batch flag. Freeze it before config,
model mapping, or backend construction. Require the restored batch-QSA flag,
original QSA F32/MPP/row-tiles/bulk/SG8 flags and existing singleton two-pass
flag. The mathematical batch identity must explicitly bind the new scope,
unchanged shader/host source pins and packed-V choice. Keep mutable counters
outside identity; singleton counters and the singleton workspace constructor
must remain byte-identical. Do not instantiate the singleton bridge Workspace
for batch: it increments singleton arena counters.

Initial eligibility: real2 or4 request lanes, uniform2048 rows, **all cohort
states** healthy/owned at length0, actual capacity>=2048 (service16384). Do not
admit dummy or synthetic lanes. Existing lane1/3, historical-prefix, smaller
window and other shapes retain the exact old branch. Preserve all source-trunk
replacement, backend/model/capacity, duplicate, pending/poisoned and token guards.

Root selected **one fully separate batch-owned509,607,936-byte Shared arena**
with all six aligned non-overlapping views. Retain the old234,356,736-byte legacy
allocation unchanged for controls/fallback. Prepared-view reuse or an overlay
that reduces the net increase is a separate optimization. No trunk-member
workspace or new weights are used.

| View | Location / offset B | Length B |
| --- | ---: | ---: |
| Prepared BF16 Q | new arena0 | 25165824 |
| Prepared index Q | new arena25165824 | 2097152 |
| Prepared block selection | new arena27262976 | 4194304 |
| Packed Q / dead-Q reused packed V | new arena31457280 | 25165824 |
| F32 QK scores / in-place global P | new arena56623104 | 402653184 |
| F32 attention | new arena459276288 | 50331648 |

Construct `DenseCoalescedWorkspace{2048,preparedQ,indexQ,selected}` and
`TwoPassWorkspace{prepared,packedQ,scoresP,rawAttention}` using literal existing
layouts. Reserve the full additional509,607,936 bytes before construction and
verify actual owner charge/delta, total workspace ledger, no denied admission,
fresh host measurement and normal pressure. The existing target singleton arena
is not borrowed. Keep one new arena for all12 layers and real lanes, reused only
after the previous lane's complete nine-dispatch packed-V sequence finishes.

No floating shader or AIR edit is necessary: all six two-pass entries already
exist in the exact current library. Keep `twopass.cpp/.hpp`, singleton bridge,
Forward arithmetic, coefficient loaders/presence508, all trained head/scalar
teacher and AB/FMA/W8 source unchanged. Likely edits are BatchPrefill.cpp,
private batch policy header, Worker strict flag/static resource status and
private readonly oracle inspector. Rebuild all actual header consumers;
conservative all50 current host TUs with authenticated Core4 and unchanged
library is acceptable. No old host object or whole-library substitution.

The current helper requires ordinary.maximumRows128,
fast.maximumRows128/maximumPartitions32. Preserve these exact service scratch
contracts. The existing authoritative coalesced plan35 dispatches validates
all original128-row windows, norm/rope/pooling/dense selection, then the helper
copies its first3 dispatches and appends6 packed-V attention dispatches.

For each lane, q/k/v/index/output views use byte offset
`lane*2048*width*2` and exact byte length `2048*width*2`, widths12288/512/512/640/
6144 respectively. Supply that lane's independent original QSA cache and all
four actual norm tensors/conventions, model epsilon/theta, and shared sticky
diagnostic view. Validate every lane's new arena/norm/cache/scratch/input/output
ranges before changing the caller graph. Validate model epsilon1e-6/theta1e7
against the existing helper's literal descriptor before use; do not silently
change those constants. Preserve helper's no-graph-mutation
on validation failure. Do not reinterpret flattened8192 rows as one sequence.

External hidden destinations must reject every one of the six two-pass views
(all six owned by the new separate batch arena) and new arena
base, together with all old storage. Add start/interior4 valid Shared
destination cases for all six (12 exact-error no-submit cases); require full
state/new-arena bytes and sameViews unchanged. Include other-lane cache overlap,
wrong backend and short/misaligned views. Guard allowances must be planned and
actual native owner charges reported separately, not guessed from view lengths.

## Qualification sequence

1. CPU closure, strict flag/dependency refusal before config/backend, lane view
   range/capacity/foreign/duplicate/inactive tests, arena partition bounds,
   exact baseline-body normalization and unchanged singleton counters/library.
2. Root captures actual current restored B2/B4 QSA q/k/v/index projection inputs
   at selected layers using owned debug GPU copies before scratch is reused.
   Record layer/lane labels, row count, cache begin/capacity, conventions,
   source/operation/library identity and real completed current profile.
   Preparers do not open capture/model data. Synthetic data remains labelled.
3. One bounded **QSA-only** oracle loads these actual same producer inputs and
   norm/cache metadata. Compare legacy batch attention against the candidate
   using unchanged original per-row/F64 source certificates and diagnostics.
   Compare the candidate bitwise with the existing singleton **QSA-only helper**
   on the same inputs, not whole singleton Forward. B1 Forward has W8 projection
   math that differs from current batch producer values and is not a valid
   whole-batch golden. Original prefix/five cache planes must be bitwise exact;
   inactive entries, future cache continuation and sticky failures are checked.
   Direct raw-V variant remains excluded. No new envelope/threshold exception.
4. Current-header full B2/B4 source-bound alternative proof: all real hidden,
   logits, complete greedy records, GDN/QSA/PLE planes, continuation/correction,
   ragged/late-drop/inactive lanes, prefix lengths, aliases, capacity, pending/
   abort/poison/rollback, current-head future behavior. One Forward/model per
   process; bounded replay partitions if needed. Worker cancel/deadline/reuse
   uses actual generation lookup boundaries and requires independent runtime
   evidence. Source-only foreign-trunk checks must not be labelled runtime proof.
5. Isolated unchanged B1 flags0 +original scalarbulk1 frozen22/recovery. For
   B2/B4, run the original22 tasks natively and preserve request budgets/graders;
   singleton teacher expectations cannot be faked as grouped API call counts.
   Grouped successful commands, lanes and real pairs need separate source-bound
   coverage with exact per-lane budget and shape evidence. Existing batch profile
   has no B2/B4 task qualification to inherit. Any new regression stays visible.
6. Only after quality/state gates, Root runs fresh standard/MTP3 B2/B4 canonical
   2048/256,temp0/cache0/context16384/cap4,1warm+3trials. Include complete packing,
   QK/softmax/packed-V/PV/unpack, all teacher/dispatch time, actual final admission,
   graceful shutdown/footer and actual native aggregate/common-span counters.
   A component saving or best single trial is not a >4K service result.

The decision is a single QSA transfer first. No closed WY/NAx, gate geometry,
LUT, coefficient-sidecar or unrelated decode route is reopened.
