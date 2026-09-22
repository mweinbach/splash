# Cache-only teacher scratch elision: proof and cost plan

Status: proposal only. No shader, runtime, production file, model payload or
artifact hash was changed or read for this audit. No GPU work was submitted.
The parent is the qualified `build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5`.

The proposed scope is only a successful singleton
`FlashMTPTeacherBulkForward::primeTeacherCache` call over complete128-row
windows, initially the canonical1920-pair body. The127-pair tail and every
generic/head proposal/Last/All/grouped/decode/verifier/trunk path stay original.
All18 arena buffers remain allocated and charged. No allocation or geometry
expansion belongs to the first experiment.

## Current timing evidence

All saved depths execute the same two physical teacher commands:1920+127,
2047 pairs and16 original chronological cache prefixes. Subtracting the saved
head-priming counters from prefill counters gives these warmed ranges:

| Depth/report | Inclusive prefill ms | Target GPU ms | Target command minus GPU ms | Teacher GPU ms |
|---|---:|---:|---:|---:|
| MTP3 canonical |505.386–513.127|486.837–493.706|6.758–7.147|5.797–5.847|
| MTP2 diagnostic |513.152–519.552|486.229–492.822|13.115–13.236|5.807–5.860|
| MTP1 diagnostic |518.515–526.467|493.780–494.495|11.263–16.212|5.813–5.969|

Main command preparation+encoding remains about1.1ms. Other target API work
beyond command wall time is about3.5–3.8ms. The substantial depth-run variation
therefore comes mostly from the target's GPU/wall gap and GPU-duration
variation, not different teacher math. Commit-to-scheduled-callback intervals
track this gap, but they are callback-arrival measurements, not timestamps of
actual GPU scheduling. There is no source-backed causal queue/clock fix here.

The old128-window diagnostic trace assigns2.438998ms to teacher
`K2560/N12288 q_proj` and1.185457ms to its `Q5/G64 K10240/N4` injection
projection. Their3.624455ms historical subtotal is **not** the cost in the
current1920-row bulk command. The current bulk-inclusive teacher GPU budget is
only about5.8ms and has no saved per-role profile. No present whole-prefill
savings should be promised from that historical subtotal.

Evidence files:

- `build/release/flash/sep21-teacher-bulk-ab-qsa-model-and-quality-v1.json`
- `build/release/flash/sep22-teacher-bulk-depth-diagnostics-v1/fixed-depth2.json`
- `build/release/flash/sep22-teacher-bulk-depth-diagnostics-v1/fixed-depth1.json`
- `build/release/flash/sep21-continuation-bulk-full512-stage-v1.json.trace.jsonl`
- `build/release/flash/sep21-teacher-bulk2048-cache-future-and-sequence-v1.json`

The exact-first bulk component already saved a measured9.419992ms API time
(15.897165→6.477173ms), replacing16 commands by2. Repeating that closed
experiment is unnecessary.

## Dataflow proof

`bulk.cpp:145–147` produces `HCRawInjection[rows,4]` and
`HCInjectionWeights[rows,4]`. The latter is never consumed by the bulk method.
It stops before attention/MLP branches and therefore never calls HC injection.
The ordinary head uses injection weights only below its cache-only early
return (`FlashMTP.cpp:659–674`, then683/687 and724/729).

Both `flash_hc_mix` and `flash_hc_mix_with_injection` write `Mixed` by calling
the identical `flash_hc_mix_element` with identical normalized/up buffers,
geometry, indices and BF16 arithmetic (`flash_hc.metal:158–201`). The extra
branch reads raw injection only to write injection-weight scratch. Replacing
that mix by the existing no-injection mix does not change `Mixed`, conditional
on preserving all failure checks described below.

`bulk.cpp:149` produces `QProjection[rows,12288]`. Its only bulk consumer is
the plane0 part of `flash_qsa_fast_prepare`: plane0 writes query scratch,
never a persistent plane (`flash_qsa_fast.metal:105–158`). The bulk prefix
builder stops before selection/attention. These queries are never consumed.

The five persistent planes depend on retained inputs:

| Plane | Producer inputs that must stay original |
|---|---|
| keys |K projection, K norm, positions, exact norm/RoPE reduction|
| values |V projection and its BF16 copies|
| raw index keys |last128 columns of IndexProjection, raw BF16 copy|
| index positions |original logical/explicit positions|
| pooled keys |original chronological prepare/pool prefixes and source norms|

First version keeps the entire IndexProjection and all index-query work. It
does not shrink that projection, pool complete prefixes together, change cache
ownership or reorder a single128-row prepare/pool pair.

`FlashMTPState::Impl` contains only owner, QSA state, logical length and poison.
No discarded query/injection buffer is persistent state. The private bulk arena
is distinct from the original head arena. Future Last/All/None/proposal calls
recompute their ordinary q/injection scratch before use. The cache-only result
publishes no hidden/vocabulary/greedy feature. Development arena inspection
must distinguish intentionally unused active scratch from cache/output state.

## Why blind elision is not exact

The existing q projection, injection projection and query preparation can set
sticky numerical diagnostics. Plane0 also checks q norm output and positions.
Removing their work can remove an overflow/nonfinite detection even when K/V
remain finite. Source deadness alone does not prove error equivalence.

The first implementation must use a **validated finite-eligibility fallback**,
not merely delete those checks:

1. Keep every existing owner/state/capacity/token/alias/extent/parameter guard
   before mutation. Unsupported roles/conventions/geometry use the original
   bulk graph. Existing immutable original/cached coefficient identity remains.
2. Establish a Root-only immutable coefficient/norm closure certificate at
   construction. It must conservatively bound the omitted q and injection
   projections and q norm/RoPE intermediates, including F32 rounding/FTZ and
   BF16 representability, from bounds on each retained upstream stage. Merely
   checking coefficients are finite is insufficient. Missing/invalid bounds
   force original graph execution.
3. Validate all actual feature BF16 words are finite and inside that certified
   input domain before selecting elision. Bad/uncertified input must execute the
   original graph, so it still gets original poison, logical-publication and
   partial-cache behavior. Do not turn its original numeric failure into a new
   pre-mutation host rejection.
4. First plan uses a CPU scan of the borrowed actual feature view as the
   simplest checkable eligibility mechanism. Its40MB canonical1920-body read,
   cache effects and latency must be included in real timing. A later GPU gate
   would add command/synchronization cost and needs a separate cost/proof; it
   is not authorized by this plan.
5. Only for eligible calls, omit the two projection dispatches and use the
   existing no-injection HC mix. Substitute a private cache-prepare entry that
   exits only plane0 before any q pointer/norm/query-scratch access. All K/V/
   index branches and their position/finite/normalization diagnostics remain
   literally original. Keep all original128-row prefix graphs and pools.

The immutable closure certificate is the main unresolved proof obligation.
Until it is specified and independently reviewed, even a finite observed input
does not authorize the cheaper graph.

## Proof required before timing or composition

- Byte-compare all five full physical cache planes, logical length, owner and
  poison state for begin0 plus offsets1/127/128/511/512 and compression
  boundaries; rows128/256/512/1920/2048 and successful partial sequences.
- Same future Last/All hidden, logits and compact greedy; truncation/reprime,
  EOS/special tokens, changed future inputs and repeated cache prefixes.
- Poison unused QProjection/HCRawInjection/HCInjectionWeights/query scratch
  with NaNs and changing bits; eligible cache/future results cannot depend on
  those bytes. All active used planes must be written; inactive1920 tails and
  all existing arena/cache-owner redzones stay intact.
- Original bad-feature NaN/Inf/large-finite/subnormal, coefficient/norm boundary
  and overflow cases must fall back. Compare original diagnostic bits, poison,
  unchanged logical length and any observable partial physical-cache state.
  Test uncertified closure, stale certificate and unsupported convention
  fallback. Keep invalid owner/capacity/token/alias faults before mutation.
- Precommit submission/cancellation failure retains original no-publication
  semantics. After healthy1920 publication, preserve the real Worker safePoint,
  cookie/generation/pointer/deadline checks and unchanged127 tail. Repeat
  qualified native cancel/deadline/delete/same-ID recovery tests for the new
  route; do not reuse old proof as runtime evidence.
- No trunk or proposal/decoder/grouped math changes. Mutable eligibility/
  fallback/dispatch counters live outside numerical identity. Existing weights,
  state buffers, arena allocation and MemoryGovernor accounting stay unchanged.

## Cost gate and stop rule

After plan approval, first add a private observation-only current1920+127
profile around the existing bulk API using the existing MetalBackend command
profiling interfaces. It must identify actual q/injection costs and all changed
dispatches. Stage profiles are diagnostic, with ordinary unprofiled control
recorded separately. No historical-profile extrapolation qualifies speed.

Then benchmark whole2047-pair sequences on actual attested trunk features:
>=150ms GPU warm per path, balanced ABBA/BAAB positions, same caches/contexts,
and no tensor hashes/copies/comparisons between timed calls. Include eligibility
scan, graph construction, both commands, state publication and tail. Follow
with canonical2K/256 same-binary ON/OFF trials and unchanged frozen22 grading.

Stop if the coefficient/domain proof cannot preserve failure semantics, or if
the new eligibility scan consumes the observed projection saving. Do not
silently weaken diagnostics or claim a few-ms teacher change explains the
measured7–16ms target GPU/wall gap. The9-trial depth report comparison is an
observation of separate process runs, not an identical-state causal depth test.
