# Separate whole-batch numerical-policy qualification

This is a new qualification route, not a change to the failed replay-v7 gate.
Root reports actual layer3/lane0 inputs pass the original global BF16 bound
(relative L2 0.00005021 <=0.0001), original sampled QK/raw/gate F64 checks, and
bit equality with the unchanged singleton QSA helper on exactly the same batch
projections. Prepared Q/index/selection and all five cache planes also match.
The extra old-SG8 EVERYROW check fails 116/2048 rows, maximum relative L2
0.0005955876 and minimum cosine 0.99999982. These are Root-provided values;
this CPU plan does not open capture or generated-response data.

Replay-v7 remains failed. Preserve its report and complete diagnostics, and
display the row failures in each new qualification summary. The original
prefill_qsa_twopass_sep21/oracle.mm requires GLOBAL BF16 relative L2 <=1e-4,
raw relative L2 <=2e-5, cosine >=.99999999, sampled F64 QK/raw/staged-gate
bounds, probability sum <=2e-6, and cache/selection/causality/fault/alias checks.
It does not contain the extra EVERYROW old-SG8 comparison. Historical frozen
PLAN/README wording calling that condition an “original per-row” gate is not
edited; this addendum explicitly supersedes that attribution for this route.
No numerical constant or original required gate changes.

The alternative can be assessed against intended arithmetic. It cannot claim
old batch row equivalence or inherit singleton task scores. Small attention
differences can change later greedy choices; global and sampled one-layer
checks do not establish whole-model generation quality.

## Independent intended-arithmetic reference

Start the reference executor from the original restored BATCH projection,
GDN/PLE/MLP/head executor. Preserve flattened rows 2048*B, original projection
choices, norm conventions, state/capacity/ownership guards and current library.
Replace only eligible fresh B2/B4 QSA emission with the unchanged packed-V
addTwoPassQSA helper on each lane's real q/k/v/index/norm/cache views, using an
independently allocated and Governor-planned 509607936-byte arena. A whole
singleton Forward is an invalid reference because W8 upstream math differs.

Compare the service candidate and reference using one process/Model/Forward
and two batch arenas with independent fresh state cohorts. Both implement
the intended NEW QSA arithmetic. Require exact full hidden/logits/complete
greedy records and all 134 physical GDN/QSA/PLE planes for each lane. Compare
topology, active/full extents, logical lengths/capacity/health and stable local
views; never compare new owner pointers across cohorts.

This full executor covers all 12 QSA sites and all real lanes on the same
batch projections. A single captured layer cannot stand in for them. Check
shared arena reuse, other-lane/cache/norm/immutable disjointness and inactive
physical bytes. B1/B3, smaller rows, mixed/nonzero prefixes and other excluded
contexts retain the exact old route and require no-new-work evidence.

Replay actual greedy continuation IDs derived from NEW-policy outputs into
both arms and compare full outputs/state again. Old/new whole-state and text
bit equality is not required because attention arithmetic intentionally differs.
Record old/new text, greedy/acceptance differences and original task outcomes.

Disk campaigns require an exact serialized whole-campaign preflight before
any payload write and replay partitions <4 GiB. In-process direct comparison
may instead hold both independently admitted arenas/state cohorts; no second
Model/Forward is constructed. Reference allocation is absent from normal runs.

## Head, state transitions and Worker

Unchanged trained-head source is not fresh proof for changed premixer features.
Feed corresponding exact candidate/reference features and incoming IDs through
current grouped teacher-cache APIs. Compare all five head cache planes, real
pair counts, lengths and complete future hidden/logit/greedy rows. Preserve
ragged priming, partial retained prefixes, rollback/reprime, EOS/budgets and
sticky/poison/alias/identity guards. Report actual grouped calls and per-lane
pairs; do not fake singleton call expectations.

Feed new-policy prefill state through existing decode/Verify/commit/correction
and future routes against intended-reference state, including inactive lanes
and late drops. Root owns actual Worker cancellation/deadline/same-ID recovery
with no output after terminal cancellation and a following healthy request.
Source-only recovery/foreign-owner checks are labelled separately.

Use cache_tokens=0, context16384 and unchanged sampler/budgets. No state/prefix
artifact is imported across numerical profiles. Broader prefix-cache sharing
requires a separate identity/compatibility proof.

## Fresh native ORIGINAL22 B2/B4 qualification

Run all 22 frozen tasks in actual B2 and B4 cohorts in BOTH standard and MTP3,
with unchanged formatted prompts, schema, budgets and graders. Run matched
old-batch baseline per width/mode; no prior singleton score is inherited.
Every task must execute in each requested width, not a leftover smaller wave.

One complete matrix is 22 waves of B identical VALID task bodies and unique
request IDs (44/88 real graded responses); add a bounded mixed-task wave for
lane isolation using original bodies. Grade every response by its original ID.
No questions, answers or budgets change. Native evidence must show actual
cohort width, 12*B encoded new lane-layer calls per eligible prefill, one
healthy forward, B*2048 prompt rows and independent healthy states. Silent B1
execution cannot qualify B2/B4.

Authenticate separate old/new profiles before the unchanged grader/comparator.
Baseline expects flag0, candidate flag1, each bound to source/library/flags/
allocation and counters. Do not bypass a shared-numerical-profile comparator
or declare the identities equal. Standard requires absent/zero MTP work; MTP3
requires real per-width priming/verification/accepted-prefix/budget coverage.
Keep original context/wire/cache/transport/error gates and every persistent
baseline failure. Require no new task regressions per width/mode.

The new qualification profile explicitly states numerical_alternative=true,
old_batch_EVERYROW_equivalence=false and whole_model_qualified=false until all
new native gates pass. Bind batch source/host/shader/library/flags plus reference
closure; singleton target identity/counters remain unchanged. Link failed-v7
and diagnostic reports. Raw/operator source checks are not task qualification.

Only after primitive original global/F64, independent full executor/head/state,
Worker recovery and native task gates pass may Root run inclusive component
timing and normal standard/MTP3 B2/B4 2048/256 benchmarks. Include every
pack/QK/softmax/packV/PV/unpack and teacher cost; each component arm warms >=150ms
GPU with 20 balanced pairs; service uses one warm wave and three uncached trials.
Keep actual aggregate/common-span counters, real admission and graceful unload.
No new performance or whole-model qualification is claimed in this plan.
