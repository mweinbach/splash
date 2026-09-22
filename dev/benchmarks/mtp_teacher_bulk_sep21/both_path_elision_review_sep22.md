# Both-path teacher scratch elision: measured cost and lifecycle review

Status: discussion/proof plan only. No elision code or additional component was
built. Agent work was CPU source/report analysis only, with no GPU/model payload
read/hash/spawn or production edit. Root completed the current role profile.

## Current measured budget

Source report: `build/release/flash/sep22-current-teacher-bulk-role-profile-v1.json`
and its `.trace.jsonl`, plus Root's `...-medians-v1.json` summary.10normal and
10stage-profiled complete1920+127 sequences use balanced AB/BA positions. Both
paths received >=150ms accumulated GPU warmup. All20profile records are
Complete, zero dropped, full metadata and valid role timestamps. DispatchBoundary
is unsupported on this device; StagePerDispatch is supported and changes encoder
boundaries. No sampling barriers were used.

| Role |1920 bulk median ms |127 tail median ms |
|---|---:|---:|
| q projection |1.2102915|0.7320000|
| raw injection4 |0.1940835|0.0735625|
| original MixWithInjection |0.0783750|0.0096455|
| whole fused cache preparation |0.2203515|0.0149580|
| chronological pool |0.0768320|0.0052290|

Q+injection diagnostic medians total**2.2099375ms across both commands**;
bulk alone1.4043750ms. First experiment would retain every prepare/pool and the
original MixWithInjection, so neither full preparation nor mix time counts as
intended savings. Whole preparation is not an isolated query-normalization cost.

The tail q role is cached BF16 M16N64 for112complete rows plus
`flash_dense_bf16_project` for15rows, consuming the same62914560-byte cached
BF16 matrix. The medians are0.2656665ms and0.4667710ms respectively. It is not an
original-Q4 fallback. The tail's full cached bindings retain the127-row views;
the kernel's inline parameters select112rows, and the scalar binding is the
disjoint15-row view. The role validation correctly identifies both dispatches.

Ordinary whole sequence mean GPU5.8610625/API6.4403499ms versus instrumented
5.9539208/6.6717542ms: profile overhead1.015843GPU and1.035930API. Matched
pair geometric ratios1.015846/1.035954. First/second-position ordinary GPU
means5.855108/5.867017ms and profiled5.963133/5.944708ms show no old first-call
2x-order effect. Profiled dispatch sums exceed command GPU duration by about
0.120ms bulk and0.032ms tail, so per-role sums are **not** guaranteed ordinary
elision savings or guaranteed additive deadlines.

The profile component's target setup uses the prior separate GDN A/B graph,
not the current Worker merged graph. Its exact teacher input/math lineage was
established previously; this remains head-role attribution, not a new Worker
throughput observation. No new input payload hash was performed in this profile.

## ROI gate

Both feature windows together contain41922560bytes. The qualified current
Worker obtains each as a view of `result.hiddenBF16`; its existing committed
feature copy keeps only one final row20480bytes. There is **no full teacher
feature copy** into which the domain check can be folded. The profiler's
one-time40MiB owned setup copy is excluded from its sequence timing and cannot
hide the runtime scan. Retain original Worker safePoints and two commands.

An integer BF16 validator can clear each word's sign bit and OR-reduce
`abs_word > 0x4380` (256). It accepts finite subnormals and either zero sign,
and rejects NaN/Inf/large finite values without scalar float conversion. A
NEON8xU16 implementation is plausible; its actual borrowed-buffer latency,
coherence/cache effects and cancellation behavior must be measured by Root.
The conditional100GB/s bandwidth floor is0.419ms, not an observed scan rate.

Even assuming the full2.210ms diagnostic sum transfers, the saved-only phase's
1.687ms median deficit leaves only~0.523ms for the scan, q sanitation and host
policy overhead. It would still retain that phase's unrelated instruction
regression and cannot qualify the phase. The qualifiedD3 slowest trial513.127ms
needs1.127ms to cross512ms, leaving~1.083ms overhead under the same optimistic
assumption. The other draft-depth runs have larger target GPU/wall gaps; this
teacher change cannot cover their6–14ms deficits.

Recommendation: proceed only through the **coefficient/range proof and measured
CPU cost gate** below. Do not build/promote elision from this timestamp budget.
Close if the exact domain is unprovable or steady-state scan/sanitation cost
consumes the available budget. Decode work remains higher impact.

## Restricted both-path scope

Initially select only the current singleton complete1920body at length0 and
its canonical127tail at length1920. All other rows/positions/unsupported
conditions execute the original graph. This keeps the two existing physical
commands,16chronological128windows,2047pairs, original state ownership and all
18bulk arena buffers/MemoryGovernor charges. Broader generic CacheOnly selection
is not part of this first experiment.

For eligible calls omit only the original q projection and raw-injection
projection, initialize/maintain their unused inputs at known positive zero, and
keep **all** original prepare/pool shaders/grid/positions/barriers and original
MixWithInjection. Original q preparation consumes zeros and writes unused zero
queries. Original injection sigmoid produces unused finite weights. K/V/index
branches and their diagnostics remain literal. All future proposal/Last/All/
None/decode/verify paths remain original and recompute their own q/injection.

Do not call existing `hc(..., false)`: its branch conditions also enable fused
down/up and cached-up-mix routes. Preserve `hc(..., true)` selection and every
retained HC projection/SiLU/mix dispatch, omitting only its raw4project under the
reviewed CacheOnly eligibility predicate. No new query-only prepare shader is
needed, and the0.235ms preparation budget remains spent.

All original host guards execute in original order before selection, including
the different tail token-write semantics. Extra eligibility conditions choose
original fallback; they do not reject formerly accepted inputs. In particular,
CPU validation needs readable Shared features and a disjoint q-clear extent;
unsupported storage/aliasing selects original execution. Bad finite-domain
inputs/coefficient certificates execute the **entire original graph**, preserving
diagnostic bits, poison, logical publication and physical partial cache writes.

## Q-zero lifecycle proof

1. Each scratch owner begins with an explicit initialized-q-zero witness only
   after its own allocated QProjection extent has been cleared. Do not assume
   a new Metal buffer or old contents are zero. No new GPU buffer is allocated.
2. The bulk owner's q plane is private and is read-only for eligible bulk
   calls. Its ordinary/fallback bulk q producer invalidates the witness **before
   submission**, even when later submission/cancellation fails. A future eligible
   call clears the active extent if its zero witness cannot cover that extent.
   Conservative whole-plane invalidation is acceptable. A canonical clear costs
   47185920bytes; construction/first warm initialization and rare fallback-reset
   costs must be recorded separately from steady state, never hidden as universal
   no-reset behavior.
3. The ordinary head q plane is shared with full/proposal/Last/All/None/ordinary
   teacher calls. Its zero witness must be invalidated by **every** original
   writer, including the small-row/qmv/cache/scalar selections and submitted
   graphs that later fail. Before an eligible127tail, clear its3121152active
   bytes when dirty. The preceding request's decode usually leaves it dirty, so
   steady-state HTTP timing includes this sanitation even though a repeated
   isolated CacheOnly-only sequence might avoid it. Marking dirty only in the
   tail path is incorrect.
   Track dirty row ranges after establishing the literal writer closure: each
   ordinary `forwardImpl` q output is the prefix[0,rows]. A128-row bitmap can
   record writes before submission and clear only `dirty & active127` rows on
   the next eligible tail, retaining dirty inactive rows. This avoids pretending
   a partially cleared128-row plane is wholly clean. A1-row proposal dirties
   24576bytes; a3-row writer dirties73728bytes. These are conditional extents,
   not measured sanitation costs or assumed current proposal geometry. Any
   unknown/aliased bridge write conservatively marks the full affected span.
   Full active3121152-byte clear remains the safe cost fallback if complete
   row-range closure is unavailable.
4. Source writer closure is anchored at `forwardImpl`'s q producer plus the
   bulk q producer. Inspect every internal/public bridge that can target the
   same q allocation and every ownership/view escape before accepting this
   closure. Batch output buffers are separate, but that fact must be verified
   from their actual bindings, not assumed from names.
5. Guard the clear and witness update under the same existing scratch-owner
   mutex and completion discipline. No clear may race an outstanding graph or
   alias the input/immutable coefficients/persistent cache. An input overlapping
   the clear extent must choose original fallback. No safePoint/reference reuse
   rule is removed to reuse a witness across requests.
6. A boolean witness is not proof after an untracked mutable-view escape.
   Existing `oracleWorkspaceBuffers()` exposes the bulk scratch to test code.
   Mutable Oracle poisoning must explicitly invalidate the witness, or exposed
   mutable views must conservatively prevent persistent zero reuse. A held view
   may not silently modify q after the engine certifies it zero. Tests must
   challenge stale/dirty witness transitions rather than assuming allocator
   zeros. This is an unresolved implementation detail for review, not an
   approved API mutation.
7. Raw-injection scratch is tiny: zero only active15360body/1016tail bytes per
   eligible call while retaining original MixWithInjection. This avoids stale
   NaN input dependence and requires no persistent raw-zero witness. Inactive
   tails/redzones remain untouched. Injection output scratch itself is dead.

The standalone timing fixture must include a representative **dirty tail**
transition before each complete sequence, outside GPU profiling if appropriate,
and time its actual sanitation inside the candidate API. A repeated zero-only
tail cannot stand in for a Worker that interleaves proposals/other requests.
Control and candidate workspaces remain distinct; no input/q copies or hashes
between timed calls. Reusing a witness reduces eligible steady-state work only
after the entire writer/escape/completion proof is sound.

## Root-only actual coefficient census

Read the **live original head objects actually consumed**, once after normal
construction and before warmup; report tiny extrema/counts/first-failure
coordinates with no tensor dump. Bounds fail to original fallback, never a new
constructor rejection. Bind the certificate to the actual tensor object/view,
logical extent, dtype, convention, saved-operand/model identity and owner epoch.

| Actual role | Logical bytes to census | Sufficient preliminary ceiling |
|---|---:|---|
| cached BF16 fc_embedding[2560,2560] |13107200|finite abs<=16|
| cached BF16 fc_hidden[2560,2560] |13107200|finite abs<=16|
| cached BF16 HC down[320,10240] |6553600|finite abs<=16|
| cached BF16 HC up[10240,320] |6553600|finite abs<=16|
| cached BF16 complete q[12288,2560] |62914560|finite abs<=16|
| original pre-fc embedding norm[2560] |5120|effective gamma abs<=16|
| original pre-fc hidden norm[10240] |20480|effective gamma abs<=16|
| original HC norm[10240] |20480|effective gamma abs<=16|
| original q norm[256] |512|effective gamma abs<=16|
| original Q5 injection scales/biases[4,160] |1280+1280|finite abs scale<=1/bias<=32|
| original U8 embedding scales/biases[248320,40] |19865600+19865600|finite abs scale<=1/bias<=256|

Total142016512bytes of actual coefficient/parameter/norm census, once. Audit
logical bytes only, not uninitialized alignment slack. No original packed code
planes need be read for these worst-code bounds. All signed/zero affine scales
are legal. Report raw and actual convention-adjusted gamma maxima; read the
head's own convention instead of inferring OnePlusWeight from a suffix.

Require descriptor epsilon after F32 conversion in[2^-20,1], validated theta
and implicit positions<=2046. The initial power-of-two DAG in
`finite_domain_plan_sep22.md` still applies to both windows: normalized inputs
<=2^14, q<=2^41, query square sums<2^90, wide margins below BF16/F32 overflow.
The tail includes cached112rows plus scalar15rows; include both literal execution
routes and every q output column in the closure. Native BF16 input/F32 destination
type/range safety is the obligation, **not universal MPP accuracy or reduction
parity**. Document permitted intermediate ranges and scalar rsqrt/trig/FTZ
behavior. A fixture pass or plausible field bound alone does not prove it.

## Gates before any elision build/composition

First review the complete writer/escape/alias/completion closure, actual Root
census and literal finite-range argument. Then Root measures the proposed
validator and dirty-tail sanitation on representative borrowed buffers; this
stage must not presume the profiler's owned-copy cache state. No elision build
is authorized merely by this document.

After proof/cost review, a bounded private component must compare all5physical
cache planes, logical/poison/failure diagnostics and future Last/All hidden/
logits/all-greedy/rollback, including original q-only/injection-only faults and
uncertified coefficient fallbacks. Exercise clean/dirty q lifecycle, ordinary
head writers, precommit failure, NaN/signed-zero/subnormal/large finite inputs,
EOS, truncate, canaries and stale witnesses. Preserve all chronological prefixes.
Actual Worker cancel/deadline/same-ID recovery plus frozen22and canonical
same-binary normal ON/OFF performance remain required before composition.
