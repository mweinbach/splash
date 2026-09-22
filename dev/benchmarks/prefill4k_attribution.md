# Full target and teacher-prime attribution

This isolated harness runs current production Flash target prefill and the same
ordered 128-row teacher-priming calls as the singleton Worker. It processes every
adjacent prompt pair exactly once. Target prefill must produce exactly one
vocabulary row per trunk chunk; teacher priming must produce none. It checks
logical offsets, arena reservations, normal sticky diagnostics, and hashes final
BF16 logits and target features after command completion.

The wrapper reads the current qualified profile through launcher validation,
requires its depth-3/40-flag policy, and resolves the two matching saved operand
and selected INT8 expert paths. It removes inherited Flash experimental flags.
Compilation, `--help`, and wrapper dry-runs submit no GPU work. The root
coordinator runs all inference serially and checks model processes unload.

Build:

```sh
make -j12 -f Makefile -f dev/benchmarks/prefill4k_attribution.mk \
  BUILD=build/prefill4k-attribution SPLASH_PRECISION=hybrid prefill4k-attribution
```

Run three uninstrumented current controls (root only):

```sh
.venv/bin/python dev/benchmarks/prefill4k_attribution.py \
  --tokens PATH_TO_REAL_CODING_PROMPT_TOKENS.json \
  --report build/release/flash/prefill4k-current-normal.json \
  --mode normal --repeats 3 --run
```

Capture descriptive per-dispatch stages in a fresh process:

```sh
.venv/bin/python dev/benchmarks/prefill4k_attribution.py \
  --tokens PATH_TO_REAL_CODING_PROMPT_TOKENS.json \
  --report build/release/flash/prefill4k-current-stage.json \
  --mode stage --run
.venv/bin/python dev/benchmarks/prefill4k_attribution_summary.py \
  build/release/flash/prefill4k-current-stage.json \
  --output build/release/flash/prefill4k-current-stage-summary.json
```

Each invocation emits source/profile/token/binary/metallib hash provenance,
command host subphases, and pipeline families. Normal runs report full call wall
time. Counter stage/dispatch modes perturb scheduling; use them for attribution
and qualify throughput separately through the actual HTTP service. Full-model
output quality and continuation equality remain required before promoting a
kernel optimization.

## Source findings and older retained evidence

`Worker::tick` passes `returnAllLogits=false`; `FlashForward` selects only the
last target row for `language_model.lm_head`. `FlashMTPLogits::None` skips the
final head mixer and vocabulary projection. Thus neither all-row target
vocabulary nor repeated vocabulary during teacher priming explains the loss.
For a singleton 2K prompt, current Worker priming submits 16 ordered commands,
covering 2,047 true pairs. The retained current default HTTP proof has about
120ms of teacher-prime GPU work per request, about 12% of its warmed ~975ms
prefill wall scope. Eliminating all teacher-priming work could not alone reach
512ms / 4K tokens per second.

The older v5 2K target stage trace has 2,993 dispatches and 846.8ms full GPU time.
Families: routed MoE 396.3ms, dense190.0ms, QSA124.0ms, GDN78.7ms, HC28.5ms,
shared experts17.1ms. Selected INT8 expert misses execute original Q4
fallback gate/up176.8ms and down76.1ms. These uniformly recurrent fallback
kernels consume about30% of total sampled GPU time. The selected hit path adds
88.4ms. Main dense leaders include BF16 M=2048/N=2560/K=6144 outputs at37.1ms
across48 layers, and HC M=2048/N=320/K=10240 at26.4ms across97 calls. QSA's
prefill row-tile kernel costs99.1ms across144 calls, with another16.2ms in online
partition fallbacks. Stage encoder boundaries alter scheduling, and this is an
older profile: these are leads to retest rather than current measured results.

## Fresh current depth-3 results

Root ran the compiled isolated harness serially on the canonical real 2K coding
fixture at `build/release/flash/prefill4k-fixture/code2048.tokens.json`.
Three normal warmed trials took an average839.0ms target call and125.0ms
teacher-prime call (target831.6ms GPU; prime120.2ms GPU). This is about2,124
input tokens per second over direct calls, excluding Worker controls and HTTP;
the current HTTP context baseline reports2,092 tokens per second.

The fresh stage pass contains17commands/3,585dispatches. Numeric audit checked
11,839floating metadata fields: allfinite, noimplausibly tiny durations, complete
metadata/status, and calibrated timestamp/duration agreement within5.75e-11s.
Final target logit and hidden hashes match the normal control. Stage target
GPU845.24ms has MoE421.24ms, dense177.35ms, QSA121.99ms, GDN77.20ms,
HC27.12ms, and sharedexpert10.40ms. Q4missgate/up214.94ms plus missdown101.28ms
is316.23ms, or37.8% of sampled targetdispatch time. These remain diagnostic
stage durations. Counterintervals can overlap at boundaries: head sampled sum
121.82ms is slightlyabove121.17ms fullcommandGPU, so the families are not
exclusive wall phases.

The initial generic classifier labels gathered head affinekernels as shared
experts. The corrected summary classifier uses selectiondepth (`threadgroups.z`)
and assigns the48genericQ4gathered kernels88.95ms to routedMoE. Teacherprime
families are MoE90.05ms, dense17.77ms, QSA9.87ms, shared2.07ms, HC1.57ms,
embedding0.16ms and inputfuse0.34ms. Raw exact kernel timings are unaffected by
the classification correction. The producer source is fixed for future builds.

## Strong teacher-prime simplification

`FlashMTPState::Impl` owns onlyQSAstate, logical length, owner and poisonflag.
QSAstate contains normalized/RoPE keys, values, raw indexkeys, pooled indexkeys
and indexpositions. Targetteacherfeatures independently provide every incoming
hidden/tokenpair; teacherpriming ignores returned headhidden. Head cache rows
are computed before attention and the MLP. Attention outputs, o_proj, mlp HC,
routed experts, sharedexpert and final hyper injection do not update any
persistent state.

A dedicated teacher-prime method can preserve the original inputprojection,
attentionHC, q/k/v/indexprojection and cacheprepare/poolprefix, then submit
without computing attentionoutput or anyMLP. This removes the88.95ms expert
kernels, around10ms attention and additional output/MLPprojections. First use
exact existing128-row geometry. Compare every persistentcachebyte and follow
with identical real MTPproposals/verified generation before promotion.

Do not globally shorten `forward(..., FlashMTPLogits::None)`: that API promises
hiddenfeatures, and other committedfold callers may consume them. Expose a
separate teacher-prime operation or explicit private selector used only by
Worker promptpriming. A later bulkcommand can preserve128-row QSAappend/pool
order and safecontrolboundaries; normal prime command overhead is only~5ms,
so eliminating unused computations is the substantial win.

## Implemented opt-in teacher cache route

`FlashMTPForward::primeTeacherCache` is a separate operation. It retains the
original input/attention HC and q/k/v/index projections, then copies only the
QSA cache-writing prefix from the authoritative attention builder. Both fused
preparation and unfused BF16/F32 routes are supported. Destination graph storage
owns parameter copies; all buffer indices, parameter size/index, preparation
pipeline names, append presence and stop boundaries are validated. Submission
retains normal sticky diagnostics, poison behavior and logical offset updates.
Generic None/Last/All forwards retain their original complete computation.

`SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY` strictly accepts unset/0/1, defaults off,
and requires MTP=1 before model/backend loading. Worker selects it only for
sequential prompt priming, including singleton tails of batched prefill. True
grouped head priming currently keeps full None forward. Status identifies the
selected route, its actual successful call counter and that scope.

Unique build: `build/prefill4k-teacher-cache/`. Native CPU selftest and eight
invalid flag/dependency startup checks passed; independent source review found
no blocker. GPU qualification is queued exclusively with the root coordinator.
The current production profile/defaults remain unchanged.

Strict oracle (root only):

```sh
.venv/bin/python dev/benchmarks/prefill4k_attribution.py \
  --build build/prefill4k-teacher-cache --teacher-oracle \
  --tokens build/release/flash/prefill4k-fixture/code2048.tokens.json \
  --report build/release/flash/prefill4k-teacher-cache-oracle.json --run
```

Then compare normal three-trial controls using this same build with and without
`--teacher-cache-only`. Oracle timings follow large cache comparisons and mix
contexts, so they provide no service throughput claim. Qualification compares
all bytes of all five persistent planes after every fold, checks BF16 cache and
result values as F32, compares future proposal hidden/full logits/greedy IDs,
verifies generic None output, EOS inputs, 128/127 and unaligned pooling windows,
8,191-pair sparse cache, capacity/input guards, NaN poison publication and
truncate/reappend at 0/1/3/4/127 plus speculative rollback.

## Root GPU qualification completed

The separate-head oracle passed all 1,008 checks: 3,101,774,848 compared bytes,
all five cache planes exact, future proposal hidden/full logits/greedy exact,
generic None contract preserved, rollback/reappend and NaN poison publication
preserved. It checked 1,542,193,152 BF16 cache words as finite F32 values. The
mixed-context oracle timings are not a throughput benchmark.

Three warmed normal trials of cache priming measured target GPU 831.39 ms
(previous control 831.59 ms), teacher GPU 11.71 ms (previous 120.21 ms), and
teacher call wall 16.21 ms (previous 124.97 ms). Direct inclusive prefill is
2,397.2 versus 2,124.5 tokens per second, +12.8%. Every target hidden/logit hash
and first greedy token matched over all three trials. Numeric metadata is
finite with no implausibly tiny values. The main target phase is unchanged
within noise, consistent with saving only unnecessary teacher computations.
A same-new-build flag-off control and full HTTP lifecycle/quality/real coding
comparison remain pending; defaults have not been promoted.
