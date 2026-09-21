# Exact vocabulary and idle-latency fixes

This round tests concrete fixes to the two v8 findings. All work is local;
original checkpoint files and the qualified v7 profile remain unchanged until
a candidate passes correctness, lifecycle, memory, and ordinary HTTP gates.

## Vocabulary coefficient contract

The rejected packed Q8 route factored FP32 scale and bias after its dot product.
The active concurrent vocabulary cache first reconstructs each coefficient and
rounds it to BF16. Direct packed decode now retains that exact BF16 boundary
before BF16 matrix multiplication.

The first GPU screen checked every635,699,200 original vocabulary coefficient
against the active cache, with zero mismatches. All five full-vocabulary
four-row tests on a captured real trained-head input also had zero BF16 logit
mismatches and exact greedy IDs. This fixes the observed coefficient error;
it does not establish full-head continuation or service performance. Direct
threadgroup decode was slower (best2.098 ms versus1.582 ms control).

An independently derived dictionary stores the256 rounded coefficients for
each of41,151 actual scale/bias pairs. Its21.069 MB BF16 table plus19.866 MB
U16 group indices, alongside635.699 MB original codes, uses676.634 MB rather
than1271.398 MB expanded BF16 vocabulary. All10,534,656 dictionary entries
passed independent nearest-even checks, with673,280 coefficients sampled
against the persisted cache. GPU lookup decode also passed the full coefficient
and output checks. Its best tested threadgroup geometry remained slower;
barrier-free cooperative register decode is the next bounded screen.

Barrier-free MPP register decode found a winning geometry: M8N32/K64 with one
SIMD group. Its normal projection median was0.992 ms versus1.578 ms control,
about37% less GPU time. Real, random, cancellation, and sparse input patterns
all produced byte-exact full-vocabulary outputs. Shader validation passed too.
The shader still reconstructs and rounds coefficients to the original BF16
cache values. Other register geometries lost, so only this measured geometry
advances to full-head continuation and service qualification.

The real full-head fixture passed both128/2048-token contexts, all distinct
lane features, future proposals, and retained-prefix overwrite. Every checked
vocabulary word, premixer output, greedy ID and full QSA plane was exact.
Whole-head GPU medians fell3.257→2.563 ms at128 and3.534→2.919 ms at2048.
The production route is `SPLASH_FLASH_MTP_Q8_BF16_REGISTER=1`, limited to joint
Last outputs with2–4 vocabulary rows and the existing verified BF16-cache
baseline. It adds no weight or workspace allocation and reuses owned padding.
It entered service qualification with its low-level default off.

Service qualification passed all22 quality/lifecycle checks with all28
normalized outputs matching v7. Same-build OFF/ON and repeated ON kept all21
benchmark outputs, draft acceptance, verifier work and memory identical.
Across1022 unchanged head-decode calls, the new route saves8.07–8.33% GPU time;
exactly380 concurrent commands use it, saving0.605–0.624 ms each. Full-request
throughput differences remain small or noisy, especially long prompts whose
prefill dominates. The exact route is now the local v8 default after109
profile/launcher tests passed. Private idle mitigations stay disabled.

Normal launcher proof selected v8 automatically, using a binary and metallib
byte-identical to the qualified experimental runtime. Four cold requests each
completed128 output tokens, selected95 register-vocabulary commands/380rows,
then returned the scheduler to healthy idle. This cold smoke rate is not used
as a warmed benchmark score. Tracing is off and the service remains on8011.

## Idle driver processing

The two captured idle requests wire the same1378 buffers, including all21
original shards106.32 GB. Driver WireMemory interval union increased416.03→
596.01 ms while command completion increased183.85 ms. The134 subsequent
Unwire Requests exactly match349.39 MB of per-request trunk state, with no
weight-shard-sized release. This is PID/temporal driver evidence, not a direct
resource/command causality join.

The opt-in request trace now serializes Metal kernelStartTime/kernelEndTime
separately from GPU timestamps, with finite/monotone checks and explicit driver
scope.528 CPU checks passed; unavailable or reversed driver intervals become
null. The default still creates no trace/profile.

A private on-demand residency refresh preserved full logits and two continuation
steps, but did not reduce total latency. After nine seconds idle it spent759.3
ms in refresh, then648.4 ms after commit, totaling1.761 seconds. The no-refresh
control totaled1.395 seconds. The apparent shorter postcommit wait merely moved
work into preparation; refresh remains private and off.

A second private loader copies original bytes once into owned Shared Metal
buffers. It preserves21 bases/3748 offsets/106,320,429,056 bytes, compares every
copied byte, then releases all source file mappings. Startup copy took27.71
seconds and comparison6.25 seconds. The final allocation ledger does not charge
another original model; one temporary source shard is protected by admission.
Idle and full-service behavior are pending Root qualification.

Owned original backing did not improve idle latency: postcommit waits were
2125.7/2191.5 ms after nine seconds idle, while immediate requests waited
2.8–4.4 ms. All six requests completed correctly. The copy cost and lack of
latency benefit reject it as a default. The next bounded control retains and
resets one previously exercised mutable request-state allocation across idle,
with fresh request identity and no provisional verification tickets.

State reuse preserved exact logits and continuation, but matched idle AB/BA
totals were mixed: mean1640.14 ms fresh versus1632.70 ms reused, with1.24–1.34
seconds postcommit driver wait remaining. No Worker pool is enabled.

The last bounded control retained two successful completed command objects and
their allocation owners across idle. All six requests passed; accounting grew
by692.49 MB within the two-state bound. Idle waits remained1243/1296 ms versus
2.6–5.4 ms immediately. No same-build-off control establishes a gain; the large
stall remains. This extra retention stays private and disabled.

## Evidence

- `build/release/flash/v9-exact-q8-bf16-primitive-screen.json`
- `build/release/flash/vocab-bf16-dictionary-v9-cpu-qualification.json`
- `build/release/flash/v9-exact-q8-lut-primitive-screen.json`
- `build/release/flash/v9-exact-q8-lut-geometry-screen.json`
- `build/release/flash/v9-idle-driver-cpu-audit.json`
- `build/release/flash/v9-idle-residency-refresh-saved-screen.json`
- `build/release/flash/v9-owned-original-initial-status.json`
- `build/release/flash/v9-owned-original-idle-http.json`
- `build/release/flash/v9-exact-q8-register-validation.json`
- `build/release/flash/v9-exact-q8-register-performance.json`
- `build/release/flash/v9-exact-q8-register-patterns.json`
- `build/release/flash/v9-exact-q8-full-head-proof.json`
- `build/release/flash/v9-persistent-state-idle-control-cpu-summary.json`
- `build/release/flash/v9-completed-retention-cpu-summary.json`
- `build/release/flash/v9-mtp-q8-bf16-register-independent-service-audit.json`
- `build/release/flash/default-v8-normal-concurrent-smoke.json`
- `build/release/flash/default-v8-normal-final-idle-status.json`
- `build/release/flash/default-v8-optimization-summary.json`
