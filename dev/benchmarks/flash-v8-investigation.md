# Continued Flash-Next optimization

This round starts from local v7 and retains the original checkpoint and saved
operand stores. GPU jobs are serial. Private prototypes do not enter the local
default until ordinary HTTP, continuation, and lifecycle checks qualify them.

## Prefill resource investigation

The prefill probe uses the frozen 128/2048-token HTTP prompts with one output
token, which excludes MTP and head priming. All fifteen requests completed,
used zero cached tokens, and left the scheduler idle. The corrected probe
waits for native terminal bookkeeping after HTTP completion and retains its
report if the final idle check fails.

The first 128-token request waited 1748.5 ms after commit returned before GPU
start, then spent 362.4 ms on GPU. Immediate subsequent 2048-token singleton
requests waited only 13.1 and 4.5 ms and spent about 846–851 ms on GPU. First
four-request prefill waited 114.5 ms; subsequent 2K four-request commands waited
21.6 and 22.0 ms. Thus prompt length alone does not explain the large stall.

The backend declares actual bound resources. Its indirect `useResources`
contains only owners of the bound argument buffer; the PLE table legitimately
references all128 source shards, deduplicated to eight original native buffers.
Direct tensor views bind their complete native owners. The top64 expert cache
still submits original-Q4 miss kernels, so the original banks remain possible
inputs. Removing these declarations would be incorrect.

A private startup experiment registered all21 original native buffers plus
the existing1113 derived allocations: 144,326,852,608 bytes in one residency
set, with no additional weight backing. Its host setup took1371.0 ms, then
first-request delay remained1895.9 ms, with2289.7 ms to first content. Full
original residency did not remove the wait and stays off.

The large wait returned on a later full-output benchmark warmup after8.886
seconds without GPU work. Target feature capture changes only the returned
buffer handle; both2048-token target graphs have2897 dispatches. MTP admission
adds19.79 MB rather than new weight backing. Idle reclaim is a hypothesis to
test, not an established driver cause.

The matched idle probe uses the same128-token prompt and zero cache reuse.
Immediate ineligible requests waited5.4–5.9 ms; after nine seconds idle, the
ineligible request waited1790.7 ms. An immediate MTP-eligible request waited
9.1 ms; after the same idle period it waited1708.8 ms. All six requests completed
and left the scheduler healthy and idle. Idle time reproduces the delay
independently of MTP eligibility; the specific driver mechanism remains unknown.

The private resource-frontier split preserved full logits and two continuation
steps. After idle, its small embedding/expand command still waited819.8 ms,
then the remaining layers waited604.4 ms. Total call time was1.778 seconds,
versus1.403 seconds for the unsplit idle control in that standalone process.
It excludes Worker batch arenas and HTTP, so this is resource attribution
evidence rather than a matched service score. Splitting redistributed the
wait and did not improve total latency; it stays private and off.

## GPU candidates

The trained-head stage trace contains40 timed dispatches with identical full
hidden/logit outputs to its ordinary replay. Vocabulary projection consumes
about45% of timed GPU work; this is a stronger head target than the previously
screened command chain. Stage profiling changes encoder boundaries and is
used only to rank work. The first report failed a strict CPU consistency check
because one summary used six-digit serialization; a separate v2 uses full
double precision and passes. The original failed report is retained.

Packed HC-down two-row reuse preserves every tested FP32 partial, BF16 raw
projection, activation, and injection bit. All six16-row cases were slower,
so this candidate remains private and disabled.

The private dense tile screen compares candidates with both original F32 QMV
and the current selected cache kernel in rotating timing order. Its54 cases
and80 cancellation traps passed the strict relative-L2 bound. QSA output Q6
with M8N32 SIMD4 repeated20–33% GPU gains over the selected route, with improved
wall time. Other formats, tails, and full-model effects remain qualification
gates; primitive speedups are not HTTP gains.

All35 Q5/Q6 source/tail cases passed both the private and integrated bridge
screen, with byte-exact outputs against the current selected cache route.
Q8 rows4/8 failed the strict comparison to raw F32 arithmetic and are excluded
from the new route. The integrated route is default-off while Root performs
full service quality and matched ON/OFF performance.

The QSA-only service passed all22 quality/lifecycle checks and kept all21
benchmark generations identical to v7. Its source/layout/acceptance histograms,
170 verifier calls, and memory ledger were unchanged. Attribution must compare
verifier time separately: the primitive route does not change prompt-sized
prefill or trained-head projection math.

The same-build ON/OFF service control found no verifier improvement:170calls
took10875.7 ms ON and10870.9 ms OFF. HTTP differences were mixed: −1.07% short
singleton, +1.41% short four-request, +1.27% long singleton, and −0.89% long
four-request. All outputs, acceptance, memory, and identity checks matched.
Despite the isolated tile win, this workload has no repeatable service gain;
the QSA route stays opt-in and v7 defaults remain active.

The original R1 Q8 vocabulary geometry screen tested22 exact-arithmetic layouts
against the current control. All full-vocabulary BF16 words, greedy IDs, and
sticky diagnostics matched, including a captured real trained-head input.
There was no stable performance gain; the original C2/SIMD8 kernel remains.
Early timing outliers are retained and excluded from steady conclusions.

The private concurrent trained-head candidate used the existing original-Q8
matrix primitive in place of the BF16 expanded vocabulary cache. At context2048
it preserved full premixer hidden state, all QSA buffers, and all greedy IDs,
but each full-vocabulary lane had relative-L2 error about0.002 against the
strict0.0001 gate. The completed report is rejected, despite faster measured
head execution. Current cached vocabulary rounds coefficients to BF16 before
the matrix product; the candidate uses original codes with factored FP32
scale/bias arithmetic. Those contracts differ. No bound was relaxed and the
candidate stays private. A future packed decode must preserve the rounded BF16
coefficient boundary before evaluating its speed and output fidelity.

Across six restored pairs, original joint-head GPU median was3.4823 ms and
candidate median2.7701 ms (20.45% faster, unqualified). Full-vocabulary error
reached0.0020585, more than20 times the unchanged limit. A packed BF16 decode
could use64-element K blocks in threadgroup memory, but its40 partial matrix
accumulations may differ from the current whole-K reduction. That is a new
numerical qualification problem, not a reason to approve the rejected route.

The rebuilt normal launcher retains the36-default v7 profile. Experimental
QSA output routing and request tracing are disabled. The final service smoke
and idle status are stored separately from the experimental benchmark reports.

## Evidence

- `build/release/flash/v8-prefill-sequence-baseline-fixed-http.json`
- `build/release/flash/v8-prefill-sequence-baseline-cpu-summary.json`
- `build/release/flash/v8-full-original-prefill-sequence-http.json`
- `build/release/flash/v8-full-original-initial-status.json`
- `build/release/flash/v8-recurring-prefill-stall-cpu-audit.json`
- `build/release/flash/trained-head-v8-v2-2048-r1-stage-summary.json`
- `build/release/flash/hc-down-packed-reuse2-r16-screen.json`
- `build/release/flash/dense-f32-tiles-v8/quick-screen.json`
- `build/release/flash/dense-f32-tiles-v8/qsa-o-q6-repeat.json`
- `build/release/flash/v8-idle-control-http.json`
- `build/release/flash/qsa-out-f32-n32-production-all-source-tails.json`
- `build/release/flash/head-q8-r1-v8-real-input-screen-cpu-audit.json`
- `build/release/flash/v8-prefill-frontier-whole-first.json`
- `build/release/flash/v8-qsa-out-n32-quality.json`
- `build/release/flash/v8-qsa-out-n32-on-http-performance.json`
- `build/release/flash/v8-qsa-out-n32-independent-service-audit.json`
- `build/release/flash/joint-head-q8-v8-ctx2048-screen.json`
- `build/release/flash/joint-head-q8-v8-ctx2048-rejection-cpu-summary.json`
- `build/release/flash/v8-final-v7-normal-smoke.json`
- `build/release/flash/v8-final-v7-normal-idle-status.json`
