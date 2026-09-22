# All-row Full512 lifecycle and continuation audit

CPU-only read-only audit of the reports and runners on 2026-09-21. No models,
GPU commands, service requests, production edits, or large operand scans were
performed by this audit. The GPU evidence below is root's saved evidence.

The candidate is not ready for default promotion: its target arithmetic is a
new numerical derivative, singleton decode regressed, and whole-target state,
joint speculative verification and phase-specific interrupted-request
qualification remain incomplete. Root's subsequent eight-case run now proves
operational concurrency, cancellation, native-work deadline and recovery.

## Current saved evidence

| Gate | Saved evidence | Scope and limitation |
| --- | --- | --- |
| Derivative identity | `ultra-locality-prefill4k-allrows-full512-first.json`: all-row and original-target-GPU-omitted markers true; derivative `2e858faa201554642a443d48d38b7302159fe1a811c028a895454cc8ce5c073d`; Full512 manifest `ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1`; 24,576 experts / 121,173,442,560 mapped bytes. | Actual loaded candidate; does not prove numerical equivalence. |
| Startup admission | `prefill4k-allrows-admission-first.json`: original GPU bytes 6,370,164,736; planned trunk 145,437,966,336; normal pressure; stage engine/host bounds fit. | Sample after original mapping and before trunk construction. Later fresh `tryReserve` checks remain authoritative. |
| Final allocation/admission | Native current/peak 158,211,801,088 bytes; engine headroom 89,178,315,162; host available 88,418,041,856 versus reserve 27,487,790,694; growth allowed; reservation and state denial counts zero. | Native allocation ledger and live host-availability sample. Neither is an OS wired-memory measurement or a refusal test. |
| Full output budgets | Four uncached 2,048-token coding requests, 256 completions each, HTTP 200, full budget, same response SHA, no Metal failures. | First sample warm-up; three warmed repeats. Same-candidate repeatability only. |
| Performance | Warm medians: prefill 2,976.02 tok/s; native prepared-prediction decode 60.55 tok/s; actual streaming decode 60.46 tok/s; accepted/proposed drafts 0.7384. | `ultra-locality-prefill4k-v11-default-proof.json` has one warm baseline sample: prefill 2,406.91, stream decode 71.96, draft acceptance 0.6667. Improved prefill and acceptance do not compensate for the observed decode regression. |
| Frozen tasks | `prefill4k-semantic-prefill4k-allrows-full512-first.json`: completed, 22/22 coverage, 20 passed / 2 failed / 0 skipped, coverage errors empty. | Task report `valid=false` because arithmetic_signed and arithmetic_inventory fail. |
| Task comparison | `prefill4k-semantic-top64-vs-allrows-full512-v1.json`: independently regraded comparison `valid=true`; no new regressions versus Top64 19/22; python_merge_counts resolved. | Numeric source quality and eight protocol/lifecycle cases explicitly remain separate requirements. This is not a general model-quality certificate. |
| Tiny tail | json_nested_copy has 2,049 prompt tokens, physical windows `[2048,1]`, 16 successful teacher calls, no miss graphs, native completion and idle status. | One-token prefill tail is observed. Small graph deltas combine tiny tails, decode and verifier; they are graph-construction counters, not independent primitive checks. |
| Initial serial cleanup | First report: 26 submitted / 26 completed / 0 cancelled / 0 failed; scheduler queued/active/prefilling/decoding/waiting zero, no command in flight. Transport ready, fresh, not recovering, pending zero, restarts zero; Metal healthy. | This first report covers serial singleton cleanup only; its B2/B3/B4 counts are zero. The later service run below adds operational concurrent and control coverage. |
| Eight actual service cases | `ultra-locality-prefill4k-allrows-full512-service-quality.json`: completed/pass/coverage_complete all true; every one of eight case error lists empty; 11 submitted / 9 completed / 2 cancelled / 0 failed; exact same all-row derivative identity; final fresh healthy idle status and zero Metal failures. | Streaming arithmetic, structured system/developer response, four concurrent requests, forced tool, tool continuation, cancellation, deadline and recovery arithmetic all passed. This is an actual all-row service result. |
| Actual concurrent hardware work | Service prefill B3/B4 counts are 1 each; decode B3=1 and B4=12; four distinct concurrent HTTP requests completed with checked answers and four-way overlap. Native allocation peak 159,921,668,096 bytes; final current 158,211,801,088; host available 83,307,151,360, growth allowed and no reservation/state denials. | Real batched AR prefill/decode is established. Concurrent drafted/accepted counts are zero; joint_cohorts_attempted and joint head command counts are zero. This does not qualify joint speculative verifier/head state. |
| Short prefill and controls | Actual all-row requests consumed 48/49/63/85/144/153-token prompts below the former 256-row threshold. Cancellation disconnected after content, with prefill and one decode command observed, native cancelled +1 and idle cleanup. Deadline returned HTTP 504/request_timeout after 85 native prefill rows, zero output and cancelled +1. Following recovery arithmetic completed normally. | Small whole-prefill, native-work timeout, cancellation and recovery are now observed. Exact 255/256/257 guards, long-prefill/teacher phase-specific cancellation and command-level cancel timing remain separate gates. |
| Service unload | Root confirmed the service wrapper auto-unloaded after completion; no frontend/native PIDs or listening service ports remained at root's check. | This is root's cleanup observation, separate from the quality JSON's pre-stop idle snapshot. Active-work teardown, unload/reload identity and physical-memory release tests remain pending. |

The general expert counters include small rows; `large_row_*` counters exclude
rows below 256. Their difference shows construction of small-row graphs. Source
and CPU partition witnesses prove the below-256 bypass and fail-closed layout
checks, not GPU numerical behavior or guard preservation.

The eight-case run is no longer a pending operational smoke. Its report has no
top-level `errors` key: all eight per-case `errors` lists are empty. It records
native work before the deadline, so this particular timeout is stronger than
the runner's allowed frontend-only case. Request command tracing is still off,
so the report does not establish the precise command/teacher phase in which a
cancel or deadline arrived.

## Existing passes that do not qualify this candidate

* `deep15-prefix-state-qualification.json` passed 64 retained-prefix cases using
  raw original coefficients, dense caches/QSA MPP disabled, and no all-row
  derivative. It is not a Full512 rollback result.
* `prefill4k-wide-state-code8192-rows8192-v1.json` passed exact/finiteness checks
  for the Top64 wide geometry, 134 persistent planes and seven AR appends. It
  contains no all-row identity, teacher state or pending verification.
* `http-direct-state-final-qualification.json` passed its original eight cases,
  but its loaded identity has no all-row numerical derivative.
* `prefill4k_allrows_http.py` freezes launch policy and file identities. It does
  not send requests or assert lifecycle behavior.
* `qualify_flash_http.py` cancels after visible content. Its deadline case may
  pass before native admission/work and only reports that scope. Its concurrent
  case requires four HTTP overlaps/IDs but reports rather than requires actual
  native joint verifier work. Root's new actual run independently demonstrates
  B3/B4 batched AR work and a native-prefill deadline, within those stated scopes.

## New artifact status

`build/prefill4k-allrows-rollback/rollback-oracle` and
`dev/benchmarks/prefill4k_allrows_rollback.md` now provide a compiled CPU-checked
one-trunk oracle: one Full512 Store, three independent request states and owned
CPU snapshots. It avoids the two-Store memory trap described below. The matrix
covers four modulo-4 prefix phases, verify4 with retain1/3/4, terminal abort0,
pending-state guards, destroyed pending ownership and seven continuations.
Same-verify4 branches with differing discarded suffixes are the primary exact
live-contamination comparison. Singleton R1/R3 versus verify R4 producer
differences are retained as separate numerical diagnostics. The invocation
witness and a partial checkpoint stream are not a completed GPU result;
GPU rollback qualification remains pending as of this update. The artifact
does not cover every verify/retained size up to16, joint lanes, MTP teacher or
HTTP lifecycle.

The new all-row I8 QMV reduction candidate is also unqualified. Its source/CPU
witness or launch preview does not inherit the current derivative's eight-case
service result. Its own primitive numerical guards, retained-prefix state,
actual service behavior and matched speed must be measured before promotion.

## Exact remaining test plan

Root must serialize all model/GPU tests and use fresh reports and a fresh
admission snapshot. Keep depth 3, teacher cache-only, PLE SSD, dense tiles and
the exact certified Full512 identity fixed. Do not enable omitted original
target resources for a reference path.

1. **Small-row primitive and prefill boundaries.** With private source-copy
   instrumentation, force target physical rows 1/2/3/4/8/15/16/17/127/128/255/
   256/257, including short valid-route tails and maximum expert/rank IDs.
   Require all 48 target layers to use Full512 and no original-Q4 target graph;
   verify input/output guards, sticky diagnostics and no nonfinite output.
   Compare against a bounded CPU/F64 derivative reference with source
   coefficient certificate; include zero, alternating-sign cancellation and
   partial-bucket fixtures. Whole-request prompts must cover exactly 255/256/
   257 and 2048/2049/4096/4097/8192/8193 tokens, uncached, checking actual physical
   windows and teacher pair counts. Below-256 rows stay Full512; there is no
   legitimate original-Q4 fallback in this candidate.
2. **Same-derivative target state and retained-prefix rollback.** Qualify the
   new one-trunk oracle with the actual all-row policy, Full512 manifest,
   current caches and source hashes, then extend its bounded matrix. Compare
   independent request states under the same numerical derivative. For
   nonzero prefixes with length modulo 4 equal
   to 0/1/2/3, force verify windows 1 through 16 and every legal retained length
   1 through the window size. Reject 17-row windows without state mutation.
   After commit, compare every GDN/PLE physical persistent plane, valid QSA
   prefix, scalar length/poison/pending flags, and copied logits/hidden; append
   the same seven frozen tokens and compare again. Primary rollback branches
   must use the same verify shape and retained tokens, differing only in the
   discarded suffix, and pass exact live-state/future-contamination checks.
   Singleton-versus-verifier row-shape differences require separate numerical
   evidence; they alone do not establish a rollback defect. Provisional QSA tails beyond
   committed length are separate diagnostics, not live-prefix equivalence.
   Add explicit abort/discard pending-verification cases where the API permits.
3. **Real joint speculative lanes and ownership.** Operational B3/B4 batched
   AR work now passes. Force actual native B2/B3/B4 target
   verifier/head work, with distinct frozen prefixes, unequal retained lengths,
   modulo-4 boundaries and mixed completion/EOS/budget. Compare each survivor
   to an independent same-derivative singleton; force cancellation/deadline and
   generation drop for a member before commit. Assert no null/stale slot is
   reused, no answer/state contamination, exact borrowing/copy ownership, and
   healthy survivor continuations. Client concurrency alone is insufficient.
4. **Cancellation/deadline during each phase.** Generic native-work deadline,
   disconnect-after-content cancellation and recovery now pass. Retain cancel/deadline
   send-and-receipt evidence plus request IDs/generations and command phase
   profiles. Cancel during a large target prefill, teacher-cache priming, and
   target verify/decode; repeat with deadlines and with a healthy peer. A
   deterministic private transport/test hook may deliver controls at the
   selected in-flight/boundary point. Require that the named phase actually
   ran, exactly one native terminal for the affected request, no subsequent
   work or tokens for its cancelled generation, and a fresh bounded idle
   snapshot. A frontend 504 or disconnect after content alone cannot pass
   prefill/teacher coverage.
5. **Memory refusal and recovery.** Test below-trunk startup budget (e.g.
   128G), then a budget that admits static workspaces but not a new eligible
   state, using the fresh exact allocation plan. An eligible state requires
   582,959,104 target bytes plus 39,010,304 MTP bytes at capacity 16384; choose
   remaining engine headroom strictly between those plans with allocation
   alignment checked. Refuse an eligible request, then complete an ineligible
   constrained-schema/AR request to show the engine is still usable. The latter
   request's AR eligibility and its actual reservation must be recorded. The
   refusal must show a real
   governor/state denial, bounded capacity exhaustion or timeout, no stranded
   queued/active/in-flight work, and healthy subsequent recovery. Preserve
   reservations; do not artificially pressure the whole host or bypass the
   governor. Existing pre-backend bad-policy refusals do not pass this gate.
6. **Unload/reload.** Root verified the completed service auto-unloaded with
   no remaining PIDs/ports. Capture HTTP/native PIDs and initial host/native memory.
   Stop at idle and during active wide work; prove both child/frontend exit,
   port closure and memory/residency release. Reload the identical candidate,
   assert the same derivative/store/file identities with a new instance ID,
   run a fixed arithmetic request plus full-budget coding request, then stop
   and verify unload again. Record physical/host memory separately from the
   native allocation ledger.
7. **Performance after qualification.** Run matched uncached full-budget
   singleton controls for 2K/4K/8K prompts and 2048/4096/8192 singleton arenas;
   preserve actual batch lanes at 2048. Separate warm-up, report at least three
   warm samples, native prefill and actual streaming decode, draft acceptance,
   host/native peak and cleanup. Repeat baseline controls if differences in
   warmth or residency remain. Default promotion requires resolved numerical
   gates and an acceptable decode result, not reaching 4K prefill alone.

### State-oracle invocation trap

`prefill4k_wide_state.py` clears inherited experiment flags and restores Top64
defaults. `--build build/prefill4k-allrows-full512` alone cannot qualify all-row.
The current oracle creates two Forward owners, each owning an expert Store;
directly enabling Full512 duplicates approximately 121 GB of immutable store
resources. Adapt it to share the certified immutable Store with independent
workspaces/request states, or construct sequential bounded snapshots, and
preflight actual ownership/allocation reservations before running. Do not
interpret its Top64 byte-exact result as evidence for this arithmetic change.
The new one-trunk rollback artifact is the bounded alternative now ready for
root's GPU qualification; its CPU preparation does not provide that result.

The generic protocol smoke already passed in root's new service report; repeat
it for any changed QMV producer, derivative or policy after that candidate is launched:

```sh
.venv/bin/python dev/benchmarks/qualify_flash_http.py \
  --base-url http://127.0.0.1:8011 \
  --output build/release/flash/CHOOSE-FRESH-allrows-eight-case.json \
  --nonce CHOOSE-FIXED-CONTROL-NONCE
```

Require complete coverage, zero skipped/failed cases, stable derivative identity
and fresh idle/healthy terminal status. This command does not implement all
seven remaining gates above and was not executed by this audit.
