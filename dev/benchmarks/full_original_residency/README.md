# Private complete original-weight residency control

This experiment is a private Worker snapshot. It changes one startup residency
selection, not the model mappings, original payload bytes, kernels, dispatch
order, argument-buffer pointers, tensor views, or recurrent-state math. The
production Worker and local v7 defaults are untouched.

Set `SPLASH_FLASH_PRIVATE_FULL_ORIGINAL_RESIDENT=1` with the normal v7 profile.
The flag is absent/off by default and rejects values other than canonical `0`
or `1`. It is mutually exclusive with `SPLASH_FLASH_ORIGINAL_TEXT_RESIDENT=1`.
The exact source/layout fingerprints, model geometry, 21 native original bases,
and 106,320,429,056 source bytes are checked before the request. This deliberately
includes the mixed PLE and vision bases omitted by the earlier 13-base test.

Existing saved-derived selection contributes 1,113 native allocations and
38,006,423,552 bytes. The expected union is 1,134 native allocations and
144,326,852,608 bytes (134.415 GiB). Those bytes are already charged by their
existing mappings; the registration deduplicates native allocations and makes
no weight copy or new backing charge.

Full-original admission reuses the original experiment's conservative host
policy: valid host measurement, normal system/engine pressure, permitted
growth, and headroom above the protected reserve for all original source bytes
plus 2 GiB. Insufficient headroom leaves saved-only residency available and
reports the reason. A successful request establishes GPU accessibility, not a
verified guarantee of physical pinning.

`/status.private_full_original_residency` reports the source/derived/registered
counts and bytes, host preflight, whether the sources joined the union, failure
reason, and host API setup duration. `startup_before_residency_ms` and
`startup_after_residency_ms` measure native startup entry to the corresponding
API boundaries. They exclude the small final Worker/Ready construction and
cannot claim asynchronous driver wiring is complete. Root must also measure
external launch-to-readiness and launch-to-first-content wall time.

## Why this control

The backend does not enumerate all model resources indiscriminately. At
`runtime/metal/MetalBackend.mm:2074` it gathers only indirect allocations of
bound argument buffers, deduplicated per dispatch. At line 2284 it declares
that list `Read`; direct bindings at line 2289 use their native owning buffers.
Original tensor views share 21 whole-shard native bases created by
`runtime/flash/FlashWeights.mm:327`.

The fused PLE gather needs dynamically indexed original source data. Its
384 pointers (`runtime/flash/FlashPLEFused.cpp:93`) deduplicate to eight native
bases covering 40,260,141,056 bytes, including co-located inactive vision data.
A narrow view of the 675,430,400-byte token embedding still binds the
5,204,606,976-byte first base. Layer 0's logical original tensors span
1,477,061,728 bytes across two native bases totaling 10,257,760,256 bytes.
These are CPU manifest sums, not physical memory traffic measurements.

The selected INT8 prefill path always encodes original Q4 miss dispatches
(`runtime/flash/FlashInt8ExpertStore.mm:259` and line 303), so a high cache hit
rate cannot safely remove original resource declarations. Removing PLE or
miss resources would make valid dynamically selected accesses unavailable.

The prior private per-tensor loader removed shard-sized WireMemory calls but
left similar total wiring time and a 1.333-second cold submission wait. Its
mixed HTTP scores did not justify changing default storage granularity. The
earlier 13-base original-text residency test excluded all PLE bases and showed
no useful matched HTTP gain; it is a different selection from this control.

Root's fixed native-v7 prefill sequence now demonstrates a first-model cost:
first 128-token singleton waits 1,748.48 ms after commit before GPU start,
then 2048-token singleton calls wait 13.06 and 4.46 ms. The first four-request
128-token cohort waits 114.48 ms, then 2048-token cohorts wait 21.60 and
22.03 ms. Ordinary CPU pipeline creation happens before command commit, so
it cannot explain this measured post-commit span. Driver preparation and
memory wiring remain hypotheses; temporal WireMemory overlap is not a
direct command/resource causal join.

A synchronous graph split could reduce the first resource frontier, but
the backend permits one outstanding ticket (`MetalBackend.mm:659`). It cannot
overlap later command preparation and earlier GPU work without a separate
backend contract change. This residency control is smaller and specifically
tests the previously omitted PLE frontier. A lower first-request TTFT must
be evaluated alongside startup duration; merely moving work before Ready is
not a reduction in launch-to-first-content time.

## CPU build and qualification

```sh
make -f dev/benchmarks/full_original_residency/Makefile \
  BUILD=build/flash-private-full-original-residency-v1 -j8 flash-next
build/flash-private-full-original-residency-v1/splash-flash --cpu-self-test
```

The fresh runtime and adjacent Metal 4.1 library compiled. All 44 inherited
Worker CPU checks passed, plus 22 private flag-process cases and 77 standalone
policy assertions under AddressSanitizer/UndefinedBehaviorSanitizer. The
CPU-only qualification and frozen artifact hashes are recorded in
`build/release/flash/private-full-original-residency-cpu-qualification.json`.
This is not GPU, service-quality, lifecycle, or performance qualification.
Root owns those serial GPU/device runs.

## Root runtime result and recurring-stall audit

Root's full-original control successfully registered the expected union. The
host residency setup itself took 1,370.978 ms; native startup through that API
was 20,134.121 ms. The first 128-token request then still waited 1,895.872 ms
after commit before GPU start and reached first content at 2,289.667 ms, versus
1,748.484 ms and 2,149.785 ms in the fixed normal-v7 baseline. Actual GPU time
was 352.942 versus 362.350 ms. Remaining sequence calls were similar. This is
a negative first-request result, not just successful admission; the private
flag remains off and does not justify additional full-residency runs.

The fixed normal-v7 trace includes a later HTTP benchmark warmup. Its target
prefill again waited 1,811.181 ms, despite earlier 2048-token calls waiting
only 4.462–13.057 ms. Critically, it followed an 8,886.056 ms interval from
the prior command's completed callback to the next backend submission.
Every other noninitial prefill followed its previous command within
27.57–109.75 ms and had a delay no larger than 114.480 ms.

The later warmup was the first MTP-eligible request. That is a confound, not
proof of head-state-induced invalidation. `captureHidden` changes only the
returned handle at `FlashForward.cpp:1173`; both 2048-token target graphs have
the same 2,897 dispatches. Worker admission always creates and zeros 134 trunk
state buffers totaling 349,388,800 bytes. Eligibility adds five head QSA buffers
and one fold-feature buffer, totaling 19,791,872 bytes. Prior immediate
four-request calls already created and destroyed roughly 1.4 GB of trunk
state successfully. The immutable residency lease is retained by the Worker
and backend until shutdown; request removal does not end or remove it.

Driver reclaim after inactivity and lazy driver preparation remain plausible
alternatives. A matched-idle ineligible request versus an immediate eligible
request is the bounded next diagnostic. The trace does not record a causal
resource join or driver wiring for this recurring span. The CPU audit is
`build/release/flash/v8-recurring-prefill-stall-cpu-audit.json`.

## Matched idle control

Root controlled idle time independently of head eligibility, reusing the same
128-token prompt in one engine instance with zero cached tokens. All six
requests completed successfully, all target graphs contained 1,709 dispatches,
and the final scheduler was healthy and idle:

| Request | Output budget | Idle before request | Commit to GPU start |
| --- | ---: | ---: | ---: |
| Initial ineligible | 1 | Initial | 1,453.060 ms |
| Immediate ineligible | 1 | 0 s | 5.446 ms |
| Idle ineligible | 1 | 9 s | 1,790.656 ms |
| Immediate eligible | 2 | 0 s | 9.121 ms |
| Idle eligible | 2 | 9 s | 1,708.755 ms |
| Immediate after head | 1 | 0 s | 5.916 ms |

This reproduces the recurring delay with MTP disabled for the request and
avoids it when MTP is newly eligible but the model is warm. Idle is therefore
a controlled trigger. The exact driver cause remains unproven: the existing
trace does not distinguish idle memory reclaim/re-wiring from process/device
wake or other driver preparation. The CPU join and status audit is
`build/release/flash/v8-idle-prefill-control-cpu-audit.json`.

One bounded private diagnostic would split the unchanged target graph after
the embedding and HC-expand dispatches. Those first two dispatches need only
the first original native base (5,204,606,976 bytes), versus all original bases
in the complete command (106,320,429,056 bytes). Two synchronous tickets retain
the current backend contract and cannot promise CPU-driver/GPU overlap. The
per-segment delay and total first-content latency can distinguish a roughly
fixed first-command wake cost from footprint-sensitive resource preparation.
A lower first-segment delay without lower total latency is merely
redistribution and must not become a default optimization. No such split,
backend concurrency change, residency refresh, or background GPU keepalive
has been implemented in this audit.
