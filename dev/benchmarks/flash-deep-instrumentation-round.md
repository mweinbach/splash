# Flash-Next instrumentation and optimization

This local round starts from the qualified v6 profile and the original
Qwen3.8-Flash-Next-oQ4e-mtp checkpoint on the M5 Ultra. No remote repository or
original checkpoint payload was changed. Measurements use the frozen HTTP plan:
128 or 2048 prompt tokens, 128 output tokens, greedy sampling, reasoning off,
one or four requests, and no cached prompt tokens. Four-request throughput is
the aggregate rate for the whole wave, including prefill and queue time.

## What the instrumentation found

The actual trained, 16-row target verification graph contains 2163 dispatches.
In the stage-sampled graph, dense projections took 23.4% of command time, HC
mixing 19.9%, MoE 20.9%, GDN 4.9%, and QSA 6.4%. Encoder changes perturb this
measurement, so these percentages identify work to investigate; they are not
ordinary HTTP performance scores. The real expert routes contained repeated
assignments, but grouping them into fewer jobs reduced occupancy and was
approximately three times slower in the tested implementation.

The native system trace also found a 1.405-second submission-to-GPU-start wait
on a prefill command. WireMemory events covered 614.5 ms of that interval,
including requests matching all 21 original weight shards. The association is
temporal; the driver records cannot establish a direct command/resource join.
A private per-tensor buffer loader removed the shard-size wiring requests but
left a 1.333-second wait and similar total wiring duration. Its steady HTTP
scores were mixed, so it remains disabled.

## Changes supported by the evidence

`SPLASH_FLASH_HC_UP_F32_MPP` uses the existing original FP32 coefficient cache
for main-model HC up/mix at 4 through 16 real rows. Its matrix tile processes
four streams together while preserving the BF16 projection, sigmoid, product,
stream-sum, and mean stages. The FP32 reduction order is an explicit numerical
alternative. The tested primitive mixed outputs were exact; that is not a
universal bit-equivalence claim. Trained MTP and prompt-size HC windows retain
their original routes.

HC-only HTTP tests kept all 21 benchmark generations unchanged and passed all
22 service checks. Across the same 170 target verification calls, GPU time
fell from 12.158 to 11.051 seconds, a 9.11% reduction. The callback wait grew
slightly, which supports a compute improvement rather than a scheduling gain.

`SPLASH_FLASH_GDN_LAZY_ROLLBACK` retains the initial FP32 recurrent state and
prepared inputs once, then replays only the recurrence when a prefix is
partially accepted. Full acceptance needs no recurrence restore command. It
avoids writing every intermediate full-state snapshot and reduces the combined
singleton/joint workspace allocation by 2,490,302,464 bytes (2.32 GiB). All 28
quality responses matched HC-only exactly, and all 22 service checks passed,
including cancellation, deadlines, concurrency, and subsequent recovery.

Relative to HC-only, lazy rollback saved 144.2 ms of verifier GPU time over the
same 170 calls. The same seven partial restores cost 8.142 ms rather than
2.340 ms, adding 5.802 ms of replay work. Draft acceptance and prefix/depth
histograms were unchanged. This is a favorable tradeoff on the tested high
acceptance workload; lower acceptance needs its own measurement.

Initial combined ON and same-binary OFF medians, tokens/s:

| Prompt / requests | OFF | ON |
| --- | ---: | ---: |
| 128 / 1 | 101.187 | 107.394 |
| 128 / 4 | 155.734 | 167.744 |
| 2048 / 1 | 67.367 | 68.868 |
| 2048 / 4 | 80.330 | 82.954 |

## Other experiment and remaining gates

The private four-proposal GPU head chain runs three unchanged R1 head bodies
in one command with GPU indirect dispatch guards for EOS, quota, and errors.
Real target features at 128 and 2048 contexts produced exact hidden states,
greedy records, QSA state, and rollback continuation logits. API host time
improved by 5.7–8.3%, while GPU time was slightly worse. This is head-only
component evidence; it is not enabled in service or at depth 15. Forced late
EOS is CPU policy coverage, not a tested late-EOS GPU execution.

Cached HC-down was screened separately. Literal row reuse was bit-exact in all
18 cases, but slower than the existing packed kernel. Whole-K and split-K MPP
variants were also slower at 16 rows, and four cases failed the strict numerical
gate. Those kernels remain private and disabled. Expanding packed weights can
increase memory traffic, and reusing rows can extend accumulator lifetimes;
neither approach guarantees a gain on a small output matrix.

The local v7 profile adds both qualified flags. Launcher/profile validation
passed 117 focused tests. The fresh runtime includes HC geometry/fallback
counters and GDN full/partial/terminal/abort counters, explicitly reporting
encoded graphs and logical byte footprints rather than physical bandwidth.

An opt-in trace is enabled with
`SPLASH_FLASH_REQUEST_COMMAND_TRACE=/absolute/fresh.jsonl`. It creates an
exclusive local file and records request/generation labels for trunk prefill,
head priming/folding/drafting, target verification, prefix restore, and AR
decode. Existing Command profiling preserves encoder boundaries. Hardware
GPU timestamps are compared with host commit timestamps only when the
Mach/steady clock bridges are valid and consistent. Scheduled callback time
is kept distinct. The default creates no trace or profiles.

The trace passed 514 CPU checks and an actual cold 2048-token singleton plus a
four-request wave: 325 records, zero missing or unexpected profiles, successful
request termination, and healthy idle status. Diagnostic HTTP rates are not
used as default performance scores. The untraced normal-launch confirmation
repeated the gain:
107.512 / 166.982 / 69.118 / 82.824 tokens/s for short singleton, short
four-request, long singleton, and long four-request workloads respectively.
The final normal build passed all 22 quality/lifecycle checks again, with zero
new task regressions. Its final status is ready, healthy, and idle; tracing is
off. The service remains on port8011 and port8000 remains closed.

The actual request trace contained 286 completed GPU profiles, all with valid
cross-clock comparisons, and 39 full-prefix resolutions without a GPU submit.
No encoder boundary was altered and no sampling barrier was inserted. Cold
2048-token singleton prefill again waited 1.428 seconds after commit returned
before hardware GPU start, then spent 834.5 ms on GPU. The subsequent 128-token
four-request prefill waited 109.3 ms and spent 283.9 ms on GPU. This confirms the
stall is observable with the new local trace, without establishing its driver
cause.

## Evidence

- `build/release/flash/v6-verify-ctx2048-singleton-R16-stage.json` and trace/routes
- `build/release/flash/v6-deep-metal-request-cpu-summary.json`
- `build/release/flash/private-tensor-vs-shard-trace-cpu-comparison.json`
- `build/release/flash/hc-up-and-private-tensor-independent-cpu-analysis.json`
- `build/release/flash/hc-lazy-combined-quality{,-comparison}.json`
- `build/release/flash/hc-lazy-combined-http-performance.json`
- `build/release/flash/hc-lazy-same-build-off-control-http-performance.json`
- `build/release/flash/mtp-gpu-chain-four-proof.json`
- `build/release/flash/default-v7-normal-confirm-http-performance.json`
- `build/release/flash/default-v7-normal-quality-qualification.json`
- `build/release/flash/default-v7-normal-final-idle-status.json`
- `build/release/flash/default-v7-optimization-summary.json`
- `build/release/flash/default-v7-command-trace-independent-cpu-audit.json`
