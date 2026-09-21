# Local v6 kernel round

The active local v6 policy retains the existing saved dense and Top64 INT8 expert
artifacts. It adds four runtime defaults after two ON runs against an OFF
control from the same combined binary and 22 bounded quality/protocol checks.
This is a small measured improvement with a short-singleton tradeoff, not a
general performance breakthrough. The final normal-launcher run passed 22 bounded quality checks and 111 CPU
checks; its actual numbers are primary below. The two candidate runs remain
separate evidence rather than substitutes for final results.

| Prompt / concurrency | Same-build OFF | Final normal v6 | First ON | ON confirmation |
| --- | ---: | ---: | ---: | ---: |
| 128 / C1 | 100.353 | 100.282 | 100.008 | 99.576 |
| 128 / C4 | 152.037 | 157.127 | 155.074 | 154.973 |
| 2,048 / C1 | 64.359 | 66.391 | 66.100 | 66.085 |
| 2,048 / C4 | 78.226 | 79.676 | 78.870 | 78.592 |

Values are complete-wave HTTP output tokens/s on M5 Ultra with 80 GPU cores,
256 GiB RAM, Metal 4.1/macOS 27, greedy reasoning disabled, zero cache reuse,
exact 128/2K prompt rows, all 128 output tokens per request, warmup and two
measured samples per run. C4 is aggregate over four independent requests.
Final normal gains versus OFF are approximately +3.16% long single,
+1.85% long C4 and +3.35% short C4, with short single −0.07%. Candidate gains
were smaller: long single near 2.7%, long C4 0.5–0.8%, and short C4 near 2%,
with short-singleton losses 0.3–0.8%. Preserve this run-to-run variation.

| New flag | Scope |
| --- | --- |
| `SPLASH_FLASH_SHARED_EXPERT_FUSED=1` | BF16 gate/up plus SwiGLU fusion with preserved rounded-dot boundaries, M32N128 for eligible prefill and existing tail fallback |
| `SPLASH_FLASH_DENSE_M64_OUT=1` | M64N128 for qualified output roles with N2,560/K6,144 and rows 512–8,192; excludes slower input/QKV/HC roles |
| `SPLASH_FLASH_GDN_BATCH_ILP=1` | Register/geometry change for uniform batched prefill, with separate request states and FP32 recurrence preserved |
| `SPLASH_FLASH_MTP_QMV_F32=1` | Existing original-Q4/G64 F32 QMV for trained-head FC-hidden proposal windows; full target still verifies greedy candidates |

These add four flags to v5's 30, making 34 static defaults plus two gated
saved-store paths. Set any new flag explicitly to `0` to opt out; setting all
four to `0` restores the v5 runtime policy. No extra weight artifacts or state
precision demotion are required. Original checkpoint and stored-format
identities remain unchanged.

All 21 performance requests in both ON runs match OFF's generated target
text, prompt-token hashes, usage and completion fields exactly. The 22-case
ON quality run also passes with unchanged normalized responses. Per-head GPU
time falls about 9% against the matched OFF control, despite four additional
head calls. Target verification has one extra cycle, and budget-end AR calls
change, so total decode improvements are not solely projection speedups.
Acceptance remains approximately 99.435% on this counting workload.

The [same-build CPU analysis](../../build/release/flash/four-routes-same-build-paired-cpu-analysis-v1.json)
records source, cache, depth, context and residency equality, expected route
changes, and command-phase timing. Main-trunk GPU/wall gains are small;
callback intervals overlap and measure CPU handler arrival rather than actual
GPU start. The stronger isolated shared/output/GDN gains do not transfer
directly to whole-request throughput.

Private QMV **and F32 control** proposal screens still pass **0/17 cases**
(17/17 failures) under the unchanged 1% hidden/state bounds. Both retain repeated argmax matches and
finite output. The prototype defaults to FC-hidden only but permits alternate
prefix lists; its JSON omits the actual selected-prefix list. Those failures
remain unresolved numerical evidence and are not erased by exact target
output. Current served QMV is explicitly FC-hidden-only, and accepted greedy
tokens are checked by the unchanged full target. The 22 fixtures are bounded
task outcomes, not a general quality evaluation or universal hidden/state
equivalence claim.

The experiment matrix retains disabled alternatives:

| Alternative | Completed evidence | Outcome |
| --- | --- | --- |
| Batch window 1,024 | 22 quality checks pass; long C4 78.541 versus this-round 80.041 baseline | Disabled |
| Original-text residency | 22 checks pass; long C4 77.853; only configured API registration is observed | Disabled |
| Top128 INT8 plus residency | 22 checks pass; long C4 78.110 versus 80.041 baseline | Top64 retained |
| Persistent INT8 job compaction | Complete producer bytes, maps, misses, canaries and payload immutability exact; isolated mixed R2K chain slower | Disabled; no full-model quality claim |
| Broad M64 dense roles | 24 cases exact, but input/QKV and HC geometries frequently slower | Only measured output roles retained |

The original config/tokenizer/index match aligned copies, and source/layout
identities remain unchanged. This CPU audit did not rescan complete multi-GB
payloads during root GPU runs. No GitHub writes are part of this work.

The [final normal performance report](../../build/release/flash/splash-normal-default-v6-http-performance.json)
completes all 21 requests at 128 output tokens and matches OFF generations
exactly. [Final quality](../../build/release/flash/default-v6-quality-qualification.json)
passes 22 cases; [CPU qualification](../../build/release/flash/local-profile-v6-cpu-qualification.json)
passes 111 checks. The [fresh final idle proof](../../build/release/flash/default-v6-final-idle-status.json)
matches engine 550771527669187, with 49 submitted=47 completed+two deliberate
cancellations, no failures, no active/in-flight request, healthy Metal and
normal observed pressure. Allocation-ledger peak is 155,472,838,656 bytes;
that value is not process RSS.
