# Saved-format Flash optimization on M5 Ultra

The selected v5 policy combines persisted BF16/F32 dense operands, a persisted
INT8 cache of 64 selected routed experts per layer, direct-device MoE
activations, temporal QSA query tiling, and a Metal residency request for the
verified saved operand allocations. Everything is local. The original
`~/.omlx/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` checkpoint and oMLX
preferences remain preserved.

The results below are from the final normal `./splash` launcher with local
v5 defaults. It passed 22 bounded quality/protocol checks and a fresh idle
verification matching its engine identity. The preceding explicitly configured
saved-policy run is retained separately, with stronger intermediate numbers;
those do not replace the final normal-launcher result.

| Prompt / concurrency | Fresh v4 control | Final normal v5 | Fresh installed oMLX |
| --- | ---: | ---: | ---: |
| 128 / C1 | 99.314 | 98.687 | 89.710 |
| 128 / C4 | 149.796 | 151.965 | 112.084 |
| 2,048 / C1 | 52.404 | 65.206 | 61.995 |
| 2,048 / C4 | 65.768 | 79.421 | 77.993 |

Values are complete-wave HTTP output tokens/s. Tests ran on Apple M5 Ultra
with 80 GPU cores and 256 GiB RAM, using Metal 4.1/macOS 27, deterministic
greedy requests, reasoning disabled, zero cached tokens, exact 128/2,048-token
prompts, 128 output tokens per request, warmup, and two measured waves. C4 is
aggregate output across four independent requests. All cohorts use plan SHA
`cb3747ba6e77cbfe2d74af976163efc92bf3300f4d1a98c72c4da88bfd6badb4`.
The final normal-launcher medians exceed this fresh oMLX reference in all four
cohorts; finite samples and this workload do not establish universal speed.

Raw reports: [final normal v5](../../build/release/flash/splash-normal-default-v5-http-performance.json), [preceding configured saved policy](../../build/release/flash/splash-top64-saved-dense-resident-http-performance.json),
[fresh native control](../../build/release/flash/splash-v4-no-store-fresh-control-http-performance.json),
[fresh installed reference](../../build/release/flash/omlx-mtp3-saved-formats-refresh-http-performance.json).

The v5 profile retains 27 v4 flags and adds
`SPLASH_FLASH_MOE_DIRECT_A=1`, `SPLASH_FLASH_QSA_ROW_TILES=1`, and
`SPLASH_FLASH_SAVED_OPERANDS_RESIDENT=1`: 30 static runtime flags. Model,
hardware, manifest and selection checks gate optional saved-store paths.
Explicit environment values remain authoritative, including `0`; unavailable
optional artifacts retain their ordinary runtime fallbacks.

Explicit opt-outs are supported: `SPLASH_FLASH_OPERAND_STORE=0` disables
saved dense mappings, `SPLASH_FLASH_INT8_EXPERT_STORE=0` disables selected
INT8 expert mappings, and `SPLASH_FLASH_SAVED_OPERANDS_RESIDENT=0` disables
the residency request. The launcher preserves these values. Disable all
three when returning to ordinary generated dense operands and original Q4
expert prefill. Empty store paths are invalid rather than opt-outs.

The dense derivative `install/local-models/Flash-Next-operands-v1` contains
509 BF16 matrices and 508 F32 matrices, with 16 KiB alignment. Their allocated
payload sizes are 8,467,251,200 and 14,391,705,600 bytes. F32 operands retain
the original affine coefficient arithmetic; BF16 operands retain the same
once-rounded coefficient boundary as the already-qualified dense cache.
Mapping these payloads skips their startup conversion graphs. It is a
separate derivative rather than an overwrite of the checkpoint.

The expert derivative `install/local-models/Flash-Next-int8-experts-top64-v1`
contains 48 layers × 64 selected experts × gate/up/down projections. The
source Q4/G64 coefficient is reconstructed with separate F32 multiply/add,
rounded to BF16, then quantized using a symmetric signed INT8 scale per
output row. Scale is F32 `absmax/127`, integer conversion is nearest-even,
and codes clamp to `[-127,127]`; a zero row uses scale one. Native MPP uses
BF16 activations, signed INT8 weights, F32 accumulation and per-row F32 scale.
The selected cache, scales and rank maps allocate 15,147,466,752 bytes.
This is a declared numerical alternative, not bit-identical to original Q4.

Only eligible blocked prefill routes use the selected INT8 experts. Experts
outside the cache keep the original affine Q4 path. Small-window/decode
routes, trained MTP weights and prompt/head semantics retain their existing
policies. Singleton drafting remains at most 15, bounded by output budget;
joint drafting remains at most three. No recurrent-state precision change
was needed.

Direct-device activations remove repeated activation staging from each K64
MoE step; sanitization and bounded global tail padding preserve valid input
reads. QSA temporal M32 tiles share K/V over neighboring dense causal query
rows in the qualified range beginning at token 512 and ending by token 2,048,
while retaining F32 online probabilities and the existing sparse fallback.

The selected stored policy reports successful active Metal API registration
for **1,113 base allocations / 38,006,423,552 bytes**. That set comprises
verified saved BF16/F32 operands and selected INT8 payload/rank allocations;
the original checkpoint and PLE mappings are excluded. Backing allocations
are already charged to the memory ledger. The API request is observed;
physical pinning is explicitly unverified. The policy's reported allocation
peak is 155,472,838,656 bytes; allocation-ledger bytes are not process RSS.

The [same-binary residency comparison](../../build/release/flash/top64-residency-paired-cpu-analysis-v1.json)
isolates ON/OFF policy without changing model math. Main-trunk GPU time
remains approximately nine seconds over warmup plus 20 measured requests,
while command wall falls from 11.210 to 10.591 seconds. The commit-to-scheduled
callback interval shrinks by 655 ms. Callbacks timestamp CPU handler arrival,
not actual GPU scheduling, and named intervals overlap.

For the preceding configured saved-policy run, true trunk prefill totals are
9.021 seconds GPU / 10.784 seconds command wall. Head priming is a separate
1.348 / 1.407 seconds;
inclusive prefill already contains it, so subtract it exactly once for trunk
attribution. Inclusive decode is 15.034 / 15.515 seconds. Relative to heap
dense operands with Top64 residency enabled, decode GPU time is essentially
unchanged, while decode wall falls by approximately 458 ms and the
commit-to-scheduled callback interval falls by approximately 492 ms. These
command intervals help locate waits; they do not identify a unique driver or
physical-memory mechanism. Final normal-launcher totals are 9.035 / 11.723
seconds trunk prefill, 1.352 / 1.449 seconds head priming, and 15.133 / 15.847
seconds inclusive decode. Do not substitute the stronger intermediate timing
for those final measurements.

The Top64 policy passed [22 bounded whole-model/protocol checks](../../build/release/flash/persisted-int8-quality-top64-v1.json),
with [no newly failed tasks](../../build/release/flash/persisted-int8-quality-top64-comparison-v1.json).
Only unconstrained JSON whitespace changed relative to the v4 baseline;
values/types remained correct. The [residency-enabled protocol run](../../build/release/flash/http-top64-residency-on-qualification.json)
passed all eight cases. Arithmetic, retrieval at several positions and four
independent lanes, JSON, tools/continuation, and held-out Python examples are
covered. These fixtures establish bounded task outcomes, not general model
quality or universal generation identity. The original water-cycle prose has
an unchanged causal wording issue; recorded prose remains reviewable.
Normal native HTTP/CLI exposes no usable teacher-forced logits or likelihood.

Intermediate benchmark status snapshots can precede final native terminal
bookkeeping despite complete HTTP streams. The [final normal-launcher idle
proof](../../build/release/flash/default-v5-final-idle-status.json) matches engine
290516427249285 and shows 49 submitted = 47 completed + two deliberate
cancellations, zero failed, no active/in-flight request, and healthy Metal.
The [final 22-case qualification](../../build/release/flash/default-v5-quality-qualification.json)
is bounded rather than a general evaluation. Short-singleton measured samples
were 102.918 and 94.456 tokens/s; the final median is slightly below fresh v4
control despite exceeding oMLX. No GitHub writes are part of this workflow.
