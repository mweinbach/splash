# QSA probability and register-PV audit

Normal production sources/defaults remain unchanged. The old online BF16PV
prototype is not the requested dense model-reference boundary.

## Verified source boundaries

The local Qwen4Exp implementation inherits Qwen3_5Attention. Its dense≤2048
eligibility stays on ordinary fast SDPA, while gathered QSA is enabled beyond
the sparse budget. The parent attention implementation invokes fast SDPA;
see [pinned mlx-vlm source](https://github.com/Blaizzy/mlx-vlm/blob/3fb24e907dec2b99ea4ec7b350f52fd6db145575/mlx_vlm/models/qwen3_5/language.py).

Official [MLX 0.32.2 NAX attention](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h)
keeps score/online-softmax fragments F32, feeds them to PV alongside BF16 V,
and normalizes/stores the final output. Its [register MMA helper](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/steel/attn/nax.h)
uses 16×32×16 single-SIMD cooperative operands. **It also sets
`relaxed_precision=true`**. Consequently F32 storage does not establish strict
F32 compute accuracy. The SDK's `MPPTensorOpsMatMul2d.h:385` names the seventh
constructor argument `relaxed_precision`; its comments permit trading precision
for speed. [Apple's Metal 4 presentation](https://developer.apple.com/videos/play/wwdc2025/262/)
also identifies this parameter as the precision control.

The local portable gathered fallback explicitly performs global F32 softmax,
casts probabilities to the query dtype, then multiplies by selected values
(`qsa_fast.py:734–744`). This BF16 probability boundary is different from both
strict Splash PV and the relaxed NAX compute path. It cannot be called the
actual dense fast-SDPA boundary merely because it appears in fallback Python.

Current Splash temporal M32 already groups 12 query heads per KV head, sharing
K/V across 32 flattened query/head rows. Host grid is
`ceil(rows*12/32) ×2 ×partitions` (FlashQSAMPP.cpp:192–196); the 24-head grid is
only the 3.53 ms final reducer. The 97.6 ms main kernel already shares each K/V
bank. No missing 12× query-head grouping exists.

## Existing BF16PV evidence

`dev/benchmarks/flash_qsa_online_bf16pv.metal:35–42,157–175` rounds tile-local
**unnormalized exponentials**, while denominator/max/merge remain F32. It
explicitly does not implement globally normalized BF16 checkpoint probabilities.

`build/release/flash/qsa-online-bf16pv-screen.json` covers 18 passing synthetic
variants: rows 1/4/8, begins 1920/4096, partitions 8/16/32. Matched F32 online gains
are only 1.015–1.039x. Relative L2 is 0.225–0.252% versus F32. There is no retained
128-row prefill, actual-activation, FP64 or generation evidence. Its oracle uses
a sampled serial F32/global-softmax reference, not the candidate's tile-local
rounding algorithm. Passing the old 1% bound is not model qualification.

A global BF16-P MPP route already exists at
`runtime/metal/kernels/shared/flash_qsa_mpp.metal:124,221`; it uses three
dispatches and approximately 36 MiB of score/probability sheets at 128 rows. No
retained materialized-route timing report was found. It is a numerical
alternative and would require its own real-activation and generation checks.

## Prepared register-PV experiment

`register_pv.metal` keeps current QK/softmax, F32 probability storage, the
original 64-token bank boundaries and `out*alpha+partial` update. Four SIMD
groups each own 16 query/head rows×128 output columns. It accumulates each
64-token partial from zero in four ordered K16 register primitives. Direct BF16
V reads remove shared V staging; PV barriers fall from 8 to 1 per bank. Source
scratch remains 20 KiB. **5,726,720 CPU lane/causality checks passed**.

The initial source-reference variant copied NAX's relaxed precision setting.
Its synthetic screen passed the primitive numerical/guard checks and kept
score/softmax statistics byte exact, but changed 90–92K BF16 cells of 786K,
relative L2≈0.132–0.139%, with only 5–17% speedup. This is a relaxed-compute
alternative, not an exact replacement. Compile-time probes confirm its left
cooperative element is Float32 and right element BF16; any reduction occurs
inside relaxed compute, not in explicit storage assignment.

`register_pv_strict.metal` is the same schedule with relaxed precision disabled.
`relaxed_pv.metal` retains current shared M32 scheduling and enables relaxation
only for PV, isolating precision from register scheduling. Both compile; GPU
results remain pending at document preparation.

## Actual operands and FP64 qualification

A private generated Forward overlay captures last 128 GPU-prepared Q rows,
rotated K and actual V through token 2048, plus gates, at layers 3/27/47.
Capture memory is explicitly added to the planner: **25.5 MiB**. Preparation
and values are captured directly from the running native graph, rather than
recomputed approximately on CPU. The collector has compiled and passed help
and dry-run; no capture GPU command has been submitted by this subtree.

```sh
.venv/bin/python dev/benchmarks/prefill4k_attention/run_capture.py \
  --tokens build/release/flash/prefill4k-fixture/code2048.tokens.json \
  --report FRESH_CAPTURE_REPORT.json --outdir FRESH_CAPTURE_DIRECTORY --run
```

Root must serialize that model load. Process exit releases the model; the
capture driver starts no HTTP server. Capture timings are diagnostic because
copy dispatches alter execution.

`fp64_reference.py` accepts the generated manifest directly. It evaluates full
FP64 causal QK, stable max-shift global softmax and PV, with independent complete
causal-row `math.fsum` audits at sampled heads/columns. FP32-P and global BF16-P
counterfactuals are distinct. It reports near-zero, ULP and worst-cell errors,
and does not assign qualification without explicit thresholds. Nominal staged
BF16 gating is labeled as an approximation to Metal precise exp, separately
from raw attention. CPU self-test and NPZ/JSON round trips passed.

```sh
.venv/bin/python dev/benchmarks/prefill4k_attention/fp64_reference.py \
  FRESH_CAPTURE_DIRECTORY/layer3/manifest.json --candidate ACTUAL_RESULT.json \
  --comparison-policy fp32-p --report FRESH_FP64_REPORT.json
```

Any nonexact result still requires a full-model quality/continuation/lifecycle
run and unload verification before a default change. No QSA change is promoted.

The compiled actual-input runner compares native baseline/candidate attention
using the service's 32-partition allocation layout. It dumps raw F32 attention,
rounded BF16 attention, and gated BF16 output; the extra debug reduction is
excluded from timed commands. Every score/softmax statistic must remain byte
exact; immutable input/cache SHA256 must survive timing. Capture must be fresh.

```sh
PREFILL4K_ATTENTION_REGISTER_PV_STRICT=1 \
build/prefill4k-attention/actual-oracle \
  build/prefill4k-attention/actual-pv.metallib \
  CAPTURE/layer3/manifest.json FRESH_ACTUAL_RESULT

.venv/bin/python dev/benchmarks/prefill4k_attention/fp64_reference.py \
  CAPTURE/layer3/manifest.json --candidate FRESH_ACTUAL_RESULT/candidate/manifest.json \
  --comparison-policy fp32-p --report FRESH_FP64_COMPARISON.json
```

Use `PREFILL4K_ATTENTION_REGISTER_PV=1` for NAX-style relaxed register compute;
`PREFILL4K_ATTENTION_RELAXED_PV=1` keeps current scheduling and relaxes PV only.
The seventh descriptor argument is the explicit precision policy in all three
variants. No nonexact variant should be described as generation-qualified.
