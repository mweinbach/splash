HC-UP source and metadata audit, 2026-09-21

This audit reads source and JSON metadata only. It does not read or hash any
model/captured payload, convert coefficients, create a GPU device, execute GPU
work, or modify production/private-worker sources.

The actual R2048 worker control is BF16 cached coefficients, not the small-row
F32 HC route. In the pinned private source, `supportsHCFused` only accepts up to
32 rows (`FlashHCFused.cpp:160-168`). `FlashForward.cpp:461-501` therefore emits
down projection, BF16 HC SiLU, up projection, then HC mix. Float cached
projections are limited to R2..16 (`FlashForward.cpp:430-445`); R2048 with
`DENSE_CACHE=1` calls `denseCache->addProjection` (`451-455`).
`FlashDenseCache.cpp:353-364,380-385` requires BF16 coefficients. HC-UP is absent
from `FlashPrefillDenseTiles.hpp:70-85`, so the caller's legacy M32N128 tile is
retained. `batchProject` delegates to that same route (`FlashForward.cpp:774-778`).
Source root: `build/adaptive-expert-tail-sg2k128-fma-sep21-worker-v1/source`.
Required flag snapshot: QMV_F32=1, FUSE_HC=1, DENSE_CACHE=1,
FLOAT_DENSE_SELECTIVE=1, PREFILL_DENSE_TILES=1.

Exact bounded fixture role:
`language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up`.

| Operand | Metadata path | Contract |
| --- | --- | --- |
| Activated HC-down | `build/prefill4k-dense/actual-activations-v1/input-1.bin` | BF16 row-major [2048,320], 1,310,720 bytes |
| Normalized hyper plane | Same directory, `input-0.bin` | BF16 row-major [2048,4,2560], 41,943,040 bytes |
| Original affine coefficients | `install/local-models/Flash-Next-operands-v1/operand-512.bin` | Explicit F32 row-major [10240,320], 13,107,200 bytes, offset 0 |
| Captured raw-dot control | Capture `output-1.bin` | BF16 [2048,10240]; BF16-coefficient control, not F32-coefficient output |

`actual-activations-v1/manifest.json:1` declares actual activations and both
projection roles. `capture-verification-v2.json:7-45` records their byte extents.
The capture source (`capture/source/FlashForward.cpp:430-435`) uses the same
normalized plane for HC-down, applies HC SiLU, then passes activated down to
HC-up. `dev/benchmarks/prefill4k_dense/capture.hpp:55-70` requires a BF16 cache
matrix and copies two-byte input/output words before/after the projection.
The captured coefficient file `weights-1.bin` matches saved BF16 `operand-4.bin`
(`actual-weights.json:40-47`). It must not be reinterpreted as F32.

The original F32 operand manifest explicitly selects F32 and
`original-affine-contractoff-f32-coefficients-row-major-v1`. Source geometry is
Q5/G64, 1 expert, K320/N10240, weight-row stride 200, parameter-row stride 10,
weight-expert stride 2,048,000, parameter-expert stride 102,400.
`FlashFloatDenseCache.cpp:253-275,343-366` validates packed U32/BF16 scale/bias
geometry and either maps saved F32 coefficients or expands into F32.
`runtime/metal/kernels/shared/flash_float_dense_cache.metal:129-139` performs
separate F32 `float(code)*float(BF16 scale)` then `+float(BF16 bias)`.
`FlashOperandStore.hpp:17-29` and `.mm:334-359,394-400` pin math version, source
identity, manifest fingerprint, format, geometry and operand semantics.

Recorded metadata identities (not recomputed in this audit):

- Source: `ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e`.
- Weight manifest: `edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0`.
- F32 operand512 SHA: `86c57dda081aebfdd7642a7f23496b6b45ecb996a45dbf1e74df9d2d3c8030da`.
- BF16 coefficient SHA: `2403b3e90d9452756dbaaac29da866510babc95a4027146d7f34896760ad92bb`.
- Activated input1 SHA: `a31906c1fdc06155147b07b6658b26ae7e1fa246b861e0fb0304c4d408d2846f`.
- Normalized input0 SHA: `fb402a313a5f18053671675c43608111370334ed6113e16dfda11d2916963b69`.
- Raw control output1 SHA: `0091bfea9c293cc9d97898fe1c84f40f771b8428864b767c4263b1cdb04b5787`.

The SiLU and later sigmoid contracts differ. HC SiLU uses fast exp, with BF16
down/4, sigmoid stages and product (`flash_forward.metal:9-34` and
`flash_hc_fused.metal:118-127`). HC modulation uses precise unary sigmoid with
BF16 exp/add/divide/subtract, BF16 normalized product, sequential BF16
stream0..3 sum, then BF16 /4 (`flash_hc.metal:33-62`). Keep every boundary;
CPU promotion is not an exact exp oracle. The original small-row F32 fused
reference also keeps raw BF16 before sigmoid (`flash_hc_up_f32_mpp.metal:38-48`).

Candidate contract: quantize original F32 coefficients directly to symmetric
I8 [-127,127] with one F32 scale per output row. Quantize activated BF16 input
on GPU with one F32 scale per input row. Accumulate the whole K320 in I32
(magnitude bound 5,161,280), apply two ordered F32 scale multiplications, cast
raw dot to BF16, then run the exact same HC modulation post shader as control.
The complete timing includes activation conversion, integer projection and
postprocessing. No generic BF16 coefficient fitting is reused for W8.

The isolated oracle should read only input0, input1 and F32 operand512 when Root
invokes `--run`. Derive the BF16 worker-control coefficients using the documented
RNE conversion and require their full SHA to equal the recorded BF16 coefficient
SHA. Require the complete actual M32N128 raw control output SHA to equal the
recorded captured output1 SHA. This guards the conversion without an additional
BF16 coefficient/output payload read. Type/shape/stride/identity mismatches
reject before GPU construction. Preview/self-test never read payloads.

Preregister full raw-dot, gate, product, sequential-sum and mixed-output relative
L2 <=0.02 and cosine >=0.9998 against the actual worker control. Separately
record sampled FP64 dots from original F32 coefficients and dequantized I8
operands, full late-scale identity, exact sampled I32 dots, source/subnormal
counts and a stated rounding envelope. Quality failure excludes promotion.
The existing capture provides raw BF16 control parity; it does not provide a
fresh actual-route captured fused mixed witness or model continuation proof.
This experiment remains a numerical alternative and model-qualified=false.
