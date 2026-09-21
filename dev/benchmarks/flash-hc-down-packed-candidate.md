Private packed HC-down row reuse reduces repeated reconstruction of the original
coefficients while preserving each row's literal lane32 dot-product chronology.
The existing FP32-cache screen lost in every configuration; this candidate keeps
packed Q4/Q5/Q6/Q8 weights and introduces no FP32 weight stream or new weights.

Each SIMD owns one output and two (mode0) or four (mode1) scalar FP32 row
accumulators. Scale/bias, packed code and `code * scale + bias` reconstruction are
shared between those rows. Each row still accumulates `k = lane + 32*j` in its
original order with separately rounded multiply and add. The original
`simd_sum`, BF16 raw dot, BF16 divide by4, compiled-fast BF16 SiLU or unary-precise
BF16 injection sigmoid and BF16 multiplication by2 remain unchanged. Injection
uses its independent original format/stride descriptor, including mixed
Q4-down/Q5-injection weights. Final HC without injection is supported.

The grid keeps 648/324 groups at R16 for two/four row reuse, compared with1,296
control groups and81 groups in the losing all16-row FP32-cache literal route.
Reconstruction operations and packed/parameter load requests fall by2/4 while
input loads and per-row floating operations remain unchanged. These are source
counts, not measured bandwidth or occupancy. Reduced parallelism and the
outlined scalar helper may still lose; only Root's GPU screen can answer that.

Compilation and CPU qualification pass with the production Metal4.1 hybrid
flags and frozen HC+lazy host objects. `CommandTiming` remains200 bytes, and
object identities are embedded in the oracle. The 8 independent CPU tests cover
737,280 packed-code addresses across12 formats and padded strides, full-K row
partials, tails through32, all144 down/injection format pairs, arithmetic
cancellation, signed zero, subnormal/overflow/NaN classification and exactly-once
writes. The private host limits rows to1..16. Compiled LLVM IR checks all24
private pipelines/helpers: no accumulator `alloca`, FMA, reassociation or
contract flags; each retains exactly2/4 original hardware SIMD sums. IR is not a
machine register/occupancy measurement.

The oracle checks original input/source SHA preservation, clean guards, sticky
diagnostics, untimed literal-witness agreement with the production control,
byte-exact raw FP32 and BF16 dots, activation and injection, debug/non-debug
agreement, and candidate-own canonical epilog agreement. It accepts no numerical
tolerance or midpoint override. Default inputs are explicitly declared
synthetic normalized BF16; `FLASH_HC_DOWN_PACKED_INPUT` accepts exact captured
BF16[rows,10240] data. This is not full-model quality evidence.

Root completed the bounded six-source R16/reuse2 screen. All six cases pass
byte-exact raw FP32/BF16 dots, activation, injection, production witness,
candidate-own canonical epilog, guard, sticky diagnostic and immutable source
checks. Every case is slower:

| Original role | Original bits / injection | Control GPU ms | Packed reuse2 GPU ms | Candidate throughput / control |
|---|---|---:|---:|---:|
| layers.1.attn_hyper_connection | 4 / 4 | 0.146854 | 0.222854 | 0.659x |
| layers.0.attn_hyper_connection | 5 / 5 | 0.235271 | 0.261542 | 0.900x |
| layers.15.attn_hyper_connection | 6 / 6 | 0.230792 | 0.257958 | 0.895x |
| layers.31.attn_hyper_connection | 8 / 8 | 0.149271 | 0.237625 | 0.628x |
| layers.19.attn_hyper_connection | 4 / 5 | 0.214125 | 0.249083 | 0.860x |
| hyper_connection_mixer | 4 / 0 | 0.115167 | 0.222333 | 0.518x |

This rejects this implementation. The approximately10–48% throughput loss does
not justify further R4/R8 or reuse4 GPU screens without a new measured cause.
Reduced grid parallelism, row addressing/live-state overhead and the outlined
helper remain hypotheses; this screen does not establish memory bandwidth,
register pressure or occupancy as the cause. No production route/default was
introduced, and the existing literal packed HC-down remains preferred.

The frozen Root-only command documents the completed run:

```sh
FLASH_HC_DOWN_PACKED_ROWS=16 FLASH_HC_DOWN_PACKED_MODES=0 \
FLASH_HC_DOWN_PACKED_PAIRS=6 \
build/flash-hc-down-packed-v1/flash-hc-down-packed-oracle \
  build/flash-hc-down-packed-v1/flash-hc-down-packed.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/hc-down-packed-reuse2-r16-screen.json
```

Frozen CPU-only qualification is
`build/release/flash/hc-down-packed-v1-cpu-ir-qualification.json`.
No production route, default, checkpoint, GitHub content or `.omlx` state changes.
