The private R1 vocabulary screen uses original packed Q8 bytes and BF16 group
scale/bias operands from `language_model.lm_head`, shape `[248320,2560]`, G64.
It adds no weight cache, conversion dispatch, or copied operand. The original
production R1 path is `flash_affine_q8_g64_c2`; the complete comparison appends
the existing two exact BF16 greedy reductions to both graphs.

All candidates preserve original F32 affine coefficient formation, the K order
`k = 64*g + 32*subblock + lane` for each lane, F32 per-lane accumulation,
SIMD32 sum and BF16 output. Geometry varies C1/C2/C4/C8 with two to eight SIMD
groups. A second loader has each lane quartet read one original aligned U32
and shuffle it within the quartet, then extract the corresponding original
unsigned byte. The explicit mathematical contract is unchanged, but GPU byte
exactness must still be verified because shader specialization can affect
compiler scheduling and floating-point instructions. Any BF16 mismatch fails
the private screen; no relaxed aggregate bound or proposal acceptance claim is
used here.

Build only, no Metal device or command submission:

```
make -f Makefile -f dev/benchmarks/flash_head_q8_r1_v8_oracle.mk \
  BUILD=build/flash-head-q8-r1-v8 SPLASH_PRECISION=hybrid \
  flash-head-q8-r1-v8-oracle -j4
build/flash-head-q8-r1-v8/flash-head-q8-r1-v8-oracle --cpu-self-test
```

The binary embeds SHA256/config provenance for six ABI200 v7 objects. The CPU
self-test passed 6,118,868 byte-unpack, lane address and output column coverage
checks. GPU execution belongs to Root and must remain serial with service and
other GPU experiments. A suggested initial Root screen uses all 22 variants,
full vocabulary, five alternating control/candidate pairs, and one normalized
random input:

```
FLASH_HEAD_Q8_R1_REPEATS=5 FLASH_HEAD_Q8_R1_PATTERNS=1 \
  build/flash-head-q8-r1-v8/flash-head-q8-r1-v8-oracle \
  build/flash-head-q8-r1-v8/flash-head-q8-r1-v8.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 FRESH_REPORT.json
```

`FLASH_HEAD_Q8_R1_INPUT_BF16` optionally supplies exact 5120 bytes of a real
completed head's post-final-RMS BF16 input; this replaces pattern 0 and retains
its source fixture provenance separately. Patterns 1..3 are alternating signs,
within-group sign cancellation and sparse input. Optional pattern 4 contains
infinity and checks nonfinite diagnostic/greedy behavior. `PATTERNS` is a count
1..5. `OUTPUTS` supports a guarded crop for shader-validation tails; a cropped
control is the production generic projection, not production R1 vocabulary
specialization, so its times cannot substantiate a full-vocabulary gain.

Variants are indexed by `(C,SG)` in order `(1,4),(1,8),(2,2),(2,4),(2,8),
(4,2),(4,4),(4,8),(8,2),(8,4),(8,8)`, with byte then shuffle for each pair.
`FLASH_HEAD_Q8_R1_VARIANTS` accepts a comma-separated subset 0..21. Timing
reports separately compare projection-only and projection-plus-existing-greedy
commands. Kernel compilation/residency warmups are excluded. `valid=true`
requires every checked output BF16 word, compact 16-byte greedy record and
sticky diagnostic word to match. No production source or default route is
changed by these artifacts, and primitive qualification is not complete-head
or HTTP qualification.

Root's first full-vocabulary synthetic screen completed all 22 variants and
compared 5,463,040 BF16 words. Every word, greedy record and sticky diagnostic
matched. The six-object provenance retained config `26ff33bbeeaa8d04`, ABI200.
Report: `build/release/flash/head-q8-r1-v8-synthetic-screen.json`; CPU audit:
`build/release/flash/head-q8-r1-v8-negative-screen-cpu-audit.json`.

This screen found no meaningful route gain. Stable complete control commands
were around 0.797ms. C2/SG2 and C2/SG4 complete candidate medians were 0.796ms and
0.801ms, versus their 0.794ms/0.800ms paired controls. C4 candidates were around
0.872–0.883ms, C8 around 1.183–1.301ms, and shuffle loads were slower. The first
C1/SG4 reported a 1.195× ratio, but both its control 1.763ms and candidate 1.475ms
were much slower than later controls. That initial clock/cache/residency
outlier cannot substantiate a gain. The original C2/SG8 remains the default.

The original projection operands contain 635,699,200 packed-code bytes,
19,865,600 BF16 scale bytes and 19,865,600 BF16 bias bytes, totaling 675,430,400
bytes; output is 496,640 bytes. Dividing that payload by 0.8ms yields an effective
payload rate of 844.288GB/s. This is a conditional arithmetic estimate, not a
hardware bandwidth or occupancy measurement: caching, transaction amplification
and instruction bottlenecks remain unresolved. It provides no percentage of a
published bandwidth ceiling. One normalized synthetic input is primitive
validation, not real proposal, complete head or HTTP qualification; it is enough
to reject these slower geometry alternatives without touching production.

Root repeated all 22 candidates using a real original trained head's exact
post-final-RMS BF16 vector after its 2048-token primed state. The 5120-byte input
SHA256 is `b06f531a8f2b14c1464db67869d484e42fe84d2e45cad936700984376b4f518f`.
All 5,463,040 vocabulary BF16 words were again exact, all 22 candidate/control
greedy tokens were 198 and diagnostics matched. Stable C2/SG4 candidate 0.78083ms
versus control 0.77975ms was the closest result; C2/SG2 candidate 0.78325ms versus
0.77875ms control and original-equivalent C2/SG8 candidate 0.78413ms versus
0.78004ms control did not improve. C4 around 0.863ms, C1/SG8 around 0.950ms and
word-shuffle alternatives lost. The first C1/SG4 again had elevated control
1.598ms and candidate 1.204ms; its 1.327× ratio remains an initial transient
rather than a steady route gain. Keep the original R1 specialization.

Real report: `build/release/flash/head-q8-r1-v8-real-input-screen.json`; audited
source bytes, complete paired medians and exactness:
`build/release/flash/head-q8-r1-v8-real-input-screen-cpu-audit.json`. This confirms
bounded real-input primitive parity; it does not introduce a new model or
service route. The private candidates remain OFF.
