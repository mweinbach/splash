This private screen tests four geometry alternatives for small-row main-model dense projections. The operands preserve the saved original F32 affine coefficient bytes, the input BF16 words, F32 accumulation and BF16 output rounding. Changing the cooperative tile is explicitly a reduction alternative, so final model equality requires separate service qualification even when the primitive passes.

The first screen covers actual source matrices for GDN QKV `[10240,2560]` at layer 0 (Q6/G64), QSA query `[12288,2560]` at layer 3 (Q4/G64), and QSA output `[2560,6144]` at layer 15 (Q6/G64). It tests R4/R8/R16. The expanded screen adds the first source representative of each format for those same three roles: 11 role/format groups in total. Each matrix uses one resident saved F32 cache at a time, with no new weight format or production routing.

Candidates are M8N32 SIMD2, M8N32 SIMD4, M16N64 SIMD8 and M8N64 SIMD2. Existing M8N64 and M16N64 SIMD4 tiles are included as controls. Smaller N increases threadgroup count, while SIMD2/8 changes cooperative work distribution. Timing rotates raw F32 quantized QMV, the candidate and the currently selected production cache policy for six warm samples each. Both GPU and wall times include padding. The selected control owns a separate padding arena, keeping candidate workspace canary checks independent of its padded extent.

Qualification requires strict relative L2 below `1e-4` against both raw F32 affine output and the current production policy, plus sampled independent serial/SIMD32 references. BF16 boundary exceptions are rejected. The inherited oracle retains exact coefficient samples, BF16 input/padding bit checks, sticky nonfinite diagnostics, output row guards, workspace canaries and exact mixed-F32 cancellation traps. Every candidate runs all cancellation traps before real source cases. Private host checks reject overlaps, short extents, unsupported dtype/geometry, and invalid tile IDs.

The immutable host object provenance records SHA256 hashes from `build/flash-default-v7-instrumented` and requires all objects to be newer than the current backend ABI header. The timing ABI is `sizeof(CommandTiming)==200`. Root alone runs GPU/model work. The agent builds and executes only CPU self-tests and plan generation.

Prepare and build:

```sh
make -f Makefile -f dev/benchmarks/flash_dense_f32_tiles_v8_oracle.mk \
  BUILD=build/flash-dense-f32-tiles-v8 \
  METAL_BUILD=build/flash-dense-f32-tiles-v8/unused-metal \
  flash-dense-f32-tiles-v8-oracle
build/flash-dense-f32-tiles-v8/flash-dense-f32-tiles-v8-oracle --cpu-self-test
.venv/bin/python -B dev/benchmarks/flash_dense_f32_tiles_v8_plan.py
```

Root runs the generated `build/release/flash/dense-f32-tiles-v8/run-quick-screen.sh`, followed by the expanded script only after reviewing the quick result. The preparer never executes GPU commands. The frozen plan hashes the binary and private Metal library. None of these files change production defaults or the source model.

Root completed the first 54-case screen and all 80 private cancellation traps. QSA output M8N32 SIMD4 was the only candidate with a repeatable substantial gain over the current production policy. Layer15 Q6/G64 repeated at 1.20–1.33× selected GPU speed and positive wall gains. Layer23 Q5/G64 also improved by 1.27–1.33× GPU. Q8 was rejected: its R4/R8 candidate comparisons against raw output exceeded the original strict `1e-4` limit, despite matching the existing selected cached output exactly. Existing Q8 selection remains unchanged.

The five actual Q5/Q6 output matrices (layers15/23/27/35/43) were then tested at R4/5/7/8/9/15/16. All 35 cases passed strict acceptance, and every BF16 output was byte-exact against the current selected F32-cache tile. The largest relative L2 difference against raw F32 affine output was `8.58777771216e-5`. Minimum measured GPU ratios across each matrix’s seven rows were 1.216×, 1.217×, 1.231×, 1.228× and 1.088× respectively. One layer23 wall sample regressed by about 0.9%; whole-model measurement decides whether this route becomes a default.

The production implementation is opt-in through strict `SPLASH_FLASH_QSA_OUT_F32_N32=0|1`, absent by default. It overrides `FlashFloatDenseCache::addSmallRows` only for original source-qualified main QSA output Q5/Q6/G64 projections at R4–16, N2560/K6144. It uses the existing padding arena and original saved F32 coefficients. The source policy enum and selection remain unchanged, keeping Q4 raw, Q8 on its original route, and all other roles/trained head routes unchanged. The new kernel dispatches80N32 groups per M8rowblock with128threads. Its scalar parameter ABI remains32bytes. No weight or workspace allocation was added.

CPU qualification passes 3,629 geometry/fallback checks and8isolated strict flag modes; the existing1,370policy checks pass both with the flag off and on. The fresh native worker CPU suite passes44checks. Two null-safe cache counters record actual graph construction without GPU profiling: `qsaOutF32N32Dispatches` and `qsaOutF32N32RealRows`. Independent source review checked the source map, flag lifetime, host/backend ownership, pad parameter copying, dispatch/grid output bounds and canaries.

A separate production bridge oracle uses the fresh integrated object and production kernel. It asserts the actual pipeline name, two-dispatch pad/matrix graph,80N32groups, M8padded row extent and route-counter increments, then compares the result against the unchanged selected primitive. Root runs `build/release/flash/run-qsa-out-f32-n32-production-all-source-tails.sh`; compiling and its150,699CPU checks execute no device work. Service quality/lifecycle and benchmark qualification remain Root’s responsibility.
