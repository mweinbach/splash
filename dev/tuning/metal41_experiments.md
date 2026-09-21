# Opt-in Metal quantization experiments

An unconfigured checkout keeps affine Q4, the measured M5 Ultra N64 decode
policy, and paired M8 gate/up for the qualified 17408-by-5120 geometry. On this
Mac, `.splash-build.mk` selects the validated hybrid build for normal `make`
and `splash serve` launches. The local preference survives rebuilds and is
excluded from Git. `make SPLASH_PRECISION=q4` explicitly selects the Q4 build.
Two separate builds materialize target/draft linear weights in memory at model
load. Embedding gathers, vision weights and the original model files retain
their existing representations. The experiment macros are mutually exclusive.

## Whole-row W8A8

`SPLASH_INT8_EXPERIMENT` reconstructs every affine coefficient `q*scale+bias`,
then rounds it to signed INT8 using a float32 maxabs/127 scale per output row.
GPU activations use a separate maxabs/127 scale for each input row. One whole-K
INT8 MPP operation accumulates exact INT32 dots, scales into float32 and returns
BF16. Gate/up projections round independently before the SiLU epilogue.

Decode uses M8/M16/M24/M32 with N64. Prefill uses M128/N128 for complete
128-row tiles and M32/N128 for smaller storage buckets. The 25,600-element K
limit bounds every dot by 412,902,400, within INT32. The implementation compiles
with Metal 4.0; the current full-runtime comparison uses uniform Metal 4.1.

```sh
make -j4 all BUILD=build/int8 \
  ENGINE_CXXFLAGS='-std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1' \
  PROD_METALFLAGS='-std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1' \
  BUILD_ID_PYTHON=.venv/bin/python
```

For Qwen3.8-27B the loader estimates 27,538,767,872 additional bytes while
retaining original Q4 views. This includes converted weights, row scales,
diagnostics and shared activation scratch. Startup preflight accounts for the
extra allocation; actual converted content hashes distinguish model/cache
identities. Numeric and shape failures set sticky GPU flags, which the backend
checks before completing a command. A later successful use cannot clear an
earlier failure.

`SPLASH_INT8_POLICY` is read once at startup and recorded in model/cache
identity. Its precision choices stay fixed per projection across prefill,
decode, request batch sizes and cached replay:

| Value | Converted projections in Qwen27B | Estimated extra bytes |
|---|---:|---:|
| `all` | All 358 target/draft linear projections | 27,538,767,872 |
| `target` | 321 target projections, including the shared vocabulary head | 25,739,640,832 |
| `hybrid` | 64 target MLP down projections | 5,759,844,352 |
| `q4` | None | 0 |

The INT8-enabled build defaults to `hybrid`; set the environment before
launching a server to explicitly select `all`, `target` or `q4` instead.
Draft bodies remain Q4 in `target` and `hybrid`; the draft borrows the target
head, whose precision therefore follows the target profile. `hybrid` also
omits the upstream output-sum tail that its INT8 down projections do not use.

The completed matched Qwen3.8-27B comparison on the current M5 Ultra measured
four-request HTTP output throughput at 149.10 tok/s for tuned Q4 and 179.07
tok/s for W8A8 (+20.1% median paired). Singleton cold HTTP throughput falls
from 73.05 to 58.05 tok/s (-20.5%). Both paths score 8/11 on the small practical
answer suite, with the same failed tasks. All 24 performance requests complete
their 1,024-token budgets with valid counters, cache behavior and runtime health.
Actual additional dense allocation is about 25.64 GiB. Raw details are in
`build/release/metal41/metal-quantization-report.md` and the evaluation JSON.

## Block-scaled FP8

`SPLASH_METAL41_EXPERIMENT` reconstructs the same affine coefficients into
E4M3 weights and power-of-two UE8M0 scales per 32 elements. BF16 activations
cast to HALF, MPP accumulates float32 and the epilogues return BF16. Scale rows
are tightly packed at K/32 bytes. This path requires Metal 4.1.

```sh
make -j4 all BUILD=build/metal41 \
  ENGINE_CXXFLAGS='-std=c++20 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0 -DSPLASH_METAL41_EXPERIMENT=1' \
  PROD_METALFLAGS='-std=metal4.1 -O3 -Wall -Wextra -Werror -Iruntime -mmacosx-version-min=27.0' \
  BUILD_ID_PYTHON=.venv/bin/python
```

The FP8 loader estimates 28,427,550,720 additional bytes. On the current M5
Ultra, the initial full-model experiment preserved all eight baseline
successes in 11 practical tasks. Four-request native decode throughput improved
about 21%, but slower cold prefill made HTTP throughput about 2% worse; singleton
decoding slowed about 20%. Changed outputs and draft acceptance affect native
throughput, so it is not an isolated kernel-speed comparison. The hardened
runtime repeated the same 8/11 result with no new failure and valid transport.

## W4A8 probe and evidence

[Splash issue 6](https://github.com/incoai/splash/issues/6) proposes retaining
four-bit weights while quantizing activations to INT8. The standalone probe
preserves the original affine weights: signed nibbles are `q-8`, with corrected
float32 bias `bias+8*scale`; INT32 dot products and activation sums restore
each 64-element group's affine contribution. This gives small numerical error
on synthetic inputs, with uneven speed across matrices and batch sizes.
Expanding the signed nibbles into INT8 without changing the per-group scales
does not remove that repeated affine work. Whole-row W8A8 instead requantizes
the coefficients to enable a single whole-K operation.

Ignored raw evidence and reproducible helpers live in `build/release/metal41/`:

- `int8-model-evaluation.json`: matched whole-model W8A8 comparison in ABBA order.
- `fp8-model-evaluation.json`: initial FP8 whole-model comparison.
- `fp8-guarded-runtime-verification.json`: hardened FP8 warmup/answer recheck.
- `metal41-q4-control-evaluation.json`: unchanged Q4 in Metal 4.1; all output
  hashes matched Metal 4.0 and decode differences were small.
- `w4a8-*-repeat32-results.json` and `w8a8-*-results.json`: synthetic projection
  probes including activation quantization; 12 alternating pairs, 32 complete
  graphs per command and exact CPU integer dot checks.
- `int8-runtime-guard-test.log` and `fp8-runtime-guard-test.log`: known-math
  outputs, all INT8 epilogues and sticky failure rejection. INT8 ran with Metal
  GPU validation enabled.

The normal `splash` command on this Mac uses a local wrapper pointing at this
checkout. Homebrew's original package files remain intact; the previous link
is recorded under `~/Library/Application Support/Splash/checkout-default/`.
The default launch was verified with `SPLASH_INT8_POLICY` unset: actual model
and target-weight fingerprints match the measured hybrid build, memory audits
pass, and the API returns correct JSON. The historical frozen benchmark images
retain their original defaults and recorded identities.

Whole-model measurements use identical tokenizer/templates and persisted 2K
prompts, reasoning off, temperature zero and full 1,024-token output budgets.
Cold-cache reuse, cached replay, counters, runtime health and resource release
are checked separately. The 11-task answer suite is a small regression check,
not a general coding or language-quality evaluation.

## Prefill tracing and the current local follow-up

Optional metadata-only CPU/command/counter tracing is documented in
`dev/benchmarks/prefill-tracing.md`; ordinary inference keeps it disabled. Real
HTTP traces showed that cold 256/2K requests execute 224+32 / 2016+32 rows.
Exact M32/N256 INT8 down projection now handles only logical/storage rows 224 or
2016 on family 10/core 80, N=5120/K=17408; the 32-row tail retains M32/N64.

The follow-up unprofiled ABBA comparison completed 48 groups/96 requests with full
1024-token decode budgets and exact matched response hashes. Relative to the
previous hybrid default, 2K cold prefill improved 6.77% single / 2.26% concurrent;
native decode remained approximately flat. Optional immutable-weight residency
reduced a first-command CPU-driver scheduling interval 273→166 ms in a separate
control, but did not eliminate it. Model fingerprints and observed tracked
allocation footprint remained unchanged. The local CLI wrapper enables residency
when unset; an explicit SPLASH_MODEL_WEIGHT_RESIDENCY=0 disables it. The running
local server was restarted and verified; original model files are unchanged.
Raw results and limits: `build/release/metal41/prefill-instrumentation-report.md`.
