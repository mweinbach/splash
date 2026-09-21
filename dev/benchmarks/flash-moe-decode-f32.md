This private decode candidate groups selected experts across 2/4/8/16 real rows
before running mixed BF16/F32 M8N64 MPP projections. It reconstructs original
Q4/G64 coefficients with contract-off FP32 arithmetic and keeps them as F32
matrix operands. Inputs remain BF16; no activation quantization or checkpoint
conversion occurs. The K64 MPP MAC order differs from selected QMV and is
explicitly recorded as a separate numerical producer.

Gate/up retain two FP32 accumulators while reusing one 16 KiB F32 B staging tile
and one 1 KiB BF16 A tile. Barriers separate gate and up staging. Dot outputs
round to BF16 before the qualified SwiGLU stages. Down consumes packed BF16
activations and scatters BF16 output into original route order, followed by the
unchanged BF16 combine. Existing stable bucket packing/jobs and source guards
are reused; private matrix grids launch at most rows×10 jobs, a safe bound
without CPU count readback. Canonical BF16 gate/up taps expose each boundary.
Production files, flags, profiles and original checkpoints are unchanged.

The oracle forces the actual `QMV_F32=1` and `EXPERT_QMV=1` baseline and asserts
its selected gate/up and down pipeline names. Every gate, up, activation, down
and combined stage must be finite with relative L2 at most 1e-3; the limit can
be tightened through `FLASH_MOE_DECODE_F32_LIMIT`. Accuracy failure saves valid
JSON with `pass=false`, preserving per-stage errors and timings. BF16 word
mismatches, maximum ULP and absolute error are reported separately. Reduced
dispatch or memory traffic is a hypothesis until Root's GPU measurements.

An untimed GPU coefficient sampler uses the same reconstruction helper as the
matrix producer and checks 98,304 exact original F32 coefficient words per
case (eight selected experts, three projections, 64×64 samples each), including
negative scales and values not representable as BF16. Independent CPU decoding
reads original unsigned nibbles/signed BF16 scale+bias. Output canaries and
independent packed route/job expectations are checked before and after matched
warm command replays. Candidate timings include bucket packing/jobs; baseline
timings include the complete selected MoE chain.

```sh
build/flash-moe-decode-f32/flash-moe-decode-f32-oracle --cpu-self-test
build/flash-moe-decode-f32/flash-moe-decode-f32-oracle --pipeline-metadata \
  build/flash-moe-decode-f32/splash.metallib

FLASH_MOE_DECODE_F32_ROWS=8 FLASH_MOE_DECODE_F32_PAIRS=6 \
FLASH_MOE_DECODE_F32_IDS=build/flash-moe-decode-f32/captured-layer0-rows8-ids.i64 \
build/flash-moe-decode-f32/flash-moe-decode-f32-oracle \
  build/flash-moe-decode-f32/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/moe-decode-f32-captured-layer0-r8.json
```

Omit `IDS` for spread/concentrated fixtures. Omit `ROWS` for all eight default
cases. Other captured files use layers 0/24/47 and rows 2/4/8/16; select the
matching source projection with
`FLASH_MOE_DECODE_F32_PREFIX=language_model.model.layers.24.mlp.switch_mlp`.
`INPUT` accepts a matching raw BF16 hidden file; hidden inputs otherwise remain
synthetic, declared in the report. `decode-fixtures.json` records capture/source
SHAs, every fixture hash, reuse counts and provenance. Rows 16 contains 15 actual
captured decode rows and one explicitly repeated final ID row.

The private combined library links 49 frozen round7 AIRs and one isolated
decode-MPP AIR. Compilation, CPU self-test and fixture extraction submit no
GPU work. Pipeline preflight checks actual static TGM/thread limits before
loading the model; Shader Validation can enlarge staging beyond the device
limit and must be reported as refusal rather than qualification. Root owns
all GPU work and must qualify coherent full-model generation before promotion.
