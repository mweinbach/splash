This isolated candidate explores M8/M16/M32 N128 geometry for the original
Q4/G64 expert projections. The current production Q4x8 N64 producer already
loops the whole K inside one threadgroup. It has no global K64 partials or fold
dispatch to remove.

The N128 candidate retains original FP32 affine coefficient reconstruction,
BF16 weight/input boundaries, ascending K64 MPP multiply-accumulate traversal,
and the existing BF16 SwiGLU/down/scatter/combine boundaries. Changing the MPP
descriptor's N dimension still requires GPU bit-exact qualification. Gate/up
reuse one staged B tile, retain both FP32 accumulators, and serialize gate and
up staging in each K64 chunk. Down uses one staged A+B tile. N128 halves the
output-column grid and repeated A staging; additional gate/up barriers may
offset that gain.

The source arrays use 17/18/20 KiB for M8/M16/M32, rather than retaining two
N128 B tiles above the device limit. The oracle checks the compiler's actual
pipeline threadgroup memory before loading the model or submitting work.
Metal Shader Validation can increase static memory; a preflight refusal is
not a qualification pass. The blocked MoE parameter ABI and stable bucket
jobs are reused verbatim. The graph transform borrows source-owned parameter
bytes, and its source graph must survive all submissions.

The private artifact is `build/flash-moe-wide/flash-moe-wide-oracle`, with
`build/flash-moe-wide/splash.metallib`. The latter links unchanged production
AIRs from `build/flash-round4/metal` plus `flash_moe_wide.metal`; production
routes and libraries are not modified. Compilation and `--cpu-self-test`
submit no GPU work. Root runs GPU qualification serially.

```sh
build/flash-moe-wide/flash-moe-wide-oracle --cpu-self-test
build/flash-moe-wide/flash-moe-wide-oracle --pipeline-metadata \
  build/flash-moe-wide/splash.metallib

FLASH_MOE_WIDE_PHASE=both FLASH_MOE_WIDE_ROWS=2048 \
FLASH_MOE_WIDE_TILE=32 FLASH_MOE_WIDE_PAIRS=6 \
build/flash-moe-wide/flash-moe-wide-oracle \
  build/flash-moe-wide/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/moe-wide-m32-n128-layer0.json
```

Use `FLASH_MOE_WIDE_PHASE=gate_up` or `down` to isolate either phase, and
`FLASH_MOE_WIDE_TILE=16` to compare M16. The default runs spread and concentrated
top-10 IDs. Select additional original layer planes with
`FLASH_MOE_WIDE_PREFIX=language_model.model.layers.24.mlp.switch_mlp` (and layer
47). `FLASH_MOE_WIDE_INPUT` and `FLASH_MOE_WIDE_IDS` accept matching captured
BF16 input and I64 route-ID files; both must be set, with exact extents.

The oracle checks every activation, canonical down, and combined BF16 byte
against the production Q4x8 producer, independent CPU bucket/job expectations,
sticky diagnostics, and output canaries, before and after alternating matched
commands. Reports include source/manifest identity, phase, actual pipeline
memory, job utilization, and both GPU and wall timings. A full-model HTTP
benchmark is required before promoting any primitive gain.
