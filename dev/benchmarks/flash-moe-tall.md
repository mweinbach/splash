The isolated M64N64 candidate increases the packed expert row tile from 32 to 64.
At 2,048 physical rows with top 10 routes, a uniformly used 512-expert layer has
about 40 rows per expert. M64 can replace two M32 jobs with one, reducing repeated
weight reconstruction/staging. Padding and register pressure can offset this
gain, so both 128-thread SG4 and 256-thread SG8 producers are provided.

The candidate preserves original aligned Q4/G64 FP32 coefficient reconstruction,
BF16 operands, ascending K64 MPP MAC traversal, BF16 SwiGLU stages, canonical
down scatter and combine. Gate/up source arrays use 24 KiB (A 8 KiB and two B 8 KiB);
down uses 16 KiB. The oracle checks actual compiled pipeline limits before loading
the model or submitting work. Shader Validation can increase static memory; a
preflight refusal is not a qualification pass.

The original isolated artifact uses two private M64 job kernels with the existing
32-byte bucket ABI, stable expert offsets, GPU job counts and fixed-capacity
emission. Its host graph owns modified parameter bytes and changes the two job,
gate/up and down dispatches. Pack, excluded-route poisoning and combine retain
their production dispatches. Unchanged parameter bytes still borrow the source
graph, which must survive submission.

After Root's original-source bit-exact qualification across layers 0/24/47,
SG8 was added as the opt-in production policy `SPLASH_FLASH_MOE_M64=1`, requiring
`SPLASH_FLASH_MOE_Q4X8=1`. The production bucket producer directly accepts tile 64,
and the blocked dispatcher selects its 256-thread SG8 kernel. Forward and batch
prefill select M64 only for at least 4,096 physical rows without a hot-expert
cache; smaller and hot-cache producers retain M16/M32. Conservative M8 scratch
capacity is unchanged. Flags remain off by default, freeze on first use, and
reject malformed values/dependencies. The route identity records this policy.
Root's primitive SG8 speedups were 1.22–1.25× on spread and 1.10–1.17× on
concentrated fixtures; SG4 lost performance and is not a production route.
Matched service tests retained M32 for singleton 2K prefill; the M64 policy
applies to larger combined cohorts. Primitive eligibility still permits
explicit M64 oracle tests from 1,024 rows.

The private library links 49 frozen production AIRs from `build/flash-round6/metal`
and the two candidate AIRs. Compilation and the CPU self-test submit no GPU work.
Root runs GPU qualification serially:

```sh
build/flash-moe-tall/flash-moe-tall-oracle --cpu-self-test
build/flash-moe-tall/flash-moe-tall-oracle --pipeline-metadata \
  build/flash-moe-tall/splash.metallib

FLASH_MOE_TALL_ROWS=2048 FLASH_MOE_TALL_TILE=32 \
FLASH_MOE_TALL_SIMDGROUPS=4 FLASH_MOE_TALL_PAIRS=6 \
build/flash-moe-tall/flash-moe-tall-oracle \
  build/flash-moe-tall/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/moe-tall-m64n64-sg4-layer0.json
```

Repeat with `FLASH_MOE_TALL_SIMDGROUPS=8`. `FLASH_MOE_TALL_TILE` selects the
production control M8/M16/M32 tile; the candidate always uses M64N64. Additional
original source planes use
`FLASH_MOE_TALL_PREFIX=language_model.model.layers.24.mlp.switch_mlp` or layer47.
Matching raw BF16 input and I64 top 10 IDs can be supplied through
`FLASH_MOE_TALL_INPUT` and `FLASH_MOE_TALL_IDS`. The oracle now uses its own
independent CPU pack reference extended to 8,192 rows; set
`FLASH_MOE_TALL_ROWS=8192` for qualification at the production maximum. Legacy
shared reference tests retain their original extents.

Every activation, canonical down and combined BF16 byte must match the current
Q4x8 control before and after alternating matched commands. Counts, offsets,
packed inputs and stable route maps are compared against the CPU reference;
M64 jobs use an independent 64-row bucket walker, separate from the control's
M32 job reference. Reports record source/manifest identities, both job counts,
capacities, utilization, actual pipeline memory and GPU/wall timings. Promote
only after source-wide primitive qualification and a matched full-model HTTP
gain.
