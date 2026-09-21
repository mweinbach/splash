# Private affine INT8-code screen

This local experiment expands each original Q4/Q5/Q6/Q8 unsigned code `q`
to the exact signed byte `c = q - 2^(bits-1)`. Original BF16 scale and bias
words remain unchanged. Activations retain their original BF16 words; there
is no fitted weight quantizer, activation quantizer, or checkpoint rewrite.

For each group of 64 inputs it computes

```
sum_x = F32 sum(BF16 x)
dot_c = MPP dot(BF16 x, INT8 c), with F32 destination
group_y = scale * dot_c + (bias + center * scale) * sum_x
y = BF16(F32 sum(group_y))
```

Code expansion is lossless; the factored reduction is a numerical alternative.
In particular, F32 `bias + center * scale` can discard a small bias. The CPU
review demonstrates this for every supported bit width: with `scale=1`,
`bias=2^(bits-25)`, `q=0`, and `x=[1,0,...]`, the original result is the nonzero
BF16 bias while this epilog returns zero. Consequently this candidate remains
private and disabled even if selected real source matrices pass the screen.
It needs a stable correction and full model qualification before promotion.

Each projection submits three dispatches: exact row padding, one SIMD F32
activation sum per row/group, and the mixed BF16/INT8 MPP dot plus affine
epilog. The selected code cache adds one byte per coefficient, versus four
bytes for the F32 coefficient cache. Source scale/bias storage is shared.
The G64 dot/epilog is repeated along K; reduced memory does not establish a
speed gain until it is measured against the current qualified source routes.

The oracle compares every real BF16 output against both the current affine
route and the F32 coefficient cache, and samples independent CPU F32 serial
and SIMD reductions. It also checks padding, sticky diagnostics, deterministic
outputs, unchanged input, code samples, guarded source-code/output allocations,
and host rejection of short, overlapping, overflowing, or malformed views.
The accuracy threshold remains `1e-4` relative L2; a failure is retained in JSON.

CPU checks create no Metal device:

```sh
build/flash-affine-int8code/flash-affine-int8code-oracle --cpu-self-test
.venv/bin/python dev/benchmarks/flash_affine_int8code_cpu.py
```

Only the root agent runs GPU jobs, with other inference jobs stopped. The
complete source screen is:

```sh
SPLASH_FLASH_QMV_F32=1 SPLASH_FLASH_EXPERT_QMV=1 \
FLASH_INT8CODE_ROWS=1,4,8,16 FLASH_INT8CODE_TILES=0,1,2,3 \
FLASH_INT8CODE_REPEATS=6 \
build/flash-affine-int8code/flash-affine-int8code-oracle \
  build/flash-affine-int8code/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/int8code-source-screen.json
```

Defaults select the full Q8 vocabulary head `[248320,2560]` and layer 0 Q6
GDN QKV `[10240,2560]`. Optional Q4/Q5 source examples are layer 1 PLE key
projection and layer 0 attention HC up projection. Select these with
`FLASH_INT8CODE_PREFIXES=language_model.model.layers.1.ple.key_proj,language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up`.
Tiles 0/1/2/3 correspond to M8N64/M8N128/M16N64/M16N128; N128 tiles require
the output width to be divisible by 128.

The private metallib links all round10 source AIRs plus the candidate AIR.
The original library and executable remain unchanged. Private artifact hashes,
source dimensions, cache sizes, and completed CPU results are recorded under
`build/flash-affine-int8code/`. Compiled artifacts and CPU results are distinct
from a completed GPU screen or a qualified model route.

## Unsigned v2 correction

The separate v2 shader stores original codes directly as UINT8 and computes
`scale * dot(BF16 x, UINT8 q) + bias * F32sum_x`. It forms neither a centered
dot nor a rounded `bias + center * scale` correction. The CPU review confirms
that this fixes the demonstrated lost bias and small-code cancellation traps;
it remains a group-factored numerical alternative because source coefficients
round individually to F32 and the MPP dot uses its own reduction order.

The v2 artifact preserves the first linked candidate. It adds exact source
shapes and typed view alignment to the descriptor guards. CPU compilation and
checks do not prove runtime UINT8 MPP precision.

The root agent first executes the hard GPU fixtures:

```sh
SPLASH_FLASH_QMV_F32=1 SPLASH_FLASH_EXPERT_QMV=1 \
build/flash-affine-int8code/v2/flash-affine-uint8code-v2-oracle \
  --gpu-self-test build/flash-affine-int8code/v2/splash.metallib \
  build/release/flash/affine-uint8code-v2-hard-traps.json
```

This loads no model. It tests five fixtures at Q4/Q5/Q6/Q8, rows 1/4/8/16,
and every tile, for 320 cases: tiny bias, mixed-sign tiny bias, small-code
contribution, mixed codes with signed scales, and a BF16 midpoint. Each output
must match the original affine kernel, F32 coefficient MPP, independent CPU
reference, and the explicit nonzero or tie-cell expectation where applicable.

After those pass, use the complete source-screen command above with the v2
oracle and metallib paths and a separate v2 report. Preserve completed source
screens separately from hard-fixture results and model qualification.
