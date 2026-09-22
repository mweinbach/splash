This private dense decode primitive requantizes one saved, actual F32
projection matrix to signed I8 coefficients and F32 row scales. It uses
unchanged BF16 activation rows from a captured prefill-2K input, rather
than a live decode activation. Eligible roles are attention, GDN and PLE;
vocabulary, router, HC down and MTP roles are excluded. Fixture metadata
and payload hashes are certified by the root preparation flow using
schema `splash-private-dense-i8-f32-row-fit-fixtures-sep21-v1`.

The candidates are:

| Variant | Suffix | Tile M/N | SIMD groups |
| --- | --- | --- | --- |
| 1 | `m8_n64_sg4` | 8 / 64 | 4 |
| 2 | `m8_n128_sg4` | 8 / 128 | 4 |
| 3 | `m16_n64_sg4` | 16 / 64 | 4 |
| 4 | `m8_n64_sg2` | 8 / 64 | 2 |

Names begin with `dense_i8_decode_sep21_`; separate F32-output audits
add `_audit`. Runtime coefficient conversion finds each F32 row's
maximum absolute value, sets scale to `maxabs/127`, and writes I8 codes
using round-to-nearest-even and clamp [-127,127]. All-zero rows use
code 0 and scale 1. Nonzero row scales are floored to the minimum
normal F32 value. Whole-K BF16×I8 dots accumulate in F32, apply the
coefficient row scale once, and cast once to BF16. Candidate activation
quantization and padding are absent; dynamic row extents mask partial
R1/R4/R16 tiles directly.

Controls include original stock `addAffine` over a manually constructed
`FlashAffineProjection`, with `SPLASH_FLASH_QMV_F32=1`, plus stock
FloatDenseSmallRows and optional BF16SmallRows. Three selected original
raw tensor spans are copied into guarded buffers and hashed before and
after timing. No full raw shard checksum or full-model read is claimed.
Before timing, 260 uniform original affine coefficient reconstructions
must match the certified saved F32 matrix bit for bit. Raw control output
must match F32 cache output within relative L2 ≤1e-4. Cache controls
include their input padding and helper dispatch costs. The R1 shipping
raw route is constructed over captured prefill inputs; live decode and
MTP acceptance remain unqualified.

Before timing, full BF16 outputs must pass preregistered relative
L2 ≤0.02 and cosine ≥0.9998 against the original F32 control. Complete
BF16 equality is also reported. Mandatory mathematical samples check
an activation-weighted F64 coefficient-quantization envelope against
original F32 coefficients. Samples cover at most 512 uniform columns
plus all tile boundaries, first/last columns and near-zero actual-output
columns, on first/last active rows. Full BF16 output comparison and
the coefficient census still cover every value. `--strict` requires full BF16 equality;
it is off by default. Rejected candidates retain failure reports with
no timing samples. Coefficient requantization remains a numerical
alternative, with model, semantic and live decode quality unqualified.

The governor reserves every allocation before construction: original
F32 weights, new I8 codes and F32 scales, optional BF16 weights and
control buffers, captured BF16 inputs, stock padded-input workspaces,
outputs, F32 audits and guards. New code and scale planes include
64-byte guards and 16KiB rounding. Up to two stock padded-input
workspaces hold `16*K*2` logical bytes each. The shape-derived plan
comes from `cache.hpp`, protecting at least 16GiB or 10% of host RAM.
Only the fixture's actual projection is loaded, not a full model.

GPU execution alone maps and verifies the certified weight and captured
input files, creates the I8 coefficients, and performs initial/final
immutability and guard checks. Balanced warmed timings rotate raw,
F32, optional BF16 and candidate categories across row shapes and
variants. Coefficient conversion, CPU scans, mathematical
certification and immutability checks remain outside timed GPU commands;
there are no CPU scans between paired timed commands.

Strict C++ and Metal 4.1 builds and CPU checks read no fixture payloads
and create no Metal device:

```sh
make -f dev/benchmarks/dense_i8_decode_sep21/Makefile -j4 cpu-self-test
```

The default host is `build/prefill4k-wide-fullcache`, whose current
headers and exported FloatDenseSmallRows/BF16SmallRows helpers match.
The candidate compiles directly with its portable `abi.hpp`. Its AIR
links with the complete original `build/flash-next` native AIR inventory.
`all` only builds; `cpu-self-test` also runs `--cpu-self-test`.

Prepare all row shapes and variants without executing GPU work:

```sh
.venv/bin/python dev/benchmarks/dense_i8_decode_sep21/run.py \
  --cases build/dense-i8-decode-sep21/actual-role-fixtures.json \
  --rows 1,4,8,16 --pairs 4 --report build/dense-i8-decode-sep21/new-screen.json
```

Add `--run` only in the exclusive root GPU window. Rows may be supplied
as CSV or separate values selected from 1,4,8,16; omission uses all four.
`--projection` selects an exact manifest name. A short first screen can
use `--projection language_model.model.layers.0.linear_attn.out_proj --rows 1,4`.
`--pairs` selects 1–32 timing repetitions and defaults to 4.
`--variant 1` through `--variant 4` selects one candidate; omission
screens all four. The runner records code, binary/metallib and fixture
manifest hashes plus command, controls and comparison limitations.
Preparation never reads the weight or captured input payloads.
