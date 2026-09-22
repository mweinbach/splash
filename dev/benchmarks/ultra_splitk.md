Private M5 Ultra split-K screen
================================

This benchmark compares the same MPP BF16 matmul tile at whole K and at K
partitions of 2048/4096. Split partitions accumulate into FP32 scratch and the
following dispatch reduces those partials, then rounds once to BF16. Whole-K
and split-K use the same operand bytes, tile geometry, diagnostics, and native
MetalBackend queue. There is no die affinity assumption or explicit placement.

Build and CPU verification, without GPU submission:

    make -f dev/benchmarks/ultra_splitk.mk -j3 cpu-self-test

Run when the GPU is otherwise idle:

    ULTRA_SPLITK_REPEATS=9 ULTRA_SPLITK_BATCH=4 \
      build/ultra-splitk/ultra-splitk-oracle \
      build/ultra-splitk/ultra_splitk.metallib build/ultra-splitk/results.json

For guard/tail validation under the Metal shader validator:

    MTL_SHADER_VALIDATION=1 ULTRA_SPLITK_SHAPE=guard \
      ULTRA_SPLITK_REPEATS=3 ULTRA_SPLITK_BATCH=1 \
      build/ultra-splitk/ultra-splitk-oracle \
      build/ultra-splitk/ultra_splitk.metallib build/ultra-splitk/guard-results.json

`ULTRA_SPLITK_SHAPE` selects a substring. The seven shapes are rows 512/2048,
K6144, N2560 at M64N128/s8 and M16N128/s4; verifier rows 4/16 at M8/M16N128/s4;
and row3/K6176/N130 for padded row, output-column, and last-K-partition tails.

The report includes warm GPU and wall medians for a complete multiply/reduce
sequence; timings are per sequence, divided by the configured command batch.
Route order rotates on each repeat. Scratch logical bytes, actual allocation
ledger delta, full padded writes, and logical reduction reads are separate.
The whole-K route binds an unused guarded dummy scratch buffer solely to keep
the kernel ABI identical; its allocation is reported separately.

The operands are deterministic BF16 data at the model's projection dimensions.
No model is loaded. Input/weight bytes must remain immutable, output/scratch
guards must remain intact, and every written output/scratch element must be
finite. Small tail cases validate all 390 output cells. Performance shapes
validate sampled columns and rows against independently computed double dots.
BF16 output must lie inside the conservative gamma-K FP32 reduction error
interval. Exact BF16 mismatches, maximum ULP, relative L2, and full-output
differences from whole-K are reported independently.

A numerical interval pass establishes a bounded matrix result, not identical
greedy model output, model quality, cross-die traffic, or HTTP speed. End-to-end
model qualification is required before changing a production dispatch policy.

Source F32 selective-verifier screen
------------------------------------

The default depth of three normally verifies four rows: pending token plus
three drafts. In `FlashForward::project`, source-qualified small-row F32 cache
routes take precedence over the BF16 cache. At N2560/K6144, QSA output Q5/G64
and Q6/G64 use M16N64 even at four real rows; Q8/G64 uses M8N64. The GDN output
Q5/G128 stays raw affine at rows4/8 and uses F32 M8N64 at rows16. The vocabulary
head has its own INT8 route at rows2..16, and small trained MTP projections use
raw affine. A BF16 tiny-row win therefore does not establish a runtime win.

Export one selected original affine projection without a Metal device:

    .venv/bin/python dev/benchmarks/ultra_splitk_export.py \
      install/local-models/Flash-Next-oQ4e-mtp-v1 \
      language_model.model.layers.23.self_attn.o_proj \
      build/ultra-splitk/qsa23-f32.bin

The artifact contains original packed weights, stored BF16 scales/biases,
source identity, prefix, and separate NumPy multiply/add F32 coefficients.
The native independent byte unpacker checks every reconstructed F32 word:

    build/ultra-splitk/ultra-splitk-oracle --source-self-test \
      build/ultra-splitk/qsa23-f32.bin

Run its matrix screen exclusively on the GPU:

    ULTRA_SPLITK_F32_SOURCE=build/ultra-splitk/qsa23-f32.bin \
      ULTRA_SPLITK_REPEATS=9 ULTRA_SPLITK_BATCH=4 \
      build/ultra-splitk/ultra-splitk-oracle \
      build/ultra-splitk/ultra_splitk.metallib build/ultra-splitk/f32-results.json

F32 mode tests the active M16N64 projection at rows4/16, exploratory M8N64 and
M8N128 at rows4, source coefficients cropped/padded for row/N/K tails, and an
independent cancellation fixture. Each complete timing includes GPU input
copy/positive-zero padding, multiply, FP32 scratch writes, and reduction.

Every output cell that differs from the whole-K control gets an independent
double dot. Both variants report exact BF16 differences, ULP, absolute error,
relative L2, and gamma-bound compatibility. A separate strict BF16 cell gate
uses `CellRelation` to require at most one ULP plus midpoint/bound agreement.
Cells with reference magnitude below the FP32 bound and wrong-sign results
are counted independently. Early versus late small positive/negative residuals
against large cancelling K partitions ensure the strict gate can expose a
lost near-zero result even when a broad absolute arithmetic bound passes.
The cancellation fixture reports failures instead of rejecting its report.

These checks are matrix arithmetic evidence. They still do not establish
identical greedy generation, quality, or a service performance improvement.

The private padding kernel uses declared bfloat copies while production uses
ushort copies with explicit nonfinite input diagnostics. This screen uses
finite input words and checks byte-exact copies plus positive-zero tails;
the difference does not affect its measured finite workload. Candidate strict
gate coverage is the sampled cells plus every cell changed from the control;
control strict diagnostics are recorded separately.

Observed source layer23 Q5/G64 result (`build/ultra-splitk/f32-results.json`):
the active M16N64 row4 path improves GPU 0.1131 to 0.0591 ms (1.91x), and wall
0.1722 to 0.1019 ms (1.69x), including pad and reduction. Five of 10,240 BF16
cells change, all at one ULP with no strict violations. Row16 improves GPU
1.88x, but its 15 changed cells include two strict violations near zero
(maximum 152 BF16 ULP); the whole-K control has one violation at 344 ULP.
The constructed cancellation fixture loses early residuals in 168/256 cells
in both the control and split variants, despite their identical outputs and
passing broad arithmetic bounds. Their strict gate correctly fails. No
production integration is justified by these isolated results alone.
