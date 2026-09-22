Private Qwen3.8-27B M5 Ultra prefill screen
=========================================

This experiment compiles the original native Q4 prefill kernels unchanged and
adds bounded projection candidates. It changes no normal production source,
model files, decode route, service defaults, or GPU die placement.

The installed target has 64 layers, hidden width5120, MLP width17408, GDN input
width16640, attention input width14336 and mixer output input width6144. The
Apple10 policy selects M32N128/four SIMD groups during prefill. Its Q4 operator
executes K64 matmuls followed by each group's FP32 scale/bias epilogue; K5120,
6144 and17408 therefore require80,96 and272 ordered group iterations. The
four-SIMD route already reads sums directly without threadgroup staging barriers.

The private routes expose concrete alternatives:

* Original Q4 M32N128/four SIMD groups is the control, including input sums.
* Q4 M64N128/four SIMD groups retains the group order and affine expression.
  Matching64-row input sums preserve the SIMD32 sum tree. Whether this row
  geometry preserves all BF16 words is measured, not assumed.
* Original M32N128/eight SIMD groups stages32 KiB of sums. A private reversible
  helper snapshot changes only its symbol name and the sum-source constant to
  read the existing device sums directly. It keeps all ordered group epilogues,
  avoids that threadgroup allocation and tests whether the SIMD-count policy
  was confounded by staged sums. Full output comparison gates exactness.
* M32N64/two or four SIMD groups and M16N64/two or four SIMD groups read device
  sums directly. At four groups, the smaller output tile lowers the per-thread
  accumulator/partial footprint. Their private helper uses the existing
  validity-aware cooperative traversal once, so padded fragment capacity does
  not read invalid scale/bias positions. M16 uses the original SIMD32 input-sum
  tree with a16-row layout. BF16 output stores use valid FP32 accumulator
  coordinates directly and make no cross-dtype fragment-layout assumption.
  Quant-group order and all affine expressions remain
  identical. Actual GPU guards and full native BF16 parity are still required.
* Whole-K MPP at M32N128, M64N128 and M128N64 uses source-reconstructed row-major
  F32 coefficients or once-rounded BF16 coefficients. F32 isolates grouping
  changes; BF16 adds coefficient rounding. Both are numerical alternatives.
  Column-fast, row-fast and row blocks of4 vary scheduling without die affinity.

Build and CPU-only checks:

    make -f dev/benchmarks/prefill4k_q27.mk -j3 all cpu-self-test
    .venv/bin/python dev/benchmarks/prefill4k_q27_export.py --check

The exporter reads the actual native layer headers/section alignment and
tile256 Q4 layout, validates the source artifact hash, and independently
certifies every emitted F32 and BF16 coefficient. The native loader repeats
all coefficient checks with an independently staged scalar multiply/add.

A source-certified N256/K6144 fixture already exists at
`build/prefill4k-q27-artifacts/prefill4k_q27_export_gdn_output_n256_v1.bin`.
Run its loader without constructing a Metal backend:

    build/prefill4k-q27/prefill4k-q27-oracle --source-self-test \
      build/prefill4k-q27-artifacts/prefill4k_q27_export_gdn_output_n256_v1.bin

Only Root runs GPU jobs, after exclusive GPU/memory timing is available:

    MTL_SHADER_VALIDATION=1 Q27_ROWS=33 Q27_SAMPLES=3 Q27_BATCH=1 \
      build/prefill4k-q27/prefill4k-q27-oracle \
      build/prefill4k-q27/prefill4k_q27.metallib \
      build/prefill4k-q27-artifacts/prefill4k_q27_export_gdn_output_n256_v1.bin \
      build/release/flash/prefill4k_q27_guard_v1.json

N256 timings cannot represent full-role occupancy. Export one full GDN output
projection (206,438,448 bytes including source and both coefficient routes):

    .venv/bin/python dev/benchmarks/prefill4k_q27_export.py --role gdn-output \
      --output build/prefill4k-q27-artifacts/prefill4k_q27_export_gdn_output_full_v1.bin

Then Root can measure all28 routes at2048 real rows:

    Q27_ROWS=2048 Q27_SAMPLES=7 Q27_BATCH=2 \
      build/prefill4k-q27/prefill4k-q27-oracle \
      build/prefill4k-q27/prefill4k_q27.metallib \
      build/prefill4k-q27-artifacts/prefill4k_q27_export_gdn_output_full_v1.bin \
      build/release/flash/prefill4k_q27_gdn_output_full_v1.json

`Q27_FILTER=q4_candidate`, `f32`, `bf16`, or a complete candidate name narrows
the screen while always retaining the control. `Q27_PATTERN=alternating` or
`sparse` changes finite BF16 inputs; random is default. `mixed` stresses ordered
reductions using finite positive/negative inputs across33 powers of two;
these are deliberate numerical stress inputs, not model activation captures.
Malformed environment
values and fixture extents reject before submission. Actual model activations
and whole-request generation remain follow-up gates.

Complete warm projection GPU/wall timings include original Q4 input-sum
dispatches, whole-K coefficient operand loads and BF16 outputs. They exclude
one-time CPU conversion/upload. Source and output guard buffers, immutable
operand hashes, every finite output word, full-output differences from native
Q4, and exact same-geometry traversal parity are checked. Sampled independent
double dots record the source affine result and each coefficient route's own
result. Source compatibility includes conservative FP32 accumulation error
plus the directly measured input-weighted coefficient-rounding loss. A
separate strict BF16-cell gate compares the kernel with its own coefficient
math; cancellation-dominated and wrong-sign cells are reported. The CPU gate
explicitly proves that losing a small residual can pass the broad arithmetic
bound while failing the strict BF16-cell gate.

The actual optional body BF16 cache would require45.43 GiB, or47.80 GiB including
the vocabulary head, plus retained original source weights (16.16 GiB for all
native maps), scratch, KV/state and admission reserves. F32 doubles the cache
footprint. No full cache is built here. Native request quality, continuation,
output hashes, lifecycle behavior and actual prefill throughput must qualify
any integration; a primitive gain does not establish4K tokens/s.

The first actual full GDN-output screen passed all source, guard and traversal
checks. Native Q4 M32/S4 took1.286 ms. Q4 M64/S4 was byte-exact but took7.150 ms.
Whole-K F32 took4.13–4.37 ms. Best BF16 M32N128 row-fast took1.228 ms, a small
4.7% primitive improvement, but changed4,037,205 of10,485,760 output words and
had sampled relative L2 about0.00209 from the source-affine BF16 reference.
The cache routes remain disabled; these data do not support building a full
47.80 GiB cache for this role. The separate SG8/direct-sum screen follows
these negative results and has not yet been run.

Root subsequently measured SG8/direct at the full GDN-output shape: it retained
every BF16 word but was slower, around0.87x the native control. The queued N64
screen now has a full source-certified MLP-gate fixture at
`build/prefill4k-q27-artifacts/prefill4k_q27_export_mlp_gate_full_v1.bin`:
N17408/K5120,584,908,848 bytes. All89,128,960 F32 and BF16 coefficients have
independent scalar certificates. Its native packed/scales/biases total50,135,040
bytes. Use `Q27_FILTER=q4_` to compare the eight quantized routes without
executing any coefficient-cache matmul. The driver still checks source cache
certificates and maps these bounded one-role operands for immutability checks;
it does not expand a whole model or change decoding.

Root's full MLP-gate small-N screen passed native BF16 parity for all eight
quantized routes. Native M32N128/S4 took3.178 ms; M32N64/S2 took3.165 ms, a1.004x
ratio consistent with noise. M32N64/S4 and M16N64/S2 took about3.66 ms, and
M16N64/S4 took4.304 ms. These geometry-only variants remain private.

Two additional `pair` routes issue two independent K64 matmuls before reading
either result, then execute both original FP32 affine epilogues in group order.
They use M32N64/S4 or M16N64/S2 to contain partial/accumulator register footprint.
The final odd K group retains one unchanged epilogue. This changes scheduling,
not scale/bias factoring or coefficient precision. Ordered source math does not
establish identical compiler-generated BF16 outputs. Use
`Q27_FILTER=pair` to retain the native control and only these two candidates.

Root's pair screen passed the synthetic odd-group shader guard. At the real
full MLP-gate2048-row shape, native took3.180 ms; M32N64/S4 pair took3.570 ms,
and M16N64/S2 pair took3.546 ms. Both pair routes changed954 BF16 words despite
ordered source epilogues. The measured routes are slower and fail native
byte parity, so no integration or GDN rerun is justified from these results.

Installed MLX backend reference
------------------------------

`prefill4k_q27_mlx_reference.py` benchmarks the original Flash source coefficients
using the bundled MLX backend and its required qwen4_exp compatibility model.
This model is not present in pristine upstream MLX-LM. Optional HC fusion,
hybrid/compiled HC, gathered QSA and eager-dispatch choices are disabled inside
the isolated process. No serving engine or model settings are loaded.

CPU-only source/model/token/version preflight succeeds with the actual bundled
Python and proves `mlx.core` was not imported:

    env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      /Applications/oMLX.app/Contents/MacOS/omlx-cluster-python \
      dev/benchmarks/prefill4k_q27_mlx_reference.py --preflight \
      --output build/release/flash/prefill4k_q27_mlx_flash_preflight_bundled_v1.json

Only Root can replace `--preflight` with `--run-gpu` and a fresh output name.
Defaults are the current frozen code2048 token file, one2048-row call, one warmup,
three measured fresh-cache requests, and64 greedy output tokens. `--max-tokens
256` increases the output budget. Prefill evaluates all KV/GDN/PLE cache state
and synchronizes; load, compilation warmup, sampling and full finite-state
validation are excluded from model-forward rates and reported separately.

The default `--lm-head-policy last` harness wrapper slices hidden states before
calling the original loaded quantized vocabulary module. This avoids a needless
2048-row vocabulary projection and matches Splash's final-logit scope;
`--lm-head-policy all` records the untrimmed installed language-call diagnostic.
Neither choice alters checkpoint coefficients.

Installed versions are MLX0.32.2, MLX-LM0.31.3 and MLX-VLM0.6.3. The official
v0.32.2 matmul source predates commit2d27ab0's separate M5 Ultra routing; packaged
binary source/backport provenance remains unproven. The vendor's Qwen4 GDN
normalization override is not called by its installed Qwen3.5 parent forward.
Thus these timings are an accurately labeled installed-backend reference,
not yet canonical Flash model-quality ground truth. No reference GPU run was
performed while preparing this harness. Root subsequently completed one warmup
and three full64-token requests: median prefill3219.1 tokens/s and ordinary
autoregressive decode31.07 tokens/s, with all four output hashes identical.
The retained report is
`build/release/flash/prefill4k_q27_mlx_reference_run_last_v1.json`.

Native Q27 coefficient bridge to MLX
------------------------------------

The exact mlx-community original Q27 checkpoint is absent from known local MLX
directories. `prefill4k_q27_mlx_layout.py` plans1,847 loaded stock qwen3_5 keys
from the installed66 target containers, including fused-projection slices and
nonquantized parameters. `prefill4k_q27_mlx_convert.py` repacks U32 integer words
and copies BF16 scale/bias words without reconstructing or requantizing weights.
Embedding is already row-major; body/head are tile256. GDN input packs qkv,z,
b48,a48, then160 unused tail rows; q-projection query/gate rows remain interleaved.

Native norm data contains operative BF16 gains after centering, and GDN stores
pre-exponentiated negative F32 a_scale. Original centered norm and A_log words
are not claimed recoverable. The converter retains exact operative gains,
exports convolution as[C,4,1], and excludes MTP so stock sanitize adds no extra1.
Exact native decay lives in a required sidecar; unused A_log slots are zeros.
The private artifact has a custom model_type that rejects accidental plain
stock loading. `prefill4k_q27_native_mlx_adapter.py` explicitly selects the stock
qwen3_5 classes and binds48 GDN parameters to the sidecar. Only compute_g changes
to exp(native_a_scale.float32*softplus(a+dt_bias)); the stock state-update kernel
is unchanged. This is a native-coefficient bridge, not full upstream raw-word
recovery or an already qualified model reference.

CPU samples certified51 tensor planes and71,040 U32/BF16 words, including small
GDN row slices and embedding/head boundaries. All48 native decay vectors were
finite negative F32. Integer inverse permutation and every emitted chunk's
readback are checked. Source headers/extents, manifest/config and all copied
tokenizer files are watched for changes. Full conversion has not been run.

Default bounded CPU preparation:

    .venv/bin/python dev/benchmarks/prefill4k_q27_mlx_convert.py \
      --report build/release/flash/prefill4k_q27_native_mlx_adapter_cpu_next.json

Only after Root grants a serialized memory/I/O slot can it add
`--run-root-memory`. Output must be a fresh workspace build directory; the
target tensor payload is15,132,806,656 bytes (14.09 GiB), excluding draft/vision.
Binary/NumPy chunk payload remains bounded around16 MiB. No original file,
installed model catalog, service settings or normal runtime is changed.

Root completed full conversion with all integer inverse/readback certificates
and preserved original sources. The same-native-coefficient stock MLX reference
then completed one warmup and three measured64-token requests, all finite with
identical greedy output hashes and all48 native-decay bindings active. Median
prefill was1883.16 tokens/s, versus the original Splash native result around2042;
ordinary target-only decode was49.17 tokens/s. This does not support a4K stock
reference expectation for Q27 on this workload. No256-token rerun is required
to validate that matched prefill comparison. Future driver preflight explicitly
checks that converted and selected native-package manifest identities match.
The existing run's pair was independently verified to match before that added
assertion. Its source version is preserved by the recorded hashes.
