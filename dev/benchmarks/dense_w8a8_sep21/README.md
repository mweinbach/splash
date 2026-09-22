Standalone W8A8 dense prefill numerical alternative. It does not edit production
or persist quantized sidecars. Compilation, CPU tests and preview never create
a GPU backend or read model/captured payloads. Only Root invokes `--run`, which
reads exactly one requested captured role. Default is QKV; other authorized
filters select Z, GDN out, or QSA Q.

Weights are quantized from the verified original saved BF16 coefficient rows,
using symmetric signed I8 codes [-127,127] and one F32 scale per output row.
Activations use a GPU BF16 row-max/F32-scale/precise-RNE I8 converter. Every timed
projection repetition includes this converter and the dense kernel. SG4/8
device I8×I8 whole-K MPP operations use an I32 cooperative destination, followed
by two separate F32 scale multiplications and one BF16 cast. K2560 and K6144
symmetric integer dot bounds are 41,290,240 and 99,096,576, respectively.

Normal timed kernels omit I32 probe writes. Separate untimed probe entries
expose the integer destination for sampled exact CPU dot checks and a complete
late-scale/BF16 numerical identity check. Thus probe instrumentation adds no
output traffic to the timed candidate. Source-compile LLVM confirms I8×I8→I32
MPP dispatch, separate scale multiplications, and omitted normal I32 writes;
hardware correctness and performance remain Root-owned.

Build/freeze with no GPU or payload reads:

```
make -f dev/benchmarks/dense_w8a8_sep21/Makefile -j4 all cpu-self-test cpu-quant-test preview freeze
```

Bounded Root-only first screen:

```
build/dense-w8a8-sep21/oracle --run --samples 10 --repeat 4 --out build/dense-w8a8-sep21/qkv-v1.json
```

`--projection-filter linear_attn.in_proj_z`, `linear_attn.out_proj`, or
`self_attn.q_proj` selects another single authorized role. `--groups 4` or `8`
narrows the candidate. `--preview` reports role/variants from metadata only.

Preregistered component quality gates are full-output relative L2 ≤ 0.02 and
cosine ≥ 0.9998 against the exact captured BF16 control. A failed quality gate
is reported independently of timing. Passing these component gates does not
qualify model quality, MTP acceptance, or authorize a production overlay.

Certificates cover original operand SHA256, BF16 captured control parity,
complete GPU activation scales/RNE codes versus CPU, coefficient and activation
source-to-dequantized FP64 error, sampled exact integer dots, full BF16 output
reconstruction from integer dots and F32 scales, and sampled original/dequantized
FP64 products. All output/I8/F32-scale/I32 guards and sticky diagnostics are
checked; complete outputs must repeat before and after timing. Immutable source
operands, I8 coefficients, weight scales and the untimed I32 probe are hashed.

After all CPU qualification, each candidate and its matched BF16 control accrue
at least 150 ms of GPU-only warmup in balanced AB/BA pairs. There is no CPU
operand/output/guard access during warmup or timing. Timings balance candidate
position and AB/BA order strata exactly; requested samples are rounded upward
to a multiple of twice the candidate count. Reports retain every GPU/wall pair,
warmup totals and order/position counts.
