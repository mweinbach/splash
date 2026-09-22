HC-UP W8A8 standalone numerical alternative

Preparation and compilation are CPU-only. Only Root invokes `--run`; that flag
reads exactly three pinned payloads for layer0 attn HC-UP: activated BF16 input1,
normalized BF16 input0, and original F32 operand512. No coefficient sidecar or
production overlay is created. See `source_audit.md` for the source contract.

The actual R2048 control is cached BF16 M32N128 SG4 whole-K GEMM plus the actual
shared HC mix kernel with the worker's 64-thread launch. Original F32 weights
are RNE-converted for that control only; the complete derived hash must match
the recorded saved BF16 coefficient hash. The complete control raw-output hash
must match the captured raw-output hash, without reading either additional
BF16 weight or expected-output payload. W8 weights are fitted directly from the
original F32 values, never from the BF16 control copy.

Candidates use SG4 or SG8 M128N64 I8×I8→I32 whole-K GEMM and two ordered F32
scale multiplications, then BF16 raw-dot. They use the same HC post shader as
control. Every candidate timing includes GPU activation row-max/RNE-I8
conversion, integer projection and postprocessing; the matched control includes
BF16 projection and postprocessing. Injection-gate projection is outside this
HC-UP component experiment and shared by any eventual worker adaptation.

Untimed integer and post-stage probes must preserve the complete normal raw and
mixed outputs. Full raw-dot, sigmoid gate, modulation product, each sequential
BF16 stream sum and final mixed output must each meet relative L2 <=0.02 and
cosine >=0.9998 against actual control. Failures remain in the report and exclude
promotion. Coefficient fitting and sampled source/dequantized FP64 errors are
reported separately; the experiment stays `model_qualified=false`.

Certificates include complete GPU activation scales/RNE codes versus CPU,
sampled exact I32 dots, complete late-scale/BF16 identity, subnormal source
counts, and a stated FTZ envelope limited to genuinely subnormal F32
intermediates. Normal values require exact late-scale identity. Timing uses
balanced candidate positions and AB/BA pairs after >=150 ms of GPU warmup per
candidate and matched control. There is no CPU payload/output/guard access
during warmup or timing. Complete output repetition, guards, immutable operands
and untimed probe immutability are checked afterward.

Build and CPU-only checks:

```sh
make -f dev/benchmarks/dense_w8a8_hcup_sep21/Makefile -j4 all cpu-self-test preview
```

Root-only bounded execution:

```sh
build/dense-w8a8-hcup-sep21/oracle --run --samples 10 --repeat 4 --out build/dense-w8a8-hcup-sep21/hcup-root-v1.json
```

No actual-route fused mixed capture or model continuation proof is available.
The existing raw control capture and identical post shader support this isolated
component oracle; they do not qualify full-model quality or service throughput.
