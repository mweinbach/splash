Private weight-only I8 dense prefill numerical alternative. BF16 activations
remain unchanged. The verified original BF16 coefficient rows are fitted to
the same symmetric signed I8/F32-row-scale format as the existing W8A8 screen.
Direct BF16×I8 whole-K MPP operations accumulate into F32, then multiply by one
F32 weight scale and cast to BF16 once. No activation converter is introduced.

SG4/SG8 M128N64 entries and an optional M64N128 SG8 entry are compiled. LLVM
confirms BF16×I8→F32 dispatch, one late F32 scale multiplication, and no raw-F32
probe access in normal timed entries. Separate probe entries expose the unscaled
F32 result only during qualification. No component gain or accuracy is assumed.

Build/CPU test/freeze without GPU or model/captured payload reads:

```
make -f dev/benchmarks/dense_weightonly_sep21/Makefile -j4 all cpu-self-test preview freeze
```

Root-only first screen, one actual QKV capture:

```
build/dense-weightonly-sep21/oracle --run --samples 10 --repeat 4 --out build/dense-weightonly-sep21/qkv-root-v1.json
```

Single-role filters may select `linear_attn.in_proj_z`, `self_attn.q_proj`, or
`linear_attn.out_proj`. The GDN-output role is a new component screen, not an
eligible whole-model route. `--output-tile` additionally tests M64N128 SG8;
`--groups 4` or `8` narrows M128N64 candidates. Default tests SG4/8 on QKV.

Preregistered quality gates remain full-output relative L2 ≤ 0.02 and cosine
≥ 0.9998 against the exact captured BF16 control. Failed gates are recorded
independently of performance. This standalone does not create a whole-model
worker or production overlay.

Every signed I8 code [-127,127] is exactly BF16 representable; the CPU proof
checks the entire range. Source BF16 A is verified unchanged before and after
timing. Probe and normal complete outputs must match; every BF16 output is
reconstructed from its raw F32 result, F32 row scale and final BF16 conversion.
Sampled raw F32 dots are checked against an FP64 quantized-coefficient dot with
a conservative worst-K sequential F32 addition bound and explicit denormal
envelope. Original FP64 dots are decomposed into quantized-dot times scale plus
a residual dot and checked within a preregistered FP64 accumulation envelope.
Source-to-quantized FP64 differences, source/scale subnormals, errors, guards,
immutable hashes, normal/probe identity and all paired samples are recorded.

After CPU qualification, each candidate and matched BF16 control accrue at
least 150 ms GPU-only warmup in balanced AB/BA pairs. Timings balance candidate
position and pair order exactly. No CPU operand/output/guard access occurs
during warmup or timing. Raw-probe output remains unchanged throughout timing.
Root alone runs serialized GPU/component qualification.
