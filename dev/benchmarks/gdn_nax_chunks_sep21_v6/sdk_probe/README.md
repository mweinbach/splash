Compile-only Metal 27 MPP feasibility proof, 2026-09-21. No Metal device,
pipeline, command buffer, GPU dispatch, or model payload was opened.

The active SDK is MacOSX27.0.sdk; the compiler is Metal 32023.921
(metalfe-32023.921.6), from Metal toolchain v27.1.266.1. All probes use
`-std=metal4.1 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0`.
The source sets safe math, disables contraction and reassociation, and requests
`relaxed_precision=false` unless a specifically named relaxed probe overrides it.

Result: **81 positive probes compile; 12 intended negative probes fail;
93/93 outcomes match the expected SDK contract.** Commands and full diagnostics
are in `build/gdn-nax-chunks-sep21/sdk-probe/results.json` and individual logs.

Each of these 9 `(M,N,K)` geometries compiles with SG4 and SG8 for all three
operand combinations: F32/F32, BF16/F32, F32/BF16, always a float cooperative
destination. This is 54 positive base probes.

| M | N | K | Intended coverage |
|---:|---:|---:|---|
| 16 | 128 | 128 | Time16 token/value tile |
| 32 | 128 | 128 | Time32 token/value tile |
| 128 | 16 | 128 | Transposed Time16 projection tile |
| 128 | 32 | 128 | Transposed Time32 projection tile |
| 128 | 128 | 16 | Time16 state update |
| 128 | 128 | 32 | Time32 state update |
| 128 | 128 | 128 | Full state-sized product |
| 16 | 16 | 128 | Time16 Gram |
| 32 | 32 | 128 | Time32 Gram |

At `(32,128,128)`, all three type combinations also compile in each of these
variants: threadgroup-pointer tensors, transposed right operand, relaxed
precision enabled, and explicit multiply-accumulate. These supply 24 more
positive probes. SG1 cooperative left inputs supply the last three positives.

The six SG4/SG8 cooperative-input probes fail at SDK implementation line 5264
with `Input cooperative tensors require a single SIMD group`. Use ordinary
device/threadgroup tensors as inputs to SG4/SG8 products; a cooperative output
is allowed. The six const-input probes fail because `const float` and
`const bfloat` do not match the SDK's unqualified element-type traits. If inputs
are semantically immutable, create tensor metadata from unqualified pointers
and keep all code paths read-only, as the existing isolated GDN experiment does.

Float coefficients do not need conversion to BF16. The SDK explicitly supports
F32/F32/F32, BF16/F32/F32, and F32/BF16/F32 in MPPTensorOpsMatMul2d.h lines 25,
36, and 39. Keep gates, beta-after-expansion, triangular solve, coefficient
matrices, and carried recurrent state in F32; native BF16 Q/K/V may participate
in mixed products with a float destination. For the strongest all-F32 arithmetic
path, expand BF16 Q/K/V exactly to float before the product and use F32/F32/F32.
That choice affects numerical association and GPU performance; this CPU proof
does not establish either.

The public constructor is
`matmul2d_descriptor(m,n,k,transpose_left=false,transpose_right=false,
relaxed_precision=false,mode=mode::multiply)`. The third boolean parameter,
immediately after transpose_right, is the relaxed-precision setting; it is the
sixth constructor argument counting m/n/k. There is no TF32 enum in this API.
Header constructor lines 385-396 prove defaults; lines 120-121 explain that
enabling relaxed precision permits an accuracy/performance tradeoff.

Apple's [MSL specification, Table 7.4, p354](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf#page=354)
specifically says enabling relaxed precision permits truncation of the float
mantissa before multiplication. Keep it false. Its
[float accumulation example, p358](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf#page=358)
uses a float cooperative destination and explicit multiply_accumulate. The
[MPP programming guide, section 2.3.4, p8](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf#page=8)
also demonstrates retaining float partial results while accumulating K chunks.
Explicitly select multiply_accumulate when reusing a retained accumulator across
multiple run calls; the default mode is multiply.

`f32-precise.ll` additionally demonstrates an emitted
`__tensorops_impl_matmul2d_op_run_cooperative_dv_f32_dv_f32_f32_v2` call with a
descriptor containing relaxed_precision=0. Corresponding mixed-type AIR/LLVM
probes identify BF16/F32/F32 and F32/BF16/F32 entry points. This supports the
API-level float input/destination contract with permitted mantissa truncation
disabled. It does not prove an internal instruction format, actual NAX usage,
GPU occupancy, resource limits, throughput, or bitwise equality to a sequential
F32 FMA loop.

Reproduce all expected outcomes with:

```sh
python3 dev/benchmarks/gdn_nax_chunks_sep21/sdk_probe/compile_probes.py
```

Optional positional substrings select only matching probes and merge their
results into the existing JSON; this is useful for bounded additions.
