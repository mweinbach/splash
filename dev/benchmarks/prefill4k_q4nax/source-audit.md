# CPU-only Q4 expert register-NAX source audit

This audit read small source files and ran CPU coordinate enumeration only. It
did not import `mlx.core`, allocate GPU resources, load a model, scan model/store
payloads, or submit GPU commands. The experiment owner builds the private
shader and oracle separately. Source feasibility is not numerical or service
qualification.

## Verified source identity

The installed package metadata reports MLX **0.32.2**. The include root is:

```
/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages/mlx/include
```

The installed native binary's exact source revision is not established by
header identity. The following small-file bytes were freshly checked against
official raw source, not inferred from the package version:

| Source | Bytes | SHA-256 | Identity |
|---|---:|---|---|
| Installed `mlx/backend/metal/kernels/steel/gemm/nax.h` | 25,621 | `fb10bcae44095e9fed5c81213d03527dd5fe5ac11be11ec7784bd357c6f317ec` | Exact pinned Ultra commit, exact v0.32.2, exact existing Steel vendor copy |
| Installed `mlx/backend/metal/kernels/steel/gemm/gemm_nax.h` | 3,580 | `8f8d11f3c29e30dbfcaa6799963edd5fb71b81f912c4f3494e294d3002e3d337` | Same existing pinned vendor copy |
| Installed `mlx/backend/metal/kernels/quantized_nax.h` | 50,027 | `3dc0dfa3dab060ce1f75dcbb9a4c7ae4546693d74f71cafe7658c1e6743af745` | Exact official v0.32.2 |
| Pinned Ultra `quantized_nax.h` | 50,017 | `d3e174be18edde8c89e630204638dc240479151a92dd0a20c655fe5be9696169` | Differs from installed only in the `sgp_sm` min overload/cast |
| Existing `build/flash-moe-fused/mlx-v0.32.2-source/quantized.cpp` | 58,218 | `44483286a359afb7f35d8a9c1658ffaec6ff6c01a7ab34ac7bb35cf6438804a6` | Exact pinned Ultra commit bytes |
| Installed `mlx-0.32.2.dist-info/METADATA` | 5,863 | `d3454348122b9b131c3b8f97587635e0b13baa6d7805da27e1334d39214f1a41` | Local version evidence |

The pinned commit is `2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99`.
Its existing provenance manifest is
`dev/benchmarks/prefill4k_dense/source_manifest.json`; the vendor's copyright
notices and MIT license remain intact.

Primary sources:

- [Pinned NAX helper](https://github.com/ml-explore/mlx/blob/2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99/mlx/backend/metal/kernels/steel/gemm/nax.h)
- [Installed-version quantized NAX source](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/quantized_nax.h)
- [Pinned sorted RHS gather dispatch](https://github.com/ml-explore/mlx/blob/2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99/mlx/backend/metal/quantized.cpp)

## The source comparison that is actually bounded

The sorted RHS affine gather dispatch chooses `BM32/BN64/BK64/WM2/WN2` when
`M/E < 64`. At 2K/top10 and 512 experts, `M/E=40`, so this source policy chooses
the smaller row tile. Its physical grid spans globally sorted row tiles; each
tile loops over consecutive expert runs. This does not prove the precise
kernel selected inside the installed binary.

The gather implementation already loads A directly from device memory, stages
Q4-dequantized B into padded threadgroup memory, then uses register NAX. Its
inner `SK32` load feeds two ascending K16 MMAs. Two SK32 iterations form each
BK64 step. A register-only Q4 miss experiment can keep Splash's stable native
jobs, packed/padded Direct-A input, original INT8 hit branch, fused gate/up,
BF16 SwiGLU stages, scatter, and canonical combine. It changes the matrix
execution and B loader, avoiding the independent sorting/bucket comparison.

## Register tile and coordinate derivation

Pinned `BaseNAXFrag::mma` has the descriptor:

```
matmul2d_descriptor(16, 32, 16, false, true, true, multiply_accumulate)
matmul2d<descriptor, execution_simdgroup>
```

The sixth descriptor argument permits relaxed precision. A strict private
variant sets it to `false`; it is a separate arithmetic comparison. Input
cooperative tensors require a **single** SIMD group in the current SDK, so the
four-SIMD tile consists of four separate single-SIMD operations, rather than
input cooperative tensors owned by `execution_simdgroups<4>`.

For SIMD group `s` in `[0,3]`, the output quadrant origins are:

```
sm = (s / 2) * 16
sn = (s % 2) * 32
```

Pinned MLX's fragment packing uses lane `l` and fragment element `i` in `[0,7]`:

```
q = l >> 2
r = ((q & 4) | ((l >> 1) & 3)) + 8 * (i >> 2)
c = 4 * ((q & 2) | (l & 1)) + (i & 3)
```

For the NT register operation, with `j` in `[0,1]` and current K16 origin `kg`:

| Register | Original physical source or output |
|---|---|
| `A[i]` | Packed A row `begin+sm+r`, inner channel `kg+c` |
| `B[8*j+i]` | Original weight row `norigin+sn+16*j+r`, inner channel `kg+c` |
| `C[8*j+i]` | Output row `begin+sm+r`, column `norigin+sn+16*j+c` |

The second B fragment adds 16 to the physical **weight row** because B is
transposed. The second C fragment adds 16 to the **output column**. Mixing those
two offsets is a silent transpose error.

CPU enumeration proved the fragment covers all 256 cells of a 16x16 matrix
exactly once, and the four output quadrants cover all 2,048 cells of BM32xBN64
exactly once. This validates the pinned source formula, not an opaque compiler
intrinsic's layout for every descriptor variant.

The SDK says cooperative layouts depend on the operation, types, and execution
scope. It offers `get_multidimensional_index` and `is_valid_element`; these are
preferable to an assumed lane formula for a new descriptor. Its public comments
do not explicitly specify whether input coordinates are physical before
transpose; the input accessor forwards the operand and full descriptor to an
opaque intrinsic. The table above is established by the pinned MLX packing
assumption. A private identity/coefficient-layout GPU probe must establish the
input axes for both strict and relaxed variants. Loading physical tensor slices
through cooperative `load` lets the SDK own layout/transpose, but B still needs
a reconstructed physical operand or staging route in that case.

The local SDK sources used are:

```
/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/MPPTensorOpsMatMul2d.h
/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/__impl/MPPTensorOpsMatMul2dImpl.h
```

## Q4 coefficient and accumulation contract

Native original source bytes at audit time:

| File | SHA-256 |
|---|---|
| `runtime/metal/kernels/common/flash_affine_mpp_common.h` | `5904d5e532130495fe1089c919e66271b77c45e372db8f76ee94e5d8c21a4f84` |
| `runtime/metal/kernels/common/flash_moe_direct_a_common.h` | `86b7b1efc9a70072337d09fc8bebea1890fb0d38493d0045e70b1fce168af772` |

The precise native operand is:

```
f32_coefficient = float(unsigned_Q4_code) * float(source_BF16_scale)
                  + float(source_BF16_bias)
bf16_operand = bfloat(f32_coefficient)
```

Keep the native helper and compiler math scope. Rounding the product to BF16
before adding bias is a different contract; replacing unsigned original codes
with centered codes is also a different contract. The installed quantized NAX
loader promotes BF16 parameters to F32, reconstructs, and converts once to the
operand type. Its high nibble uses `(scale/16)*(byte&0xf0)`; the native normalized
code helper avoids an unnecessary alternative reconstruction expression.

Each K64 group contains four ascending K16 steps. Gate/up share the same loaded
A registers. Their F32 accumulators persist across the entire K dimension;
there is no BF16 accumulator conversion at K16/K64 boundaries. At the epilogue,
preserve the native once-rounded gate/up, compiled BF16 sigmoid, BF16 SiLU, and
BF16 final multiply. Down converts its final F32 accumulator once, then scatters
using the unchanged route map. Combine remains unchanged.

Pure independent register operations have no shared-memory producer/consumer
dependency, so they need no threadgroup staging barriers. This does not imply
cross-die affinity. It trades staging writes/barriers for duplicated B reads and
reconstruction across the two M16 quadrants: two copies of each coefficient for
one BM32 tile. With the pinned fragment packing, each four-code run is aligned
and could be loaded from one U16, but the first safe experiment should establish
native cooperative layout before introducing that packing shortcut.

## Qualification scopes and guards

Report these independently; one successful scope does not establish another:

1. **Source operand identity.** Check original unsigned Q4 code, signed BF16
   scale/bias, reconstructed F32 bits, and once-rounded BF16 coefficient bits
   against the native helper. Cover signed parameters, cancellation, rounding
   ties, nonfinite traps, both halves of B, and first/final G64 groups.
2. **Strict native stage identity.** Compare all live gate/up/activation/down/
   combined BF16 cells against native strict M32K64, plus independent bucket/job
   and route ownership checks. A strict flag and ascending K visits do not
   guarantee byte equality after intrinsic reduction grouping changes.
3. **Relaxed arithmetic.** Report complete output differences and independent
   numerical checks for `relaxed_precision=true`. Same coefficients do not
   make relaxed intrinsic accumulation exact.
4. **Original-F32-source dot windows.** Use independent double accumulation of
   original source-affine coefficients, the native F32 reconstruction, and
   once-rounded BF16 operands. Record the input-weighted coefficient rounding
   loss separately from dot arithmetic error. The existing Q27 source oracle
   demonstrates this split in `referenceCell` and `Errors::add`. Report maximum
   absolute error, own-operand BF16 ULP, source interval violations, and sign/
   residual failures for cancellation-dominated near-zero cells. Relative L2
   alone cannot qualify them.

Preserve active-job/expert/offset bounds, byte-stride alignment and overflow
validation, reserved-field rejection, sticky finite diagnostics, exact live
row masks, route-map bounds, and allocation guards. Direct-A's globally
initialized +63 rows keep physical full-tile reads in bounds; dummy rows may
belong to later experts and must never be stored for the current expert.
Retain immutable source/store verification and mapped-resource replay lifetime
checks. Keep the first experiment on original Q4 misses so saved INT8 hit
coefficients and cache coverage remain the control.

No GPU numerical result, speedup, or 4K-token/s service claim was produced by
this audit.

## Failed v1 audit and isolated v2 correction

The first strict MISS-only GPU screen stopped with a sampled raw-linear NaN.
The failed build is preserved in `build/prefill4k-q4nax-failed-v1`; its report
and invocation remain unchanged. CPU inspection with `xcrun metal-nm` found
that the fast and audit AIR objects each defined identical weak symbols for
the four instantiated `q4nax_gate` and `q4nax_down` helpers. Their bodies differ
because only the audit versions write the raw F32 projection buffers. The
kernel entry points had distinct names, but the helpers did not. Linking the
fast object first can therefore coalesce an audit call onto the fast helper
and leave its NaN sentinels untouched. Static evidence establishes the symbol
collision; a fresh GPU run is required to confirm it explains the failure.

The new `build/prefill4k-q4nax-v2` renames every audit `q4nax_` helper as well
as its kernel entries to `q4naxaudit_`. Kernel arithmetic and buffers remain
unchanged. The oracle writes a separate diagnostic file before certification,
retaining paired GPU/wall timings, full-chain differences and sticky state.
If a raw-linear sample is nonfinite, it additionally records its plane,
expert, packed row, output column, job range, first80 raw values and first64
input BF16 words before throwing. The runner accepts an explicit `--build`
and records these source hashes and the separate diagnostic path.
