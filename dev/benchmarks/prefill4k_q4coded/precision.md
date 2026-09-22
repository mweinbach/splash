# Original Q4/G64 unsigned-code expert precision audit

This is a bounded source/numerical audit. It submits no GPU work, loads no
model, scans no model/store payload, and changes no production source or
profile. Preserving original Q4 codes and BF16 source parameters does not
preserve the current blocked-prefill arithmetic.

## Source and arithmetic contracts

`runtime/flash/FlashWeights.hpp:26-42` describes raw MLX affine tensors with
byte strides and no implicit StorageN padding. The loader at
`FlashWeights.mm:546-578` requires U32 packed weights and separate BF16 scale
and bias tensors. Expert geometry at `:645-647` is Q4/G64, E512, gate/up
N640/K2560 and down N2560/K640.

For valid expert `e`, output channel `n`, input channel `k`, and G64 group `g`:

```
packed offset = e * weight_expert_stride + n * weight_row_stride + k / 2
parameter offset = e * parameter_expert_stride + n * parameter_row_stride + 2*g
q[k] = (little_endian_U32 >> (4*j)) & 15       # eight aligned nibbles
s[g], b[g] = exact F32 promotions of source BF16 words
```

See `flash_moe_q4x8.metal:59-79`. Codes are unsigned `[0,15]`; scales and
biases retain their signs. This raw affine layout is distinct from the
packaged interface in `common/moe_expert_slab.h`.

The blocked/Direct-A prefill producer reconstructs each coefficient in F32,
rounds it once to BF16, and feeds BF16 A and BF16 coefficients to an
F32-accumulating K64 MPP operation (`flash_moe_blocked.metal:54-70`,
`common/flash_affine_mpp_common.h:34-40`):

```
c[k] = RN32(RN32(q[k]*s[g]) + b[g])
w[k] = RN_BF16(c[k])
Y_B = sum_k exact_real(x[k] * w[k])
```

For Q4 codes and BF16 scale, `q*s` has at most 12 significant bits, so it is
exactly representable in F32 whenever finite; even the smallest nonzero
BF16 value times a Q4 integer is representable. Under a no-overflow guard,
contracting that particular reconstruction multiply/add does not change its
result. Do not generalize this fact to the candidate epilog.

The proposed candidate instead computes, in ascending G64 order:

```
D[g] = sum_{k in g} exact_real(x[k]*q[k])
T[g] = sum_{k in g} exact_real(x[k])
C[g] = s[g]*D[g] + b[g]*T[g]
Y_A = sum_g C[g]

Dhat[g] = F32-accumulating BF16 x U8 MPP group dot
That[g] = F32 group sum of unchanged BF16 activations
phat[g] = RN32(s[g]*Dhat[g])
vhat[g] = RN32(b[g]*That[g])
Chat[g] = RN32(phat[g] + vhat[g])
Yhat = ascending_group_RN32_sum(Chat[g])
P_A = RN_BF16(Yhat)
```

Use safe math, `contract(off)` and `reassociate(off)` for this declared
candidate operation sequence. Centering codes, reconstructing an unsigned
dot from a centered dot, fitting scales, or BF16 group accumulation are
different candidates.

The canonical affine vector path is a second reference: it retains each
coefficient in F32, accumulates `float(A[k])*coefficient`, and rounds the
completed projection to BF16 (`flash_affine.metal:88-97` and `:378-395`). The
candidate distributes/reorders its affine operations as well. Label which
reference a comparison uses. `FlashInt8Head.hpp:19-20` already labels grouped
F32 reconstruction a numerical alternative.

## Absolute projection envelope

Let `u=2^-24`, `eta=2^-150`, `gamma(m)=m*u/(1-m*u)`, and
`delta(m)=m*eta/(1-m*u)`. The conservative F32 rounding rule is

```
abs(RN32(z)-z) <= u*abs(z) + eta
```

The formulas below are conditional on finite intermediates, correctly
rounded F32 additions/multiplications (or FMA with no larger error), and
gradual underflow. An API requesting an F32 accumulator does not establish
every detail of MPP's hardware accumulation or denormal handling. The actual
GPU oracle must check that behavior. A GPU output outside this envelope
fails; compilation and a CPU proof do not qualify MPP precision.

For each input row, expert, output channel and group, compute positive norms

```
Lq[g] = sum_{k in g} abs(x[k]*q[k])
Lx[g] = sum_{k in g} abs(x[k])
Cabs[g] = abs(s[g])*Lq[g] + abs(b[g])*Lx[g]

eD[g] = gamma(64)*Lq[g] + delta(64)
eT[g] = gamma(64)*Lx[g] + delta(64)
```

BF16 times original Q4 integer is an exact F32 product whenever finite.
`gamma(64)` conservatively covers an ordinary sequential or tree reduction
of 64 terms; it does not assume an undocumented SIMD reduction tree. The
additive underflow term is conservative even though this specific input
lattice often makes it unnecessary.

Each separate scale/bias product has an absolute error bound against its
exact-real target, including its input dot/sum error:

```
ep[g] = (1+u)*abs(s[g])*eD[g] + u*abs(s[g])*Lq[g] + eta
ev[g] = (1+u)*abs(b[g])*eT[g] + u*abs(b[g])*Lx[g] + eta
eC[g] = (1+u)*(ep[g]+ev[g]) + u*Cabs[g] + eta
```

For `G=K/64` groups (40 for gate/up, 10 for down), ascending accumulation
then obeys

```
E_A = (1+gamma(G))*sum_g eC[g]
      + gamma(G)*sum_g Cabs[g] + delta(G)
abs(Yhat-Y_A) <= E_A
```

This includes the group-dot reduction, group sum, two F32 products, their
addition, and the final group accumulation. Bounds proportional only to
`abs(Y_A)` miss epilog cancellation and are invalid.

The changed BF16 coefficient boundary contributes an independent term:

```
E_coeff = sum_k abs(x[k]) * abs(w[k] - (q[k]*s[g]+b[g]))
abs(Yhat-Y_B) <= E_A + E_coeff
```

Use the actual reconstructed BF16 `w[k]` for a tight certificate. An analytic
fallback, with `uB=2^-8` and `etaB=2^-134`, is

```
R[k] = abs(q[k]*s[g]) + abs(b[g])
ec32[k] = u*R[k] + eta                    # q*s exact when finite
ecB[k] = ec32[k] + uB*(R[k]+ec32[k]) + etaB
E_coeff <= sum_k abs(x[k])*ecB[k]
```

The analytic bound can be very conservative for large opposing affine
parameters. It is an absolute bound, not a universal relative-error or
model-quality bound.

To compare against a captured F32 blocked-MPP accumulator `Z_B`, include
its own accumulation error. A conservative model covering separate product
and addition as well as FMA is

```
Lw = sum_k abs(x[k]*w[k])
E_B_gpu = gamma(2*K)*Lw + delta(2*K)
abs(Yhat-Z_B) <= E_coeff + E_A + E_B_gpu
```

For the final BF16 projections `P_A=RN_BF16(Yhat)` and
`P_B=RN_BF16(Z_B)`, add both final rounding errors:

```
abs(P_A-P_B) <= E_coeff + E_A + E_B_gpu + rA + rB
rA = abs(P_A-Yhat)                       # exact in FP64 if captured
rB = abs(P_B-Z_B)
```

If only finite BF16 projection outputs are captured, a safe observable
rounding bound is `(uB*abs(P)+etaB)/(1-uB)` for each projection. This follows
from `abs(P-z)<=uB*abs(z)+etaB`; it also covers a zero/subnormal output.

Gate/up projections are separately rounded to BF16 before compiled BF16
SwiGLU; down uses the candidate activation and rounds before route combine.
The down projection bound is valid for a fixed shared input activation. A
linear projection envelope does not bound a comparison whose down inputs
already differ, the fast/BF16 sigmoid, routing, or recurrent/attention state.
Capture/check the linear projections when claiming this certificate, then
retain full activation/down/combine absolute errors and whole-model quality
tests as separate evidence.

## FP64 implementation must round bounds outward

BF16 inputs/parameters and U8 codes promote exactly to FP64. Their pairwise
products need at most 16 significant bits and have exponents within FP64's
normal range. Finite K2560 norms/products remain far below FP64 overflow.
Their sums can still round in FP64; using ordinary FP64 arithmetic alone
does not make an envelope rigorous.

For each nonnegative addition, multiplication, or division in a bound,
round upward: compute the FP64 result and apply `nextafter(result,+inf)`.
For `gamma`/`delta`, round `m*u` upward and `1-m*u` downward before upward
division. Treat zero products/additions exactly when both factors/terms are
known zero if desired; adding one upward ULP remains safe. Nonfinite bound
arithmetic is a rejected diagnostic, not permission for every output to pass.

A tight FP64 coefficient-error certificate can avoid an exact-rational
library. Let `v=q*s`, which is exact in FP64, `a=RN64(v+b)`, and
`d=RN64(w-a)`. With `u64=2^-53`, an outward upper bound is

```
coeff_error_up = abs(d)
                 + u64*(abs(w)+abs(a))
                 + u64*(abs(v)+abs(b))
```

The two added terms cover the FP64 subtraction and affine sum; no FP64
underflow term is needed for these BF16-derived coefficients. Use upward
operations for every positive term. Multiply by `abs(x)` upward and sum
upward to construct `E_coeff`. Exact-rational dyadic checks or an
error-free floating expansion may instead give a tighter result.

If the oracle's reference is a computed FP64 dot
`Y64_B=sum_k FP64(x[k]*w[k])`, its products are exact but its additions may
round. Include `E64=gamma64(K)*Lw`, with outward arithmetic and
`gamma64(m)=m*2^-53/(1-m*2^-53)`, in the threshold. For example:

```
abs(Yhat-Y64_B) <= E_A + E_coeff + E64
abs(P_A-Y64_B) <= E_A + E_coeff + rA + E64
```

The FP64 diagnostic's final subtraction/absolute value can itself round.
Compare its outward upper error value against the outward bound, or add its
FP64 rounding term explicitly. Do not silently use a tolerance of zero
against an unqualified FP64 sum. Require cardinalities and maximum
error-to-envelope ratios separately for every projection/pattern/channel,
including exact-zero reference channels.

## Required cancellation and overflow fixtures

All unspecified x/q values below are positive zero; every fixture is one
G64 group embedded in both legal K geometries. The nonzero inputs and source
parameters are exactly BF16.

1. **Removed BF16 coefficient rounding.** `x=[1,-1]`, `q=[0,1]`,
   `s=2^-9`, `b=1`. Blocked coefficients are both BF16 one, so `Y_B=0`.
   Unsigned affine gives exactly `Y_A=Yhat=-2^-9`. The tight coefficient
   term equals `2^-9`; this expected difference must not be called a
   bit-exact success or a kernel bug.
2. **F32 group-sum cancellation with no coefficient-boundary error.**
   `x=[1,2^-24]`, `q=[1,0]`, `s=1`, `b=-1`. Both source coefficients
   are already exact BF16 `(0,-1)`, so `Y_B=Y_A=-2^-24` and
   `E_coeff=0`. `RN32(1+2^-24)=1` ties to even; the unsigned epilog
   instead returns zero. The F32 sum/epilog terms must cover this error.
3. **Unsigned bias preservation.** `x=[1]`, all q zero, `s=1`,
   `b=2^-21`. Reference and unsigned epilog retain `2^-21` exactly. A
   centered Q4 formulation loses this bias when forming `b+8*s` in F32.
4. **Finite inputs with an overflowing sum.** Every x is `2^123`, all
   q zero, `s=1`, `b=0`. The reference coefficient/dot is zero, but the
   F32 group sum overflows and `0*inf` is NaN. Require sticky numerical
   rejection; a finite-input guard alone is insufficient.
5. **Finite inputs with an overflowing code dot.** Every x is `2^119`,
   every q is 15, `s=0`, `b=0`. The reference dot is zero; the code dot
   overflows before scaling by zero, while the group sum itself is finite.
   Require rejection.

Also exercise both signed-zero words, negative scale/bias, alternating-sign
and cross-group cancellation, BF16 subnormal inputs/coefficients, BF16
projection midpoint values, maximum expert/selection IDs, and partial bucket
tails. Move nonzero terms to different SIMD lanes to expose reduction-order
assumptions. Denormal tests either establish the declared arithmetic behavior
on the actual GPU or require a clearly recorded unsupported/fallback route.

## Typed extraction, source guard, and lifetime checklist

- Hard-gate U32 packed weights and BF16 scale/bias; check rank/shape,
  logical bytes, real buffer lengths, checked offsets, readable mappings,
  and source identities. `FlashMoEBlocked.cpp:114` and
  `FlashWeights.mm:487` contain current extent/alignment gates.
- CPU extraction uses `memcpy` into U32/uint16 words, nibble shifts, and
  `bit_cast<float>(uint32_t(bf16_bits)<<16)`. Do not alias BF16 bytes as
  F32. If supporting F32 parameters later, independently carry the actual
  scale and bias element widths/strides; the current shared stride cannot
  safely represent mixed parameter dtypes. `FlashDType` has no F16.
- Preserve exact A source words and U8 q `[0,15]`; initialize positive-zero
  padding. Retain per-replay Direct-A sanitization and its 63 initialized
  guard rows (`flash_moe_direct_a.metal:49,93`), stable independent jobs,
  job/rank bounds, and masked stores. No stale activation can enter an MPP
  tail read.
- Require signed finite source s/b and finite A. Require finite actual-code
  reconstructed F32/BF16 reference coefficients. A conservative source
  guard can validate finite `RN32(15*s)` and finite
  `RN_BF16(RN32(RN32(15*s)+b))`, in addition to s/b; affine monotonicity
  then covers every Q4 code. It may reject unused extreme codes and should
  fall back rather than claim the whole source is invalid.
- Check group dot, group sum, each correction product, corrected group,
  running total, final F32/BF16 projection, and SwiGLU outputs. Sticky
  finite diagnostics use integer exponent masks `0x7f800000` for F32 and
  `0x7f80` for BF16 (`flash_affine_mpp_common.h:12-22`). Masking an invalid
  value to zero does not turn that replay into a successful numerical test.
- `FlashInt8Head` is a structural reference only: its host gate is Q8/G64,
  its byte view does not expand Q4 nibbles, and its shader hardcodes BF16
  parameters/two-byte offsets. Its current per-group epilog lacks explicit
  scale/bias/product/corrected-result guards (`flash_int8_head.metal:131`);
  copy the structure, not that omission.
- Preserve source hashes and independent full output checks, canaries,
  admission failures, mapping-owner destruction/replay, and diagnostics.
  Full-model quality, service lifecycle, and matched performance remain
  required after a primitive passes. This audit is not evidence of any of
  those GPU/service outcomes.

## Bounded CPU validation performed

Two exact-dyadic cancellation fixtures above and 48 random finite cases at
K64/K128/K640/K2560 passed the outward projection envelope under simulated
sequential IEEE F32 arithmetic. Exact rational dyadics independently computed
the blocked reference and coefficient errors. The largest random-case
error/envelope ratio was 0.864663. The boundary fixture produced
`reference=0`, `candidate=-0.001953125`; the group-sum fixture produced
`reference=-5.960464477539063e-8`, `candidate=0`. The two overflow fixtures
also confirmed the intended overflowing/finite group-sum distinction.
This CPU arithmetic simulation is not an MPP emulator or GPU precision result.
Separately, 1,007 finite randomly selected BF16 scale/bias/Q4-code triples
passed the tight outward FP64 coefficient certificate against exact rational
dyadics; 17 nonfinite source/reference triples were rejected by guards.
