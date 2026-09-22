# Floating-point envelope for the v1b vector GEMV screen

The initial v1a reference omitted possible underflow in intermediate additions.
Its observed-error gate rejected a concrete counterexample, but its generic
gamma formula did not cover all arithmetic Metal permits. This v1b correction
was prepared before GPU measurements. Kernels, BF16 stage tolerances, timing
controls and independently reported strict failures remain unchanged.

[Apple's 2026-06-04 Metal specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
permits subnormal operands and results to flush (§§8.1,8.5), and supports
either ties-to-even or toward-zero arithmetic rounding (§8.2). Basic F32
addition and multiplication are correctly rounded (§8.4). BF16 conversion
uses ties-to-even (§8.6). The following envelope is our conservative derivation
from those guarantees, rather than an assumption that safe math preserves
subnormals or forces native F32 ties-to-even arithmetic.

Let `lambda=2^-126`, the smallest normal F32 value, and `u=2^-23`, which covers
both arithmetic rounding modes. An addition can flush two operands, changing
their sum by less than `2*lambda`, then round it and flush a subnormal result.
It admits the model

```text
fl(a+b) = (a+b)*(1+delta) + eta
abs(delta) <= u
abs(eta) <= (3+2*u)*lambda <= 4*lambda
```

Each vector output contains `K` additions into the four per-lane partial
chains, `3*L` component-collapse additions and `L-1` additions in its final
XOR dependency tree. Its total operation count is at most `N=K+4*L-1`.
Every contribution crosses at most `D=ceil(K/(4*L))+3+log2(L)` additions.
These are logical dependencies of one output; duplicate XOR computations
for other lanes do not add operations to that output's tree.

Using `gamma(D,u)=D*u/(1-D*u)`, later rounding amplifies each local absolute
flush error by at most `1+gamma(D,u)`. The dot-error allowance therefore adds

```text
dotFlushAllowance = 4*N*lambda*(1+gamma(D,u))
dotBound = productErrorUpper
         + gamma(D,u)*(sumAbsProductsUpper+productErrorUpper)
         + dotFlushAllowance + compensatedF64ReferenceUncertainty
```

Normal BF16×symmetric-I8 products need at most15 significant bits, so those
accepted products are exactly representable in F32. Source/product subnormals,
nonfinite values and possible product/intermediate overflow remain exceptional.
Before applying the relative-error theorem, the upward envelope
`sumAbsProductsUpper+productErrorUpper+dotBound` must stay within finite F32
range. This dominates every subset/partial/component/XOR node. Finite captures
alone are insufficient because toward-zero overflow may saturate to finite
values. The existing product-overflow report field also covers this conservative
intermediate-overflow risk in v1b.

For positive normal stored scale `s`, the raw-dot operand of the final
multiplication may itself flush. Let `B=dotBound+lambda`. Late scaling uses

```text
scaledBound = s*B + gamma(1,u)*s*(abs(referenceDot)+B)
            + lambda + F64ReferenceScaleProductUncertainty
```

The first extra `lambda` covers the dot operand; the second covers the scaled
result. The upper scaled envelope `s*(abs(referenceDot)+B)` must also remain
within finite F32 range. Subnormal scale operands stay exceptional, including
when their true product would be normal. The scalar GPU sample uses the same envelope with
`D=N=K`. Nonnegative F64 bound operations advance upward with `nextafter`;
estimated positive sums are inflated by their F64 uncertainty before use.

The fixed CPU counterexample has K640, all codes1, scale1, and source entries
`0x0081,0x8080,0x0080` followed by zero. All nonzero source/products and the
final reference dot are normal, but component collapse can form `2^-133`.
Permitted flushing loses that value. The corrected absolute envelope covers
the loss while preserving the failed strict-sensitive BF16 result separately.
Additional CPU cases cover source subnormals hidden by zero/large coefficients,
a subnormal scale whose mathematical product is normal, and normal products
with a finite final dot but potentially overflowing intermediate sums.

Passing sampled absolute bounds and the registered dense global/per-route
relativeL2/cosine gates qualifies this numerical alternative primitive only.
It does not convert a strict-sensitive failure into a pass, establish exhaustive
per-dot correctness, or qualify model quality and speculative state behavior.
