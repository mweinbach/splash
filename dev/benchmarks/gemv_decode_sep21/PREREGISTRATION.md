# Vector I8 decode numerical alternatives: frozen before GPU measurement

This experiment changes only reduction order and GPU work assignment. Source
BF16 activations, symmetric signed-I8 expert coefficients, persisted F32 row
scales, canonical original I64 route IDs, BF16 projection rounding and compiled
BF16 SwiGLU stages remain fixed. No activation quantization or new coefficient
cache is admitted.

Existing scalar direct-I8 QMV C1/C2 are prior work, not new variants. The new
variants load adjacent `char4` coefficient and `ushort4` BF16 source vectors,
maintain four independent F32 partial accumulators per lane, serially sum those
four partials, then perform explicit XOR butterfly reduction. SIMD32 assigns
four outputs per 128-thread CTA. SIMD16 assigns eight outputs; both halves reuse
the same BF16 vector through SIMD shuffles while keeping reductions separate.

For normal finite products, BF16 × symmetric I8 contains at most15 significant
bits and is exact in F32. Each contribution experiences no more than
`ceil(K/(4*Lanes)) + 3 + log2(Lanes)` additions. The sampled F64 certificate is
product-rounding error plus this accumulation gamma bound against the sum of
absolute products, plus compensated-F64 uncertainty. Late F32 row scaling
includes its own rounding term. Source subnormal/product-underflow/overflow and
nonfinite cases remain exceptional and never qualify from the finite bound.
Scalar GPU samples use a separate sequential-K certificate. Absolute bounds,
sensitive-dot BF16/sign/zero checks and exceptional classifications remain
reported separately from aggregate quality; no broad norm erases a failed
sensitive dot.

Frozen stage thresholds versus the original gathered SG4 producer are relative
L2 error at most `1e-4` and cosine similarity at least `0.999999`. They apply to
both global arrays and each canonical route for BF16 gate/up/down projections,
compiled activated values, and full-chain down outputs. Zero-reference routes
require exact zero norm; all compared values must be finite. No threshold will
be changed after measurements. Raw/scaled F32 equality is reported but not
assumed; these remain numerical alternatives requiring full-model quality and
state/rollback/speculation qualification before integration.

Timed controls are old bucketed MPP and original gathered SG4, with identical
original fixtures and one admitted readonly layer shared by every variant.
Each variant must accumulate at least150ms of GPU warm work. Timing order is
balanced; timed samples contain shipping chains only and no CPU writes to
operands/diagnostics between samples. Untimed scalar/dense taps, references,
payload verification and poison/invalid/duplicate tests do not enter timings.

Invalid ID/rank gates poison every output with canonical NaN and sticky bit1
while still scanning their hidden source for bit4. Invalid down routes skip
arbitrary intermediate input, poison every output and preserve sticky bits5.
Duplicate valid routes remain separately computed with sticky bit1. Nonfinite
source operands become positive zero and preserve sticky bit4. Every fixture,
ID/rank test buffer, output and writable scratch allocation has canaries;
the certified readonly weight layer uses immutable hashes. Poisoned-output
replay must reproduce every live element without touching source weights.

Only Root invokes GPU mode. This subtree performs sealed CPU-only builds and
reads no model payloads.
