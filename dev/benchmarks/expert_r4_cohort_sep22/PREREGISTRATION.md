# FIRST N16 R4 expert primitive: frozen before GPU measurements

Only N16, physical rows4, selections10, experts512, SIMD32,128 threads and ten
persistent cohort slots are admitted. No whole-worker integration, other new
tiles, activation quantization, coefficient cache or sidecar is present.
Original BF16 hidden/intermediate values, signed-I8 stored codes, positive F32
row scales, I64 IDs, rank lookup, late F32 scaling and compiled BF16 SwiGLU
stages remain fixed.

The new association is exactly `(p0+p1)+(p2+p3)` for adjacent char4 products,
then one F32 lane accumulator, followed by descending XOR masks16,8,4,2,1.
Contraction/reassociation are disabled. Each original route has independent
accumulators even when one coefficient vector serves up to four routes.
The new dependency depths are27 for K2560 and12 for K640; logical dependency
addition counts are2591 and671, respectively. Normal in-range BF16×symmetric-I8
products need at most15 significant bits and are exact in F32. Arithmetic FTZ
and overflow still require the complete error envelope.

The independent F64 reference retains u32=2^-23 for permitted RN/RTZ modes,
4*minNormalF32 per dependent addition for operand/result FTZ, upward positive
bound arithmetic, compensated-F64 uncertainty, raw-dot operand and final-scale
FTZ allowances, explicit F64 scale-product uncertainty, and conservative
intermediate/scaled nonoverflow envelopes. Source subnormals, subnormal scales,
nonfinite/overflow risks are exceptional. Per-dot sign/zero/cancellation and
strict-sensitive results remain separately failing statuses when appropriate.
Existing opaque MPP references and controls remain visible.

BF16 projected gate/up/down, compiled activation and own full-chain down must
pass **both global and each original canonical route** relativeL2≤1e-4 and
cosine≥0.999999 against current direct gathered SG4. Zero-reference routes
require exact zero norm, with no norm floor. No threshold may be changed after
measurements. Dense probes must reproduce shipping compiled SwiGLU and own
activation down exactly; scalar GPU samples independently change reduction
order. Raw/scaled F32 differences remain numerical alternatives, not exact
producer identity or model-quality qualification.

The compact planner uses exactly6,796 logical existing-metadata bytes in an
isolated guarded oracle allocation. It initializes all fields, validates IDs
and ranks, preserves stable expert/flattened-route order and every duplicate
ownership, and splits buckets into at most4-row cohorts. It scans all four
hidden rows for sticky4 even when all IDs/ranks are invalid. Inputs are never
mutated; nonfinite producer operands become positive zero in registers.
Invalid gates fully poison canonical output with NaN/bit1, with numeric bit4
only from actual evidence. Invalid down routes poison NaN/bits5 and never read
their arbitrary intermediate. All metadata/operand writable views remain
pairwise disjoint and disjoint from immutable layer/rank allocations.

One graph orders plan→gate→down through Metal serial compute dispatches.
Shipping gate/down require exactly400/1,600 threadgroups through the actual
`threadgroups_per_grid` built-in, plus the matching metadata epoch/ready stage.
A partial gate cannot publish readiness. Shipping uses one stage-ready writer,
not contended global completion counters. Untimed audit entries share the
same arithmetic, add a device threadgroup finish barrier and one counter per
CTA, and require all400 gate completions before down can read A. Audit/shipping
BF16 output equality and counters400/1,600 are mandatory. Missing/reordered
plan/gate, partial grid, wrong epoch and corrupt live map tests fail closed.
The unmodified backend source is sealed and freshly CPU-compiled against the
same frozen headers, establishing its serial encoder contract for this binary.

Root alone may invoke GPU mode. The first matrix includes U10/20/30/40 overlap,
old failing mixed R4 fixture, slot permutation and rank permutation. Current
direct gather and original native bucket MPP are matched controls, sharing
one admitted readonly layer. All semantic, metadata, audit, scalar/F64, BF16,
canary, immutable-source, stale-plan and malformed ownership checks precede
timings. Timing is refused on any new-primitive gate failure. Every variant
accumulates at least150ms GPU warm work, then balanced three-way shipping-only
samples (default18); no CPU data/diagnostic touches occur inside those loops.
Audit/projection/scalar/reference/hash work remains untimed.

A passing synthetic primitive establishes neither real normalized activation
fidelity nor model quality. Those tests, frozen22 state/rollback semantics,
acceptance and whole-model performance remain Root's later promotion gates.
