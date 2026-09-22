# New-policy proposal: T32 chunk / V32 tile native fallback

No kernel is implemented here. The performance-rejected v6, its thresholds,
proofs, and actual-capture evidence remain unchanged. The HC whole-worker draft
has been stopped and cannot be frozen for performance composition.

Root's actual prepared layer-0 capture observed 39/48 full-row replayed heads
(15 range, 14 cancellation, 10 norm range), making v6 about 3.37 ms versus
native about 1.90 ms. Exact native replay quality passed. This evidence motivates
changing replay **granularity**, not loosening the numerical checks.

## One workgroup can own the complete local decision

Keep four SG8 workgroups per head, each owning 32 value coordinates. Those
coordinates have independent recurrent rows once Q/K, alpha, and beta are
prepared. For each 32-token chunk, compute the existing range/norm/cancellation
predicates and 0.095 L2 threshold into one local threadgroup reason mask.
Range is shared by all four tiles of the head/chunk; norm/cancellation/nonfinite
remains local to the value tile that generated it. Unsafe tiles replay only
that 32-token chunk. This is a distinct policy; it does not preserve v6's
whole-head replay decisions or whole-prefix bit identity.

The persistent recurrent state must remain untouched until the local decision
is final. Retain the candidate update cooperative tensor in registers, replace
each valid element with the exact existing
`endPrefix * incomingState + update` result, and perform all nonfinite,
subnormal/saturation and local reason checks before any persistent write.
A uniform threadgroup barrier/decision follows. Safe tiles commit the retained
state. Unsafe tiles run native recurrence from the unchanged persistent incoming
F32 tile state. No whole-head snapshot, restore dispatch, or end-of-R replay is
needed. A nonfinite result cannot be detected only after committing half a tile.

Speculative BF16 output writes must either be deferred too or overwritten for
every owned chunk/coordinate by native replay before the next chunk and before
command completion. Audit history/delta/pre-output slots need the same rule.
Other value tiles' rows and all recurrent padding remain untouched. There is
no cross-workgroup spin barrier; every mutable row has one workgroup owner.

## Captured native arithmetic fits the SG8 workgroup

An unsafe V32 tile can process four V8 waves with the captured native recurrence,
using two Time16 token blocks in each wave. Each value row retains one SIMD32
group, four consecutive keys per lane, identical scalar expression order,
SIMD reduction, BF16 beta, F32 state, and BF16 output boundary. Calling the
captured V16/512-thread helper directly from 256 threads would be incorrect.
The V8/Time16 template association is independently byte-qualified per row.

Native scratch is Q4352 + K4352 + V512 + decay64 + beta32 = 9312 bytes. Existing
WY scratch is 8516 bytes. A shared, aligned phase scratch arena of their maximum
(plus the local mask/broadcast) fits below 32 KiB. Distinct allocations or native
Time32 staging would waste memory or exceed the bound if an incoming16KiB
snapshot were also kept. With deferred persistent writes, that snapshot is
unnecessary in timed kernels; an audit-only snapshot may help prove the branch.
Cooperative tensor liveness across the decision barrier must compile and be
resource-qualified rather than assumed.

## Reject preparation before expensive transforms

Prepare actual raw alpha/beta words and the unchanged per-chunk range predicate
first. After a uniform decision/barrier, range-ineligible chunks can skip Gram,
inverse, W/U, QK, and norm-cache products. Native replay reads raw mixed/decay/beta
data. Store an explicit per-chunk flag and defined unused-transform coverage;
report unsupported/unrepresentable coefficient diagnostics as unused/unavailable
without converting their old raw failures into passes. Safe chunks still use
the exact existing transform arithmetic and cached norm order.

## Required new proof before timing or model composition

1. CPU source/ABI/overflow/alias bounds, uniform barriers, shared scratch layout,
   untouched-state-until-decision, and SDK compile/resource witness.
2. For each unsafe tile/chunk, exact F32 state, BF16 output, and audit tapes
   versus captured native chunk recurrence from the **same incoming F32 seed**.
   Cover partial chunks, real zeros/post-zero underflow, norm FTZ/saturation,
   cancellation, mixed safe/unsafe tiles, padded strides, and guards.
3. Safe candidate output/state arithmetic and reason predicates versus the
   existing local v6 math. Report changed replay scope explicitly. Keep all
   original raw F64 gates and unavailable conditioning cases separate.
4. Continued hybrid sequences, actual captured cold/carried input, then a
   balanced >=150 ms warm timing with real chunk/tile native-row counts.
5. Only after component improvement: new static policy identity, governor
   accounting, full-model logits/state/MTP behavior and frozen semantic suite.

Native chunks start from a hybrid incoming F32 state that can contain rounding
from earlier WY chunks. Exact local native equivalence does not establish a
whole-history F64 certificate or equality to a fully native prefix. That limit
must remain explicit. Speed and fallback frequency are unknown until Root's
serialized actual-input measurement.
