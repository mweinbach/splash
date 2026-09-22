# CPU-only eligibility and native-replay audit

2026-09-21. This is an independent mathematical/design review, not a GPU
qualification result. At initial review time v3 shader/encoder sources were not
present. The statements below are proof obligations for the new implementation;
they must be checked against its sealed source and runtime comparisons before a
branch-equivalence result is claimed. No Metal device, GPU dispatch, model
payload, public source, or production source was accessed or modified.

An in-progress v3 source read subsequently confirmed sticky per-head reasons,
actual-decay finite[0,1] checks, a zero-reset segment scan, and a tile/chunk L2
Cauchy cancellation selector. At this read the restore/replay encoder path was
still being implemented, so source correspondence and branch equivalence remain
pending. The rounded F32 segment scan needs a conservative normal-boundary
margin for its mathematical underflow claim; the unscaled squared-norm selector
also needs a policy for nonzero terms that underflow to zero. These observations
are not a sealed-source result and should be rechecked after implementation.

The frozen v2 native reference reviewed here is `frozen/canonical.metal`,
SHA-256 `e7f3afe8450a339a16d418d43ea7b625979741f484d171ac675a221a597975dc`.
Its ABI hash is `ed92edcf1d837506ce5079ecec66d60a4db851a7140cd71893c47b71440e54f9`.
This identifies the reference; it does not transfer any prior certificate to
the W/U algorithm or the guarded v3 worker.

## Range eligibility

Eligibility must read the actual F32 decay for the current `(lane,value_head,
row)`; fixture names, stale maximum-state estimates, and presumed source ranges
are not eligibility inputs. Nonfinite raw inputs or out-of-domain decay values
must be explicitly rejected or sent to the native branch according to a stated
domain contract. If the admitted domain is `0 <= alpha <= 1`, verify it from
the actual decays rather than assuming it.

An exact zero decay legitimately erases preceding state. A mathematical product
of nonzero decays that enters the F32 subnormal/zero range does not: rounding the
product before applying it to a large state can erase a representable result.
Track products in normalized mantissa/integer-exponent form, or use an equivalent
range-safe representation. Do not let a rounded F32 zero become the authority
for whether the exact product is zero. The minimum normal F32 magnitude is
`2^-126`; in the `frexp` representation `m*2^e`, `0.5 <= abs(m) < 1`, its exponent
is `e=-125`. Account conservatively for the rounding of a mantissa tracker near
this boundary; simply testing the rounded product is not an exact certificate.

The origin prefix alone is insufficient. After a zero decay, origin prefixes
remain zero while relative products governing later updates can underflow. For
the verified `[0,1]` decay domain, reset a separate nonzero-segment range tracker
to one at each exact zero, and flag when its longest nonzero-segment product
becomes unsafe. With factors of magnitude at most one this conservatively covers
suffix products within that segment. Keep any flag sticky across all chunks.
Without this domain restriction, scan every direct relative-product interval:
an earlier gain can make the origin product look safe while a suffix is unsafe.

A concrete counterexample embeds into one key/value coordinate of K128/V128:

```
alpha = [0, 2^-100, 2^-100]
beta  = [1, 0, 0]
k = q = 1; v[0] = 2^100
native post-update state = [2^100, 1, 2^-100]
origin prefix = [0, 0, 0]
relative end-key factor P(2,0) = 2^-200 -> F32 zero
```

All listed powers are representable source BF16/F32 values. A guard that only
checks the origin prefix misses this loss; a zero-reset segment tracker catches
it. The source must also check finite W/U/end-key/score/prefix values, and any
range assumptions needed to rule out intermediate overflow. A finite prepared
matrix alone does not prove safe state products or history.

## Cancellation selector

For `D=U-W*S0^T`, near cancellation in the outer subtraction is a useful replay
selector. The local risk scale should include projection conditioning:
`abs(U)+sum_k(abs(W_k*S0_k))`, rather than only `abs(U)+abs(projected)`. The latter
can hide cancellation within the projection itself.

A Cauchy upper scale `abs(U)+norm2(W_row)*norm2(S0_row)` is also a conservative
mathematical substitute for the absolute dot-product sum. A per-value-tile,
per-chunk L2 selector can aggregate these scales and D residuals. Compute its
norms with range-safe normalization or flag nonfinite/underflowed intermediates;
unscaled F32 sums of squares can overflow or disappear. Aggregating a tile does
not make the selector a per-element bound or a whole-history certificate.

For F32 use unit roundoff `u=2^-24`, not machine epsilon `2^-23`, and
`gamma_n=n*u/(1-n*u)` with `n*u<1`. An illustrative conservative scalar rounding
surrogate is `gamma_(2K+1)*scale`; at K128 gamma257 is approximately `1.5319e-5`.
Its operation count is not a proven model of MPP's internal implementation.
Compare normalized residual/scale to the registered threshold to avoid overflow
or underflow in `gamma*scale`. Reject nonfinite scale; when residual is zero and
scale is positive, select replay. An epsilon denominator floor can conceal
precisely the tiny residual being screened and must not admit such a case.

This remains an explicit **selector heuristic**, not a universal numerical
certificate. It omits errors in Gram, triangular inversion, W/U, direct decay
products, and the incoming carried state. Current-state magnitude does not make
historical state error zero. A whole-history W/U certificate would have to
propagate all those errors through D, output, and carried-state updates from the
known incoming state. No old direct-solve certificate is inherited here. No
branch-frequency or guard-overhead estimate is established by this CPU review.

## Full-head replay ownership and equivalence

Use sticky atomic flags indexed by `(lane,value_head)`, one flag for all 128
value coordinates and all chunks. A risk found by any value tile selects the
whole head. Keep an immutable snapshot of every input head state before any WY
write: 48*128*128*4 = 3,145,728 bytes per lane, excluding guards/padding. Snapshot
and restore addressing must respect `recurrent_lane_stride_bytes`, including
padding if the accepted ABI permits it.

The required order is snapshot -> initialize flags -> preparation -> WY apply
-> finish all flag-producing dispatches -> restore flagged full heads -> replay
all rows through native -> expose committed outputs/state. Device/encoder
barriers must make flags, snapshot, prepared tensors, and restored state visible
at their consumers. Atomic OR makes flag discovery sticky; it does not replace
the dispatch ordering/barriers. The restore cannot race with any WY state read
or write. A later cancellation flag requires **fullRows** replay from the
pre-WY snapshot, not replay of a suffix from already approximated state.

Replay uses the same raw BF16 q/k/v/beta and F32 decays and the same head mapping
as frozen native: `key_head=head/3`, mixed stride10240, output stride6144, state
layout `[head,value,key]`. For each value coordinate the native reduction tree is:

1. SIMD lane `l` owns key coordinates `4*l+i`, i=0..3, and loads exactly those
   four F32 state values from the restored initial state.
2. In token order, multiply each state value by its actual F32 decay, accumulate
   memory in i=0..3 order, then perform the same `simd_sum`.
3. Form `(float(BF16_value)-memory)*float(BF16_beta)` in the same expression
   order. Do not turn this into a different beta placement, FMA, or scalar
   128-term reduction.
4. Add `k[i]*delta` to each state value in the same order, accumulate the query
   dot in i=0..3 order, perform the same `simd_sum`, then convert to BF16 once.
5. Store the four F32 state values with the same indexing and contraction /
   reassociation / math-mode settings as frozen native.

Canonical Values16 partitioning provides 16 SIMD groups/512 threads per group;
8 groups cover all 128 value coordinates. Other tile sizes require an explicit
proof that lane ownership and every per-value reduction remain identical.
The claim is equivalence to this frozen native implementation for the same
incoming F32 state and source inputs. It is not equivalence to F64 or a
sequential scalar F32 dot.

Replay must overwrite **every** flagged head's fullRows BF16 output and final
F32 state; no speculative WY value may survive. For head-zero audit dispatches,
replay only head zero even if preparation flags the other heads: the existing
unmodified-other-head state and output-poison invariants remain mandatory.
When audited head zero is flagged, replay must also overwrite every row/value
delta, pre-BF16 output, and row/value/key history slot using the actual native
state/delta, not the WY reconstruction. Audit additions may observe native
arithmetic but cannot change its expressions or reduction order. NaN sentinels
and guards establish write coverage only when all expected slots are checked.

Separate speculative/raw WY diagnostics from committed native diagnostics if
replay recovers a nonfinite intermediate; retain the raw failure in reporting.
Invalid raw inputs cannot be hidden by clearing diagnostics. Mixed flagged and
unflagged heads must retain disjoint state/output/audit ownership. Continuation
starts from the actual committed F32 carry; its independent F64 truth starts
from the previous F64 truth, never from rounded GPU carry.

## Evidence and gate reporting required for the new branch

The v3 branch proof needs new source correspondence and runtime comparisons to
frozen native from the identical snapshot. Required comparisons for flagged
heads are byte-identical final F32 state/BF16 output, and native-reduction F32
delta/preoutput/history when the audited head is flagged. Exercise a forced
native branch as well as guard-selected range, postzero range, cancellation,
mixed-head, tail, multi-lane, and continued sequences. Check immutable source
and snapshot hashes, allocation canaries, and unchanged heads. Include snapshot,
guard, restore, and fullRows replay in worker timings and report flag frequency.

The original F32 gates remain relative L2 `1e-4` and max absolute
`5e-4*max(1,reference_peak)` for every originally checked F32 field and
continuation. Native itself may fail the F64 delta gate on a cancellation
fixture; native equivalence does not convert that raw failure into a pass.
Report raw F64-gate status separately from frozen-native branch equivalence.

Prepared W near `6e-61` cannot be represented by one F32 value. On a proven native
branch it may be labeled `unused_transform_on_native_replay`, while its original
raw W relative-gate failure remains reported. Do not alter the reference,
silently omit the metric, relax its tolerance, or say the original transform
gate passed. Similarly, distinguish `raw_quality_gate_pass` from
`native_replay_equivalence_pass` and a scoped committed-output qualification.
Unflagged WY validation remains new W/U history evidence; a finite heuristic
ratio cannot supply a universal certificate for that branch.

Until v3 source and its new branch evidence satisfy these obligations, this
document establishes eligibility/replay design requirements, not that the
guarded standalone or whole worker has qualified.
