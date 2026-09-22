# Bounded correction audit; no fallback implemented

Root first measured T16 SG8 at 1.632833 ms versus native 1.897208 ms, with
preparation, GPU seed reset, at least 150 ms warm GPU work, and ten balanced
pairs. Evidence: `sep21-gdn-wy-bench-r2k-v1.jsonl`. T32 is compiled in an isolated
v2 source/build. Root requested timing <=1.3 ms before investing in integration
or correction. No speed, branch-frequency, or fallback-overhead estimate below
is measured.

## What the actual v1 numerical failures mean

- `cancellation_repeated_key`: prepared W/U/end-key/score/prefix are exactly the
  F64 truth in this fixture, but D subtraction/projected state changes rounding.
  Delta max absolute error is 4.470348e-8 and relative RMS 2.43586e-4, exceeding
  the unchanged 1e-4 gate. The native GPU algorithm independently has a known
  delta relative-gate failure around 1.4583e-4 / 1.49e-8 absolute. A serial
  fallback alone therefore cannot promise a universal pass against F64.
- `prefix_underflow_surviving_update`: final state, delta, and output are exact,
  but isolated prepared W has F64 magnitude 6.223015e-61. No single F32 value
  can represent that magnitude, so its relative error is 1. This is an actual
  intermediate gate failure, not a state/output regression. Calling it a pass
  would silently discard a gate. A future guarded branch must explicitly state
  which transform is unused, or use an exponent representation; neither is
  implemented here.
- `extreme_range_prefix_underflow`: initial F32 state is about 2^100 and two
  tiny decays produce a mathematically tiny prefix. Materializing that prefix
  as F32 first gives zero, although sequential decay of the large carried state
  leaves a representable 7.888609e-31 value. A scalar prefix is not a sufficient
  representation for arbitrary incoming F32 state.

The heterogeneous actual timed SG4/SG8 pipelines passed their full 48-head
state/output/coefficient checks and carried continuation. That result has a
specific normal-input scope and does not erase the failures above.

## Credible range fallback

Preparation can conservatively flag a chunk when a nonzero mathematical
prefix would become subnormal/zero in F32. Track a normalized mantissa and an
integer exponent with `frexp`/renormalization while reading the actual F32
decays. An actual zero decay is distinguished from nonzero decay underflow:
an actual zero legitimately removes previous state. Flagging every nonzero
prefix below the normal F32 range is conservative and independent of an
assumed bound on incoming state. This avoids a stale or unknown-history proxy.

The state phase would replay the *whole value tile/chunk* through the captured
canonical decay -> SIMD memory reduction -> BF16 beta delta -> rank-one update
-> SIMD output reduction sequence. All decisions must precede writes to state,
output, or audit history. Every source BF16/F32 boundary, contraction setting,
and reduction tree must be retained. Values can be processed in smaller
canonical tiles while preserving each row's SIMD tree; this must be checked
byte-for-byte against the captured native kernel and continued state. The
original state is already on device. No full-state host copy is required.

This fixes the demonstrated local prefix-state reassociation loss if the
fallback starts from the same F32 state as native. It does not establish an
independent whole-history error bound after preceding matrix chunks. Extra
preparation comparisons are small in count, but branch costs and frequency
need actual measured evidence.

## Cancellation requires an error certificate or an accuracy alternative

A heuristic `abs(D) < epsilon*(abs(U)+abs(projected))` catches the current
fixture but is not a certificate for all W/U transform errors or prior state
errors. Do not ship that heuristic as a universal numerical guarantee.

A concrete conservative certificate is possible, but it is additional work:

1. Bound Gram, direct relative products, L, and the triangular inverse using
   actual source values. Propagate nonnegative errors through each inverse
   column, including L error times previous inverse error. A matrix norm alone
   without the actual inverse/history is insufficient.
2. Propagate bounds into W and U. For a dot, use an upper bound of the form
   `gamma_(2K) * sum(abs(a*b))` plus coefficient/input error terms, where
   `gamma_n = n*u/(1-n*u)`. A real MPP implementation must justify its rounding
   model/count. Without a proven subnormal contract, an absolute per-operation
   allowance up to the minimum-normal F32 scale must also be included.
3. Track an incoming-state error bound from the known initial F32 state and
   every prior chunk. For D, account for U error, W error times state magnitude,
   state error times W magnitude, product/reduction error, and subtraction
   rounding. Unknown history makes this certificate unavailable; do not reset
   it to zero from current-state magnitude.
4. Aggregate `||E_D||_2` and a lower bound on true delta norm obtained from
   `max(abs(D_approx)-E_D,0)`. Admit a matrix chunk only when this proves the
   existing 1e-4 relative and absolute gates. Otherwise fall back or explicitly
   report the certificate unavailable. Extend the same proof to state/output.

For well-conditioned normal deltas, the bound may be smaller than 1e-4.
For near-cancellation the bound is unavailable/too large and will select the
fallback. Those are hypotheses, not measured admission rates. A native replay
preserves native behavior but still may fail the original F64 delta gate; only
a separately valid conditioning certificate can explain that failure without
changing the gate. The old direct-solve certificate cannot simply be relabeled
for W/U.

Compensated projection/subtraction might improve local cancellation accuracy,
but it does not remove error in already-carried F32 state or establish history
precision. An exponent representation can address prefix range, but feeding
such coefficients into the existing F32 MPP product requires a new scaling
design. Neither is a justified one-line repair.

Proceed only if Root's T32 measured performance justifies this work. Until then,
v2 remains a diagnostic numerical alternative with unchanged failing gates.
