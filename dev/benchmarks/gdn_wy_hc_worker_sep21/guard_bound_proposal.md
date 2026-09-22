# Guard refinement assessment — proposal only

The actual-input local component is rejected: cold 4.15973 ms versus native
1.89794 ms, carried 4.19475 versus 1.89875 ms. Cold decisions selected 9446/12288
native tiles: 3568 range, 2572 norm, 3306 cancellation. Same-incoming-seed native
chunk equality, future candidate-own-carry quality, canaries and immutable data
passed. Evidence: `build/release/flash/sep21-gdn-tile-chunk-actual-capture-v1.jsonl`.
These counts do not reveal the actual scalar contributions that triggered the
checks. No new implementation or GPU run is proposed now.

**A bounded norm repair is mathematically credible, but insufficient to admit
the complete WY algorithm.** A vanished squared term need not invalidate a
row norm. Let eta be the minimum normal F32 value (2^-126), conservatively
covering permitted FTZ; use worst-case RTZ u=2^-23. For K=128 finite inputs,
no overflow/saturation, and observed positive square-sum s_hat, a conservative
arithmetic-local upper bound is

`sum(x_i^2) <= (s_hat + 2*K*eta) / (1 - gamma_(2*K))`,
`gamma_n = n*u/(1-n*u)`.

The absolute allowance covers product and positive-sum underflow; the gamma
factor covers ordinary rounding. Its square root, conservatively rounded
upward, can replace “any subnormal square => reject” while retaining an upper
Cauchy norm. Actual overflow/finite saturation stays rejected. This changes the
norm policy, not the raw F64 tolerances. It needs an independently justified
MPP/scalar rounding model and representable conservative evaluation. It does
not bound errors already present in the state or W coefficient values.

**Range admission needs the actual incoming state and update magnitudes.** A
discarded prefix contribution is bounded by `abs(prefix_error)*norm(S0)`;
discarded relative update contributions need the corresponding delta/key
magnitudes, not merely a small prefix. A test can admit only when an upper
bound on the total omitted contribution is below both the unchanged absolute
budget and `1e-4` times a lower bound on the true result norm. A zero lower norm
means unavailable, not a denominator floor. This rejects the demonstrated
huge-incoming-state/tiny-prefix case and can admit a genuinely negligible term
beside normal nonzero updates. Current reports do not contain the needed
per-tensor contribution bounds, so claiming admission rates would be speculation.

**This does not resolve cancellation or whole history.** The 3306 cancellation
tiles remain rejected unless preparation errors in L/inverse/W/U and carried
state errors are propagated into a delta bound. The earlier certificate for a
different algorithm cannot be reused. A Cauchy upper bound may overestimate
projection magnitude, but shrinking it heuristically would not prove the
existing F64 gates. Native itself also fails a known cancellation-relative
fixture gate; reference equality must remain distinct from accuracy.

Recommendation: close this GDN performance branch for the present goal. Keep
the bounded norm/range ideas as a future new-policy study, requiring source
norm/error observations and preparation/history error propagation before any
implementation. The exact teacher optimization is the current route to the
prefill target. No tolerance, sealed kernel, selector decision, or whole-worker
composition is changed by this assessment.
