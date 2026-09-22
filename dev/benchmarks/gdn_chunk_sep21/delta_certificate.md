# Separate FP32 delta certificate

This audit preserves the existing `1e-4` delta relative-RMS quality guard. A
failed old guard remains a failed old guard. The certificate diagnoses whether
an absolute delta discrepancy is consistent with a specified FP32 arithmetic
path, and records conditioning and incoming-state evidence separately.

No numerical constant is fitted to the observed GPU or CPU errors. The only
precision constant is FP32 unit roundoff `u = 2^-24`. Operation counts determine
`gamma_n = n*u/(1-n*u)`. BF16 q/k/v/beta and F32 alpha/initial state are the
immutable, quantized problem inputs; their quantization is not charged as an
implementation error against a reference using those same inputs.

## Why the cancellation fixture fails a delta-only relative guard

For `cancellation_repeated_key`, the canonical scalar F32 serial recurrence
versus the independent F64 recurrence has delta relative L2 error
`5.447417445517276e-4`, although its maximum absolute delta error is only
`4.470348358154297e-8`. Thus the old guard also fails on a normal F32 serial
implementation; this observation does not authorize changing it.

Define the cancellation operand scale per coordinate as

```
c_t = beta_t * (abs(v_t) + abs(memory_t)),
memory_t = alpha_t * S_(t-1) * k_t.
```

The fixture has `||c||2 / ||delta_F64||2 = 11091.08887700653`. Even one rounding
step at the operand scale corresponds to `u*11091.088877 = 6.6106858703e-4`
relative to delta. Memory dot rounding and the incoming state add further
terms. A relative error measured only against the small subtraction residual
does not measure error at the scale of the operands that formed that residual.
The implementation reports both the old relative error and the separate
condition-aware quantities; `abs(error)/(u*c)` alone is a diagnostic, not the
complete certificate, because it omits dot/coefficients/state propagation.

In this fixture alpha=beta=1 and the repeated key has exactly unit norm. The
chunk lower-system matrix `B = I+L` is the all-ones lower triangle. Its inverse
has diagonal 1 and first subdiagonal -1, so `||B^-1||inf=2` for a multirow
chunk. `kappa_inf(B)=2*C`: 32 for C16 and 64 for C32. The dominant delta
conditioning here comes from subtraction, not a nearly singular lower system.

## Exact chunk problem and state provenance

Within a chunk, with state layout `[V,K]`, define

```
A_t = product(alpha_0 ... alpha_t)
P_tj = product(alpha_(j+1) ... alpha_t), P_tt=1
G_tj = dot(k_t,k_j)
L_tj = beta_t * P_tj * G_tj, j<t
b_t = beta_t * (v_t - A_t * dot(S0,k_t))
B = I+L
B * delta_exact = b.
```

The residual API computes

```
r_t = delta_candidate_t + sum(j<t,L_tj*delta_candidate_j) - b_t.
```

For the local arithmetic diagnosis, `S0` must be the actual incoming candidate
state, independently captured before each chunk's projection. The preferred
input is flattened `[ceil(T/C),V,K]` F32 state. For comparison to the full F64
serial truth, the audit also computes the incoming-state difference and adds
its effect to the forward error bound. This prevents a local residual from
hiding errors carried from previous chunks.
The incoming-state quality guard remains independent: a large observed
incoming-state discrepancy can produce a large delta bound even when local
solve arithmetic is consistent, and is not excused by this certificate.

Reconstructed per-token history may differ from the actual MMA state used by
the next chunk. When only history is provided, the API marks it as a proxy. A
rigorous global certificate is unavailable unless a proven componentwise
proxy uncertainty is supplied. The first chunk's immutable initial state is
known independently. When neither candidate input nor history exists, later
chunks use reference input only for a conditional diagnostic, explicitly
marked unavailable for qualification.

## Operation-count rounding bounds

Assume finite values, alpha/beta in `[0,1]`, keys whose prequantization norm is
at most one, BF16 RNE source inputs, and normal representable intermediate
products/reductions. BF16 unit roundoff is `2^-8`, giving the default key-norm
limit `1+2^-8`. The formulas use the measured absolute operand sums, not a
fitted state/activation scale.

For exact incoming candidate state `S0`, let

```
s_t = dot(S0,k_t)
D_t = sum_k abs(S0_vk*k_tk)
E_s,t = gamma_K * D_t
E_A,t = gamma_(t+1) * abs(A_t)
```

`gamma_K` is the standard dot-product bound and covers ordinary multiply/add
or FMA dot arithmetic. A different precision, approximate dot, or nonstandard
MMA accumulator must be modeled separately. The prefix count includes the
first multiply by one conservatively. Products are built directly; the
certificate does not permit prefix division through zero/underflow.

The decayed projection and RHS bounds are

```
E_m,t = abs(A_t)*E_s,t
        + E_A,t*(abs(s_t)+E_s,t)
        + u*(abs(A_t)+E_A,t)*(abs(s_t)+E_s,t)

E_b,t = beta_t*E_m,t
        + gamma_2*beta_t*(abs(v_t)+abs(A_t*s_t)+E_m,t).
```

The last line includes subtraction and beta multiplication, with operand
magnitude retained even under cancellation. This bounds the implemented
`fl(beta*fl(v-fl(A*fl(dot))))` relative to the exact RHS.

For each lower coefficient, let

```
D_G,tj = sum_k abs(k_tk*k_jk)
E_G,tj = gamma_K * D_G,tj
E_P,tj = gamma_(t-j) * abs(P_tj)

E_L,tj = beta_t * (abs(P_tj)*E_G,tj
                    + E_P,tj*(abs(G_tj)+E_G,tj))
         + gamma_2*beta_t*(abs(P_tj)+E_P,tj)*(abs(G_tj)+E_G,tj).
```

This covers two coefficient multiplications and either coefficient product
ordering used by the shader. Beta and source k are exact problem inputs.

For the direct forward row solve, candidate magnitudes provide an operation
bound independent of the observed errors:

```
E_solve,t = gamma_(2*t) * (abs(b_t)+E_b,t
                           + sum(j<t,(abs(L_tj)+E_L,tj)*abs(delta_candidate_j)))

R_t = E_b,t + E_solve,t
      + sum(j<t,E_L,tj*abs(delta_candidate_j)).
```

Two rounding steps per term cover a nonfused multiply/subtract. The API may
use one only when the compiled direct solve is known to fuse those operations.
An inverse/WY/matmul solve has a different operation path and cannot reuse this
direct-row bound without a separate derivation. A rounding-consistent local
residual satisfies `abs(r_t) <= R_t` plus the reference evaluation allowance.

The componentwise incoming-state RHS contribution is

```
J_t = beta_t*abs(A_t) * sum_k(abs(S0_candidate_vk-S0_truth_vk)
                              + proxy_uncertainty_vk)*abs(k_tk).
```

## Forward propagation and conditioning

The conservative recursive forward bound is

```
E_t = R_t + J_t + sum(j<t,abs(L_tj)*E_j).
```

This is the requested coefficient/state/triangle propagation bound. It can be
very pessimistic: all `L_tj=1` makes the comparison recurrence grow like `2^t`,
despite the signed matrix inverse being bidiagonal. The audit reports this
bound but also computes the stronger signed inverse bound

```
E_sharp <= abs(B^-1) * (R+J).
E_posterior <= abs(B^-1) * (abs(r)+E_residual_eval+J).
```

The posterior bound alone cannot distinguish an incorrect algorithm: a large
observed residual merely produces a large posterior bound. The useful separate
tests are the residual's consistency with the operation-count bound `R`, and
the observed delta error normalized by the independently derived `E_sharp`.

The implementation evaluates centers in `long double` and reports that type's
actual mantissa bits (53 on this host). It charges gamma bounds for center
evaluation, including prefix, Gram, projection, RHS, and residual evaluation.
Computed product centers use `gamma_n/(1-gamma_n)*abs(computed_product)` to
bound their exact-product uncertainty. Nonnegative bound aggregates are
inflated for their own fixed rounding paths.

A numerically computed signed inverse `X` is certified through
`F=I-B_exact*X`. The audit bounds `q=||F||inf`, including coefficient-center
uncertainty and residual-evaluation rounding. If `q<1`,

```
||B^-1||inf <= ||X||inf/(1-q)
abs(B^-1)*h <= abs(X)*h + ||X||inf*q/(1-q)*||h||inf.
```

The second term charges inverse computation error. The full comparison also
adds the F64 target's consistency difference from the exact local lower solve
plus its evaluation bound. Thus an F64 serial baseline is not treated as
symbolically exact.

## Explicit limitations

- A candidate must actually use the specified FP32 direct-dot/direct-row solve
  path. Compiler fast approximate arithmetic or reduced accumulator precision
  needs another model.
- Relative gamma-only bounds do not cover underflow/FTZ/DAZ or overflow. The
  implementation detects nonzero subnormal/overflow-risk prefix, coefficient,
  projection, and solve products conservatively, and marks the certificate
  unavailable. It does not silently enlarge a tolerance for those cases.
- Prefix-underflow stress fixtures remain independently valuable. A zero
  prefix can lose an old large state even while the serial recurrence retains
  a representable result. That is a representation failure, not a conditioning
  explanation.
- The original quality gate is still reported separately and unchanged. A
  numerical certificate is evidence for reviewing an ill-conditioned fixture;
  it is not an automatic substitute for the preregistered gate.

## API and CPU reproduction

```
clang++ -std=c++17 -O3 -ffp-contract=off -DGDN_CPU_ORACLE_NO_MAIN \
  dev/benchmarks/gdn_chunk_sep21/cpu_oracle.cpp \
  dev/benchmarks/gdn_chunk_sep21/numeric_certificate.cpp \
  -o /tmp/gdn_chunk_sep21_numeric_certificate
/tmp/gdn_chunk_sep21_numeric_certificate > /tmp/gdn_chunk_sep21_numeric_certificate.jsonl
```

For use in the GPU harness, additionally compile with
`-DGDN_NUMERIC_CERTIFICATE_NO_MAIN`. Call `certify_delta(fixture, serial_F64,
candidate_evidence, options)`, then `delta_certificate_json(result)`.
`include_elements=true` includes every coordinate's RHS, incoming-state,
coefficient, solve, residual, recursive, inverse, and posterior bound.

The CPU example's incoming states are known because CPU `chunked()` copies
its final history verbatim into its next actual state. That source identity
does not hold automatically for the GPU MMA carry.

Local CPU reproduction: 38 of 46 examples meet the normal-range assumptions;
all 38 pass the separate certificate. Eight underflow stress cases are
explicitly unavailable. Cancellation C16/C32 have relative delta errors
`2.4358599247698565e-4` / `2.1370737174202813e-4`, maximum absolute error
`4.470348358154297e-8`, maximum residual/rounding-bound ratio about `0.021602`,
and maximum error/inverse-forward-bound ratio about `0.019885`. These CPU
results are not a certification of GPU traces whose actual chunk inputs have
not yet been independently captured.

Verification also passed a clean AddressSanitizer/UndefinedBehaviorSanitizer
run and a no-main library compile. A negative control adding `0.001` to one
candidate delta rejected both rounding consistency and the forward bound
(residual/rounding-bound ratio about 110.33). History-only evidence with no
proxy uncertainty was correctly marked unavailable.
