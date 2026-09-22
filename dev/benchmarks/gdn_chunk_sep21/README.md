# Chunked GDN prefill experiment

This directory is isolated from the production runtime. It implements a separate
prefill recurrence with BF16 source queries, keys, values and beta, F32 decay,
and F32 state, solve and accumulation. It deliberately changes floating-point
association and is not a byte-exact replacement for the canonical recurrence.

The algebra follows the chunk/WY approach in the primary
[FLA naive GDN reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py),
with direct relative decay products instead of ratios of cumulative products.

For a chunk with incoming state `S0`, define
`A(t) = product(alpha[0:t])` and
`P(t,j) = product(alpha[j+1:t])`, with `P(t,t) = 1`.
The unit-lower-triangular solve is

```
delta[t] + sum(j<t) beta[t] * P(t,j) * dot(k[t],k[j]) * delta[j]
  = beta[t] * (v[t] - A(t) * S0 * k[t])
out[t] = A(t) * S0 * q[t]
       + sum(j<=t) P(t,j) * dot(q[t],k[j]) * delta[j]
S_end = A(end) * S0 + sum(j) P(end,j) * delta[j] * k[j]^T
```

Zeros and underflowed cumulative prefixes never appear as divisors. Updates
after a zero or underflowed prefix remain represented by their own relative
products. Prefix-first multiplication can still lose range for artificially
huge incoming state; the CPU audit reports this case separately.

The first three shader variants use SG4 matrix operations and F32 state in
threadgroup memory. The two register variants retain the state in F32
cooperative tensors, partitioned into eight value rows per SIMD group. Apple
permits cooperative tensors as MPP inputs only within a single SIMD group, so
register state projection/update use four independent SG1 operations while
Gram and output products use SG4.

`candidate.metal` includes normal and separate audit entries. Audit entries are
restricted to synthetic value head zero and dump every full F32 token state,
delta and output before BF16 conversion. They are never timed.

Build and CPU verification:

```
make -f dev/benchmarks/gdn_chunk_sep21/Makefile -j2
build/gdn-chunk-sep21/cpu-oracle
```

The checked CPU report covers 23 generated immutable fixtures, chunk sizes 16
and 32, tails around chunk boundaries, K128/V128, cancellation, alpha zero/one,
decay underflow, beta zero/one and a distinct 19-token continuation. Across
56,674,048 comparisons, F64 maximum absolute error was `8.88e-16`; F32 chunk
versus serial maximum absolute error was `4.77e-7`. The ordinary gates passed,
and the expected extreme-state range loss was detected. ASan/UBSan passed.

Only the GPU coordinator should run these, after acquiring its serialized slot:

```
build/gdn-chunk-sep21/metal-oracle --resources
build/gdn-chunk-sep21/metal-oracle --quality
build/gdn-chunk-sep21/metal-oracle --quality --lanes 2
build/gdn-chunk-sep21/metal-oracle --bench --rows 2048
```

The synthetic oracle compares F32 history, delta, output before BF16 conversion,
and carried state with independent F64 scalar recurrence, alongside canonical
Metal recurrence. It checks independent continuation from each native carried
state, complete canaries, prepared-input SHA256 hashes and untouched heads.
Its BF16 mismatch rates are observations rather than F32 tolerance gates.
Benchmarks use nine randomized matched ABBA/BAAB pairs after warm-up and measure
only the 48-head recurrence; the reported row rate is not model prefill speed.

Required remaining qualification includes actual prepared activations, all
F32 carried state and BF16 outputs/history, normalization/gating output,
longer continued sequences, full-model logits/token behavior and matched
end-to-end prefill/decode measurements. Source compilation and synthetic
mathematical equivalence alone do not satisfy those gates.
