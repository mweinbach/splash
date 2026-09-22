# Full512 signed INT8 expert-cache precision audit

This is a source and retained-report audit, not an inference result. The audit
submitted no GPU work, loaded no model, ran no coefficient conversion, and
changed no production source or profile. Full512 is an arithmetic-changing
alternative to the current Top64 prefill policy; it must not be described as
bit-exact to that policy.

## Reference and candidate arithmetic

The existing converter uses the original contiguous affine Q4/G64 planes,
with little-endian nibbles in each U32. Its coefficient reference is

```
w[j] = BF16_RNE(F32_add(F32_multiply(float(q4[j]), float(source_scale)),
                      float(source_bias)))
```

The source scale and bias are BF16 promoted exactly to F32; multiplication
and addition are separate F32 operations. Each finite BF16 result is promoted
exactly for the subsequent quantizer. This is the blocked-MoE BF16 operand
boundary, not an assertion that every earlier affine execution route has
identical arithmetic.

For each output row, the default `symmetric_int8` policy is

```
R = max(abs(w))
sigma = F32_divide(R, 127)             # nonzero row
c[j] = clip(RNE_integer(F32_divide(w[j], sigma)), -127, 127)
```

All-zero rows instead use `sigma=1` and all codes zero. Full512 must retain
`group_size=0`, `fit_rounds=0`, no activation calibration, and the same
positive F32 row-scale policy if it is to preserve the Top64 payload on
the already-selected experts. Least-squares refitting, clipping, grouped
scales, or a different source reconstruction would be separate experiments.

The GPU hit kernels use BF16 activations and signed I8 operands with F32
accumulation, multiply the completed dot by the row scale, then round to
BF16. They do not reconstruct BF16 weight operands before the dot. Gate and
up are separately rounded to BF16 before the compiled BF16 SwiGLU stages;
down is rounded before canonical route combination. Even exact preservation
of every existing Top64 coefficient does not preserve the other 448 experts'
arithmetic or the resulting prefill state.

## Coefficient absolute-error certificate

The proposed streamed check

```
abs(F64(c[j]) * F64(sigma) - F64(w[j])) <= 0.5 * sigma + 32 * u * R
u = 2**-24
```

is a conservative bound for this default quantizer. It is valid for finite
BF16 reference values and correctly rounded F32 scale/division with gradual
underflow. It is not a GPU output-error bound.

For a nonzero finite BF16 row, `R >= 2**-133`. Let `eta=2**-150`, half the
smallest positive F32 subnormal. Correct F32 rounding obeys the conservative
absolute bound `abs(RN32(x)-x) <= u*abs(x)+eta`. Therefore

```
sigma >= (R / 127) * (1-u) - eta
R / sigma <= 127 / (1-u-127*eta/R)
          <= 127 / (1-u-127*2**-17) < 127.124
```

F32 division cannot move that quotient to 127.5. Thus the nearest-even
integer result is already in `[-127,127]`: the final clamp is inactive,
including for BF16 subnormal rows. No unaccounted clipping term is needed.
Integer rounding then contributes at most `0.5*sigma`, and F32 division
contributes at most `u*abs(w[j])+sigma*eta`. Since `sigma<R` and `eta<u`,

```
abs(c[j]*sigma-w[j]) <= 0.5*sigma + u*R + sigma*eta
                     < 0.5*sigma + 2*u*R
                     <= 0.5*sigma + 32*u*R
```

F64 represents both the BF16 reference and stored F32 scale exactly. The
product of a signed 7-bit-magnitude integer and a 24-significant-bit F32
scale needs at most 31 significant bits, so its F64 product is also exact.
F64 subtraction can round, but the certificate's slack is much larger than
that diagnostic rounding. Compute the bound and comparison in F64, not F32.

The certificate should cover every coefficient while each existing
eight-expert conversion chunk is live. Also independently require finite
reference/scale values, exact expected scale bits (`F32(R/127)`, or one for
zero rows), codes in `[-127,127]`, and zero codes in zero rows. A maximum
error bound alone does not certify the quantization policy. Record per-layer
and per-projection row/element counts, maximum absolute error, maximum
error-to-bound ratio, and the full-cache manifest/source/plan identities.
Assert the completed certificate's cardinalities: 120,795,955,200 coefficient
elements, 94,371,840 rows, 15,099,494,400 preserved Top64 code bytes, and
47,185,920 preserved Top64 scale bytes. Reporting counters without requiring
the full expected totals could conceal skipped chunks or projections.

For ordinary normal scales, the leading bound is approximately `R/254`, or
0.394% of the row's largest coefficient. This is an absolute bound relative
to the row maximum, not a 0.394% relative bound on each coefficient. Small
weights may round to zero, and outliers can dominate a row scale.

## Cancellation and GPU numerical checks

For exact-real dots with the same BF16 input `a`, the coefficient certificate
implies

```
abs(sum(a[j]*c[j]*sigma) - sum(a[j]*w[j]))
    <= sum(abs(a[j]) * abs(c[j]*sigma-w[j]))
    <= B_row * sum(abs(a[j]))
```

GPU F32 accumulation, the final F32 scaling operation, BF16 rounding, and
SwiGLU introduce additional terms. A conventional `gamma_K` accumulation
bound can be used only with its explicit finite/no-overflow arithmetic
assumptions; the CPU coefficient certificate does not establish MPP's actual
accumulation behavior. Validate that behavior through actual GPU comparisons.
There is no universal relative dot-error bound: cancellation can make the
reference dot zero. Downstream routing, recurrent/attention state, and greedy
choice can amplify small changes. None of these coefficient bounds imply a
whole-model quality guarantee.

A concrete BF16/Q4 cancellation fixture uses a row whose first G64 group
has coefficient one, second G64 group has coefficient `1/128`, and remaining
groups have coefficient zero. Constant group biases with Q4 codes zero
express it directly. Set `a[0]=1`, `a[64]=-128`, and all other inputs zero.
The reference dot is zero. Rowwise I8 gives codes 127 and one respectively,
so its ideal candidate dot is `sigma*(127-128)=-sigma`, approximately
`-0.007874`. This is expected quantization error, not proof of a kernel bug.
Check absolute error against the audited coefficient result and accumulation
envelope; a relative-error test against zero is inappropriate.

Additional primitive fixtures should include zero and signed-zero inputs,
negative coefficients and affine source scales, row outliers, values near
integer half-step rounding boundaries, finite BF16 subnormal coefficients,
alternating-sign cancellation, maximum legal expert/rank IDs, partial bucket
tails, and numerical overflow/rejection behavior. Compare gate/up dots when
available as well as the complete activation/down/combine chain. Retain
per-channel absolute errors and zero-reference counts alongside aggregate
relative L2/cosine; aggregate metrics can hide fragile channels.

The stock expert-store oracle's 0.10 relative-L2 and 0.99 cosine defaults are
producer sanity guards explicitly labeled as not model-quality qualification.
The stock oracle also requires ten uncached expert IDs before selecting a
pattern. Full512 must use an isolated all-hit adaptation without fabricating
misses. Preserve independent stable buckets/jobs, unequal route weights,
sticky numerical diagnostics, canaries, source/store immutability, admission
rejections, and replay after source/store owners are destroyed. Exact Top64
payload-subset equality should compare each original expert's codes and F32
scale bytes using expert ID rather than assuming compact ranks coincide.

## Fresh whole-model comparison and service gates

Saved `persisted-int8-quality-top64-comparison-v1.json` has 22/22 full-plan
cases, zero new task regressions, and one text-formatting difference in
unconstrained JSON. It does not qualify Full512. Reuse the frozen plan
`persisted-int8-quality-plan-v1.json`, whose content hash is
`a4135cc4b5d8bf7c13e22c5dfdad68ece414b3dd69bdfded63a77e007aefb466`,
but collect a fresh Top64 baseline and Full512 candidate with the same source,
build, all unrelated flags, MTP depth three, context, and cache policy.

The existing harness covers eight protocol cases and fourteen supplementary
cases: deterministic arithmetic, prose, JSON/schema, held-out Python behavior,
three retrieval positions, distinct concurrent retrieval lanes, forced tools,
and continuation. It checks HTTP/SSE terminals, token accounting, zero cached
tokens, stable identities, request counters, and a fresh idle native state.
Require `valid=true`, `full_plan_coverage=true`, no skipped/failed cases, and
zero new task regressions; exit zero alone permits partial coverage. Read
both factual-prose outputs for contradictions and coherence. Exact text
equality is useful diagnostic evidence and is not a universal requirement for
this numerical alternative.

| Additional gate | Required evidence |
| --- | --- |
| Loaded derivative | Assert persisted cache enabled, exact new manifest and plan hashes, 24,576 target experts, 121,173,442,560 mapped bytes, and numerical-alternative status. A route substring is insufficient. |
| Physical prefill boundaries | Exercise 255/256/257 rows and 2,048/2,049/4,096 rows. Confirm which chunks use I8 and which short tails use original Q4. |
| Concurrent state | Distinct long prompts with actual native batch counters, mixed lengths, and staggered completions; assert no answer/state contamination. |
| Prefill cancellation | Disconnect/deadline during a large I8 prefill, then confirm native terminal accounting and idle cleanup. The existing cancellation probe disconnects after content and exercises decode cleanup. |
| Admission failure | A constrained budget rejects before mapping or returns bounded capacity exhaustion without stranded queued/active/in-flight work. |
| Lifecycle | Capture frontend/native PIDs; stop at idle and during active work, prove both exit and port closes, reload the same cache, then repeat a deterministic request. |
| Idle/residency | Record requested/successful residency registration, governor headroom, and a request after the maintenance interval. Registration does not prove physical pinning or cross-die locality. |
| Performance | Separate matched full-budget HTTP tests with uncached prompts; report prompt rows, output tokens, concurrency, warm-up, prefill, decode, acceptance, peak memory, and idle cleanup. |

The stock metadata loader and converter allow only up to 128 selected experts,
so Full512 support belongs in the isolated experiment until its numerical and
service gates pass. Original Q4 weights remain needed for decode and small
prefills, and trained MTP weights are unchanged. Prefill state changes can
nevertheless alter later decode outputs and speculative acceptance.

The fresh SSD baseline `prefill4k-current-context-baseline.json` reports peak
120,129,617,920 bytes. Replacing Top64's 15,147,466,752 planned bytes with
Full512's 121,174,228,992 planned bytes extrapolates to 226,156,380,160 bytes,
leaving 21,233,736,090 bytes below the 247,390,116,250-byte governor budget.
This is a planning estimate; candidate startup admission and observed request
peaks remain required. The earlier resident-PLE Top128 ledger should not be
used as the current SSD profile's memory ledger.

The prepared private converter's streamed F64 coefficient checks and
per-expert Top64 subset checks follow this numerical contract. Its strengthened
preflight now requires the recorded Top64 cache to be enabled and checks its
manifest hash, 3,072-expert count, mapped bytes, original source identity,
cache manifest hash embedded in the kernel route, and enabled SSD streaming
against the preserved sidecar. The retained
`prefill4k-fullcache-preflight512.json` records the same-source SSD estimate
and identities above.

An isolated hook now runs before the base converter's atomic rename. It
requires every expected coefficient/row/subset-byte count, rechecks the
preserved Top64 payload and manifest snapshots, and writes the completed
coefficient certificate inside the temporary directory. The default hook
fails closed if a certificate callback is missing. Thus the certificate
and final preserved-store checks precede publication of the new sidecar.
Source review verified this ordering. The parent coordinator reports that
the strengthened inspect-only path and CPU self-tests passed; this audit did
not execute them. Heavy conversion has not started and remains pending the
root's shared memory slot. Native GPU and whole-model quality remain
unqualified.

HTTP and `flash_precision_quality.py` expose no usable teacher-forced target
likelihood/logit metrics. The newly prepared teacher-cache oracle checks MTP
cache-prime equality and future proposals; that is a different question from
Full512 target precision. Task successes must not be reported as measured
perplexity or logit preservation. A broader claim would require a dedicated
fixed-prefix target-logit/likelihood comparison and a representative corpus.

Only the root coordinator runs heavy hashing/conversion, GPU oracle calls,
or model services, serially and after reserving the shared memory slot. This
audit's proof and planned gates are preparation, not evidence that the cache
has been converted, admitted, numerically checked on GPU, or qualified.
