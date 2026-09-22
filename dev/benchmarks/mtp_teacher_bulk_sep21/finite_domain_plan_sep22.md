# Conservative finite domain for cache-only scratch elision

This refines the proposal; it is not an implemented eligibility policy or a
completed MPP numerical proof. No coefficient payload was read by this agent.
Root must audit actual immutable coefficients before selecting any finite
domain. Missing evidence means original graph execution, not weakened checks.

## Domain and immutable closure

Initially select only the canonical singleton1920-row body, begin0, implicit
positions0..1919, original128-row chronological prefixes and existing trained
head conventions. All existing host guards still run before selection.

The borrowed BF16[1920,10240] input must contain finite words with absolute
value at most256 (`2^8`). Validate actual **stored** BF16 words, including
subnormal/negative-zero cases. Do not infer eligibility from the trunk's
diagnostic status. This scan reads39321600bytes and its latency/cache effects
belong in inclusive timing. An ineligible input executes the entire original
bulk graph so original poison, diagnostic bits and partial-cache writes survive.

Root's immutable certificate needs the actual objects consumed by this graph:

- Global pre-fc embedding/hidden norm, attn HC norm, and q norm. Compute gamma
  using the actual Direct/OnePlusWeight convention in F32; require finite
  effective gamma with `|gamma| <= 2^4`. Dtype/convention/shape are part of the
  certificate. Auditing raw `weight` magnitude alone is insufficient.
- Actual cached BF16 fc_embedding, fc_hidden, HC down/up and complete
  q_projection coefficient planes, including both256-column halves per head.
  A simple sufficient bound is finite `|coefficient| <= 2^4`, so each consumed
  row's absolute sum is `<2^18` since K<=10240. Alternatively certify stricter
  per-row absolute sums directly. The complete q output12288 is relevant:
  projection diagnostics cover the unused attention-gate half too.
- Original Q5/G64 raw-injection BF16 scale and bias planes. They may be signed
  or zero. A sufficient separate bound is finite `|scale| <= 1`,
  `|bias| <= 2^5`; every F32 reconstructed coefficient is then `<2^6`, and its
  row absolute sum `<2^20`. Separate parameter bounds prevent large cancelling
  scale/bias intermediates from being hidden by a small reconstructed result.
- Original U8/G64 embedding parameters. Finite `|scale| <= 1`,
  `|bias| <= 2^8` imply finite BF16 embedding values at most `2^10`, including
  F32 reconstruction/storage margin. This can be certified from the two
  parameter planes without reading all635699200code bytes. Metadata reports
 19865600bytes per embedding parameter plane; their census is Root-only.
- Actual descriptor epsilon **after F32 conversion** in `[2^-20,1]` and theta
  in the already validated `(1,1e12]` interval. Current config metadata states
  epsilon1e-6, which lies inside this domain. Neither metadata nor the proposed
  bounds establish actual coefficient maxima; those remain unaudited.

The certificate must bind original model identity, actual cached-operand
identity, tensor objects/extents/dtypes, conventions, immutable ownership and
the exact graph/descriptor policy. A stale/missing certificate, unsupported
role or coefficient census failure selects original execution. The audit may
run once at construction, before warming. It does not add a full-model sidecar,
persist an uncharged coefficient copy, or rescan large coefficients per call.

Original parameter bounds can replace cached-plane scans only after replaying
the literal F32 dequantization plus BF16 conversion with conservative interval
rounding. The scale/bias extrema formula needs F32 reconstruction margin and
monotone rounding to a representable power-of-two ceiling; this is not a
BF16 coefficient rounding-parity assertion. Source/converted objects must have
the existing immutable identity witness.

## Norm bound, including epsilon and FTZ

A loose `input_max * rsqrt(epsilon)` bound compounds through the DAG and is
unnecessarily unusable. Instead use the original nonnegative RMS traversal.
For representable nonnegative F32 a/b, rounded `a+b` is at least each operand:
an existing operand is itself a representable candidate, and the exact sum
cannot lie below it. Thus any tree of such additions retains at least its
largest normal stored square; there is no accumulated lower-bound loss below
that term. Normal terms/partials cannot be lost to FTZ. The exact float SIMD
sum/rsqrt contract still belongs in independent review; this is a range
argument and does not assume CPU/GPU reduction parity.

Split at `|x| <= 2^-8`: positive normal epsilon at least2^-20 gives
`inverse <= 2^11`, so even fully flushed tiny squares yield product at most8.
For larger x, its square and square/D (D<=10240) are normal. Conservatively
allow a factor2 lower margin at square/divide, producing denominator at least
`x*x/(4*D)`. A factor2 upper rsqrt/multiply margin then gives
`|x * inverse| <= 4*sqrt(D)`. This avoids relying on subnormal division for
the quotient bound. An
effective-gamma ceiling G and BF16 storage margin give the conservative output
ceiling `8*G*sqrt(D)`. For D<=10240 and G<=16, choose the representable
power-of-two bound `2^14`. For q's D256 choose `2^12`.

Before using this quotient argument, bound every square-sum **above** to exclude
overflow. All reductions are nonnegative, so no cancellation-based argument
permits overlooking a large partial sum. Preserve the exact epsilon add and
all BF16 boundaries. A source/API contract that cannot justify the stated
rsqrt/FTZ margin means fallback, not an empirical universal claim.

## DAG ceilings

Let a projection's input bound be B and its certified row absolute sum be L.
Use `4*B*L` for output storage: this allows the finite F32 partial/reduction
rounding and BF16 conversion margin. The shape and any original scalar/MPP
path are part of the execution closure. These are safety ceilings, not claims
of identical reduction error or universal MPP accuracy.

| Original stage | Conservative stored/intermediate absolute ceiling |
|---|---:|
| Actual teacher feature |2^8|
| Embedding reconstruction/BF16 |2^10|
| Global embedding/hidden norms |2^14|
| FC embedding/each globally-normalized hidden stream |2^34|
| BF16 FC fuse into four hyper streams |2^36|
| HC norm output per stream |2^14|
| HC down |2^34|
| Original divide-by4/BF16 sigmoid/SiLU output |2^35|
| HC up |2^55|
| Raw-injection projection |2^36|
| Injection divided input |2^35|
| Injection sigmoid/output weight |1 /2|
| HC mix (allowing a deliberately loose factor2 at each BF16 product/add) |2^21|
| Complete q projection12288 |2^41|
| Query normalized BF16 |2^12|
| Query RoPE F32/BF16 |2^16|

Embedding/pre-fc RMS square sums are below2^34. Fused-HC RMS square sums are
below2^86. Q RMS square sums are below2^90. The largest certified projection
partial absolute sum is below2^55; all are far below F32/BF16 overflow. Proof
must use the literal K/dtype/descriptor and addition depth, not a measured
typical amplitude. Rounded subnormal terms add negligible absolute margin to
these large representable ceilings, but still belong to interval replay.

The sigmoid's exponential is allowed to become positive infinity for large
finite input: the original kernel deliberately turns this into denominatorInf,
tail0 and finite saturated gate. It never sets a diagnostic on that intermediate.
Do not claim every exponential remains finite or reject valid saturated HC
math. For finite source, the exact original BF16 exp/add/divide/subtract
branches produce finite gates in[0,1]; products/sums above remain bounded.

Only implicit positions inside the original capacity are initially eligible.
With theta>1, frequency is bounded near[0,1], and positions<=1919 yield finite
angles. Conservative scalar pow/trig margins allow `|cos|,|sin| <=2`; two
rotated products, sum/subtract and BF16 conversion fit the2^16 ceiling. Retained
K/index branches still check the same positions. No explicit-position geometry
expansion belongs to this first proposal.

## Remaining proof and diagnostic obligations

The ceilings give a credible wide-margin finite domain, but do not establish
the opaque matmul implementation's universal rounding behavior. Document the
actual BF16 input/F32 destination MPP descriptor's finite execution contract
and permitted intermediate range; verify bounded fixtures/actual coefficients
with the existing strict component guard. A fixture pass alone is not a
universal proof. If the execution contract cannot support this range closure,
retain the original diagnostic-producing operation or abandon elision.

An eligible call may omit q/injection/query-norm checks only after proving
every omitted original diagnostic would remain zero in this domain. Retained
stages must still run unchanged; a retained K/V failure must preserve original
diagnostic bits, poison, logical length and physical partial cache state.
The omitted checks set the same sticky numeric bit, but that fact alone is not
permission to ignore a q-only failure.

Run original-vs-selected bad-input/uncertified-coefficient fallback, all5cache
planes, poison unused scratch, future Last/All logits/greedy, truncate/EOS,
precommit failure and real Worker post1920 cancel/deadline/same-ID recovery
before timing/composition. Keep all18buffers and every chronological128 prefix.
If the40MB eligibility scan erases the current measured saving, close the
experiment. A different GPU certificate/gate needs a separately reviewed graph,
allocation, failure and timing plan.
