# Source-only R4 expert coefficient reuse proposal

Prepare one distinct primitive: **four-route cohorts, SIMD32, char4 weights,
balanced dot4 followed by one F32 accumulator, N16, 128 threads, ten persistent
cohort slots**. Do not rebuild the existing R1 expert vector or the failed
independent-route R4 vector. No kernel or worker has been built for this
proposal. No GPU work or model/capture payload reads or hashes were performed.

The source census below is from
`build/prefill-qsa-twopass-sep21-worker-v1/source`; it identifies actual current
executor branches. It does not claim to know the runtime expert overlap. Root
provided the existing approximately6.9ms standard/12ms verifier MoE budgets
and the prior R4 failure; no capture/report payload was opened to infer a fix.

## Current routes and existing controls

| Executor | Source selection | Current small-row expert path |
|---|---|---|
| Singleton trunk, including MTP verifier | `FlashForward.cpp:1282` | Full512 + gathered enabled + physical rows≤frozen cap; default cap4 |
| Autoregressive real batch | `FlashBatchForward.cpp:457` | Same test against active lanes≤cap |
| Joint target verifier | `FlashBatchVerify.cpp:573` | Same test against flattened lanes×window rows≤cap; larger flattened windows use buckets |

Thus **cap4 selects direct gather, not bucket-native**. Gate/up is
`{10,R,10}`, down is `{40,R,10}`, both128 threads
(`FlashInt8ExpertStore.mm:452,476`). R4 executes400 gate/up +1,600 down CTAs
per layer, or96,000 across48 expert layers, in two dispatches per layer.
Each route binds A as `{K,1}` despite the M16/N64/dynamic-K/SG4 descriptor
(`flash_gathered_mpp.metal:80–86`); its other logical rows are masked.
No factor16 physical-work or speedup claim follows from that opaque lowering.

The bucket-native fallback already reuses a coefficient tensor across its
valid expert rows (`flash_int8_expert_store.metal:307–318`). It retains M16/N64
and can have1–4 valid rows in a normal R4 expert bucket. Its control also pays
six histogram/prefix/map/pack/job setup dispatches
(`FlashMoEBuckets.cpp:147–180`, `FlashMoEBlocked.cpp:253`), plus gate/up,
excluded-down poison, down preparation, and down: ten expert dispatches per
layer. Small-row producer launch capacity is40
(`FlashInt8ExpertStore.mm:133`), independently of the live job count.
Both the current gathered cap4 path and this native bucket path must remain
matched controls. The new design is not the first coefficient-sharing path.

## Distinct arithmetic and execution layout

Keeping the failed vector's four independent component accumulators would
produce the same per-route association when inputs are unchanged. Merely
sharing its coefficient loads cannot repair that R4 fidelity failure.

Instead, for each adjacent four-weight packet, compute exact representable
F32 products, then fixed `((p0+p1)+(p2+p3))`, and add that block sum to **one**
F32 lane accumulator. Reduce its32 lanes with the fixed descending XOR tree.
Contraction and reassociation remain disabled. This changes association and
therefore requires a new source-bound numerical identity and fresh gates.
There is no claim that it fixes the comparison against opaque MPP rounding.

One SIMD group handles four adjacent output columns; four SIMD groups provide
N16. For each column it loads one `char4` coefficient vector, converts it once,
and uses it against up to four original BF16 route inputs. Gate/up share the
loaded BF16 vectors. Down uses each route's own canonical intermediate.
The stored layout remains `[rank,output,K]`; source addresses are naturally
char4/ushort4 aligned because K2560/K640 and the existing buffer guards satisfy
their4/8-byte alignment. No transpose, coefficient cache, activation
quantization, new sidecar, or global BF16 staging is involved.

Use a single compact R4 metadata-plan dispatch with the existing blocked
scratch counts, offsets, route map, inverse map, jobs and job count. Sort the
at most40 valid flattened routes by original expert ID and stable flattened
route order. Validate I64 IDs and rank bounds before any coefficient address.
Create cohorts of at most4 routes. For ordinary unique-per-row top10 output,
one expert has at most4 routes, so the live cohort count G equals unique expert
count U, with10≤U≤40. Malformed duplicates remain distinct; split larger
buckets into4-route cohorts, with G=Σceil(count[e]/4)≤40.

The current `ComputeDispatch` has fixed host threadgroup dimensions
(`MetalBackend.hpp:173`). Avoid a40-job grid with many empty CTAs: launch ten
persistent cohort slots per output tile and loop
`job=slot; job<jobCount; job+=10`. N16 gives gate `{40,10,1}` and down
`{160,10,1}`, exactly2,000 CTAs per layer, with1–4 jobs per CTA. It has three
dispatches per layer including the compact plan. For U10, each CTA processes
one four-row cohort; for U40 it processes four single-row cohorts. There is
no host readback or new indirect-dispatch API. Scratch payload written by the
plan is at most6,796 bytes within existing arrays.

| Output tile | Columns/SIMD | Gate+up F32 accumulator scalars/thread | Down scalars/thread | Fixed CTAs/layer |
|---|---:|---:|---:|---:|
| N8 | 2 | 16 | 8 | 4,000 |
| **N16** | **4** | **32** | **16** | **2,000** |
| N32 | 8 | 64 | 32 | 1,000 |

These are accumulator state counts, not measured physical register allocation.
Full-cohort BF16 vectors add16 F32 temporaries; streamed gate/up coefficients
add8 plus product/control temporaries. The corresponding old four-partial
four-row design would require64/128/256 gate+up accumulator scalars. N16 is
the first bounded choice: it matches the current CTA count and removes the
large state footprint. N8/N32 should not be built before N16's result warrants
them.

## Traffic and work census

Per expert: gate+up codes3,276,800 bytes, down codes1,638,400 bytes, total
4,915,200 bytes; all row scales15,360 bytes. Four rows×ten selections produce
40 route pairs. Current direct gather has196,608,000 logical code bytes/layer.
The cohort design loads4,915,200×G logical code bytes. GPU caches may already
reduce the current external memory traffic to near the unique-expert amount;
the guaranteed source-level reduction concerns loads/conversions, not a
promised DRAM saving.

| Normal U=G | Mean rows/cohort | Proposed code MB/layer | Code GB across48 layers | Logical coefficient-load ratio |
|---:|---:|---:|---:|---:|
| 10 | 4 | 49.152 | 2.359296 | 0.25 |
| 20 | 2 | 98.304 | 4.718592 | 0.50 |
| 30 | 1.33 | 147.456 | 7.077888 | 0.75 |
| 40 | 1 | 196.608 | 9.437184 | 1.00 |

Values use decimal MB/GB. Pure coefficient memory floor is
`4,915,200*ΣG[layer]/BW`; actual overlap and BW are deliberately unknown here.
Do not multiply the whole12ms MoE budget by this ratio as a performance claim:
router/shared/combine, cache effects, plan cost, register pressure and FP32 work
remain.

R4 computes196,608,000 coefficient products per layer regardless of reuse.
Balanced dot4 plus lane accumulation uses another196,608,000 F32 additions.
The physically executed32-lane XOR reductions add24,576,000 additions:
approximately417.792 million F32 operations/layer, or20.054 billion/window
across48 layers, before scale/SwiGLU/control. The logical dependency tree has
31 reductions per output; the actual shader executes five on every lane.

N16 source-level BF16 input loads total65.536MB/layer, versus262.144MB for the
old C1-style independent-output vector. Most of the actual input footprint is
only20,480 hidden bytes plus51,200 intermediate bytes, so those load counts
primarily identify instruction/conversion pressure. MPP input load behavior is
opaque and is not assumed to equal a naive tensor byte count. Canonical outputs
remain51,200 activation bytes +204,800 down bytes/layer.

## Exact product proof and new numerical gates

A finite normal BF16 value has an integer significand of at most8 bits.
Symmetric signed-I8 magnitude≤127 has at most7 bits. Their product has at most
15 significant bits, below F32's24, and is exactly representable whenever its
exponent fits F32. BF16 subnormals lie on the2^-133 lattice, which is a multiple
of F32's2^-149 lattice, so representability itself also holds for in-range
subnormal products. **Overflow remains possible**, and Metal may flush source
or intermediate subnormals even when they are representable. This is an exact
representation proof, not permission to ignore those arithmetic cases.

For the new association, each contribution sees at most
`D=ceil(K/128)+2+5`:27 additions for K2560 and12 for K640. One output's logical
addition dependency count is `N=K+31`: each dot4 packet has3 tree additions and
one accumulator addition, plus the31-node final tree. Reuse does not sum
different rows together, so it adds no cross-row numerical term.

Carry the rigorous v1b envelope over with these new D/N values: u32=2^-23
covers supported RN/RTZ; `4*N*minNormalF32*(1+gamma(D,u32))` covers operand and
result FTZ; preserve compensated-F64 uncertainty and upward positive envelopes;
late scaling covers raw-dot operand/result flushing and F64 reference-product
uncertainty. Source/subnormal scales and conservative intermediate/scaled
overflow risks remain exceptional. See the existing
`gemv_decode_sep21_v1b/FTZ_CERTIFICATE.md` and
[Apple MSL §§8.1–8.6](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).

Frozen BF16 projected gate/up/down, compiled activation and full-chain down
gates stay global **and per canonical route** relativeL2≤1e-4/cosine≥0.999999.
Zero-reference routes require exact zero norm. Retain per-dot F64 bound,
sign/zero/cancellation, strict-sensitive and exceptional results separately.
Passing a broad bound never rewrites a failed strict or BF16-stage status.

## Ownership, diagnostics and first qualification plan

The compact planner must inspect all four hidden rows for sticky4 even if every
ID is invalid. It does not mutate finite BF16 source bits. Live producer
operands turn nonfinite source values into positive zero in registers. Every
invalid original gate route is fully NaN-poisoned with bit1; invalid down
routes are fully poisoned with bits5 and never read their arbitrary
intermediate. Gate/down CTAs distribute excluded-route poison separately from
their cohort loop, through inverse-map sentinels, so poison cannot race a live
cohort destination. Duplicate valid routes compute separately with bit1.
There must be no early numeric bit5 during the gate-only diagnostic phase.

First scope is singleton target `verify()` with exactly4 physical rows; prompt,
prefill, R1, other verifier sizes, batch and trained MTP head paths remain
original. Full model GDN/PLE/QSA provisional-state and restore logic are reused,
not replaced by a new cache representation.

Before any integration, Root should run one-layer matched current-gathered,
native-bucket and N16 cohort controls on fixed synthetic overlap cases U10,
U20/U30 and U40, with permuted route order/ranks and the preserved failing
independent-R4 case. Pair dense/scalar taps against identical operands, retain
compiled SwiGLU/down consistency, source immutability/canaries, stale-plan and
poisoned-output replay, invalid/all-invalid/nonfinite/+0/corrupt-rank/duplicate
ownership cases. No quality threshold changes are permitted. Accumulate
≥150ms GPU warm per variant and balanced shipping-only samples with no CPU
buffer touches inside timings.

Only after primitive gates pass: source-bound alternative identity, unchanged
workspace/allocator closure, real normalized activation qualification, matched
whole-model MTP performance, frozen22 semantic/state/rollback cases and
acceptance measurements. A faster kernel that fails the original R4 BF16 gate
is rejected. If fidelity remains the obstacle, the distinct lower-risk control
is a compact metadata prelude feeding the existing coefficient-sharing M16
bucket operation; that preserves its arithmetic source, but still needs its
own exact stage/diagnostic comparison against current gather. Neither design
has an accuracy or speed qualification from this source-only proposal.
