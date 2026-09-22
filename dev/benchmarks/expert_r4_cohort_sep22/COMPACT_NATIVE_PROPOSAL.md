# Lower-risk compact metadata feeding the existing M16 bucket producer

Keep the rejected N16 F32 primitive rejected. Root reported one BF16 down
mismatch on route34, with per-route relativeL2≈0.0002096 exceeding the frozen
1e-4 threshold despite passing the global norm. No tolerance or association
change is proposed here. No new kernel, variation or worker has been built.

The first candidate should replace **only the five integer bucket setup
dispatches with one compact integer planner**, then reuse every current pack,
gate/up, poison, prepare-down and native down kernel. This is six dispatches
instead of ten. It provides the strongest first comparison because both MPP
producer sources and their entire surrounding BF16 preparation remain literal
controls. Three dispatches are possible only after additional store/poison
fusion; they are not the first lowest-risk candidate.

## Existing source contract

Source parent remains `build/prefill-qsa-twopass-sep21-worker-v1/source`.
`FlashMoEBuckets.cpp:147–180` emits histogram, prefix, stable map, pack, job
prefix and jobs. `FlashMoEBlocked.cpp:253–272` replaces pack with
`flash_moe_direct_a_pack` and adds63 padding rows. Store gate/up is one
dispatch (`FlashInt8ExpertStore.mm:313`). Down currently emits excluded-route
poison, `flash_moe_direct_a_prepare_down`, and native down scatter
(`FlashInt8ExpertStore.mm:334–344`). Total: ten.

Native `int8_expert_store_gate<16,4>` and `down<16,4>` bind device BF16 A with
exact dynamic valid bucket rows and device stored signed-I8 B; use whole
dynamic K, M16/N64, SG4 and multiply mode; apply the original F32 row scale
once, then original BF16 rounding/compiled SwiGLU and canonical scatter
(`flash_int8_expert_store.metal:259–374`). Leave those operations unchanged.
Native metadata already reuses coefficients for1–4 normal R4 expert rows;
compact setup removes its expensive preparation, not a previously absent
mathematical reuse capability. No padded-row physical speedup is claimed.

## Compact integer planner: same native format

R4 has at most40 flattened routes. Produce exactly the old histogram counts,
exclusive expert offsets, stable ascending-flat-route map, inverse map,
M16 job offsets, job count and jobs. Preserve original I64 ID bounds and
per-original-row duplicate diagnostics. Every valid duplicate remains a
separate route. No activation or coefficient values are read by this planner.

**Jobs must advance16 rows, not4.** The previous vector cohort's four-row
jobs cannot feed the native M16 consumer: its `valid_rows=min(16,end-begin)`
would overlap neighboring jobs. Use G=Σceil(count[e]/16), old tile_rows16
and old job_capacity `ceil(40/16)+511=514`. Existing blocked scratch has this
capacity; initialize unused job records to the old invalid sentinel. Normal
unique-per-row routing has G=U∈[10,40]. Duplicate-malformed buckets can have
more than4 rows and must retain the original M16 partitioning.

The old integer map counts valid IDs independently of rank lookup. Rank
permutation must not change ownership or stable order. Rank bounds remain
checked by the existing native job helper before coefficient pointer formation.
Do not silently add rank filtering to the compact plan: that would change the
old native map/poison behavior. The current gathered path explicitly poisons
bad-rank routes, whereas the native helper can skip a bad-rank job after bit1;
this pre-existing malformed-rank difference must stay visible in comparisons,
not be disguised as exact gathered equivalence. Healthy Full512/permuted-rank
input remains the supported numerical-fidelity comparison.

Reuse the existing GPU pack unmodified. It copies all finite BF16 words as
ushort bits, makes nonfinite operands positive zero with sticky4, checks every
original hidden row even if all IDs are excluded, fills dead packed rows and
63 guard rows with zero, and preserves route-map dead-tail sentinels.
For R4 it initializes103×2560×2=527,360 packed-input bytes. The existing down
prepare sanitizes only the live packed activation and initializes all103×640
rows, preserving finite signed zero/subnormals and padding.

The native packing layout, offsets, job boundaries, input base/alignment,
coefficient views, scale views and rank views must match the old native control
exactly. No new sidecar, activation quantization, coefficient cache, packed
matrix representation or scratch allocation is involved.

## Dispatch options and actual cost

| Candidate | Dispatch sequence | Dispatches/layer | Main additional risk |
|---|---|---:|---|
| **First: compact integer setup** | compact metadata → original pack → original G/U → original excluded poison → original down prepare → original native down | **6** | Only new metadata generation |
| Later combine preparation | compact metadata → original pack → original G/U → combined down prepare/poison → original down | 5 | Integer/BF16-copy helper fusion and phase checks |
| Later combine metadata/pack | compact metadata+pack → original G/U → combined down prepare/poison → original down | 4 | A single planner CTA must copy527KB; lost packing parallelism may erase savings |
| Later three-stage version | compact metadata+pack → G/U with prepared-output store → down wrapper with excluded poison | 3 | Producer output-store policy/context changes, although dot math can remain native |

The first six-stage layout changes no producer or preparation kernel.
Original setup launches512 histogram +1 prefix +512 stable-map +103 pack
+1 job-prefix +3 jobs CTAs=1,132. Compact metadata plus original pack is
1+103=104 CTAs, removing1,028 setup CTAs and four dispatches per layer.
Across48 layers that removes49,344 setup CTAs and192 dispatches relative to
native bucket control. Current direct gather still uses only two dispatches;
the compact candidate has four more than that shipping comparator.

Native gate/down launch remains `{10,40,1}`/`{40,40,1}`,128 threads,2,000
producer CTAs/layer with50×G live and the rest bounded early returns. No
per-output metadata rebuild or new fixed N16 software-dot computation is used.
Coefficient code footprint remains4,915,200×G logical bytes/layer, plus
15,360×U row-scale bytes. Current gather can already get cache reuse; no DRAM
or end-to-end speed promise follows from this footprint.

Five/four stages combine independent down tasks safely: one CTA per packed
row sanitizes its own640 values while, for original route indices<40, also
poisoning that route's canonical down if inverse is invalid. Packed activation
and canonical down are disjoint. Numeric bit5 occurs only after gate/up.

Three stages cannot be obtained by simply deleting preparation. Original
native down requires a device buffer with nonfinite intermediate values already
zeroed. Sanitizing that global input independently in every output-tile CTA
would create read/write races; a threadgroup barrier cannot order other CTAs.
A local fallback would change tensorA's address space and opaque lowering.
The plausible three-stage version therefore stores prepared positive zero for
nonfinite G/U outputs and adds excluded-route poison in the down wrapper.
That needs separate raw-G/U and prepared-G/U stage witnesses; it is not literal
producer-source identity and may change compiler lowering. Do not implement it
before the six-stage result justifies the added risk.

## Required comparisons before timing or integration

First prove compact metadata byte-for-byte against the old six-dispatch setup:
all512 counts,513 offsets/job offsets,40 map/inverse entries, native M16 jobs,
job count and sentinels under U10/20/30/40, oldmixed, slot/rank permutations,
invalid IDs and duplicates including buckets>16. No model/capture payload is
required to prepare those source-generated cases.

Keep the same baseline/native buffer bases for alternating plan variants, so
both run the same original native producer with identical packed BF16 inputs.
Require exact native-control raw/scaled F32 and BF16 gate/up/down probes,
compiled SwiGLU, prepared intermediate and canonical full-chain down, sticky
diagnostics, canaries, source immutability and poisoned-plan replay. Compare
current direct gather separately with its existing unchanged global/per-route
BF16 gates. The previously failing route34 fixture remains mandatory.

Gate-only canonicalization for invalid IDs needs an untimed phase-correct
observer: existing `flash_qmv_probe_unpack_activation` ORs5 on exclusion, so it
cannot be used to claim shipping gate-only bit1 behavior. Preserve the native
producer's actual diagnostics and report observer behavior separately.

Validate every metadata/operand extent, alignment, lifetime and writable/
immutable disjointness. Publish metadata readiness only after the planner
finishes; rely on the sealed serial compute pass between planner, original
pack, gate, poison, preparation and down. Missing/reordered/partial setup must
fail closed. All numerical/state gates stay fixed. Only after exact native
stage parity and gathered fidelity pass: ≥150ms balanced shipping-only timing,
then genuine normalized decode input, frozen22 state/rollback semantics,
whole-model acceptance/performance and source-bound identity handling.
