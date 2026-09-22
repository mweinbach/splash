# Proposed native R8/R16 component; no extension build authorized

Extend only the qualified parallel INTEGER planner's fixed route count to two
new bounded components: physical R8/S10 and R16/S10. Keep the original native
M16/N64/SG4 producers and original pack, excluded poison and prepare-down
kernels literally unchanged. The comparison target is the current ten-dispatch
native path. Its existing gathered cap4 does not cover R8 or R16.

The R4 singleton-verifier worker remains a separate optional profile. This
proposal does not broaden that worker's caller scope or enable a batch route.
After component qualification, a separate reviewed selector can distinguish
ordinary batch-forward lanes8/16 from batch-verifier flattened rows8/16. The
current source uses the native path in FlashBatchForward.cpp and
FlashBatchVerify.cpp when these dimensions exceed the frozen gather cap4.
Any future selector must exclude all prefill, singleton AR/R1, MTP seed/head,
and unrelated physical widths, and leave state/rollback ownership untouched.

| Native metadata or storage | R4 qualified | Proposed R8 | Proposed R16 |
|---|---:|---:|---:|
| Routes (R times10) | 40 | 80 | 160 |
| Native M16 job capacity (ceil(routes/16)+511) | 514 | 516 | 521 |
| Existing M8 job backing capacity | 516 | 521 | 531 |
| Existing backing records beyond native capacity; untouched | 2 | 5 | 10 |
| Packed/prepared operand rows (routes+63) | 103 | 143 | 223 |
| Cached I64 IDs in threadgroup bytes | 320 | 640 | 1280 |
| Total threadgroup bytes (IDs+512U32 counts+16U32 scan values) | 2432 | 2752 | 3392 |
| One-expert worst-case duplicate bucket jobs | 3 | 5 | 10 |
| Native matrix STEP | 16 | 16 | 16 |

All512 counts,513 offsets and job offsets, every route-map/inverse entry,
jobCount and all516/521 native job records must exactly match native metadata.
Stable packed order remains expert ID then original canonical route index.
No rank filtering is allowed: native metadata includes every valid ID, even
when a malformed rank causes an original producer to skip it. Report that
existing native versus gathered malformed-rank difference explicitly. Duplicate
diagnostics apply to equal IDs in the same original row's ten slots. Negative,
out-of-range and very large I64 IDs stay excluded with the native sticky bit.
All-dead metadata must empty old ranges/jobs and preserve all inactive sentinels.

Use one CTA256 as in the qualified component. Load cached IDs cooperatively:
routes80/160 fit in its first80/160 threads. Each thread still owns two of512
expert counts and scans the cached IDs. Original-route owners compute the
stable position as the number of valid smaller IDs plus prior equal IDs.
The original two-level integer SIMD prefix scans and barriers remain, including
the barrier that prevents the second scan's scratch reuse from racing the
first scan's readers. Each expert emits disjoint STEP16 jobs. For a legal
80/160-route duplicate bucket, starts0,16,32,48,64 and through144 respectively
must match old native jobs, even though such same-row duplicates set sticky1.

This costs40/80/160 global I64 ID loads, then cached integer work. The expert
count census is512*routes cached comparisons, with two experts per thread;
stable-order work is at most routes squared cached comparisons distributed
across original-route owners. New global buffers, allocations, readback,
activation quantization, sidecars and coefficient caches remain forbidden.

The native ten-stage chain contains five metadata dispatches plus the original
pack and four float stages. Replacing its five metadata dispatches with one
removes four dispatches and1028 launched metadata CTAs per layer at all three
widths: histogram512, prefix1, stable-map512, job-prefix1, job emission3 become
plan1. Across48 expert layers that is192 fewer dispatches and49344 fewer
metadata CTAs. Original matrix, padding, poison and preparation work stays the
same. This is a launch/setup reduction, without a claim of fewer matrix tiles.

Root's measured R4 old-native minus parallel-compact GPU median is0.124 to
0.128ms per layer across six synthetic patterns. Applying that difference to48
layers gives a conditional5.95 to6.13ms per native model pass. It is a proposed
measurement target for R8/R16; larger cached integer work, real route overlap,
clock state and graph execution can change it. No batch speed or end-to-end
throughput has been measured for either extension.

Before timings, use admitted existing native scratch and probe destinations
for both old and compact controls. Compare metadata, all packed padding,
original raw/scaled F32 and BF16 gate/up/down projections, literal compiled
SwiGLU, raw and prepared activation, full-chain canonical down, diagnostics,
guards, stale-plan replay, malformed geometry and source immutability. Exercise
U10/20/40/80 for R8 and U10/40/80/160 for R16, mixed overlaps, permutations,
first/last canonical slots, >16 same-expert buckets, all-invalid IDs, same-row
duplicates, poisoned scratch, nonfinite hidden rows even with no live routes,
rank holes/duplicates and the earlier rejected R4 fixture embedded unchanged.
If gathered probes are retained as a secondary diagnostic control, preserve
the existing global AND per-route L2<=1e-4/cosine>=.999999 gate, with exact zero
norm for zero-reference routes. Native-control exactness remains mandatory.

Only after every pre-timing gate passes should Root time balanced old-native
and compact shipping chains, each warmed for at least150ms GPU time with no
CPU operand/diagnostic reads in either loop. Source/dependency closure,
uniform geometry and metadata->pack->gate->poison->prepare->down ordering need
independent CPU review first. Root alone accesses GPU/model payloads.

Whole-model adoption additionally requires real normalized activations,
batch lane compaction/future/cache lifetime and MTP accept/truncate/rollback
proof, unchanged frozen22 semantic inputs/graders, actual acceptance and
matched end-to-end timing. These gates are separate from a synthetic primitive
win. No R8/R16 source or executable has been built by this proposal.
