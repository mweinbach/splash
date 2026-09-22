# Distinct parallel integer planner, preserving the exact six-stage native graph

Root rejected compact-v1 performance: old native≈0.332ms, compact≈1.0547ms,
current gather≈0.2237ms. Numerical/metadata/stage/safety gates passed. No v1
integration or further overlap runs are warranted. The source bottleneck is
`plan.metal:34–50`: one thread executes the512-expert×40-route stable map search
(20,480 repeated device I64 loads/comparisons), plus serial prefix/emission;
the other255 threads wait. The observed chain difference≈0.7227ms is a net
setup penalty, not an isolated measured planner duration.

The v2 proposal retains one256-thread CTA and the SAME global metadata,
native514 sentinels, pack, M16 producers, poison, preparation and scratch bases.
Only its integer work becomes parallel. No producer, numerical gate, rank
behavior, activation representation, global allocation or sidecar changes.

1. The first40 threads cache the40 I64 IDs into320 threadgroup bytes. Initialize
   metadata/sentinels in parallel, then synchronize. Check each original route
   for bounds and previous duplicates within its own ten-slot row; preserve
   duplicate ownership and sticky1. Invalid setup empties live ranges via one
   writer with sticky2 exactly as v1.
2. Thread t owns adjacent expert IDs2t and2t+1. Scan40 cached IDs for exact
   counts, storing a512-U32 threadgroup count/prefix array and device counts.
3. Each SIMD32 group has64 consecutive experts. Scan the pair sums with
   `simd_prefix_exclusive_sum`; eight group totals get a second SIMD scan.
   Write exact513 route offsets. Reuse the same integer scan scratch for
   `ceil(count/16)` pair sums, writing513 job offsets/jobCount. No floating
   operation or rank lookup occurs.
4. Each of40 route threads derives its stable packed position from
   `number(valid IDs<my ID)+number(earlier equal IDs)`, writes its unique map
   destination and inverse. Invalid entries retain UINT_MAX. This matches the
   original expert-ascending/flat-route-ascending order without512 serial scans.
5. After device visibility synchronization, parallel threads emit active jobs
   by the original upper-bound-search formula over job offsets. At most40 jobs
   need nine search steps. All514 unused records remain `{UINT_MAX,0}`.
   A40-route duplicate bucket produces three starts0/16/32, never four-row jobs.

Threadgroup storage is320 bytes IDs +2,048 bytes reused count/prefix scratch
+64 bytes for eight group totals/prefixes: at most2,432 bytes, with no new global
resource. Full metadata/scratch/immutable disjointness and exact one-CTA geometry
remain mandatory. Barriers separate cache/init, count/prefix stages and job
emission; every barrier is reached uniformly. Device serial dispatch ordering
continues to protect original pack and producers.

Total comparisons remain≈20,480 distributed count comparisons plus1,600
stable-route comparisons. The critical count path is80 cached comparisons per
thread, then integer SIMD scans and at mostnine job-search steps, rather than
20,480 serialized device comparisons. This is an algorithmic work/span
improvement, not a predicted speedup or a measured register/occupancy result.
The6/10/2 dispatch counts, active coefficient reuse, packed byte footprint and
2,000 native producer CTAs/layer remain unchanged.

An absolute requirement is beating current gather≈0.2237ms, not merely old
native≈0.332ms. If original other stages total T, the new planner must satisfy
`planner+T<0.2237ms`; old setup timing is unknown, so source cannot establish
that break-even budget. Retain all exact native metadata/padding/F32/scaled/BF16,
raw/prepared activation, compiled SwiGLU/full-down/diagnostic/replay gates plus
current gathered global/per-route1e-4/.999999 checks and malformed-rank
difference reporting. No timings occur on failed gates. Use a fresh v2 seal
and the same balanced≥150ms-warm shipping-only oracle; keep v1 untouched.

Secondary integration eligibility if a later primitive wins: qualified teacher
v5 currently selects gathered I8 at `FlashForward.cpp:1297–1304`, active-lane
batch at `FlashBatchForward.cpp:457`, and flattened joint verifier at
`FlashBatchVerify.cpp:573`. First whole-worker scope would be ONLY singleton
`verification && physicalRows==4`; retain every prompt, R1, nonverify/batch,
other verifier size and trained head call. `FlashWorker.cpp` actually calls
`forward_.verify` for positive MTP depth and normal `forward` for depth0.
Canonical combine must use the native scattered-down buffer for this branch.
No implementation is prepared because v1 lost.

Root supplied≈12ms MoE within37.1777ms verifier and46.7825ms whole cycle at
2.42857 mean outputs. Even eliminating all MoE work would leave≈34.7825ms/cycle
(≈69.82t/s), assuming acceptance and all other costs unchanged; halving this
MoE budget suggests≈59.55t/s, not the86t/s scenario for halving the ENTIRE
verifier. These are conditional source-budget illustrations, not end-to-end
performance or acceptance claims. No profiler/model/capture payload was read.
