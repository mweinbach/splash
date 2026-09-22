# FIRST six-dispatch compact integer native M16 R4 component

Replace only the five integer histogram/prefix/stable-map/job-prefix/job stages
with one compact R4 planner. Original BF16 pack, native whole-K M16/N64/SG4
gate/up, excluded-route poison, BF16 down preparation and native down scatter
remain literal original kernels. No software dot, new N tile, fusion,
quantization, sidecar, coefficient cache or whole-worker integration is allowed.

Use unchanged FlashMoEBucketParams/FlashMoEBucketJob layouts, physical rows4,
selections10, experts512,40 routes, tile_rows16 and native job_capacity514.
Jobs advance16, including duplicate buckets with more than16 original routes.
Compare all512 counts,513 offsets/job offsets,40 map/inverse entries,514 jobs
and every sentinel/jobCount/sticky value. The M8 allocator's last two backing
job records remain untouched. Dead route-map tail is authoritative after the
same original pack, which initializes it. Rank filtering is forbidden in the
integer planner because original metadata includes every valid ID regardless
of rank. Existing malformed native-rank skip/bit1 behavior differs from current
gathered NaN/bit5 behavior and is explicitly reported, not treated as parity.

Old/compact plans use the SAME original baseline scratch bases, original
producer parameters and the SAME probe destination bases. Alternating stages
overwrite/rebuild exact identical inputs rather than changing address-space or
padding layouts. Native packed BF16 inputs contain103 rows: every finite source
word is bit-preserved, nonfinite becomes positive zero with sticky4, dead/padded
rows are zero, and all four hidden rows are inspected even if all IDs invalid.
Raw native BF16 G/U activation is observed before preparation; prepared
intermediate is observed separately. Literal compiled SwiGLU and down-control
pairings remain mandatory.

Require exact old/compact raw/scaled F32 and BF16 gate/up/down probes, raw
compiled activation, prepared activation, canonical full-chain down, metadata,
all103 packed/prepared rows, diagnostics, canaries, source immutability and
poisoned-plan replay. Current gathered SG4 remains separate control with frozen
global AND per-route projected/activation/down relativeL2≤1e-4 and cosine≥.999999;
zero-reference routes require exact zero norm. Include the rejected software-dot
oldmixed case (route34/id431), U10/20/30/40, permutations and malformed metadata/
ID/duplicate/rank cases. Per-dot F64/sign/cancellation/strict/exceptional results
remain visible; passing norms do not erase their failed statuses.

Every writable metadata/operand view has sufficient Shared aligned extent,
pairwise disjointness and disjointness against immutable payload/ranks. No new
allocation is added to the native graph; test outputs/guards fit the admitted
bounded oracle allowance. The compact planner requires exact one-CTA256-thread
geometry. Invalid geometry clears fixed validated live metadata to empty via
one writer with sticky2, so original consumers cannot read stale ranges/jobs.
The sealed freshly compiled backend uses ordered serial compute dispatches:
planner→original pack→original gate→original poison→original preparation→down.

All metadata, raw/prepared stage, gathered-fidelity, malformed, alias, stale
replay, immutable/hash checks precede timings. Root alone invokes GPU/model
operations. Refuse timing on any new component gate failure; every variant
accumulates≥150ms GPU warm work and balanced shipping-only samples (default18,
three-way multiples3), without CPU data/diagnostic touches inside either loop.
No primitive result qualifies actual decode input, MTP state/rollback, frozen22
semantics, acceptance or end-to-end performance. Those remain Root's later gates.
