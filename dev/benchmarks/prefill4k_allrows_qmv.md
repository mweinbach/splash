# Private Full512 gathered I8 QMV for target rows 1–16

This component prepares a small-row target-only alternative to the all-row
Full512 MPP route. It removes the bucket histogram/offset/pack/job/scatter
prelude from those projections, leaving two gathered dispatches: gate/up with
SwiGLU, then down. No speed or model-quality claim is established yet.

The initial component compiled and passed CPU guard checks at
`build/prefill4k-allrows-qmv-component-v2`. No GPU work or model load occurred.
The build contains the CPU helper, a standalone two-kernel metallib, shader AIR,
and the transformed Store object. It is **not a runnable worker**. The composing
parent adds executor routing and the AIR to its new private full worker build.

## Integration API

Compose `prefill4k_allrows_qmv.transform(relative,text)` after the existing
`prefill4k_allrows_store.transform`. Stage all entries from `extra_files()` into
the same private source tree before compiling. Only Store `.hpp`/`.mm` are
transformed; the original Store MPP methods and the trained MTP are untouched.

```cpp
// Canonical BF16 input: [rows,2560]; original IDs: I64[rows,10].
// Intermediate and expertDown are existing executor-owned scratch.
if (rows <= 16 && store->gatheredQMVEnabled()) {
  store->addGatheredQMVGateUp(graph, layer, mixed, originalExpertIDs,
      canonicalIntermediate, diagnostics, rows, 10);
  store->addGatheredQMVDown(graph, layer, canonicalIntermediate, originalExpertIDs,
      canonicalExpertDown, diagnostics, rows, 10);
  // Existing addCombine reads canonicalExpertDown and originalExpertIDs.
} else {
  // Existing all-row Full512 bucketed MPP route.
}
```

The **strict** `SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV=0/1` flag is frozen when the
Store is constructed and checked again through `gatheredQMVEnabled()`. Unset
selects flag0. Any other value throws. Call `gathered_i8_qmv::requested()` before
creating the backend in the combined Worker so malformed configuration is
rejected before a device probe. Keep all existing all-row omission/Full512
startup checks mandatory for both flag values.

Flag0 retains the original MPP numerical derivative hash. Flag1 appends
`small_row_policy=<kPolicy>\n` to the identity seed before hashing. The shared
header/Python policy is:

```text
private-gathered-signed-i8-bf16-input-f32-lane32-strided-dot-late-row-scale-bf16-dots-swiglu-sg4-c1-rows1to16-v1
```

Expose that policy and flag separately in status. The existing general Store
semantics string still identifies the MPP large-row/control route; it alone
does not describe the flag1 small-row reduction. Existing aggregate hit/row
counters include QMV; four new graph-construction counters identify it:
`gathered_qmv_gate_up_graph_calls`, `gathered_qmv_gate_up_graph_rows`,
`gathered_qmv_down_graph_calls`, `gathered_qmv_down_graph_rows`.
They do not claim successful GPU completion.

## Data, staging, memory and ownership

Read `rank = ranks[originalExpertIDs[row,slot]]`, never use the original ID as
the persisted rank. The 512-entry mapping is checked by the shader. Every
original ID remains signed I64 until validated in `[0,512)`, so `2^32` cannot
wrap into a valid ID. One SIMD group computes each output column; four SIMD
groups form a 128-thread group. K is lane-strided with stride32, reduced in F32
with `simd_sum`. Gate/up and down each multiply the complete F32 dot by exactly
one positive finite F32 output-row scale. Gate/up round individually to BF16,
then use the original compiled fast-exp BF16 sigmoid and two BF16 products.
The canonical outputs are `[rows,10,640]` and `[rows,10,2560]`.

Exact group grids are gate/up `{160,rows,10}` and down `{640,rows,10}` with
`{128,1,1}` threads, totaling **8,000 CTAs per physical row**. This is more CTAs
than the capped tiny M16 route; it is not a launch-count optimization. Its
potential benefit is removing the separate bucket prelude and avoiding matrix
arithmetic on padded M16 rows. The GPU experiment must determine whether that
outweighs the extra QMV threadgroup overhead.

This changes the F32 reduction order versus MPP M16. Matching operand/staging
types does **not** establish numerical parity, greedy output parity, speculative
acceptance parity, or throughput. CPU `exp` cannot qualify Metal fast-exp
rounding thresholds.

No new model buffers, rank buffers, mappings or workspace are allocated by this
component. The Store methods use its immutable original Full512 buffers
internally. Graph buffer handles retain each base allocation and readonly host
mapping until graph disposal. No raw immutable handle accessor is exposed.
Geometry/view checks admit only rows1..16, E512, H2560, I640 and K10, sufficient
Shared views with proper alignment/address bounds, and pairwise disjoint
input/IDs/output/diagnostics. Every graph operand must also be disjoint from
every Store immutable base/rank allocation. Current governor reservations
remain unchanged. The parent may retain allocated bucket scratch for other
row counts; bypassing its dispatches does not assert those bytes were released.

## Invalid fixtures and paired qualification still required

Nonfinite hidden or intermediate BF16 values are replaced locally with zero
and sticky diagnostic4, preserving finite signed zero. Hidden values are
inspected even when every original ID is invalid. Invalid IDs or corrupt/missing
Full512 ranks flag1; the affected canonical route is poisoned with NaN, and
down flags5 as the existing excluded-route poison does. Duplicate valid IDs
flag1 and compute finite canonical routes normally; the existing combine
emits final NaN. All diagnostics use atomic OR so prepopulated bits survive.

Root should first run a primitive paired oracle in the **same process/store**,
with the existing `addMoEBlockedPack` + Store M16 gate/up/down as reference and
the two new gathered methods as candidate. Inspect both canonical activated
and down buffers, not only combined output. Use rows1,2,3,4,8,15,16; concentrated
and spread original IDs; nonidentity rank fixtures; uneven scales; cancellation
near zero; signed zero; and the compiled sigmoid threshold. Explicitly cover
IDs -1,512,2^32, duplicates, missing/corrupt ranks, nonfinite hidden with all IDs
invalid, nonfinite activated inputs, prepopulated diagnostic bits, guard regions
and readonly input digests. Record stage-specific diagnostic differences rather
than assuming malformed-input parity from the CPU tests.

Then run same-build flag0/flag1 matched 22-case service quality, state continuation,
same coding-prompt acceptance, uncached prefill/TTFT and warm decode measurements.
Retain large-row prefill/control coverage and show dedicated QMV counters prove
small-row execution. Do not promote based on a component timing alone. Stop and
unload the test models when finished.

## CPU-only reproduction

Use a new build path if input sources or component files have changed; staging
refuses to overwrite an existing destination and rejects repeated transforms.

```sh
make -f Makefile -f dev/benchmarks/prefill4k_allrows_qmv.mk -j4 \
  PREFILL4K_QMV_BUILD=build/prefill4k-allrows-qmv-component-v3 \
  prefill4k-qmv-component-cpu
```

The CPU helper verifies ABI/launch extents, unsupported row/selection counts,
canonical sizes, insufficient/misaligned/overflowing/overlapping views,
original-I64 ID limits, nonidentity ranks, corrupt ranks, duplicate diagnostic
separation, sticky bits, BF16 ties/signed zero, finite-operand sanitization and
the importance of late scaling. The Python check verifies source drift,
duplicate-transform rejection, unchanged original MPP methods/MTP, unchanged
allocation/mapping call counts, immutable-disjoint guards, policy agreement,
and conditional derivative separation. Neither loads model payloads, creates a
Metal backend, nor submits GPU work.
