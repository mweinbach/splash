# Independent private direct-gathered MPP candidate

This candidate preserves the original signed-I8 MPP producer descriptor and
uses one valid original row/slot per tensorA instead of sorting into expert
buckets. C2 remains unchanged and unpromoted. The gathered candidate is
unqualified until Root measures exact old-MPP parity and service quality.

The component compiled with warnings as errors at
`build/prefill4k-allrows-gathered-mpp-component-v3` (Store object, AIR/metallib).
The native one-layer oracle compiled and passed CPU checks at
`build/prefill4k-allrows-gathered-mpp-one-layer-v2`. No GPU work, model loads,
model payload reads or hashes were performed by this subtree.

## Store integration

Compose `prefill4k_allrows_gathered_mpp.transform(relative,text)` after the
original all-row Store transform, never after C1/C2's Store transforms. Stage
all returned `extra_files()` before compiling. The transform adds:

```cpp
bool store->gatheredMPPEnabled();
store->addGatheredMPPGateUp(graph, layer, originalMixedBF16, originalIDsI64,
    canonicalIntermediate, diagnostics, rows, 10);
store->addGatheredMPPDown(graph, layer, canonicalIntermediate, originalIDsI64,
    canonicalExpertDown, diagnostics, rows, 10);
```

Rows1..16 only; canonical layouts remain `[rows,2560]`, `[rows,10,640]` and
`[rows,10,2560]`. The unchanged combine reads original IDs/route weights and
canonical expertDown. The strict `SPLASH_FLASH_ALLROWS_GATHERED_MPP=0/1` flag
freezes at construction and is rechecked by the getter. Flag0 retains original
MPP control identity. Flag1 appends the independent gathered policy to the
numerical derivative seed. Validate `gathered_mpp::requested()` before backend
creation in the composed Worker. Dedicated Store graph-construction counters
are `gathered_mpp_{gate_up,down}_graph_{calls,rows}`; they do not assert GPU
completion. No new Store buffers, mappings or global workspace are allocated.
The reused host view guards have an independent namespace and macro-safe
invalid-ID constant; no QMV reducer is called.

The transformed graph's kernel names are checked against shipping declarations
by CPU source tests. Actual dispatches are gate/up `{10,rows,10}`, down
`{40,rows,10}`, both128 threads:500 CTAs per physical row. The parent executor
must select this branch only for rows<=16 and retain the old bucket route for
other row counts. Source/metadata guards require contiguous E512/H2560/I640/K10,
sufficient aligned Shared views, pairwise disjoint inputs/IDs/output/diagnostics,
and disjointness against every immutable layer/base/rank allocation. Graph
MetalBuffer copies retain mappings until graph disposal.

## Descriptor and finite input policy

Finite tensors use unmodified source device BF16 A with extents `{K,1}`,
strides `{1,K}`, and device signed `int8_t` B with extents `{K,64}` and the same
strides. The descriptor is exactly `M16/N64/Kdynamic`, false/true/false,
mode **multiply**, SG4, cooperative F32 destination. Each projection runs one
whole-K operation. Dot × one stored F32 row scale → BF16 projection remains
fixed. Gate/up then use the old compiled BF16 fast-exp sigmoid, BF16 gate ×
sigmoid, and BF16 × up. Down emits canonical BF16 rows directly.

Every group scans all its source activation bits. A threadgroup atomic/barrier
makes the finite/fallback branch uniform across all four SIMD groups. Malformed
nonfinite operands are locally staged to positive zero with sticky4; all finite
BF16 bits, including negative zero and subnormals, are copied as ushort bits.
Finite source tensorA remains in device storage. Static local fallback storage
still occupies5,120 bytes gate/up or1,280 bytes down plus4-byte flag, even when
unused; scan/barriers and occupancy may cost enough to erase dispatch gains.
The runtime-phased projection probe reserves5,124 bytes in both phases.

Signed I64 IDs are validated before casting/rank lookup or forming B pointers.
Duplicate valid IDs flag1 but remain separately computed. Gate still scans H
when the route ID/rank is invalid, then fully poisons its canonical activation
tile with NaN/ID1. Invalid down routes skip arbitrary intermediate reads and
poison NaN/ID|numeric5. Missing/corrupt Full512 ranks flag1. The standalone raw
projection probe deliberately poisons invalid taps with NaN/5; malformed-input
stage parity is therefore explicitly pending, not assumed from valid cases.
Malformed geometry flags2. Diagnostics remain atomic OR/sticky.

Changing A's valid row count to1 and old local row slots0..15 into logical row0
may change opaque MPP lowering/tile-K/reduction despite descriptor equality.
The threadgroup fallback invokes a different SDK address-space implementation
than finite deviceA. No exact tree or speed claim follows from this source.

## Frozen one-layer Root oracle

```sh
make -f dev/benchmarks/prefill4k_allrows_gathered_mpp_oracle.mk -j4 \
  GATHERED_ORACLE_BUILD=build/prefill4k-allrows-gathered-mpp-one-layer-v2 \
  gathered-mpp-one-layer-cpu
```

Only Root invokes GPU mode, using a **new report path**:

```sh
build/prefill4k-allrows-gathered-mpp-one-layer-v2/oracle --gpu \
  build/prefill4k-allrows-gathered-mpp-one-layer-v2/splash.metallib \
  /absolute/path/to/Full512-store 0 2 /absolute/path/to/new-report.json \
  --pattern mixed --pairs 3
```

Arguments are library, certified store, layer0..47, rows1..16 and NEW report.
`--pattern mixed` recreates the normalized synthetic/ID frame used in the
previous oracle, including its known baseline strict failure. `repeated` selects
the same top10 experts across all rows, including expert0/511, forcing the old
bucket tensor to have validRows=rows and logical row slots0..rows-1. `spread`
and `permuted` exercise sparse occupancy and canonical slot ownership.
`--rank-shift 0..511` optionally makes a cyclic synthetic ID→persisted-rank map
before its immutable witness is taken; it does not claim original-model route
semantics. `--hidden` and `--ids` accept exact-sized BF16/I64 source fixture
files, bounded to2MiB. `--pairs` is1..9. Genuine normalized decode activation
provenance remains independently required before promotion.

GPU mode maps only the certified2,524,446,720-byte selected layer, shares its
codes/scales/ranks between variants, and validates the same metadata, whole/plane
hashes, padding, code range, finite positive scales and immutable file/mapping
as the previous one-layer loader. Exact ledger is2,524,463,104 bytes plus128MiB
bounded scratch reserved until allocations finish. There is no48-layer Store
or original target Q4 model load. Those scans occur only in Root's GPU process.

Old MPP runs the actual bucket producer, gate/up, down sanitization and scatter.
Its canonical activation is shared by both standalone-down projection probes.
For each projection, untimed taps expose canonical raw F32 dot, late-scaled
F32 and BF16 projection. The frozen gate requires **zero U32/U32/U16 mismatches**
against the old MPP tap, and exact diagnostic sentinel. It also requires exact
fused activation and own-input full-chain down agreement, plus exact compiled
standalone `addSiLUMultiply` on the tapped rounded gate/up. Every comparison is
bitwise; no norm tolerance can admit a mismatch.

Both variants retain the independent compensated F64 approximate reference,
explicit F64 uncertainty, conservative opaque-MPP gamma(K) absolute bound and
finite/sign/exceptional review. Existing old-MPP F64 strict failures remain
full failed Summary/first-failure telemetry, with
`baseline_f64_strict_pass:false` and a separate candidate strict status. Exact
old-MPP reproduction is a separate pass axis; it never relabels an old strict
failure as F64 qualification or changes a tolerance.

After capture, **all** bucket scratch planes and candidate outputs are poisoned
with0xA5 and only the candidate's two dispatches rerun. Both outputs must match
their prior bytes and diagnostics, demonstrating no stale bucket/downPrepare
dependency. Timed matched warm alternating chains begin only after exact
old-MPP parity, frozen finite/sign/absolute bounds, producer stages, canaries
and input immutability pass. Every timed submission's diagnostic is checked;
final outputs must retain initial bytes. Timings include bucket prelude/
downPrepare/scatter for old MPP and only two gathered dispatches for candidate.
Probes, comparisons, metadata and hashes are excluded.

Exit0 means the frozen **exact old-MPP comparison** and timed checks passed.
It does not imply `baseline_f64_strict_pass`, actual decode activation quality,
whole-model quality, acceptance, rollback/state continuity or promotion.
Exit2 writes mismatch/failure telemetry with timings blocked; exit1 is setup
failure. The original C2 failure reports, source and policy are unchanged.

## Remaining qualification domain

Root should first pair rows1/2/4/16, then remaining rows1..16, with repeated,
mixed/sparse and permuted routes. Nonidentity ranks, invalid/duplicate IDs,
missing/corrupt rank, hidden nonfinite with all IDs invalid, intermediate
nonfinite, signed zero/subnormal and canary fixtures remain mandatory. This
initial finite oracle rejects malformed IDs rather than claiming that suite
already ran. It reports actual modulo128 A placements, initially64-byte guards;
all even BF16 residues0..126 and independently varied old packed/control offsets
still require a bounded follow-on before claiming the complete2-byte-aligned
host admission domain. The SDK does not document an ordinary BF16/I8 tensor
128-byte base restriction; do not invent one without measurement.

Only after exact projection parity on source-attested normalized activations
should Root run the frozen22-case model quality, greedy/acceptance, state/
rollback and full-request performance tests. Unload all test model processes
when complete. This subtree performed CPU compilation/source checks only.
