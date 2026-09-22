# Independent chunk-local GDN review

CPU/source/report review only. No model tensor payload, Metal device or GPU
execution was used for this review. Sealed v6 source/library and the actual
capture wrapper remain unchanged.

Root's actual layer0 report reproduces recorded native cold state and BF16
recurrence exactly. All three existing actual-input quality screens pass, but
39/48 heads select fullRows replay: 15 range, 14 cancellation, 10 norm, zero
nonfinite; those reason sets are disjoint in this capture. Cold timing is
3.36729 ms candidate versus 1.89660 ms native; native-carried timing is
3.36825 versus 1.89635 ms. This rejects full-head replay for performance.

The report has head-level OR flags, so it does **not** establish chunk/tile
fallback frequency. A single unsafe chunk currently causes all 64 chunks in
that head to be replayed. Do not infer that 81.25% of chunks are unsafe.

## Recommended bounded prototype

Use independent `(lane, head, V32 tile)` workgroups, preserving the current four
SG8 workgroups per head and all current MPP dimensions/operand types. Each
workgroup processes chronological T32 chunks and owns exactly its 32 independent
value-coordinate state rows. There is no recurrent coupling between value
coordinates: their shared Q/K/decay/beta inputs are immutable. Whole-chunk raw
range failure selects all four V32 tiles; state/norm/cancellation failure may
select only the tile whose unchanged predicate failed. The `.095` cancellation
ratio, range margins, nonzero-square underflow checks and all other conservative
scalar predicates remain unchanged.

Materialize static range reasons by `(lane,chunk,head)` in preparation rather
than OR-ing them across all chunks. Cache W norm/reasons exactly as v6. Merge
cached W reasons into each tile's local chunk reasons at the same mathematical
phase as v6; do not drop them because the incoming state appears small.
Cancellation and state/output/update nonfinite reasons belong to the current
tile/chunk. A local threadgroup atomic OR followed by a uniform barrier gives
all SG8 threads one decision. No cross-workgroup polling or spin barrier is
needed or legal.

Defer **every** persistent recurrent-state write until that decision. The
current state-update destination contains 32*128/256 = 16 F32 elements/thread.
Retain the candidate final-state values in that cooperative register tensor,
classify all of them, merge local flags, and commit only if safe. Keep the state
matrix input as the existing device-F32 tensor to avoid adding an unqualified
address-space/backend change. If any local predicate rejects, persistent state
still contains the exact hybrid incoming F32 seed and the native branch loads
it directly. Native then overwrites **every** BF16 output slot of the current
T32/V32 tile and commits its final F32 state. Earlier speculative output writes
must not be consumed before this overwrite/commit. This makes an additional
16 KiB snapshot unnecessary in the timed path; an audit-only snapshot can prove
incoming-seed provenance.

If implementation instead speculatively writes state, it needs an immutable
incoming tile snapshot and complete restore before native. Never replay from a
partly committed WY state. The no-copy proposal is valid only when source and
runtime tests prove no persistent write precedes the final uniform decision.

Execute native for four V8 waves under SG8, each with two chronological native
Time16 subtiles. For each value row, preserve four key coordinates per SIMD
lane, decay-before-memory, i=0..3 scalar addition order, SIMD32 reduction,
`(value-memory)*beta`, state update expression, query reduction and single BF16
boundary. V8 threadgroup tiling changes cooperative input loading but no
per-value mathematical dependency. Do not claim exactness until native-branch
runtime proof verifies those bytes from the identical incoming seed.

## Threadgroup and count bounds

Existing SG8 WY scratch is 8,516 B. Native V8/Time16 staging with original
Kstride136/Vstride16 requires 4,352 B Q + 4,352 B K + 512 B V + 64 B decay +
32 B beta = 9,312 B. A deliberately shared scratch arena therefore needs
`max(8516,9312)` plus new local reason/alignment bytes. If an incoming snapshot
is retained, total is 16,384 + 9,312 = 25,696 B before those extras, within
32 KiB. Declaring snapshot, WY arrays and native arrays independently would use
34,212 B and exceed that budget. Native Time32 staging plus snapshot is
35,008 B, also too large. Two Time16 subtiles avoid this problem. Alias phases
explicitly; do not rely on compiler lifetime allocation to reuse distinct TG
arrays. The compiled pipeline's actual static TG memory remains a Root check.

There are 64*48 = 3,072 chunk-heads and 12,288 chunk/V32 tiles. Optional static
chunk-reason storage is 12,288 B; optional committed tile/chunk-reason storage is
49,152 B. Keep counters outside route identity. The old report's 39 flagged
heads do not give the actual frequency: at least 39 head-chunks were involved,
but as many as 2,496 may have been. For a tile-local interpretation of these
disjoint old reason sets, at least 15*4 + 14 + 10 = 84 tile-chunks triggered;
the new hybrid trajectory may change later reasons, so this is an old-report
diagnostic bound, not a forecast. Original replay does 39*2048 = 79,872 native
head-row steps; a sparse chunk policy could remove much of that work, but no
speedup estimate is justified until Root measures actual chunk reasons.

A whole-head chunk policy retaining four separate V32 workgroups requires an
encoder barrier after all tiles discover flags, then snapshot/restore and
native replay before the next chunk. With preparation once and four phases per
chunk that is at least 257 dispatches; separating the guard would add more.
Snapshot read+write traffic alone is 64*2*3,145,728 = 402,653,184 issued bytes.
A single workgroup per full head avoids that synchronization but serializes its
four WY tiles and provides only 48 independent workgroups. That may reduce
available parallelism on the 80-core GPU. Neither alternative is the preferred
first performance experiment.

## New proof contract

The old whole-head byte-identity certificate does **not** transfer. A local
native chunk is exact only against frozen native executing those same raw
rows from the **identical current hybrid incoming F32 tile state**. Earlier WY
chunks may already differ from original all-native history. Report scoped
`native_chunk_same_seed_byte_exact` separately from whole-sequence F32/BF16
quality and the strict F64 fixture metrics. No universal whole-history
accuracy certificate is supplied by the selector.

Required new evidence: forced native all-chunks must reproduce original native
fullRows bit-for-bit; forced alternating WY/native chunks must pass scoped
same-seed branch identity; guard-selected range/postzero/norm/cancellation,
tail, multiple lanes, mixed tiles and carried continuations must pass. Native
fallback must overwrite complete state/output/audit coverage with no surviving
speculative values. Preserve canaries, immutable source/initial/audit-seed
hashes, sticky diagnostics and unchanged-other-tile proof. Continue reporting
raw failed F64 transform/delta fields as failures or unused-on-native, never
silently replace them with native equivalence.

Whole-sequence gates remain F32 relative RMS 1e-4 and max absolute
5e-4*max(1,reference_peak), BF16 relative RMS .003 and max absolute
.004*max(1,reference_peak), without widening. Run actual captured cold and
carried trajectory screens from both native and candidate carries, with
whole-sequence comparison against original native as well as scoped branch
identity. Only then run inclusive >=150 ms warmup + exactly balanced 10-pair
ABBA timing, recording per-chunk/per-tile reasons and native rows replayed.
