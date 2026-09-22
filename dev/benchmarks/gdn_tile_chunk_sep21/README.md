# Private T32/V32 tile-local GDN component

CPU-prepared numerical experiment. Root alone runs Metal and opens captured
model inputs. Sealed v6 and the performance-rejected whole-worker draft remain
unchanged. No model composition is qualified by this component.

Each SG8/256-thread workgroup owns 32 recurrent value rows and loops over T32
chunks. The range, norm, finite/saturation, and `.095` Cauchy/L2 predicates are
retained from v6, including raw source-bit zero/subnormal distinctions, post-zero
nonzero segments, and MSL permitted RTZ/FTZ scope. Replay granularity is a new
policy. Range selects all four tiles of a head/chunk; later checks select the
individual tile/chunk. The selector is not a whole-history accuracy certificate.

Candidate final state stays in the update cooperative registers until every
local reason has been classified. A uniform decision commits safe tiles; unsafe
tiles execute captured native recurrence from untouched current F32 state.
Native handles four V8 waves with Time16 token blocks and overwrites every owned
BF16 output and compact audit-history/delta/pre-output slot. There is no GPU
cross-workgroup spin barrier, full-R replay, or timed incoming-state snapshot.
Incoming audit seeds use UInt bit copies before any persistent write.

Range preparation happens before expensive matrix work. A range-rejected
head/chunk leaves defined-zero transforms marked unused/not computed. Such
coverage is not a raw F64 coefficient pass. The independent raw F64 gates remain
relative RMS <=1e-4 and absolute <=5e-4*max(1, reference peak). BF16 has separate
RMS .003 and absolute .004*max(1, peak) gates. Native may itself fail the original
cancellation-relative F64 gate; local byte equivalence does not erase that fact.

Native scratch is 9312 bytes. Shared phase scratch is 2332 UInt words (9328 bytes)
plus a separate four-byte threadgroup reason mask: 9332 declared bytes. The
coefficient stride is 13408 F32 words; R2048/B1 coefficients are 164757504 bytes,
range words 64*48*4=12288 bytes, and committed decision words 64*48*4*4=49152 bytes.
Separate16KiB-rounded admission is 164823040 bytes. Compact audit seed tape is
4194304 bytes; full-head proof seed tape is 201326592 bytes. Those tapes are not
present in timed mode.

`TileChunkParams` is 56 bytes: captured 48-byte GDN params plus mode/reserved.
Mode0 applies guards, mode1 forces native, mode2 forces native on odd chunks,
mode3 forces WY for safe-only math proof. Decisions store reason bits0..3,
native-path bit0x100, and forced-path bit0x200. Mode3 does not make skipped range
transforms valid; comparisons use only range-safe chunks.

The public performance scope is R64..2048/B1; smaller rows are permitted solely
for local/tail oracles. No multiple-lane support is claimed. The build's default
only compiles; resources/quality/proof/bench modes require explicit Root action.

Required before timing: actual resource bounds, complete native forced-prefix
equality, same-incoming-hybrid-seed local native state/output/tape equality,
safe local v6 math comparison, mixed reasons/tiles/zeros/tails/continuation,
canaries and immutable data. Actual cold/carried capture then measures tile/chunk
replay counts and inclusive >=150 ms GPU warming with balanced ABBA timing.
