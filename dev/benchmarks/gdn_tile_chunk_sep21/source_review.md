# Per-tile T32 guarded GDN source review

Independent CPU-only review of the actual component source. No GPU/model payload,
public runtime, sealed v6, or whole worker was changed. Parent owns the kernel,
ABI and Makefile; SDK agent owns the oracle. This is a separate numerical policy:
native rejected chunks start from the current hybrid incoming F32 state, which
may contain rounding from earlier accepted WY chunks. It is not universal
whole-native-prefix/F64 equivalence. Whole-worker integration remains forbidden
until actual-input speed wins.

Two material issues found during review were corrected by Parent: the initial
WY branch now reads immutable `initialRange` instead of racing the mutable local
reason, and a device-memory barrier now orders speculative output/audit writes
before native overwrite. Audit/probe seed capture also has an explicit device
fence before any accepted state commit.

## Uniformity and state ownership

- Each group owns `(head,tile)` with 32 distinct value rows. Tile t owns state
  bytes `head*65536 + t*16384 .. +16383`; four tiles partition each head exactly.
  No cross-group decision, spin wait, or whole-head synchronization is needed.
- Preparation initializes its group-local reason, loads gates, scans range, and
  barriers before reading the final stable reason. There are no later reason
  writers before its uniform return. Rejected chunks skip Gram, inverse, W/U,
  norm cache, QK and E. Zeroed transform slots remain defined; computed prefix
  slots are unused. Existing range predicates and raw alpha/beta word tests stay.
- Apply's initial predicate uses immutable preparation data. Local OR writers
  may then run inside WY, but all complete before the commit decision barrier.
  The local reason is stable during commit/native decisions and native waves;
  reset happens only at the next chunk. Valid threads take the same branch and
  reach its threadgroup barriers together. Shape/mode/audit rejection is uniform.
- Candidate next state stays in the update cooperative tensor. There is exactly
  one explicit persistent WY state store, inside the accepted commit branch.
  Rejected paths therefore call native from the unchanged incoming chunk state.
  BF16 output and F32 audit writes may be speculative but native rewrites every
  active token/owned coordinate; the pre-native device fence orders both writers.
- The local atomic reason is separate from the reused raw phase arena. Native
  query/key/value/decay scratch cannot overwrite it. The shared-memory barrier
  before reuse protects dead WY scratch; the device+group barrier at chunk end
  protects the next chunk's state reads. No reason is read after a concurrent OR.

## Native wave, pointer, stride, and tape contract

Native wave w=0..3 uses eight SIMD32 value rows with translated `group.y=4*t+w`.
Its rows are `32*t+8*w .. +7`; 256 threads match V8. The literal helper executes
two T16 blocks at a full T32 chunk, or its original bounded tail loops. Every
value row retains original four-key-elements/lane order, SIMD sums, F32 state
updates and the BF16 output boundary.

Native parameters use `cp.rows=count<=32`, `cp.lanes=1`, translated group.z=0.
Mixed/decay/beta/output are rebased by begin times 10240/48/48/6144 respectively.
Audit history/delta/preoutput are rebased by begin times 16384/128/128. Explicit
B1 entry validation makes this correct; passing a nonzero original lane while
using chunk-local rows would have indexed `batch*count` rather than `batch*R`.
This component does not support B>1. Any multi-lane layout estimates are not a
supported-kernel proof.

Padded recurrent strides remain valid aligned metadata, but B1 uses lane zero.
All state owners stay within the tight 3,145,728-byte lane extent and never write
trailing padding. The CPU address model covers three aligned strides, 14 row/tail
sizes and all head/tile/wave owners. Tails 1/15/16/17/31/32/33 and R2047/2048
are included. The audit entry rejects nonzero heads, and its native wave rewrites
all head-zero history/delta/preoutput coordinates owned by that tile. Audit/probe
incoming seeds copy uint words, preserving original F32 bits before state commit.

Native helper arithmetic is literal captured canonical code, with only helper
prefix changes (`gds_`→`gtcn_`/`gtca_`) and constant-to-thread parameter address
space adaptation for the local parameter block. Canonical dispatch wrapper
comments/attributes are omitted; they are not native arithmetic. V8 byte
equivalence from the captured hybrid seed still requires Root GPU proof.

## Packed ABI and exact storage

`TileChunkParams` is 56 bytes: literal `FlashGDNParams` occupies 48, mode is at48,
reserved at52. Main component requires reserved0. Modes0/1/2/3 are guarded,
forced-native, odd-chunk forced-native, and forced-WY diagnostics. Mode3 on
range-rejected chunks has unavailable zeroed transforms and is explicitly
unqualified/unused for unsafe math comparisons.

| Pipeline | Buffer slots | Parameter slot |
| --- | --- | ---: |
| Prep | mixed0, decay1, beta2, prepared3, range4, diagnostics5 | 6 |
| Main apply | mixed0, decay1, beta2, state3, output4, diagnostics5, prepared6, range7, decisions8 | 9 |
| Audit apply | main0..8, history9, delta10, preoutput11, incomingSeeds12 | 13 |
| Probe apply | main0..8, incomingSeeds9 | 10 |

One prep leader writes each range UInt word, index `chunk*48+head`. One apply
leader writes each tile UInt word, index `(chunk*48+head)*4+tile`. Reason bits
are low0xF, native bit0x100, forced bit0x200; no sparse/dynamic tape allocator.

| Resource at R2048/B1 | Exact logical/declaration bytes |
| --- | ---: |
| Range decisions: 3072 UInt words | 12,288 |
| Tile decisions: 12,288 UInt words | 49,152 |
| Combined decisions | **61,440** |
| Prepared W/U/E/score/prefix/norm/reason | 164,757,504 |
| Audit incoming seed tape | 4,194,304 |
| All-head probe incoming seed tape | 201,326,592 |
| Preparation TG scratch, including local atomic | **12,804** |
| Apply rawScratch[2332] plus separate local atomic | **9332** |

Active native scratch is9312 bytes; the declared raw arena is9328, with local
atomic4 outside it. This is not an 8516+9312 simultaneous allocation. Pipeline
resource padding, register spilling, occupancy and timing are not proven by
these source counts.

## Independent v6 math controls and collision closure

`frozen/v6_math_control.metal` authenticates the sealed v6 source and keeps its
entire pre-entry helper prefix literal after every `gwy_`→`gtcv6_` rename. Prep
control uses old slots0..5/params6, passing `control.gdn`. Audit control uses old
slots0..10/params11 and maps original grid.y0 to `control.reserved` tile0..3.
These controls are compared only for a selected safe V32 tile, range0, from the
same current hybrid incoming seed. They do not qualify unsafe mode3 or full
native history. The control wrappers compile with Metal4.1/O3/Werror.

Candidate user helpers (`gtc_`, `gtcn_`, `gtca_`) and controls (`gtcv6_`) have
separate ODR names. The CPU witness emits actual candidate/control LLVM IR,
checks their user helper domains are disjoint, and compares shared weak SDK
definition bodies/attributes. Entry renaming alone is insufficient: earlier
controls had incompatible `linkonce_odr` helpers and layout aliases.

`source_witness.py` records compiler commands, source/IR hashes, literal helper
comparisons, independent C++ ABI compilation, 42 address/stride cases and 86,976
tile checks. Its source identity binds the kernel, ABI, helper/control/provenance
hashes. Numerical policy is
`gdn-t32-v32-sg8-local-guard-095-deferred-state-commit-native-v8-t16-four-waves-from-hybrid-seed-v1`.
Counters/decisions themselves are observations, not static identity inputs.

CPU success permits Root component diagnostics only. Native V8 byte proof,
selected-safe-WY equivalence, actual-input performance and the unchanged raw F64
tensor gates remain separate results. No whole-worker command or speed claim is
made here.
