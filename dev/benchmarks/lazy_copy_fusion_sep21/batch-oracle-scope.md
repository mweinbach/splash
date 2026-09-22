# Native joint GDN wrapper oracle

`batch_oracle.mm` is a separate synthetic qualifier for the frozen worker
`build/lazy-copy-fusion-sep21-v2`. It includes `gdn_fixture.hpp` with its main
renamed and leaves `oracle.mm` unchanged. Preparation ran no GPU command and
read no model payload. Compile, immutable closure sealing, and GPU execution
remain the parent's responsibility.

Compile against the frozen worker headers and **all 50 nonworker objects** from
its `link-inputs.json`, matching the layer/whole oracle closure. Generate
`LazyCopyFusionBatchBuildProvenance.hpp` with the JSON string constant
`kLazyCopyFusionBatchBuildProvenance`; freeze this source, the fixture, every
transitive header, all linked object bytes, and the actual metallib.

`--cpu-only` returns before executable hashing or `MetalBackend` construction.
It checks the fixture's scalar identities plus the joint row/packed-stride and
retained-concatenation planner contracts and reports zero backend constructions
and zero commands. That result is not GPU wrapper qualification.

The GPU path calls the **actual patched `addBatchVerifyGDN` wrapper**, rather
than reproducing its preflight or pack/scatter implementation. Baseline and
candidate records capture constructor flag 0 and 1 respectively. Packed state
uses convolution stride 65,536 and recurrent stride 3,145,728, with untouched
convolution padding and final guards; each request owns separate guarded state
buffers. Both records reserve the same six R4/L4 arenas, even for smaller cases.

The 18-case matrix covers rows 2/3/4, lanes 2/3/4, and cold/carried initial
state. Each case uses two changed projected-input sequences, full acceptance,
and two rotated mixed retained-count cohorts. Captured Persistent512 states
and independent history concatenation supply prefix expectations. Checks require
zero differing bytes for raw QKV, every prepared array, initial FP32/history
snapshot, recurrence rows, output, diagnostics, full packed state/padding,
separate request states, and all six complete arenas including inactive bytes
and guards. Partial replay scatters only partial live lanes with the native
`flash_forward_copy_words` ABI; full/terminal lanes receive no commit scatter.
Changed ordinary one-row future GDN continuation runs only nonterminal lanes.
All six arenas remain unchanged through commit and continuation.

Host negatives test ordinary/foreign QKV, offset, short, wrong-row and oversized
views, output/mixed/Z aliases, a request-state alias, a weight alias, and Z
overlap with each arena. Every rejection must preserve the existing sentinel
dispatch, zero ticket, and idle record before **any pack dispatch**. An exact
own RawQKV view is accepted into an unsubmitted graph and then aborted. These
tests exercise the wrapper's exception permitting only work index 0 paired
with its precise owned RawQKV allocation; all other overlap rules remain active.

Usage: `batch-oracle --cpu-only` or
`batch-oracle FROZEN_METALLIB FRESH_REPORT_JSON`. Every submitted command checks
the 200-byte timing ABI and plausible finite timing; no timing improvement is
qualified. This scope excludes whole target logits, QSA/PLE caches, trained MTP
acceptance, model quality, service performance, and full target verifier flow.
