# Private wide persistent-state continuation oracle

The oracle compares the exact same source weights and frozen tokens through
independent 2048-row and 4096/8192-row FlashForward owners and request arenas.
It snapshots only completed healthy plain-forward boundaries. Every borrowed
last output is copied immediately, before any subsequent trunk call.

This is an execution-state oracle. A greedy output hash, cosine similarity, or
all-finite result alone does not pass it. It does not instantiate MTP teacher
priming or the HTTP scheduler and does not qualify prefill performance.

Build and CPU checks submit no GPU work:

~~~sh
make -f Makefile -f dev/benchmarks/prefill4k_wide.mk \
  -f dev/benchmarks/prefill4k_wide_state.mk prefill4k-wide-state -j 8
build/prefill4k-wide/state-oracle --cpu-self-test
build/prefill4k-wide/state-oracle --help
~~~

The read-only accessor reuses the existing request-state test friendship.
prefill4k_wide_state_overlay.py adds friendship to a separately copied Forward
header under build/prefill4k-wide/state-source. The oracle includes the private
Forward implementation, and the link excludes ordinary FlashForward.o and
FlashWorker.o. All remaining Flash objects/headers are from the private wide
build; core Metal and MemoryGovernor objects use the normal build. No production
source or shared private-overlay source is patched.

Refresh the private wide overlay and rebuild whenever source candidates change.
The make target depends on its manifest. To build against the parent's combined
wide/fullcache build instead, set PREFILL4K_WIDE_BUILD=build/prefill4k-wide-fullcache
and PREFILL4K_WIDE_FULLCACHE=1. Use the pure wide build and current Top64 package
for geometry qualification so a Full512 numerical alternative does not confound
the comparison.

The GPU matrix is six separately launched processes, serialized by root:

| Frozen prompt tokens | Candidate arena rows | Baseline prompt commands | Candidate prompt commands | Prompt checkpoints |
| ---: | ---: | ---: | ---: | --- |
| 2048 | 4096 | 1 | 1 | 2048 |
| 2048 | 8192 | 1 | 1 | 2048 |
| 4096 | 4096 | 2 | 1 | 4096 |
| 4096 | 8192 | 2 | 1 | 4096 |
| 8192 | 4096 | 4 | 2 | 4096, 8192 |
| 8192 | 8192 | 4 | 1 | 8192 |

Each uses capacity **16384**, then performs seven identical one-token appends:
tokens[0], tokens[1], 248044, 198, tokens[2], 248046, tokens[3]. These exercise
three-row GDN and nine-row PLE history carry, EOS segmentation, special tokens,
and several completed/incomplete QSA four-token pooled-block boundaries. The
2048 fixture is an independent-arena control rather than a wider physical call.

Exact command, shown for one case; substitute all six matrix values and choose
a fresh output each time:

~~~sh
.venv/bin/python dev/benchmarks/prefill4k_wide_state.py \
  --build build/prefill4k-wide \
  --tokens build/release/flash/prefill4k-fixture/code8192.tokens.json \
  --rows 8192 \
  --report build/release/flash/prefill4k-wide-state-code8192-rows8192-v1.json
~~~

The driver is a dry run by default: it writes an invocation witness but creates
no Metal backend. **Root may add --run** to actually execute this serialized
GPU process. A dry-run witness occupies that report name; use a new name when
executing. The driver clears inherited Flash experiment flags and resolves the
current local profile plus its saved Top64 operand paths. The separate dense
tile experiment is explicitly off; --dense-tiles on tests its interaction in a
separate report.

The .checkpoints.jsonl companion reports all **134 persistent planes** at
every checkpoint:

- 36 GDN F32 [48,128,128] recurrent planes and BF16 [3,10240] convolution
  histories, plus each lane stride.
- 12 QSA layers' BF16 keys/values/raw index/complete pooled keys and I64
  positions, plus QSA capacity.
- PLE I64 [2] token history and BF16 [9,10240] convolution history.
- Request length, capacity, poison flag, and pending-verification flag.

For each plane, both active extents and full allocated physical bytes are
compared. Unused capacity, incomplete pooled slots, and rounded allocation
padding are shown separately from live semantics. Full allocation SHA-256,
differing byte/word counts, first eight exact word differences, nonfinite
counts, maximum absolute/RMS error, relative L2 to reference, and ordered
BF16/F32 word distance are retained. Floating error summaries are diagnostics;
they never replace exact comparisons or suppress a mismatch.

The same checkpoint compares every word of the last BF16 vocabulary logits,
pre-mixer Hyper [10240], and post-mixer normalized Hidden [2560]. Public
FlashForwardResult.hiddenBF16 exposes the pre-mixer stream only; the private
read-only friend is necessary for final normalized Mixed scratch.

Two Forward workspaces, all optional operand allocations, two request states,
and two CPU-owned physical snapshots are reserved before construction through
the normal MemoryGovernor. Construction refuses if admission cannot support
independent arenas; the oracle does not bypass policy or clone state across
owners.

Exit **0** means all compared allocated persistent bytes, scalar metadata, and
last outputs were byte-exact and finite at every checkpoint. Exit **2** means
the full matrix case completed but differences/nonfinite values were recorded.
It continues all seven appends after a numerical mismatch. Exit **1** is an
operational exception; an existing partial JSONL trace is not a completed pass.
Timings are serialized diagnostic trunk timings interleaved with CPU copies and
hashing, excluding teacher priming; use the separate normal/HTTP attribution
for speed measurements.

CPU self-test checks BF16 infinities, signed-zero differences, a last-word
BF16 difference, one-ULP F32 recurrence differences, I64 tail differences,
live/padding separation, the exact six-case schedule, capacity, and mismatch
JSON. A compiled artifact and these CPU checks provide no GPU numerical result.
