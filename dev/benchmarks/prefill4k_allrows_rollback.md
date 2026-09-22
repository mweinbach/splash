# One-trunk allrow prefix rollback and continuation oracle

This private oracle uses **one FlashForward and one Full512 store**, with three
distinct request identities and 402 physically disjoint persistent Shared
buffer ranges. It never constructs a second 121 GB expert store. CPU snapshots
copy all 134 request planes and preserve no borrowed state/output views.

The numerical derivative is explicitly
2e858faa201554642a443d48d38b7302159fe1a811c028a895454cc8ce5c073d.
The driver selects the current local profile, Full512 allrow source omission,
Dense1, SSD PLE, and disables original/idle residency. This is a private
numerical alternative. It makes no claim about universal exact state equality
between row-shape producers, HTTP, MTP teacher priming, or speed.

Prepare a fresh isolated copy after the source handoff:

~~~sh
.venv/bin/python dev/benchmarks/prefill4k_allrows_rollback_overlay.py \
  --source-build build/prefill4k-allrows-full512 \
  --output build/prefill4k-allrows-rollback
make -f Makefile -f dev/benchmarks/prefill4k_wide.mk \
  -f dev/benchmarks/prefill4k_allrows_rollback.mk \
  BUILD=build/flash-next SPLASH_PRECISION=hybrid \
  PREFILL4K_WIDE_BUILD=build/prefill4k-allrows-rollback \
  PREFILL4K_WIDE_FULLCACHE=1 prefill4k-allrows-rollback-cpu -j 4
build/prefill4k-allrows-rollback/rollback-oracle --help
~~~

These commands copy only source/metadata and compile; CPU tests create no
Metal backend, load no model, read no model payload, and submit no GPU work.
The copied header grants test friendship only in this new directory. The
oracle includes its copied Forward.cpp and excludes ordinary FlashForward.o
and FlashWorker.o from linking. All other host objects use matching copied
headers. Existing allrow sources and production files remain untouched.

Root-exclusive invocation:

~~~sh
.venv/bin/python dev/benchmarks/prefill4k_allrows_rollback.py \
  --build build/prefill4k-allrows-rollback \
  --tokens build/release/flash/prefill4k-fixture/code2048.tokens.json \
  --report build/release/flash/CHOOSE-FRESH-allrows-rollback-v1.json
~~~

The invocation driver is dry-run by default. Add --run only when root
serializes actual model/GPU work. A dry-run witness occupies its report name;
choose a new name for execution. The driver checks source metadata digests and
never scans the heavy expert/checkpoint payloads itself. Actual native startup
retains its ordinary model/store integrity validation.

The case matrix is **four base pooling phases × four retention counts**:

| Base length | Trial rows | Retained | Primary branches | Unmutated reference |
| --- | ---: | --- | --- | --- |
| 2048, 2049, 2050, 2051 | 4 | 1 | A/B verify4, same first token, differing discarded EOS/special suffixes, then commit1 | C forward1 |
| 2048, 2049, 2050, 2051 | 4 | 3 | A/B verify4, same first three tokens, differing discarded fourth token, then commit3 | C forward3 |
| 2048, 2049, 2050, 2051 | 4 | 4 | A/B identical verify4 and full commit4 | C forward4 |
| 2048, 2049, 2050, 2051 | 4 | 0 | Reject commit0, block peer forward while pending, abort terminal, reject poisoned resume, allocate a new healthy A and restore owned base bytes | Untouched B |

Each case appends seven identical one-token inputs. The proposal includes
248044 and 248046, and future inputs include EOS, a subsequent token, and the
special token. Four additional controls destroy a pending A, then compare seven
healthy peer singleton appends against untouched C. The next healthy forward
discards the expired shared tape; no state from the destroyed request is
promoted. Requests are reset from owned snapshots between cases, preserving
their distinct identity and shared trunk owner.

The A/B same-verify4 comparison is the primary future-contamination check.
Both use the same row-shape producer; only tokens that should be discarded
differ. C's forward(rows=retained) and later singletons remain separate exact
and numerical diagnostics because existing R1/R3 versus R4 producer differences
are known. Those differences alone do not establish a rollback defect.

Every checkpoint reports live and full allocation word/byte differences,
SHA-256, first eight exact differing words, finite checks, maximum/RMS absolute
error, relative L2, and ordered BF16/F32 word distance for all persistent GDN
recurrence/convolution, QSA cache planes, and PLE histories/convolution. It also
reports all selected last logits, pre-mixer Hyper and post-mixer hidden.
All four verify rows are copied immediately; the retained row r-1 is selected
from those owned copies after commit. Commit does not recalculate logits.

QSA tails after a partial commit deliberately retain discarded trial bytes.
Those allocated-byte differences remain visible and do not count as live
prefix contamination. Complete pooled key extent is floor(length/4), and the
next append overwrites stale token rows/recomputes newly completed blocks.
All allocated values are still checked for finiteness.

Retained0 is **terminal abort**, not healthy zero-retain restoration:
the aborted state has length base+4, poison=true, pending=false, and retains
trial-consumed caches. The report emits all terminal allocated-plane
differences from base, then tests a newly allocated healthy state.

Normal MemoryGovernor admission reserves one workspace/store, verification
storage, three request states, and six CPU physical snapshots plus output
margin before construction. No store duplication or admission bypass occurs.

Exit0 means the primary same-shape **live** state/output checks and explicit
guard/fresh-reset/cancellation controls were exact and finite. It does not mean
all stale physical tails matched or all row-shape producers matched. Exit2
completes every case and records primary differences honestly; numerical
metrics never replace exact checks. Exit1 is an operational exception and a
partial JSONL trace is not a completed result. Compilation and CPU self-tests
provide no GPU rollback qualification.
