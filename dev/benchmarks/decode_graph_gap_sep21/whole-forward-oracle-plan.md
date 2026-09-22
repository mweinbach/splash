# Sequential exact whole-Forward qualification plan

This is a CPU/source audit, not a GPU result. No model payload was read. The
frozen base is `build/moe-pointwise-sep21-worker-v1/source`.

## Feasibility and diagnostic surface

Use one live `FlashForward` at a time. Persistent request storage is already
accessible from a test-only `FlashDeepPrefixOracle` definition through the
existing friendship in `FlashForward.hpp:71` and
`FlashRequestStateInternal.hpp`. No production/public API change is required.
The friend can enumerate every physical Shared buffer, read pending/owner/
identity metadata, and restore a healthy resolved base while preserving the
new destination's owner and distinct identity. The existing deep-prefix and
all-rows rollback oracles show this pattern.

Capture hidden/logits/greedy immediately from `FlashForwardResult`, before
commit or any later trunk call: all are borrowed workspace views. Serialize
their exact real extents, not the head's maximum allocated rows. Post-mixer
hidden inspection is optional and would require Forward friendship in a private
copied header plus including its implementation/filtering ordinary Forward.o;
the requested pre-mixer hidden/logits/compact greedy need neither change.

Keep `SPLASH_FLASH_GDN_LAZY_ROLLBACK=1` fixed: its existing flag getter caches a
process-static value (`FlashGDNLazyRollback.cpp:161-171`). The new copy-fusion
candidate flag must be strictly parsed and captured as a const per-instance
option by the private Forward/lazy-record constructor. Changing the environment
after construction must not alter the existing object's route. Test invalid
flag values before constructing a backend.

## Pass lifetime and admission

Run control and candidate as sequential subprocesses with the same frozen
binary/library and complete environment except the candidate flag. A second
Forward on the same backend can retain the first pass's cached operands through
residency registrations: `MetalBackend.hpp:120-124` says those registrations
remain until backend stop and command-ticket consumption. Subprocess exit gives
a clear lifetime boundary. If an in-process design is used, scope/destroy the
backend, weights, residency lease, all results, cached-operand vectors and every
state before constructing the next pass; `stop()` makes that backend unusable
for new submissions.

Use current Full512 pointwise policy, capacity4096, maximumRows2048 and
maximumVerifyRows4. Apply real governor admission before construction using
workspacePlannedBytes + expertCachePlannedBytes + floatDenseCachePlannedBytes +
int8HeadPlannedBytes + selected Dense/blocked-MoE plans + two request states +
bounded CPU comparison scratch. Preserve original-text omission/current saved
operand policy. Assert actual allocation and numerical route identity after
construction. This correctness run has no timing claim.

## Exact file protocol

Each checkpoint includes capacity, length, poisoned/pending flags, per-layer
state geometry, plane label/type/live-byte extent/physical-byte extent and file
SHA256. Stream every full physical plane to a fresh bounded file, atomically
publish its manifest only after all writes succeed, and checkpoint failures.
Compare candidate bytes directly to control files in <=1MiB chunks, including
padding and every inactive suffix. Hashes record provenance and do not replace
byte comparison. Reject missing/extra planes, truncated/oversized files and
metadata mismatch. Scan F32 and BF16 planes for finite values; I64 history and
positions are integer data. Do not compare raw owner or identity addresses
across processes; compare ownership/disjointness and within-pass stability.

The inventory is 134 persistent planes: 36 GDN convolution/recurrent pairs,
12 QSA groups of five (keys, values, raw index keys, pooled keys, I64 positions),
and two PLE planes. At capacity4096 these occupy exactly232,603,648 bytes
(221.828125MiB); capacity8192 occupies349,388,800 bytes (333.203125MiB).
Two capacity4096 requests add443.65625MiB. Avoid whole-cache CPU vectors.

## Initial bounded matrix

Prefill canonical2048 tokens once per pass and require byte-exact base/output
agreement. Test base lengths2048+phase0..3 to cross every QSA pooling phase.
Keep one healthy peer and one trial request, restoring the same resolved base
from files between cases while preserving each request's owner and identity.

For each phase, verify the same fixed four inputs, spill full state and all
four output rows, commit retained1/2/3/4, spill state, then append three fixed
identical singleton inputs and spill state plus hidden/logits/greedy after each.
Require full-4 commit to produce zero GPU timing. Separate tests cover abort0
(commit0 must reject, abort marks terminal poison without rolling back length),
fresh healthy reset followed by the same three singleton inputs, and destruction
of a pending trial followed by those inputs on the healthy peer. This is116
full-state checkpoints:26,982,023,168 bytes (25.129GiB), plus small outputs and
metadata. Preregister a32GiB control-spill bound; stream-compare candidate
without writing another full set. Temporary guard snapshots add at most one
request-state file, well within that bound.

Before/after byte checks cover retained0/>rows rejection, default/moved-from
state, same-owner wrong-peer commit, non-matching-peer abort no-op, peer trunk
blocked while a tape is live, poison rejection, move/ownsState stability, and
physical range disjointness. A true foreign-owner request can be retained from
a previous destroyed Forward for an in-process guard-only test, budgeting one
additional221.828125MiB; otherwise report that foreign-owner construction was
not exercised. Do not invent a second live model just for this guard.

Control-vs-candidate uses identical verify shape and input, so every allocated
byte is a hard equality gate. Any optional `forward(rows=retained)` producer or
different-discarded-suffix control must report live-prefix equality separately
from stale physical QSA suffix differences, as the existing rollback oracle
does; those different producer shapes cannot establish universal exact parity.

Layer-oracle qualification should cover wider row counts/all retained prefixes;
expand the whole-model matrix to R8/R16 only if that qualification and the first
R4 whole-model gate pass. A successful layer oracle, compile or CPU protocol
self-test is not a successful whole-model qualification.
