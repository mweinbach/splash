# Paused draft: guarded WY v6 over HC worker

**KNOWN REAL-INPUT PERFORMANCE REGRESSION. Do not compose or run this draft.**

Root stopped this integration after actual-input capture rejected v6 performance:
39/48 heads replayed (reported reason counts: 15 range, 14 cancellation, 10 norm).
Native took approximately 1.90 ms versus WY 3.37 ms for cold/carried state.
Exact fallback quality passed. Strict F64 qualification remains false.

These are Root-reported component observations; this integration performed no
GPU work, model/payload access, or serving. The earlier zero-flags primitive win
does not establish performance on these captured inputs or in a whole worker.

## Preserved draft state

`worker_bridge.hpp`, `worker_overlay.py`, `worker.mk`, `worker_policy_cpu.cpp`,
and `worker_witness.py` are uncompiled integration drafts. Root-owned
`telemetry.hpp` and `telemetry.metal` are ready, with Root-reported standalone
Metal compile and CPU proof passing. No new worker build directory, worker
freeze, composed library/executable, or whole-worker qualification was created.

The bridge draft provisions a fixed 167,919,616-byte R2048/B1 arena, including
164,757,504 coefficient bytes, 3,145,728 snapshot bytes and the rounded flag
page. The 256-byte telemetry view uses existing padding after 192 flag bytes;
it adds no allocation. It accumulates scheduled/eligible/applied/replayed heads,
overlapping reason counts and the 16-bin mask histogram on the GPU. A single
copy after a completed trunk command updates cached status, with no per-layer
host reads. The draft moves mutable counters outside numerical identity.

The planned route is singleton MAIN nonverification R64..2048, strict
`SPLASH_FLASH_GDN_PREFILL_WY_SEP21=0|1` requiring staged=1 when enabled. Flag zero
retains parent FMA. It is not a batch/decode/verify/lazy-rollback/replay change.
The intended parent is `build/prefill-hc-inject-norm-sep21-worker-v3`; neither it,
sealed v6, nor the old v3 WY worker was edited.

## Incomplete and unvalidated

- Witness adaptation to the new source paths, telemetry ABI/identity, four new
  AIRs and HC parent closure is unfinished.
- Final v6-seal authentication and telemetry pins are unfinished; no freeze ran.
- Generator anchors and compiler/link/dependency closure are not validated.
- CPU policy/native self-tests and actual governor checks have not run here.
- No Root 2K/256-token or 22-semantic-case whole-worker command is proposed.

Do not treat draft source as a sealed artifact. Root now prioritizes a
chunk-specific native fallback proposal/new policy. Resuming integration needs
an explicitly selected replacement policy with real-input performance evidence;
the v6 full-window fallback performance failure must remain visible.
