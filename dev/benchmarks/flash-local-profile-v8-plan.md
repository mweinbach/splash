# Local v8 proposal: exact BF16 vocabulary reconstruction

This CPU proposal preceded Root's service qualification and local promotion.
The accepted launcher and `.splash-local-profile.json` now match
`m5-ultra-flash-next-v8` with 37 static defaults. See
`flash-local-profile-v8.md` for accepted behavior and evidence.
`_local_profile_v8_candidate()` returns an independent review copy with the
same static defaults; normal serving reads the accepted profile directly.
No model or GPU execution was performed by the profile agent.

The proposed route applies only to trained joint-MTP `Last` vocabulary
projections with two through four real lanes and the existing cached BF16
vocabulary baseline. It reconstructs those exact BF16 coefficients in registers
from the original Q8/G64 codes and BF16 scale/bias, using M8N32/K64/SG1 F32
accumulation. It reuses existing padding and allocates no weights or scratch.
It does not select the rejected raw-F32 Q8 arithmetic candidate or the QSA N32
experiment. The singleton head, target vocabulary and prefill `None` logits
keep their existing routes.

Implied register defaults follow explicit zeroes for `DENSE_CACHE`, `BATCH_MTP`,
`MTP`, and `BATCH`. `DENSE_CACHE` supplies the cached vocabulary and its padding;
`BATCH_MTP` constructs the joint head. `MTP` is a worker prerequisite. `BATCH`
is inherited launcher policy: its opt-out suppresses an implied `BATCH_MTP`,
although the native joint executor can run with an explicitly enabled
`BATCH_MTP` while ordinary batching is disabled. Both transitively implied
parents are listed directly because the launcher merger does not recursively
inspect adjusted defaults. Explicit child values remain authoritative,
including a contradictory `1`, empty value, or malformed value for native
validation. `INT8_HEAD`, `GPU_GREEDY`, `BATCH_MTP_PREFILL`, `BATCH_PREFILL`,
`QMV_F32`, and `FLOAT_DENSE_CACHE` are independent.

Historical v5/v6/v7 review helpers retain their exact 30/34/36 static defaults
even under a hypothetical active v8. Existing source/layout/architecture and
hardware gates, dense/TOP64 artifact pins, exact store `0` opt-outs, corruption
rejection, 2048-row prefill windows and singleton depth15 remain unchanged.
Private loaders, command retention, TOP128, GPU prefill copy and QSA N32 are
absent from the proposal.

The prepromotion profile suites passed 98 tests with one optional native-checker
test skipped. After activation, all 109 profile and launcher tests passed using
the existing CPU-only exact-zero selector checker, with no skips. The 11 new
tests verify independent copies, accepted-v8 and historical-v7 profile gates,
every operative parent opt-out, explicit overrides, independent switches,
historical key sets and preserved artifact metadata. The existing saved-operand
and local-launcher suites cover source/hardware/artifact gates.

Root qualified complete service output/state/lifecycle behavior and matched
same-build ON/OFF performance before authorizing activation. The accepted
launcher metadata and `.splash-local-profile.json` were updated together, and
active profile expectations changed without changing historical key sets.
Root owns normal serve readiness and idle recovery proof using the matching
rebuilt binary/metallib.

The machine-readable proposal is
`build/release/flash/local-profile-v8-promotion-plan.json`; its static candidate
is `build/release/flash/local-profile-v8-candidate.json`.
