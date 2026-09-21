# Local v7 defaults

The active launcher and `.splash-local-profile.json` now match v7 with 36 static
defaults. It adds `SPLASH_FLASH_HC_UP_F32_MPP=1` and
`SPLASH_FLASH_GDN_LAZY_ROLLBACK=1` to v6. Saved dense/TOP64 pins, original
source/layout/norm/hardware guards, 2048-row prefill windows, residency defaults,
and exact saved-store zero opt-outs are unchanged. TOP128 and raw-text residency
remain manual experiments; private per-tensor/indirect modes are not enabled.

Root's combined ON run measured 107.394/167.744 tokens/s for short
single/four-request workloads and 68.868/82.954 for 2K workloads. Same-build
OFF measured 101.187/155.734 and 67.367/80.330. The combined route passed all
22 quality/lifecycle checks and all 28 responses matched the HC-only route.
After instrumentation, the normal launcher confirmed 107.512/166.982 tokens/s
for short single/four-request workloads and 69.118/82.824 for 2K workloads.
All 22 quality/lifecycle checks passed again, with zero new task regressions.
The normal server remains ready and idle on port8011, with tracing disabled.
The fresh diagnostic build and normal launcher binary/metallib are byte-identical.

The directly changed verifier GPU work fell about9% in the matched run. Lazy
rollback saved2,490,302,464bytes of workspace while preserving FP32 recurrent
state. Compared with HC-only, it saved144.2ms of verification work and added
5.8ms of partial-prefix replay. The gain depends on draft acceptance; a new
low-acceptance workload needs its own measurement.

HC-up's implied default requires FUSE_HC and FLOAT_DENSE_CACHE. It uses existing
original-F32 coefficients and small-row workspace for R4..16 HC-up matrices;
unrelated DENSE_CACHE/QMV_F32/MTP settings are not prerequisites. Lazy rollback
requires FUSE_GDN; GDN_STAGED, batching and MTP are unrelated to its helper.
Explicit parent0 disables an implied child; every explicit child value remains
authoritative. Historical v5/v6 review copies retain 30/34 static defaults.

The lazy singleton/joint memory planners reserve their arenas before
construction. Production backend/ComputeDispatch ABI is unchanged. A complete
fresh host rebuild and matching metallib are required because lazy BatchVerifyGDN
helper signatures and shaders changed. None of the private backend or loader
variants are part of this promotion.

The existing private proposal-head synthetic 1% error bound remains failed;
full service coherence and autoregressive parity remain separate evidence.
Checkpoint payloads, original oMLX preferences and remote roster are untouched.
