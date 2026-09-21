# Historical local v6 defaults

Active defaults are now v7; see `flash-local-profile-v7.md`. The historical v6
launcher/profile matched 34 static
defaults. It adds `SHARED_EXPERT_FUSED`, `DENSE_M64_OUT`, `GDN_BATCH_ILP`, and
`MTP_QMV_F32` to v5. Optional saved dense operands and top64 INT8 experts keep
their existing source/layout/manifest/plan pins. Prefill windows remain 2048
rows. Top128 is manual; original-text residency stays off.

Same-build ON confirmation measured 99.576/154.973 tokens/s for short
single/four-request workloads and 66.085/78.592 for 2K workloads. OFF measured
100.353/152.037 and 64.359/78.226. The initial ON run was 100.008/155.074 and
66.100/78.870, so the modest concurrency and long-prompt gains repeated while
short singleton throughput varied slightly.

Shared-expert fusion and the dense output route require an implied DENSE_CACHE
default. GDN_BATCH_ILP requires GDN_STAGED and BATCH_PREFILL; MTP_QMV_F32 requires
QMV_F32 and MTP. An explicit parent `0` disables an implied child. Explicit child
values remain authoritative, including `0`. Saved-store exact-zero opt-outs,
absent-artifact fallback, corrupt-metadata errors, and the single hardware probe
remain unchanged.

Root observed all 21 target outputs unchanged and all 22 service quality and
lifecycle checks passing, plus 128-token autoregressive parity. The private
synthetic proposal-head 1% error screen still failed. These are different
claims: service coherence and target parity do not turn that failed synthetic
bound into a pass. The proposal F32 route retains its declared arithmetic
alternative and needs that limitation preserved in future tuning.

Static/effective review snapshots are
`build/release/flash/local-profile-v6-candidate{,-static}.json`. Dynamic weight
paths remain outside the accepted static profile, preserving conversion/Q4
fallback when an optional artifact is absent. Original checkpoint payloads and
the remote model roster are untouched.
