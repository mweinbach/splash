# Local idle residency maintenance

The server can avoid the measured post-idle rewiring penalty by periodically
exercising its immutable resource owners while truly idle. This is a local
worker feature; it does not change model weights, arithmetic, the OS collector
or global GPU memory limits.

The command reads one32-bit word from each of1134 existing native owners,
including original weights and verified saved operands, then validates its own
guarded output. These owners total144,326,852,608 bytes (134.4 GiB). It adds
25,456 bytes of diagnostic allocation under a conservative32 KiB admission,
and no weight backing. The shader's4536 logical read bytes are not a measured
physical bandwidth number.

Maintenance is default-off at the native layer. It arms only after a successful
user request that completed GPU work, runs every500 ms only with no active,
queued or pending request/command, and checks the incoming queue again just
before submission. The reader remains free to queue requests. A request may
overlap one already-started maintenance command;500 ms is a cadence, not a
latency bound. Pressure or insufficient host reserve suspends it. A command
whose wall duration reaches the interval disarms it until the next successful
user request. Healthy diagnostic failures disable maintenance safely; an
unhealthy backend retains the normal failure behavior.

The production switches are:

```
SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=1
SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS=500
```

An explicit maintenance0 opts out. Canonical intervals100..1000 ms are accepted.
Unknown model identity refuses explicit enablement. The qualified model missing
its complete saved operand/expert union continues normally with maintenance
unavailable and a status reason.

## Measured private proof

With identical128-token prompts, output budgets1/2, and zero KV reuse, nine-second
idle first-token latency fell from1692/1737 ms to389/410 ms. Queue time remained
about23–25 ms; the delay was removed rather than moved into admission.
All corresponding request bodies, outputs and token usage matched.

After30 seconds idle, a16-output request reached first content in397 ms versus
377 ms immediately. System wired memory stayed148.32–148.38 GiB at10/20/30 s.
During measured idle intervals maintenance GPU execution duty was about0.004%,
with roughly0.13% wall-time duty. This is GPU command timing, not a measured
power/energy saving. Keeping the existing144.3 GB weight backing wired reduces
RAM available to other applications; pressure guards intentionally permit
unwiring when memory is needed elsewhere.

All22 private service quality/lifecycle cases passed, including concurrent
requests, tools, structured output, cancellation, deadlines and recovery.
Production same-binary OFF/ON confirms the nine-second mean first-token latency
change1635.02→395.82 ms (75.79% less). Queue time changes only+0.84 ms, all six
outputs/inputs/usages match, and target driver waits fall from1241–1277 ms
to7.6–9.3 ms. No resource padding/state math changes. The first cold request
still pays preparation; maintenance arms after warmup rather than hiding that
cost in startup. A production30-second probe reaches first content in392.94 ms
versus370.45 ms immediately, with exact16-token outputs.

Production lifecycle qualification passed all22 checks with zero new task
regressions. The same-binary full-output HTTP comparison preserves all21
benchmark outputs and native work. Active throughput differs by small mixed
percentages; both runs have Command metadata tracing enabled and are matched
diagnostic controls, not untraced publication scores. Maintenance avoids active
work and model phase counters remain separate from its own command counters.

Local v9 now enables the feature with38 static flags and a256 GiB memory floor,
preserving historical gates. All121 profile/launcher CPU tests pass with no
skips. The normal launcher selects the same binary and metallib as the qualified
production runtime, byte-for-byte. With tracing off, its first cold request
still takes1574 ms, while the next request after9 seconds idle reaches first
content in396 ms versus370 ms immediately. All16-token outputs match, and the
service returns to healthy idle on8011 with maintenance enabled. OS settings
and GitHub remain unchanged.

This is a server-level mitigation of the observed OS reclaim penalty, not a
permanent physical-pinning guarantee. Host pressure, sleep or long scheduling
gaps can suspend it and allow a later cold preparation cost. Disable it with
`SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=0` to prefer lower idle wiring/use.

## Evidence

- `build/release/flash/v11-private-maintenance-on-idle-http.json`
- `build/release/flash/v11-private-maintenance-on-long-idle-http.json`
- `build/release/flash/v11-private-maintenance-positive-screen-cpu-audit-v2.json`
- `build/release/flash/v11-private-maintenance-on-quality.json`
- `build/release/flash/v11-idle-maintenance-production-compilation-qualification.json`
- `build/release/flash/local-profile-v9-promotion-plan.json`
- `build/release/flash/v11-production-maintenance-off-idle-http.json`
- `build/release/flash/v11-production-maintenance-on-idle-http.json`
- `build/release/flash/v11-production-maintenance-on-long-idle-http.json`
- `build/release/flash/v11-production-maintenance-idle-cpu-audit-v2.json`
- `build/release/flash/v11-production-maintenance-on-quality.json`
- `build/release/flash/v11-production-maintenance-on-quality-comparison.json`
- `build/release/flash/v11-production-maintenance-on-http-performance.json`
- `build/release/flash/v11-production-maintenance-off-http-performance.json`
- `build/release/flash/v11-production-maintenance-matched-http-cpu-audit-v2.json`
- `build/release/flash/default-v9-normal-idle-proof.json`
- `build/release/flash/default-v9-normal-final-idle-status.json`
- `build/release/flash/default-v9-build-identity-comparison.json`
- `build/release/flash/default-v9-idle-fix-summary.json`
