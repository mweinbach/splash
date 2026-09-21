# Private immutable-resource maintenance

This default-off Worker overlay tests a local remedy for the idle GPU rewiring
delay. Production source, local defaults, OS settings and model weights are
unchanged. Root owns all GPU and HTTP execution.

Build:

```
make -f dev/benchmarks/idle_residency_maintenance_v11/Makefile \
  BUILD=build/flash-private-idle-maintenance-v11 -j8 flash-next
build/flash-private-idle-maintenance-v11/splash-flash --cpu-self-test
```

Use the existing qualified v8 model and all persisted operand/expert paths. The
private route refuses any different source, layout, model geometry, original
base list or immutable union. The exact selection is all21 original bases plus
the main and head `cachedOperandsOnly()` base/rank lists:1,134 unique disjoint
Shared native owners,144,326,852,608 bytes. No mutable request, target or head
workspace participates. Nothing copies/reformats weight payloads.

```
SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY_MAINTENANCE=1
SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY_INTERVAL_MS=500
```

The first flag accepts only unset/0/1. The optional interval accepts only
canonical decimal100..1000 milliseconds and defaults500. Readiness does not
run this command or shift the first-request wire cost into startup. It arms
after a real user request emits tokens and completes successfully after GPU
work; constructors, cancelled requests and maintenance do not arm it.

Each command makes one uint read per owner,4,536 total source bytes, through
one reflected readonly Tier2 argument buffer. One256-threadgroup computes a
32-bit checksum and writes each mixed word/count into its own guarded16KiB
diagnostics buffer. CPU validation checks every output word and all guards.
Only two native diagnostic buffers are added, admitted before construction:
an argument buffer rounded16KiB and one16KiB output; the actual ledger delta is
reported. No extra weight backing or second weight charge is created.

The existing Worker thread submits one synchronous command at its idle safe
point. There is no timer thread, external automation or command mutex. All
active/pending/live requests must be empty, no command ticket or sparse unmap
may be outstanding, execution must be healthy and the reader queue empty.
Pressure decisions sample the governor afresh; measurement validity, growth
allowance, Normal effective/system pressure, zero reservations and strictly
more than hostReserve+2GiB available are required. A failure skips maintenance
and moves the due time by one interval. Long cold maintenance commands disarm
the route until another successful user request. Healthy diagnostic errors
disable maintenance while preserving user service; native GPU command errors
retain the backend's normal unhealthy behavior.

The reader queue is rechecked after status publication immediately before
submission. A request arriving after that check can wait for one synchronous
maintenance command. The500ms interval is not a bound on command duration or
request latency: driver rewiring can still take a second after pressure or
OS scheduling gaps. Status reports actual maximum/last/cumulative GPU and wall
times, cold misses, suspensions, failures, source count/bytes and added ledger
bytes. `/status.private_idle_residency_maintenance` is separate from model
request timings. Optional request tracing labels maintenance with its own
phase/role and zero request lanes/rows, consuming its profile immediately.

Pure CPU policy tests and the actual extracted Scheduler timer tests must pass
before Root tests idle9/30s, wired memory, correctness/lifecycle, concurrent
throughput and maintenance duty. Public Metal APIs still do not guarantee
physical pinning; the remedy requires measured evidence before promotion.

## Production integration after the private proof

Root's private9/30-second idle proof succeeded, so the same owner-touch and
actual timer logic now live in `runtime/flash/FlashIdleResidency*` and the
production shared shader. Runtime defaults remain off pending full production
service qualification; local hardware profile activation belongs to Root.

```
SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=1
SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS=500
```

Production status is `idle_residency_maintenance`. It reports `requested`,
`available` and `failure_reason`. This qualified model may run with optional
saved operand/expert stores disabled: maintenance then skips with a clear
reason. A different source/layout/original model geometry still refuses the
explicit flag. The helper uses the actual native ledger delta, observed25,456
bytes, bounded by its conservative32,768-byte reservation.

The frozen production runtime is
`build/flash-idle-maintenance-production-v11-v1/splash-flash` with the adjacent
Metal4.1 library. Root must validate the exact same binary with the flag off
and on; public token/weight/math behavior is unchanged. Tests under
`dev/tests/flash/idle_residency_*test.cpp` include the actual production policy
and Scheduler headers rather than copies of their logic.
