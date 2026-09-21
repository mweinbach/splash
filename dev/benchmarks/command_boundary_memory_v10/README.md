# Private command-boundary memory query control

This backend-only diagnostic isolates the four `MTLDevice.currentAllocatedSize`
reads made immediately before commit, immediately after commit, in the scheduled
callback, and in the completed callback. It changes no encoders, command
ownership, resource bindings, pipelines, math, sparse events, ticket lifecycle,
or public C++ ABI. The copied normal v8 metallib is byte-identical.

`SPLASH_FLASH_PRIVATE_SKIP_COMMAND_MEMORY_QUERIES=1` skips these reads.
Missing or `0` retains them. Invalid values reject before device creation.
Allocation/pipeline/sparse lifecycle samples and explicit `refreshMemoryStats`
queries remain intact, as do the allocation ledger and MemoryGovernor admission
and host-pressure checks. Cached device totals remain last observed totals;
the diagnostic does not promise a fresh command-lifecycle sample or replace
admission accounting. Skipped timing spans remain unavailable and counts zero.

Both modes add one relaxed atomic counter per attempted boundary. There are no
file writes, logs, new mutexes, or heap allocations at command boundaries.
Backend async-state teardown prints one JSON line to stderr with queried/skipped
counts in `[precommit,postcommit,scheduled,completed]` order. Backend state already
survives until the final Metal callback, so this output follows safe retirement.

CPU sanitizer policy checks and ABI200 checks pass. Frozen runtime:
`build/flash-private-command-memory-v10/splash-flash` and adjacent metallib.
The build manifest records the original normal v8 binary, all reused host
objects, fresh backend object, private source snapshots, and library hashes.
No production/default/profile/model source is modified and no GPU was executed.

Existing evidence already makes a blocking getter an unlikely explanation.
Ten saved whole-graph samples show each query taking at most 1.292 microseconds
while the kernel-driver interval lasts up to 1.340 seconds. Pre/postcommit
reads precede the driver interval; callbacks sample after scheduling/completion.
This control can exclude an unobserved deferred getter side effect, but query
duration must not be interpreted as a demonstrated source of the delay.

The existing driver windows wire the same 1378 resources / 145,943,855,104
requested bytes. Wire active unions are 416.03 and 596.01 ms; the envelopes
contain another 537.01 and 539.76 ms between Wire events. Those gaps are not
proven idle and could contain uninstrumented driver work, lock waits, or
descheduling. A native CPU stack profile over immediate/idle pairs is the next
measurement that can distinguish them. Resource/connection IDs were unavailable,
so the association remains PID-scoped and temporal.

Reproducible CPU analysis:

```sh
.venv/bin/python -B dev/benchmarks/command_boundary_memory_v10/audit.py \
  --output build/release/flash/v10-existing-boundary-memory-cpu-audit.json
```

Root alone runs GPU experiments. Compare this same private binary OFF/ON using
the frozen 128-prompt / one-output workload, zero reused prefix, immediate versus
nine-second idle pairs, exact expected output, and scheduler idle after each.
Use command-level profiling only when comparing driver interval and query counts;
no counter sampling or stage boundaries need to change. Check teardown counters.
