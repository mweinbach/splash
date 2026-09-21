# Private completed native command retention

This diagnostic tests whether keeping the last two successful native compute
command objects alive changes repeated idle Metal Wire Memory work. It adds no
GPU commands, encoders, kernels, weight copies, or changes to model math.

`SPLASH_FLASH_PRIVATE_COMPLETED_COMMAND_RETENTION=1` enables the private backend.
Absent or `0` remains off; every other value rejects before device creation.
An optional `SPLASH_FLASH_PRIVATE_COMPLETED_COMMAND_RETENTION_TRACE` must name a
fresh file; JSONL gives accepted/dropped command count and unique retained
allocation base count/bytes. Counts concern retention, not physical residency.

Only MetalBackend::Impl strongly owns the store. The completed callback captures
a weak pointer to that store, avoiding store→command→callback→store cycles.
Before publishing command completion to the normal ticket, a successful native
command plus its existing deduplicated C++ allocation owners enter a two-slot
FIFO. Holding allocation owners keeps zero-copy mmap backing alive and keeps
normal allocation accounting accurate. Shared weight/scratch allocations do
not gain duplicate ledger charges. Up to two old request states can remain
owned, at most 698,777,600 bytes in this round's capacity8192 trunk-only fixture.

The FIFO has its own mutex; no ticket, gate, or command mutex is acquired while
holding it. Displaced owners retire outside the FIFO mutex. Stop closes and
empties the store before stopping async state; racing callbacks cannot refill a
closed store. Diagnostic allocation failure closes/empties retention without
turning a successful model command into a failure. Failed/uncommitted commands
are not retained. Normal ticket consumption still clears its original vector
and opens the submission gate.

The SDK only promises command resource ownership required for execution. A
completed command may shed its own resource list even while its object remains
alive. This diagnostic also keeps C++ owners, and tests the mechanism rather
than assuming it guarantees GPU VM residency.

Frozen local binary and Metal4.1 library:
`build/flash-private-completed-command-retention-v9/splash-flash` and adjacent
`splash.metallib`. The backend was compiled freshly; other normal public ABI200
objects and library bytes came from the frozen idle-refresh v9 runtime. That
runtime's refresh backend is replaced; refresh behavior is not present here.
No production source, ABI header, local profile, or original model was edited.

Root alone executes the GPU. Compare this private binary flag0/flag1 with
128 prompt / one completion token, zero prefix cache reuse and nine seconds
idle; inspect fresh request traces/driver exports for actual wiring reduction.
Preserve output and lifecycle checks before any promotion. JSONL writes run in
the completion path and are diagnostic overhead; steady service timing should
also be measured without the optional sink.

CPU qualification is in
`build/release/flash/private-completed-command-retention-v9-cpu-qualification.json`.

## Root measured result

Six deterministic HTTP probes passed, each with128prompt tokens and zero cached
reuse. Retention accepted11native completed commands, dropped none, and never
held more than2. The final FIFO held839unique bases /115,188,141,312bytes;
these are shared owners, not additional weight allocations. Normal dense ledger
rose692,486,144bytes, within the two full trunk state bound698,777,600bytes.

After nine seconds idle, commit-return→actual-GPU-start remained1243.103ms for
a one-token request and1296.015ms for a two-token eligible request. Immediate
requests measured2.608/5.421/5.097ms. GPU target work was325–333ms. This confirms
the completed-object/allocation lifetime extension did not remove the idle wait.

There is no same-build flag0 comparison in this screen; differences from older
v8 samples cannot be credited as a retention speedup. These probes used command
tracing and retention JSONL, and no driver export was recorded; this screen does
not directly recount Wire Memory events or replace full model qualification.

The diagnostic remains private/default-off. Detailed measured audit:
`build/release/flash/v9-completed-retention-cpu-summary.json`.

Root shutdown then emitted two idempotent stopped records, both closed with
zero commands, bases and bytes; the retention store drained.
