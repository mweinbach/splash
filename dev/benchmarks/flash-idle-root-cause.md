# Idle-delay root-cause investigation

The recurring delay is before GPU execution, while the Metal/IOGPU driver
prepares the resources bound by the next command. Idle loss of physical wiring
and subsequent resource-dependent driver preparation are directly observed;
the combined evidence identifies idle reclamation followed by costly rewiring.
The exact proprietary collector policy is not exposed by the public API.
All diagnostics are local; the qualified v8 model math/profile is preserved.

## Idle threshold and actual execution

Fifteen fixed128-token/one-output requests, greedy reasoning disabled with
zero prompt cache reuse, reproduce a transition between1.005 and2.005 seconds
of inactivity. Excluding the first uncontrolled cold request, median
commit-to-hardware-GPU wait is4.71 ms below the threshold versus1296.63 ms
above it. Actual GPU work is333.07 versus343.34 ms. Metal kernelStartTime to
kernelEndTime explains essentially the added interval. Pipeline compilation,
HTTP scheduling and slower GPU math cannot explain that postcommit interval.

CurrentAllocatedSize reads at command boundaries take microseconds in the
captured profiles; no user stack sample shows a blocking allocation-size
getter. The native main thread waits for CommandTicket completion, and Metal's
submission thread enters IOGPUCommandQueueSubmitCommandBuffers through an
IOKit kernel trap. The mapping thread's permanent listener does not establish
that it is actively processing mappings; user-space stacks cannot resolve the
kernel's internal work.

## Wired pages change while data remains resident

After a real request, system wired memory stays near149.38 GiB through1.25
seconds, falls to107.36 GiB at1.5 seconds and49.17 GiB at1.75 seconds, and
reaches13.73 GiB by5 seconds. The decline is145,926,045,696 bytes, within
17.8 MB of the145,943,855,104 bytes requested by native driver WireMemory
events in the earlier captured command. These are different runs and
system-wide counters, so this is a close corroboration rather than an exact
resource/PID join.

File-backed resident-page accounting rises by144.31 GB as wired pages fall.
Only802,816 bytes of page-ins occur during that window, with zero page-outs,
swap-ins or swap-outs. The pages largely remain in RAM while becoming unwired;
this is not evidence of a146 GB model reload or disk read. The349.39 MB request
state is explicitly released, but weight buffers and their residency lease
remain alive. The much larger timed unwiring is not application destruction
of original weights.

## Minimal resource-volume experiment

A standalone Metal probe executes no inference kernels. Each command reads
one32-bit word per bound native allocation and validates guarded outputs.
All kernels/resources are prepared before idle. The same106.32 GB model can
be loaded but left unbound by the tiny command.

| Bound native resource bytes | Wait after3 s idle | Wait after a tiny GPU wake |
| --- | ---: | ---: |
| Tiny-only/no model | 9.34 ms | — |
| Tiny/model loaded | 9.11 ms | — |
| 5.20 GB /1base | 59.30 ms | 65.40 ms |
| 20.78 GB /4bases | 175.30 ms | 167.15 ms |
| 106.32 GB /21bases | 848.34 ms | 854.81 ms |

Immediate repeats of the same large resources wait about0.11 ms. Actual GPU
execution is measured in microseconds; preparation, encoding and commit are
negligible. A tiny wake executes quickly yet does not make the subsequent large
resources ready. Thus global hardware wake alone is insufficient, and simply
having a large mapped model in the process is insufficient. Bound-resource
preparation is the expensive operation. Resource count and bytes vary together
in this screen, so their individual costs are not independently separated.

## Public policy and remaining scope

The system has256 GiB physical memory; Metal reports239,143,780,352 bytes of
recommended working set. The measured145.94 GB resource frontier is below
that recommendation. Kernel settings remain unchanged: wired_limit_mb=0,
wired_lwm_mb=0, dynamic_lwm=1, disable_wired_collector=0. Zero selects the
platform's default policy; it does not mean a zero GPU budget.

Apple documents CPU work required to establish GPU residency and warns that
requests may be postponed under competing memory needs. Current MLX explicitly
caps residency-set sizes because macOS makes residency decisions per set.
Its set_wired_limit manages MLX allocator membership, not an OS sysctl.
These documented mechanisms explain why a retained requested set is not proof
of permanent physical wiring. A private one-set versus8 GiB-capped-set control
uses identical resources and initialization order to test grouping. One set
and25 capped sets both contain1134 existing allocations/144,326,852,608 bytes;
both lose nearly all extra wiring by3 seconds idle. After9 seconds, one-set
postcommit waits are1378/1396 ms, versus1805/1760 ms capped. All requests remain
correct, immediate waits remain a few milliseconds, and no extra backing is
created. Grouping does not fix the observed reclaim and stays private/off.
Its2-second request was fast and its VM decline completed near3 seconds,
showing that the observed expiry window varies with conditions; the1–2 second
threshold is specific to the normal-profile sweep rather than an exact universal
timeout. No collector toggle, global limit write or background keepalive
is part of this investigation.

The actionable cause is the huge native resource frontier that must be prepared
again after idle. A harmless four-byte view still refers to its whole native
owner. Every original MoE bank remains a possible miss input, and PLE indirect
lookup legitimately declares all source owners. Reducing declarations safely
requires a different loading/access strategy that binds only resources actually
needed; simply dropping declarations would make GPU access invalid. A public
residency request alone does not guarantee this OS leaves every page wired.

The qualified v8 defaults, model payloads and OS settings remain unchanged.
Normal untraced service is restored after diagnostics, with final readiness,
smoke and idle proof recorded separately.

## Evidence

- `build/release/flash/v10-idle-threshold-cpu-audit.json`
- `build/release/flash/v10-idle-wired-pages.json`
- `build/release/flash/v10-wired-page-reclaim-cpu-audit.json`
- `build/release/flash/v9-idle-driver-cpu-audit.json`
- `build/release/flash/v10-resource-root-cause-cpu-summary-v2.json`
- `build/release/flash/v10-idle-driver-user-stacks.txt`
- `build/release/flash/v10-residency-one-set-idle9-http.json`
- `build/release/flash/v10-residency-capped-sets-idle9-http.json`
- `build/release/flash/v10-residency-grouping-cpu-comparison.json`
- `build/release/flash/v10-idle-root-cause-summary.json`
- `build/release/flash/v10-restored-v8-idle-status.json`

Primary references: [Apple residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets),
[MLX residency policy](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/resident.h),
[MLX allocator](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp).
