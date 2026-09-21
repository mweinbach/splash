# Private residency-set granularity diagnostic

This private backend and Worker snapshot tests whether the large immutable
residency group changes the driver’s recurring cost after inactivity. It does
not enable a performance route or modify production source/defaults.

Use the same frozen binary, model package, profile and full-original union:

```
SPLASH_FLASH_PRIVATE_FULL_ORIGINAL_RESIDENT=1
SPLASH_FLASH_PRIVATE_RESIDENCY_SET_CAP_BYTES=0
```

The one-set control uses cap `0`. The capped control uses
`SPLASH_FLASH_PRIVATE_RESIDENCY_SET_CAP_BYTES=8589934592` (8 GiB).
Both use identical initialization: create an empty set, issue one standing
`requestResidency()`, add existing deduplicated native allocations, commit, then
attach to the same legacy command queue. The full-original private Worker keeps
its source/layout/geometry and host-reserve admission checks. Native allocation
selection, views, payload bytes, model graph and recurrent math are unchanged.

The deterministic first-fit planner groups whole native allocations, refusing
an individual allocation larger than a nonzero cap or more than 32 sets. It
never slices/copies weights. Its empty-group, parse, integer overflow, exact
capacity, 32-set limit and random partition checks passed 147,335 assertions
under ASan/UBSan. No weight backing or double weight ledger is added; set
metadata belongs to the driver and can change its measured allocation total.

The lease and backend retain the same native allocations through shutdown.
Per-set requested/attached state supports safe cleanup after partial startup
failure. `/status.private_full_original_residency` adds actual set count, cap,
per-set caller-ledger resource bytes and a false physical-pin-verification field.
The existing registered-union count/bytes remain authoritative. Actual physical
residency cannot be queried through these public getters.

Build and CPU qualification:

```
make -f dev/benchmarks/partitioned_residency_v10/Makefile \
  BUILD=build/flash-private-partitioned-residency-v10 -j8 flash-next
build/flash-private-partitioned-residency-v10/splash-flash --cpu-self-test
```

All 44 inherited Worker CPU checks passed. The frozen binary and adjacent
Metal4.1 library are in `build/flash-private-partitioned-residency-v10`.
GPU, idle, full service and performance validation belong to Root and are
pending. There are no OS policy/sysctl changes and no background keepalive.

## Verified primary-source contracts

Apple’s SDK states `recommendedMaxWorkingSetSize` is an approximation of a
performance-safe working set, not physical memory size or a definitive hard
limit. `maxBufferLength` is the maximum single resource length and does not
state a residency budget. `currentAllocatedSize` counts resource allocation;
it is not physical wired memory.

The Apple residency article says the framework performs CPU work to make
allocations resident, by default at first command-buffer commit, and that this
can delay GPU submission. `requestResidency()` can perform it ahead of time but
may postpone work when other apps have competing memory needs. Residency sets
can keep allocations resident indefinitely; this is not an API guarantee that
they can never be evicted or must stay physically pinned under OS policy.
Queue association is snapshotted onto command buffers at commit.

Metal4 command allocators manage encoded-command memory. Metal4 queues use the
same residency-set protocol and 32-set queue cap as legacy queues. Public docs
do not promise that switching command models bypasses the OS residency policy.

Current primary MLX implementation is materially different from a device-limit
setter: `set_wired_limit()` only changes MLX’s allocator budget and resizes its
committed residency sets. It does not write an OS sysctl or set a private device
working-set limit. Current MLX bounds group size with `MLX_RESIDENCY_SET_MAX_PCT`
(default5% of recommended working set). Its source comment explicitly says
macOS makes residency decisions per set and that losing one group under GPU
memory pressure makes that group’s allocations require residency again. This
comment supplies the hypothesis tested here; it does not independently prove
Splash’s driver uses precisely that policy after idle.

- [Apple residency article](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets)
- [Apple Metal4 architecture](https://developer.apple.com/videos/play/wwdc2025/205/)
- [Apple working-set explanation](https://developer.apple.com/videos/play/tech-talks/10580/)
- [MLX allocator source](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp)
- [MLX residency group source](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/resident.h)
- [MLX residency initialization](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/resident.cpp)

## Root’s current diagnostic implication

Root reports a system-wide VM wired-memory collapse from about149.6GiB after
model execution to13.75GiB by3 seconds idle, beginning around1.5 seconds. This
is temporal system-wide evidence, not an individual resource or command join.
Together with retained model allocations/leases and recurring next-command
WireMemory spans, it strongly favors driver residency reclaim/remapping over
model reload. System page-in/swap-in and task footprint deltas are a useful
orthogonal check of actual paging. The proprietary collector’s precise trigger
and policy are not documented by a public Apple API and remain unproven.


## Root’s completed grouping result

Both conditions registered exactly1,134 existing native allocations totaling
144,326,852,608 bytes. Cap0 used one set and cap8GiB used25 sets, each within
its cap. Both retained an active standing residency registration and all20
corresponding requests had equivalent output bytes, zero cached prompt tokens,
128 actual prompt rows and1,709 target dispatches. Both finished healthy idle.

Grouping did not prevent reclaim or the recurring stall. The two9-second-idle
requests averaged1,386.89ms target commit-to-GPU wait with one set and
1,782.18ms with25 sets. Their actual target GPU execution averaged344.00 and
344.76ms respectively. Driver kernel-processing timing accounts for nearly all
of each extra wait. HTTP first-content averages were1,764.48 and2,158.77ms.
These are traced diagnostic results, not steady-state throughput scores; process
order and background activity prevent a precise attribution of the difference
between conditions. There is no justification to enable capped grouping.

During the5-second VM windows, wired memory fell by145.47GB(one set) and
146.32GB(capped). Page-ins were only19.73MB and6.90MB respectively; swap-ins,
swap-outs, page-outs and compression all stayed unchanged. File-backed nonwired
resident page counts grew by approximately144GB in each condition. This
systemwide evidence supports unwiring while keeping payload resident in RAM,
followed by the next command’s expensive driver wiring/mapping preparation.
It does not identify the proprietary collector’s exact policy or causally join
individual pages to a command.

The reproducible CPU audit verifies56 invariants and freezes artifact hashes:
`build/release/flash/v10-residency-grouping-cpu-comparison.json`. Recreate it with
`.venv/bin/python dev/benchmarks/partitioned_residency_v10/audit.py` after Root’s
reports and JSONL traces are present. No further GPU experiment, OS policy
change, production edit or default change resulted from this control.
