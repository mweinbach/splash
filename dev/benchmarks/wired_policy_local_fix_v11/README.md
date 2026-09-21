# Temporary OS policy experiment

This is a prepared fallback experiment, **not an applied or qualified fix**.
It changes only `iogpu.disable_wired_collector` from its current value to `1`
while one bounded diagnostic command executes, then restores the exact original
value. It never writes `wired_limit_mb`, `wired_lwm_mb`, `dynamic_lwm`, launchd,
`sysctl.conf`, or the normal Splash profile. Administrator execution is external;
the Python tool never invokes sudo or asks for a password.

## What is established

The installed macOS SDK declares `MTLDevice.recommendedMaxWorkingSetSize` and
`maxBufferLength` readonly. There is no public setter for those properties. The
former is an approximation for good performance, not a public permanent-wiring
reservation. `MTLResidencySet.commit` says that it *tries* to make additions
resident. Retaining a residency request cannot be equated with retaining all
physical wiring. Actual normal-profile measurements demonstrate idle unwiring
despite live model resources and a standing residency lease.

MLX's `set_wired_limit` controls which allocations MLX places in public Metal
residency sets. It is not a per-process equivalent of the IOGPU collector
sysctl. Splash already uses those public sets, so importing MLX's API would not
provide a new supported guarantee. See [MLX residency implementation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/resident.h)
and [MLX allocator](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp).

Apple's proprietary IOGPU collector implementation is not in the inspected
public SDK or Apple OSS material. The low-water mark and dynamic-low-water mark
sysctl names do not establish their exact semantics or interaction. No
primary Apple documentation/source was found validating that `wired_lwm_mb`
would impose a permanent pinned floor. Setting a large value by analogy to
generic memory watermarks would therefore be an uncontrolled guess. Raising
`wired_limit_mb` addresses a ceiling, whereas the measured live frontier is
already below the device's recommendation.

The collector boolean is a narrower single-variable hypothesis: disabling that
idle actor may prevent the measured decay, but this has **not been tested on
this device**. It is a system-wide private policy rather than a public per-process
API. If effective, it can retain approximately146 GB of additional GPU wiring,
leaving those pages unavailable for ordinary OS reclamation while the native
resources are alive. This does not allocate another146 GB, but it changes the
reclaimable state of existing memory. Other GPU applications may be affected.

## Read-only preview

From the repository, with the project Python:

```sh
.venv/bin/python -B dev/benchmarks/wired_policy_local_fix_v11/temporary_policy.py
```

The preview prints current policy values and system VM counters. It does not
touch Metal or send an HTTP request. `potential_extra_wired_bytes` is derived
from the previous driver trace, not a guaranteed increment for this test.

## Optional administrator test

Only the serial GPU coordinator should arrange execution. First capture the
normal-policy control using the existing fixed128-token/one-output idle
threshold probe and opt-in native command trace. Keep the same server/model,
trace settings, fixture and desktop background for the temporary policy arm.
The child below performs the requests; the wrapper controls only the policy.

```sh
sudo /Users/mweinbach/Projects/splash/.venv/bin/python -B \
  dev/benchmarks/wired_policy_local_fix_v11/temporary_policy.py \
  --apply-temporarily --run-root-gpu --timeout-seconds 120 \
  --report build/release/flash/v11-temporary-collector-transaction.json -- \
  /Users/mweinbach/Projects/splash/.venv/bin/python -B \
  dev/benchmarks/flash_idle_threshold_probe.py \
  --output build/release/flash/v11-temporary-collector-idle-http.json \
  --run-root-gpu
```

This example is reviewable and **has not been executed**. The wrapper refuses
non-administrator execution, missing coordinator acknowledgment, existing report
files, or an already disabled collector. It saves the exact original restore
command to disk before writing. Success, ordinary exceptions, SIGINT, SIGTERM,
SIGHUP and child timeout all reach restoration. SIGKILL, a kernel crash, or a
power failure cannot execute cleanup; the saved report supplies manual restore
arguments, and no persistent configuration is installed.

The initial control and final normal-policy arm should measure native
postcommit-to-GPU delay, actual GPU time, system wired pages after0/1/3/9seconds,
page/swap deltas, HTTP correctness and a healthy idle scheduler. A successful
wrapper transaction alone is not proof of a model fix. Do not promote the toggle
to an OS default based on one trial.

## CPU qualification

```sh
.venv/bin/python -B -m unittest \
  dev.benchmarks.wired_policy_local_fix_v11.test_temporary_policy
```

The tests use in-memory settings only. They cover successful restoration,
child failure, interruption, already-disabled/unknown values, a write followed
by failed readback, and concurrent administrator changes. No GPU or sysctl writes
are involved.
