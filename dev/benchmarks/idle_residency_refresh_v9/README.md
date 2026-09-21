# Private on-demand idle residency refresh

This backend snapshot tests one mechanism: after more than five seconds since
the prior actual command completion callback, issue another `requestResidency()`
on the retained startup set before preparing the next command. No background
work, replacement set, teardown, weight copy, graph/kernel change, hazard change,
unretained references, or recurrent-state conversion is introduced.

`SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY_REFRESH=1` enables this private control;
absent/`0` remains off, any other value is rejected. The initial command has no
prior completion and does not refresh. Every eligible submission uses the same
committed/attached set, under commandMutex then gateMutex, while no outstanding
ticket exists and the backend is healthy/running. Stop cannot race teardown.
An API exception propagates before submission while retaining the original
lease. Shutdown still issues one original endResidency/remove.

Apple permits requestResidency after the set commit but does not guarantee that
residency preparation finishes synchronously. Repeated calls are consistent
with that precondition; nested request/reference counting is not documented.
[Apple requestResidency API](https://developer.apple.com/documentation/metal/mtlresidencyset/requestresidency/)

`SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY_REFRESH_TRACE=/absolute/FRESH.jsonl`
records every submission's next sequence, enable decision, refresh count,
actual-completion/refresh timestamps, idle interval, host API duration, and
exact existing registered native-base count/bytes. The default registration
contains saved-derived operands only: 1,113 bases / 38,006,423,552 bytes. Those
exclude the original 106,320,429,056 bytes; a negative saved-only result cannot
establish that refreshing original resources would also fail.

The submission timing starts before the refresh API, so normal command-profile
preparation includes its host cost. The dedicated oracle additionally measures
total Forward call wall time; HTTP Root probes must measure complete TTFT and
request duration. A shorter postcommit wait alone is not a speedup.

## Frozen artifacts and CPU checks

Fresh runtime and adjacent Metal4.1 library:
`build/flash-private-idle-residency-refresh-v9/splash-flash`.
Dedicated oracle: `build/flash-private-idle-residency-refresh-v9/idle-residency-refresh-v9-oracle`.

Built entirely with fresh normal public ABI200 objects. Worker CPU44 checks,
strict flag/idle policy974 assertions under ASan/UBSan, and --help pass without
GPU/device execution. CPU evidence/hashes:
`build/release/flash/private-idle-residency-refresh-v9-cpu-qualification.json`.
Production files, local v7 profile, original model mappings are unchanged by
this experiment. Root owns GPU exactness/performance/service tests.

## Root-only oracle

Apply the normal v7 environment and derived-store paths, stop Root's serving
process, then run serially:

```sh
build/flash-private-idle-residency-refresh-v9/idle-residency-refresh-v9-oracle \
  build/flash-private-idle-residency-refresh-v9/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/flash-v6-verifier-attribution/fixture/ctx128-width1-lane0.tokens.json \
  build/release/flash/FRESH-idle-refresh.json
```

Four fresh-state 128-token prefills run off/on immediately, then on/off after
separate nine-second idle intervals. All full BF16 vocabulary rows and two
continuation rows must match byteexactly; each target graph remains one command.
`FLASH_IDLE_REFRESH_IDLE_SECONDS=0..15`, `FLASH_IDLE_REFRESH_FIRST=off|on` can
change controls. The oracle can add the existing qualified full union using
`SPLASH_FLASH_PRIVATE_FULL_ORIGINAL_RESIDENT=1`; it checks source/layout,
21 original bases/106,320,429,056 bytes and host protected reserve. It excludes
Worker batch arenas/HTTP and therefore remains diagnostic evidence.

For HTTP saved-only tests use the fresh private runtime with normal v7 profile,
`SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY_REFRESH=1`, request-command trace and a fresh
refresh trace. The existing flash_idle_prefill_probe.py measures initial,
immediate, nine-second-idle ineligible/eligible requests and confirms idle
scheduler/zero cache reuse; Root must compare against same private build flag0.

## Root runtime result: saved-only refresh rejected

Root completed the saved-derived union oracle with full248320BF16 logits and
two continuation rows exactly equal across all four calls. After nine seconds
idle, refreshON took1760.68 ms total, including759.28 ms host API work and
648.40 ms postcommitwait. RefreshOFF took1394.71 ms total and1037.32 ms
postcommitwait. A shorter postcommitspan moved work into the host API and
made the fullcall365.97 ms slower; this is a negative result, not a speedup.

The38,006,423,552-byte saved-only registration excludes all original sources,
so this does not establish that another selection would behave identically.
It supplies no reason to enable refresh or add backgroundactivity. The flag
remains private/off. Completed Root report:
`build/release/flash/v9-idle-residency-refresh-saved-screen.json` and
`.commands.jsonl`.
