This private candidate keeps the qualified v5 selected-INT8 hit and original
Q4 direct-miss math while changing only GPU job traversal. Current production
launches both classes over the full fixed job capacity for gate/up and down.
Many groups immediately return because the job is inactive or belongs to the
other weight class.

A fixed persistent grid reads the GPU-produced job count and traverses
`job = group.y; job < count; job += grid.y`. Every valid job remains owned by
exactly one group for each output-column tile. Branches and loop decisions are
uniform across the threadgroup, and a threadgroup barrier separates iterations
before scratch reuse. An initial masked helper invocation preserves parameter
and diagnostic checks even for an empty job list.

The hit and miss kernels remain separate. Lean BF16 × signed-INT8 whole-K hit
kernels therefore do not inherit the Q4 miss kernel's 16 KiB gate/up or 8 KiB
down weight staging. The existing M/N64/K traversal, F32 scale multiplication,
BF16 boundaries, sanitization, excluded-route poisoning and canonical scatter
remain unchanged. The candidate allocates no new operand or job backing and
uses no CPU count readback. Load balance and cache reuse may change; a speedup
is a hypothesis until Root measures it.

Current `ComputeDispatch`/`CommandGraph` support host launch dimensions only.
Argument-buffer indirect-resource support is hazard/ownership tracking, not
GPU indirect dispatch. A compact-list alternative can safely reduce static
hit/miss capacities to `ceil(routes/M)+H−1` and
`ceil(routes/M)+(512−H)−1`, but requires a GPU partition pass. A single fused
weight-class branch can halve launches while giving every INT8 hit the miss
path's shared-memory footprint. Neither alternative is selected here.

The private oracle compares every live activation, canonical expert-down and
combined BF16 byte against the current v5 mixed INT8/Q4 producer before and
after alternating matched commands. It also checks independent bucket/job
expectations, sticky diagnostics, all scratch/output guards, source and sidecar
SHA immutability, and retained mapping/parameter lifetime.

All 41 dependency objects and 56 baseline AIRs came from fresh
`build/flash-default-v5`. The compile-time `sizeof(CommandTiming)` is 200.
Every published timing sample must be finite and greater than 1e−9 seconds;
old 16-byte timing ABI objects are not accepted. Object/source/library hashes
are recorded in `build/flash-int8-persistent-jobs/private-freeze.json`.
Compilation and CPU self-tests submit no GPU work; Root runs screens serially.

```sh
FLASH_I8_JOBS_LAYERS=0 FLASH_I8_JOBS_ROWS=2048 FLASH_I8_JOBS_TILE=32 \
FLASH_I8_JOBS_PATTERN=mixed FLASH_I8_JOBS_GATE_Y=64 \
FLASH_I8_JOBS_DOWN_Y=16 FLASH_I8_JOBS_PAIRS=4 \
build/flash-int8-persistent-jobs/flash-int8-persistent-jobs-oracle \
  build/flash-int8-persistent-jobs/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  install/local-models/Flash-Next-int8-experts-top64-v1 \
  build/release/flash/persistent-jobs-m32-g64-d16.json
```

Use `LAYERS=47`, `ROWS=8192 TILE=64`, or `PHASE=gate`/`down` to isolate scope.
Grid widths accept 1..128; useful bounded screens are gate 32/64/128 and down
8/16/32. Patterns include `hit-concentrated`, `hit-spread`, `miss-only`, `mixed`
and `spread-all`. The default runs layers 0/47 and all five patterns.

`FLASH_I8_JOBS_ROUTE_IDS` can select
`build/flash-int8-persistent-jobs/captured-layer0-r2048-ids.i64` or its layer47
counterpart with the matching single-layer setting. These real 512-expert
route distributions were extracted from the existing source-matched prefill
calibration. ID-only screens retain synthetic activation values and are
labeled accordingly. `FLASH_I8_JOBS_INPUT` additionally accepts a real BF16
activation fixture with exact extent; it requires route IDs.

Production files, defaults, original checkpoints and saved formats are not
modified by this candidate. A meaningful primitive gain and full-model
qualification are required before promotion.
