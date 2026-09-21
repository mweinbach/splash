# Local v9 proposal: preserve GPU accessibility while idle

This CPU-prepared proposal was activated locally after Root's production
qualification. The accepted launcher and `.splash-local-profile.json` match
`m5-ultra-flash-next-v9` with 38 static defaults: the 37 accepted v8 routes plus
`SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=1`. The helper
`_local_profile_v9_candidate()` still returns a separate review copy without
writing files, changing process environment or submitting GPU work. See
`flash-local-profile-v9.md` for accepted behavior and measured evidence.

The v9 hardware gate requires the exact qualified source, architecture, M5
Ultra identity and at least 256 GiB physical RAM. This is the memory capacity
on which the complete retained immutable union was measured. Historical
v5/v6/v7/v8 review copies keep their 30/34/36/37 flags and 192 GiB gates even
when generated under a hypothetical active v9. Source, aligned layout, saved
operand and TOP64 expert identity pins stay unchanged. Model math, quantization,
FP32 recurrent state, prefill windows and singleton draft depth stay unchanged.

After a real user request successfully emits tokens and completes GPU work,
the production Worker may submit one idle maintenance command every 500 ms.
The command reads one uint32 from each immutable owner through a reflected
readonly Tier2 argument buffer, then verifies the count, checksum and output
guards. The selection is 21 original weight bases plus verified main/head
cached operands: exactly 1,134 owners and 144,326,852,608 bytes, about 134.4
GiB. The command reads 4,536 source bytes; it does not rewrite weights or use
mutable request/model workspaces. Only an argument buffer and bounded output
buffer add native backing, and their actual allocation delta is admitted and
reported. The default interval comes from the runtime; it is not a 39th static
flag. An explicit `SPLASH_FLASH_IDLE_RESIDENCY_INTERVAL_MS` remains unchanged
for strict native validation of canonical decimal 100 through 1000.

Maintenance requires all active, pending, live and incoming request queues
empty, no command ticket or sparse unmap outstanding, and healthy execution.
The Worker samples the governor again before each command. Both system and
effective pressure must be Normal, growth and the host measurement valid,
reservations zero, and available memory strictly greater than the host reserve
plus 2 GiB. Pressure suspends maintenance. A cold command lasting at least one
interval disarms it until another successful user request. The first request
keeps its normal startup wiring cost; readiness does not run a hidden warmup.

This trades idle GPU work and retained wired memory for a faster request after
idle. It does not guarantee physical pinning or a power reduction; energy has
not been measured. A request arriving after the final queue check can wait
for one maintenance command, and 500 ms is the scheduling interval, not a bound
on that command's driver delay. `/status.idle_residency_maintenance` reports
actual timing, owner counts, added backing, pressure suspensions and failures.
No system memory limits, external timer or service automation changes.

An implied maintenance default follows explicit zeroes for `DENSE_CACHE`,
`FLOAT_DENSE_CACHE`, `BLOCKED_MOE`, `OPERAND_STORE` and `INT8_EXPERT_STORE`,
because they determine the qualified owner union. Explicit maintenance values,
including contradictory `1` or invalid values, remain visible to native
validation. The saved operand residency lease, MTP and batching switches are
independent. Changes such as `FLOAT_DENSE_SELECTIVE=0` remain caller choices;
the runtime validates the actual union rather than inventing extra launcher
prerequisites. A qualified source missing the required saved union skips
maintenance with a status reason and continues normal model service. An
unqualified source/layout/original geometry under explicit opt-in is refused.

Optional dense and TOP64 paths remain dynamic qualified defaults, never
mandatory paths in this static proposal. Exact store `0` disables their default
inspection and survives merging; a missing store does not create a path.
Existing present-artifact corruption rejection remains in place.

CPU qualification covers independent copies, inactive gate rejection, the
255/256 GiB boundary, source/CPU mismatches, parent opt-outs, explicit flag and
interval preservation, historical routes/gates and artifact identity pins.
Root separately proved same-binary OFF/ON idle delay reduction, all 22 complete
HTTP output/lifecycle cases and matched single/concurrent request controls.
The profile agent's 121 CPU tests passed with no skips, including the native
exact-zero saved-store selector checker. Pressure and owner-union fallback
policy have CPU qualification; the normal server's final readiness and idle
proof remain Root-owned evidence rather than a profile-agent GPU claim.

Prepared review artifacts:

- `build/release/flash/local-profile-v9-candidate.json`
- `build/release/flash/local-profile-v9-promotion-plan.json`
- `build/release/flash/local-profile-v9-cpu-qualification.json`
