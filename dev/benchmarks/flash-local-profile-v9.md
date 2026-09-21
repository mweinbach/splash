# Local v9 defaults

The accepted launcher and `.splash-local-profile.json` match
`m5-ultra-flash-next-v9` with 38 static defaults. It adds only
`SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=1` to the 37 v8 routes and qualifies
the exact existing Flash-Next source on an Apple M5 Ultra with at least 256
GiB RAM. The model payloads, saved operands, precision, FP32 recurrent state,
2048-row prefill windows and singleton draft depth 15 remain unchanged.

With the same production binary and identical uncached requests, mean first
content after nine seconds idle fell from 1,635 ms with maintenance OFF to
396 ms ON, a 75.8% reduction. Immediate requests stayed around 371 ms. All
six output bodies, usage records, request inputs and model identities matched;
the improvement was not moved into queue time. After 30 seconds idle, the ON
request took 393 ms versus a 370 ms immediate repeat. System wired memory
stayed around 147.4–147.7 GiB in that 30-second interval. Those are observed
systemwide counters, not a per-resource physical-pinning guarantee. The first
request still pays the normal initial wiring cost.

After a real successful user request, the Worker submits a maintenance command
every 500 ms when all request queues are empty and no GPU command, ticket or
sparse unmap is outstanding. It reads one uint32 from each of exactly 1,134
immutable owners: 21 original weight bases and the verified saved main/head
operands, totaling 144,326,852,608 bytes, about 134.4 GiB. Each command reads
4,536 source bytes and verifies a checksum, owner count and all output guards.
It adds 25,456 native diagnostic bytes on this machine and no weight backing.
The measured idle commands averaged about 0.02 ms GPU time and 0.6–0.7 ms wall
time, with the reported GPU duty around 0.004%. These timing measurements do
not establish energy consumption.

This uses periodic idle GPU work and retained wired memory to avoid the next
request's rewiring delay. A fresh governor check requires valid Normal system
and effective pressure, growth allowed, zero reservations, and available
memory strictly greater than the host reserve plus 2 GiB. Pressure suspends
maintenance. A command lasting at least one interval disarms it until another
successful user request. A late-arriving request can wait for one synchronous
maintenance command; the interval is not a hard bound on driver delay. The
route does not change macOS memory limits or create an external timer.

All 22 production HTTP quality/lifecycle cases passed with zero new regressions
and exact comparison outputs. The same-binary active benchmark retained exact
outputs, usage, request inputs, MTP acceptance and phase call counts. No
maintenance commands ran during that active workload. Single/concurrent
throughput differences were between −1.69% and +1.37%; two samples and
sequential process order do not support a precise active-throughput claim.
Both controls used request tracing, so these are diagnostic controls. The
fix's measured benefit is the request after idle.

All 121 focused profile/launcher CPU tests passed with no skips, including the
native exact-zero saved-store selector checker. An implied maintenance default
follows explicit zeroes for `DENSE_CACHE`, `FLOAT_DENSE_CACHE`, `BLOCKED_MOE`,
`OPERAND_STORE` and `INT8_EXPERT_STORE`. Explicit child and interval values
remain authoritative for strict native validation. Historical v5/v6/v7/v8
helpers retain 30/34/36/37 static defaults and their 192 GiB gates. Optional
artifact paths remain dynamic; a qualified model without the complete derived
union serves normally with maintenance unavailable and a status reason.

`/status.idle_residency_maintenance` reports owner bytes, commands, actual
GPU/wall timing, added backing, pressure suspensions, cold misses and failures.
Disable the local fix with `SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=0` before
launching. Normal native default is still OFF outside the qualified local
profile. Original checkpoints, oMLX preferences and GitHub are untouched.

Evidence:

- `build/release/flash/v11-production-maintenance-idle-cpu-audit-v2.json`
- `build/release/flash/v11-production-maintenance-on-long-idle-http.json`
- `build/release/flash/v11-production-maintenance-on-quality-comparison.json`
- `build/release/flash/v11-production-maintenance-matched-http-cpu-audit-v2.json`
- `build/release/flash/local-profile-v9-cpu-qualification.json`

Normal launcher readiness and idle verification passed on8011 with tracing
disabled. The binary/metallib match the qualified runtime byte-for-byte.
After9 seconds idle, first content arrived in396 ms versus370 ms immediately;
all16-token outputs matched and maintenance stayed healthy. Proofs are
`default-v9-normal-idle-proof.json` and `default-v9-normal-final-idle-status.json`
in the same evidence directory. The profile preparation/activation used CPU
commands only; Root executed the serialized GPU qualification.
