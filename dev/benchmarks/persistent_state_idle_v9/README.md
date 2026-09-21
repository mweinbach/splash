# Private persistent-state idle control

This oracle uses unchanged production host objects, ABI200 and Metal4.1 graph.
Only its private existing friend helper resets one previously exercised request
state. No Worker pool, production source edit, background command, new residency
set or numerical conversion is introduced.

One persistent capacity8192 state remains alive during ALL controls, including
fresh-state cases. Warm fresh/reuse calls precede two matched nine-second-idle
AB/BA pairs: fresh/reuse/reuse/fresh, optionally reversed. This controls memory
retention separately from reuse of the 134 native buffer identities.

Before reuse, the oracle requires healthy owned state, no provisional trial,
a healthy backend and no outstanding ticket/unmap. It never executes verify or
batch verification. It allocates a fresh verification identity before mutation,
validates every host-readable Shared plane, CPUzeros all physical bytes including
padding, restores first2 I64 PLE-history words to descriptor.pleHistoryEos,
and publishes fresh identity/length0. Owner, capacity, QSAcapacities and GDN
strides remain intact. Poisoned states are rejected, never revived.

The exact native footprint is134 buffers /349,388,800 bytes:
72 GDNplanes115,605,504B;60 QSAplanes233,570,304B;2 PLEplanes212,992B.
Native pointer/size metadata must remain unchanged through every sample.

Six target calls must remain one command /1709 dispatches. Full248320 BF16
vocabulary words and two subsequent full-vocabulary continuation rows must
match exactly against the fresh control. The report separately records state
allocation/reset, Forward and TOTAL preparation+Forward elapsed time. Lower
postcommit delay alone is insufficient. Raw command profiles include host,
hardware and driver kernel times with calibratedclockbridges.

The main/head arenas and saved-derived residency match earlier standalone
controls but exclude Worker's batch arenas and HTTP. This is an idle mechanism
probe, not service throughput or pool qualification. A default pool requires
positive evidence plus admission/pressure/cancel/deadline/lifecycle work.

Fresh frozen oracle:
`build/flash-private-persistent-state-idle-v9/persistent-state-idle-v9-oracle`.
Adjacent normalMetal4.1 library `splash.metallib`.
Fresh44 host objects compile/link; --help executes withoutdevice. GPUexactness
and latency remain Root's gates. CPU evidence/hashes:
`build/release/flash/private-persistent-state-idle-v9-cpu-qualification.json`.

Root alone runs after stopping its servingprocess, applying normalv7 profile
and derived-store paths:

```sh
build/flash-private-persistent-state-idle-v9/persistent-state-idle-v9-oracle \
  build/flash-private-persistent-state-idle-v9/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/flash-v6-verifier-attribution/fixture/ctx128-width1-lane0.tokens.json \
  build/release/flash/FRESH-state-idle.json
```

`FLASH_STATE_IDLE_SECONDS=0..15` defaults9.
`FLASH_STATE_IDLE_FIRST=fresh|reuse` reverses both idlepairs.
The report path and REPORT.commands.jsonl must be fresh.

## Root runtime result: no idle fix

Root completed all six calls with exact full-vocabulary BF16 outputs and two
continuations. All graphs had1709 dispatches and one command; the persistent
134 native planes/349,388,800 bytes retained their identities throughout.

| Idle AB/BA sample | State | Preparation+Forward total | Host state preparation | Commit→GPU wait |
| --- | --- | ---: | ---: | ---: |
| A1 | Fresh |1666.44 ms |20.76 ms |1292.13 ms |
| B1 | Reused |1584.42 ms |4.03 ms |1240.29 ms |
| B2 | Reused |1680.97 ms |2.86 ms |1340.04 ms |
| A2 | Fresh |1613.84 ms |22.15 ms |1258.26 ms |

Mean total latency was1640.14 ms fresh versus1632.70 ms reused, a mixed
~0.45% difference across onlytwo samples each. Reuse saves roughly18 ms in
host allocation/initialization, but the longdriver preparation delay remains
and the secondpair reverses direction. This is **not an idle-stall fix** and
does not justify a Worker pool or default change. The probe specifically
controls persistent-memoryretention; simply keeping349MB alive was common
to both routes. No backgroundkeepalive was introduced.

Completed Root GPU evidence:
`build/release/flash/v9-persistent-state-idle-control.json` and
`.commands.jsonl`. CPU-only validated summary:
`build/release/flash/v9-persistent-state-idle-control-cpu-summary.json`.
