# Independent completed-report latency audit

This CPU-only audit joins Root's completed production ON/OFF HTTP reports to
native command profiles. The two production conditions used the same frozen
binary and model routes. The unmodified v8 report provides an additional prior
baseline. Full service quality and ordinary throughput qualification are
separate evidence.

| 128-token prompt, fresh state | Maintenance off | Maintenance on |
| --- | ---: | ---: |
| Nine-second idle, one output token, HTTP first content | 1643.24 ms | 388.96 ms |
| Nine-second idle, two output tokens, HTTP first content | 1626.81 ms | 402.68 ms |
| Mean idle HTTP first content | 1635.02 ms | 395.82 ms |
| Mean idle server queue to execution | 21.53 ms | 22.37 ms |
| One-token idle target commit to actual GPU start | 1277.30 ms | 7.59 ms |
| Two-token idle target commit to actual GPU start | 1240.56 ms | 9.33 ms |

The observed end-to-end first-content reduction is 1239.20 ms, or 75.79%.
The native submission-entry waits are also only8.73 and10.53 ms with the route
enabled, and admission denials remain zero. This removes the observed driver
wait instead of moving it into request admission or queueing. Initial cold
request wiring remains: the ON condition's first target still waits1135 ms.

After30 seconds idle, the16-token fixture returns first content in392.94 ms
(immediate repeat370.45 ms). The target waits8.39 ms from commit to GPU start,
with349.45 ms GPU execution. All target samples retain1709 dispatches. All six
nine-second HTTP fixtures retain identical request bodies, output bytes,
response hashes, usage and zero cache reuse across OFF, ON and original v8.
The long-idle fixture retains identical16-token output across all three calls.
All reports end with a healthy, drained native service.

The enabled route accesses one word from each of1134 immutable owners every
500 ms when the worker is idle and pressure/headroom allow it. It retains GPU
accessibility for144,326,852,608 bytes (134.41 GiB) of existing weight and saved
operand resources. It allocates25,456 bytes of new diagnostics, within the
conservative32 KiB admission plan, and creates no extra weight backing.

During the nine-second audit span,34 maintenance commands total0.8924 ms GPU
time and22.5523 ms wall time. GPU duty is0.00415% of that measured21.495-second
span; maintenance wall fraction is0.1049%. During the two separately measured
ten-second idle intervals, GPU duty is0.00395% and0.00425%; wall fraction is
0.1160% and0.1330%. Hardware trace sums exactly match the status counters.
Maintenance has one dispatch, zero request lanes/rows and no request IDs;
all seven user timing phases remain unchanged between the idle snapshots.
These are measured time costs, not an energy measurement.

At10/20/30 seconds, system wired memory measures147.68/147.37/147.74 GiB.
These global VM counters corroborate continued accessibility; they do not
identify individual resources or prove guaranteed pinning. Pressure
suspensions, command failures, and cold misses are zero in these observed
normal-pressure windows. The policy suspends rather than fighting pressure.
An arriving request can overlap one synchronous maintenance command; the
largest observed wall duration1.665 ms is not a hard latency bound. A driver
rewiring miss after pressure or scheduling gaps may still be long, and
physical pinning is not guaranteed by this technique.

Primary completed artifact:
`build/release/flash/v11-production-maintenance-idle-cpu-audit-v2.json`.
The report includes per-request HTTP and target command timings, status
counter deltas, measured duty intervals, VM scope, and artifact SHA256 values.
`audit.py` reads completed reports only and performs no GPU work.

The completed matching HTTP control uses the same frozen plan, real MTP
policies and command tracing in both conditions. Its128-output-token workloads
show−1.69% for short C1, +1.37% for short C4, +0.027% for long C1 and +0.199%
for long C4. There are only two samples per cell and consecutive process order;
these are diagnostic throughput controls, not an untraced score or a proven
small speed change. Zero maintenance commands execute during either active
benchmark span. All measured request bodies, output text and usage are exact;
all user phase command counts,170 verifier cycles,2125 drafted tokens,2110
committed accepted drafts,2108 emitted accepted proposals, depth histograms and
accepted-prefix histograms agree. The separate completed artifact is
`build/release/flash/v11-production-maintenance-matched-http-cpu-audit-v2.json`.
Root separately completed all22 production service quality/lifecycle checks;
their zero-regression comparison is
`build/release/flash/v11-production-maintenance-on-quality-comparison.json`.
