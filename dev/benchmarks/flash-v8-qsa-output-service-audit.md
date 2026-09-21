# QSA output N32 independent service audit

The QSA output N32 route passed the bounded service qualification, but the same-build measurement does not support making it a default. Keep `SPLASH_FLASH_QSA_OUT_F32_N32=0` unless a later matched full-service screen demonstrates a meaningful gain.

All 22 quality and lifecycle cases passed, with no new task regressions and exact comparison outputs. All 21 benchmark generations, including warmup, match the accepted v7 and same-build OFF runs for text, reasoning, tool calls, finish reason, usage, request bodies, and tokenized prompt hashes. Native target verification work is unchanged: 170 commands, 2,125 drafted tokens, 2,110 committed drafts, and identical acceptance and proposal-depth histograms. MTP head commands, original model/layout identities, saved operand/expert identities, allocation ledger, and the 152,982,536,192-byte peak match. Retained idle status snapshots confirm no outstanding native work or Metal failures.

The ON run recorded 1,040 candidate calls covering 15,490 real rows since startup; the benchmark interval contributes 845 calls / 13,385 rows and OFF contributes zero. These are main-model graph construction counters, not completed GPU dispatch counts. Source policy restricts the route to main-model QSA `o_proj`, source Q5/Q6 group64, N2560 K6144, R4..16. The trained MTP head, R1 decode, large-row prefill and original coefficient bytes retain their existing routes.

| Context | Concurrency | ON throughput relative to same-build OFF |
| --- | --- | --- |
| 128 | 1 | −1.072% |
| 128 | 4 | +1.407% |
| 2048 | 1 | +1.268% |
| 2048 | 4 | −0.891% |

The matched target-verifier GPU totals are 10,875.711959 ms ON versus 10,870.897542 ms OFF over 170 commands: ON is 4.814417 ms slower, a 0.0443% increase. Relative to the earlier accepted-v7 run, ON reduced the same phase by only 41.964875 ms (0.3844%). The preceding apparent prefill-wall gain is unrelated to this small-row route and does not survive the same-build control; neither head drift nor prefill drift establishes a QSA optimization benefit. Two throughput samples per cell do not establish statistical significance.

The CPU-only audit records phase deltas rather than comparing cumulative totals accumulated during the preceding quality suite. Trunk prefill subtracts MTP head priming once, because batched head priming is already a subset. Recompute the report without loading a model or submitting GPU work:

```sh
.venv/bin/python -B dev/tools/audit_flash_v8_qsa_service.py
```

Evidence: `build/release/flash/v8-qsa-out-n32-independent-service-audit.json`, `v8-qsa-out-n32-{on,off}-http-performance.json`, their idle snapshots, `v8-qsa-out-n32-quality.json`, and `v8-qsa-out-n32-quality-comparison.json`.
