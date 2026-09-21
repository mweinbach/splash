# Exact Q8/BF16 joint-head independent service audit

The exact coefficient reconstruction fixes the numerical failure of the previous packed-Q8 head route. In the current same-build service screen, it preserves all generated outputs and saves 237.055750 ms of head decode GPU time. Concurrent HTTP gains are smaller and variable because prefill and target verification dominate the full request.

The private primitive compares all 635,699,200 reconstructed coefficients against the qualified BF16 cache, with zero mismatched words. Captured and finite synthetic R2/R3/R4 output lanes are also word exact. The R2/R3 nonfinite diagnostic fixture matches greedy and sticky diagnostics; it is reported separately from finite numerical accuracy. The strict relative-L2 bound remains 0.0001.

The full trained four-lane head screen preserves premixer words, full QSA buffers, greedy records and every lane's vocabulary. At contexts 128 and 2048, six paired normal commands reduce head GPU medians by 21.33% and 17.40%, respectively. Eight future continuation cases and eight truncate-overwrite prefixes pass, with two step-zero continuation anchors recorded separately. These are isolated head measurements, not whole-service claims.

All 22 service quality/lifecycle cases pass. All 28 normalized records match accepted v7 for text, reasoning, tool type/name/arguments, finish reason, usage, HTTP status and errors. Generated request/tool IDs, timestamps, stream chunk boundaries and performance metrics are excluded from semantic equality. The factual paragraph remains coherent and identical to accepted v7.

The matched HTTP screen uses contexts 128/2048, concurrency 1/4, 128 output tokens, greedy reasoning disabled and zero cross-request KV reuse. All 21 generations, including warmup, match the same-build OFF run in request bodies, prompt hashes and outputs. Acceptance/depth histograms and verifier work match. Model, unchanged route and persisted store identities, the allocation ledger and peak memory match. Both retained idle snapshots are healthy with no outstanding work. Tracing is disabled.

| Context | Concurrency | Initial ON throughput relative to OFF |
| --- | --- | --- |
| 128 | 1 | +1.086% |
| 128 | 4 | +6.006% |
| 2048 | 1 | +0.060% |
| 2048 | 4 | +0.370% |

The short concurrent OFF samples are 162.125 and 175.960 tokens/s, so the apparent 6% gain is sensitive to one slow control sample. The repeated ON short-concurrent median is 177.014 tokens/s, 4.716% above that control median, while its long-concurrent median is 86.536 tokens/s, only 0.125% above OFF. Two samples per cell do not establish statistical significance. The local default is qualified by exact outputs and repeatable head GPU savings; these HTTP screens do not establish a robust 4–6% end-to-end gain.

Direct phase evidence is stronger: equal 1,022 head-decode commands consume 2,847.453042 ms OFF versus 2,610.397292 ms ON, an 8.325% reduction across singleton and joint head work. Exactly 380 completed joint vocabulary commands, covering 1,520 real rows, use the new route; OFF records zero. The approximately 0.624 ms saving per optimized command agrees with the private head screen. Batched prompt-head priming uses `None` logits and contributes no register-vocabulary calls. Register counters increment only after successful command completion and sticky-diagnostic verification.

The route owns no additional weights or scratch: it reads original Q8 codes, scales and biases and reuses the existing padding arena. The BF16 cache remains for fallback. Scope is joint trained MTP head `Last`, 2..4 real vocabulary lanes; singleton decode, prompt priming and target verification retain their existing routes. Their timing changes should not be credited to this optimization. Trunk prefill wall time is 25.926 ms higher ON in this screen, and its host time is 135.041 ms higher.

The repeated ON measurement reproduces exact 21-generation equality, acceptance work, route identities, unchanged memory and 380 completed register commands / 1,520 real rows. Its 1,022 head commands save 229.742668 ms of GPU time against OFF, an 8.068% reduction, or about 0.605 ms per optimized joint command. Target verification changes by only +3.516 ms, while trunk prefill GPU/wall time rises by 31.026/43.098 ms. This separates the repeatable head gain from full-request timing drift. The confirmation service snapshot is idle and healthy.

The earlier `v9-v7-baseline-http-performance.json` remains excluded as a promotion control because CPU xctrace finalization overlapped its measurement. Recompute the independent report without loading a model or submitting GPU work:

```sh
.venv/bin/python -B dev/tools/audit_flash_v9_exact_head_service.py
```

Evidence is frozen in `build/release/flash/v9-mtp-q8-bf16-register-independent-service-audit.json`, with hashes of the private proofs, quality reports, matched ON/OFF reports and retained idle snapshots.
