# Batched results (September 22)

`splash_tuning_sep21.py --mtp 4 --contexts 2048 --batches 1,2,4 --workloads coding
--output-tokens 256 --warmup 1 --trials 3 --lane-variation seeded`, medians of
three valid trials. Aggregate decode is total post-first-emission tokens over the
client wave interval; prefill is the native prefill command rate.

| Build | Width | Prefill tok/s | Per-lane decode | Aggregate decode |
| --- | ---: | ---: | ---: | ---: |
| v13 (qualified) | 1 | 4,014 | 62.0 | 62.0 |
| v13 (qualified) | 2 | crash | crash | crash |
| v13 (qualified) | 4 | crash | crash | crash |
| GDN fix only + ≤5-row kernels | 2 | 2,811 | 38.2 | 75.8 |
| GDN fix only + ≤5-row kernels | 4 | 2,911 | 26.7 | 103.8 |
| v14 | 1 | 4,097 | 75.0 | 75.0 |
| v14 | 2 | 2,862 | 50.7 | 99.3 |
| v14 | 4 | 2,955 | 30.9 | 121.2 |
| v15 | 1 | 4,196 | 124.2* | 124.2* |
| v15 | 2 | 2,925 | 56.2 | 111.5 |
| v15 | 4 | 3,007 | 35.0 | 137.0 |
| v16 | 1 | 4,208 | 84.6 | 84.6 |
| v16 | 2 | 2,935 | 59.2 | 117.1 |
| v16 | 4 | 3,015 | 37.7 | 143.9 |

v13 fails the first two-lane request with `batched GDN validated producer
contract changed`. Single-lane decode on the canonical prompt varies with that
one greedy trajectory's acceptance; see README for the ten-prompt comparison.

\* v15's single-lane canonical run follows a different greedy path (70 cycles
instead of 100) because its prefill summation order changed. v16 is back on
v14's 100-cycle path; see the README for multi-prompt comparisons.
