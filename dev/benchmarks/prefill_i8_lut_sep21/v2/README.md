V2 fixes one source-proven v1 oracle expectation: native MoE buckets diagnose
duplicate legal expert IDs and retain both canonical routes. Only out-of-range
IDs are excluded. V1 mistakenly excluded later duplicates. The independent
CPU bucket reference already implements the native behavior; explicit small
goldens now test the corrected histogram, offsets, maps, jobs and diagnostics.

V1 sources/builds are preserved. V2 copies the v1 Metal library byte-for-byte;
no runtime/kernel, coefficient, numerical fidelity rule or threshold changes.
All malformed checks now precede timing, then clean states are restored.
Durable `REPORT.checkpoints.jsonl` events record initial/final raw and scaled
fidelity, eligibility, actual/expected malformed counts and offsets, totals,
excluded/duplicate counts, IDs, jobs, maps and sticky diagnostics before a
requirement can throw. V1's buffered empty-case failure report cannot establish
candidate fidelity or whether eligible timings ran.

```sh
make -f dev/benchmarks/prefill_i8_lut_sep21/v2/component.mk all cpu-self-test
.venv/bin/python dev/benchmarks/prefill_i8_lut_sep21/v2/run.py \
  --guard-only --report build/release/flash/sep21-i8-lut-native-guards-v2.json
```

Only Root adds `--run` in its exclusive GPU window. Guard-only loads no model
or compressed coefficient operands, measures no candidate fidelity, attempts
no timings, and checks the three malformed ID fixtures on spread/hot patterns.
It admits a bounded1 GiB allocation plus the same host reserve as v1.

After the narrow guard check passes, Root can qualify the existing layer pack:

```sh
.venv/bin/python dev/benchmarks/prefill_i8_lut_sep21/v2/run.py \
  --pack-dir build/prefill-i8-lut-sep21-layer00-v1 \
  --variant 2 --pairs 4 --pattern spread-all \
  --report build/release/flash/sep21-i8-lut-r2k-v2.json
```

Omit `--variant` and use `--pattern both` for the complete eight-variant screen.
Fresh primary report, checkpoint and invocation paths are required. Full v2
retains the v1 source-code reconstruction certificate and all strict fidelity,
guard, balanced timing,150 ms warm-up and post-timing checks.
