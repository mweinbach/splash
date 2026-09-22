Private actual GDN comparison; root exclusively schedules GPU execution.

Current V16/T16 uses512 threads/16 SIMD32 groups with four F32 state values per
thread. It is not the QSA large-accumulator case. The fresh-ABI V3 synthetic
report already tested V32/T16 and V32/T32 at SG8 and found them slower; V64 was
worse. Those variants are not repeated here.

Two unused comparisons are prepared, with T16/T32 variants:

- V16/SG16 declares the actual512-thread cap and loads each query only during
  result folding; state/k lifetimes, chronological token order and both SIMD32
  reductions stay unchanged.
- V16/SG8 owns two independent value columns per SIMD, retaining the same384
  CTAs and ordered operations for each column. It shares q/k across the two
  columns and has eight F32 state values per thread. This trades per-thread
  state against scheduler/staging work; a speed gain is not established.

Both retain contraction/reassociation disabled. No global state approximation,
time reordering, changed dot tree or normal production default is introduced.

Build the private layer-zero snapshot executable and recurrence oracle:

```sh
make -f Makefile -f dev/benchmarks/prefill4k_dense/gdn_capture.mk \
  BUILD=build/prefill4k-dense-runtime -j6 prefill4k-gdn-capture
make -f dev/benchmarks/prefill4k_dense/GDNMakefile -j6 all cpu-self-test
```

Capture a real2048-row layer-zero sequence and its initial/final state, prepared
mixed/decay/beta arrays, z/norm, BF16 recurrence and final BF16 GDN output:

```sh
.venv/bin/python dev/benchmarks/prefill4k_dense/gdn_run_capture.py \
  --tokens build/release/flash/prefill4k-fixture/code2048.tokens.json \
  --report build/prefill4k-gdn-pair/capture-normal-v1.json \
  --outdir build/prefill4k-gdn-pair/actual-layer0-v1 --run
```

Verify complete capture logits and hidden hashes against the normal native
baseline. The paired component oracle first requires production recurrence,
final state and final GDN output to match the recorded model exactly. It then
tests all private variants on the cold initial state and on the nonzero state
carried after that actual2K sequence, appending the same frozen prepared input.
Every full BF16 output and F32 state byte, input/weight hash and canary is checked.

```sh
build/prefill4k-gdn-pair/oracle build/prefill4k-gdn-pair/gdn.metallib \
  build/prefill4k-gdn-pair/actual-layer0-v1/manifest.json \
  build/prefill4k-gdn-pair/actual-screen-v1.json
```

Timings require CommandTiming ABI200, finite GPU/wall durations between1ns and
600seconds. Each route has two warm-ups and nine alternating matched pairs.
The older `gdn-ilp-layout-prefill-timing.json` records subnormal~3e-314 timing
values and cannot establish throughput.

The first actual screen showed clear control drift (3.214ms early versus about
2.02ms later), so its largest apparent ratios are not promotion evidence. The
randomized driver uses one backend and one fixture for every route, eliminating
buffer-address and variant-order differences. Each cold/carried case runs all
five routes in random order for six complete warm-up cycles, then nine matched
cycles. Every route and control runs once per cycle with identical reset state.
The output includes raw GPU samples, route order, paired ratios and positive
pair counts, with the same exactness/timing gates.

```sh
build/prefill4k-gdn-pair/randomized-oracle \
  build/prefill4k-gdn-pair/gdn.metallib \
  build/prefill4k-gdn-pair/actual-layer0-v1/manifest.json \
  build/prefill4k-gdn-pair/randomized-complete-v1.json
```

Append `--recurrence-only` and choose a fresh report path to exclude the
unchanged output normalization/gating dispatch. This mode still validates the
full captured baseline first, but its timed scope matches the stage trace's
recurrence-only interval. Comparing complete-graph samples directly with
68.36ms/36 from the recurrence-only trace would mix scopes.
