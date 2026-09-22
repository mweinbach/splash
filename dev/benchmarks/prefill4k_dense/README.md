Private dense-prefill screening for the actual Flash-Next roles in
`build/release/flash/prefill4k-current-stage.json.trace.jsonl`. No normal
production source or default is changed by these tools.

The2048-row diagnostic trace attributes about177ms to dense operations. Actual
cached BF16 shapes include K2560→N10240 (39.6ms), K6144→N2560 (28.6ms),
K10240→N320 (24.9ms), K2560→N6144 (23.0ms), and K2560→N12288 (15.6ms).
The remaining roles include HCup, QSAkey/indexer, shared down, and PLEvalue.

CPU-only preparation and build:

```sh
make -f dev/benchmarks/prefill4k_dense/Makefile -j 6 all cpu-self-test
.venv/bin/python dev/benchmarks/prefill4k_dense/prepare.py
```

Preparation verifies whole coefficient payload SHA256 and independently unpacks
260 original affine coefficients per role. The10 selected real matrices passed
2600 source comparisons. An initial screen can use these real weights with
deterministic synthetic input:

```sh
build/prefill4k-dense/oracle \
  --fixture-manifest build/prefill4k-dense/actual-weights.json \
  --rows 2048 --samples 7 --repeat 4 \
  --out build/prefill4k-dense/actual-roles-screen-v1.json
```

Root exclusively coordinates every GPU run. To obtain exact inputs and outputs
from the running2048-token coding prompt, build the private capture overlay:

```sh
make -f Makefile -f dev/benchmarks/prefill4k_dense/capture.mk \
  BUILD=build/prefill4k-attribution -j 6 prefill4k-dense-capture
.venv/bin/python dev/benchmarks/prefill4k_dense/run_capture.py \
  --tokens build/release/flash/prefill4k-fixture/code2048.tokens.json \
  --report build/prefill4k-dense/capture-normal-v1.json \
  --outdir build/prefill4k-dense/actual-activations-v1 --run
```

Compare the capture run's final logits hash with the normal attribution baseline.
Then screen actual activations:

```sh
build/prefill4k-dense/oracle \
  --fixture-manifest build/prefill4k-dense/actual-activations-v1/manifest.json \
  --rows 2048 --samples 7 --repeat 4 \
  --out build/prefill4k-dense/actual-activations-screen-v1.json
```

The captured-production output must match the standalone production baseline
byte-for-byte before any candidate is timed. Candidates cover M32/M64/M128,
N64/N128,4/8 SIMD groups, column/row traversal and swizzle2/4/8. K512 candidates
retain a single F32 accumulator across ascending512-element blocks and cast once
to BF16. Changing geometry, SIMD group count, or Kblocking may change reduction
rounding; the report records full BF16 differences and sampled scalar-double
error rather than declaring those arithmetic changes qualified. Traversal must
be byte-exact within each geometry/SIMD/Kstrategy. Guard and sticky diagnostic
checks cover partial output intervals and malformed parameters. Timing samples
rotate candidate order and include GPU plus wall time per projection.

The K512 MPP experiment is analogous to MLX's BK512 tuning. Steel uses smaller
register MMAs with relaxed precision; this harness retains strict precision.
Neither traversal nor geometry requests die or memory affinity. Any full-model
promotion needs final-logit/hidden-state and service qualification separately.

The production candidate is off by default. `SPLASH_FLASH_PREFILL_DENSE_TILES=1`
selects measured M128N64 whole-K tiles only for2048-row cached affine projections.
The public original-BF16 router route retains its existing tiles. Selected
K→N shapes and SIMD/traversal policies are:

| K→N | SIMD groups | Traversal |
|---|---:|---|
|10240→320|8|column-fast|
|2560→10240|4|swizzle4|
|2560→6144|4|swizzle4|
|6144→2560|4|swizzle4|
|2560→12288|4|swizzle8|
|2560→512|8|swizzle2|
|2560→640|8|swizzle8|

No K512 reduction is selected. The actual-capture bridge hard-fails on any
BF16 difference from the normal production output:

```sh
build/prefill4k-dense/oracle --production-prefill \
  --fixture-manifest build/prefill4k-dense/actual-activations-v1/manifest.json \
  --rows 2048 --samples 7 --repeat 4 \
  --out build/prefill4k-dense/production-prefill-bridge-v1.json
```

The complete worker, library, normal attribution oracle and private capture
oracle are built in `build/prefill4k-dense-runtime`. The explicit qualification
launcher records dense and teacher overrides; the normal attribution launcher
removes inherited flags and cannot activate the new selector by an environment
assignment alone:

```sh
.venv/bin/python dev/benchmarks/prefill4k_dense/run_qualification.py \
  --tokens build/release/flash/prefill4k-fixture/code2048.tokens.json \
  --report FRESH.json --dense-tiles 0 --teacher-cache-only 0 \
  --warmup 1 --repeats 3 --run
```

Use fresh report paths for each off/on comparison and include final logits,
hidden hashes, output tokens, and service lifecycle qualification. Root
coordinates every GPU run.
