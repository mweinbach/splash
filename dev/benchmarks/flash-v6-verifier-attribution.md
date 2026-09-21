# V6 target verification attribution

The private oracle uses actual target premixer features and the checkpoint's trained MTP head. It teacher-primes the head from each exact saved prompt, folds the real target anchor, and produces real speculative tokens. Only the target verification command is profiled. It validates and commits the model's actually accepted prefix afterward.

Build CPU-only:

```sh
.venv/bin/python dev/benchmarks/prepare_flash_v6_verify_snapshot.py --check
make -f Makefile -f dev/benchmarks/flash_v6_verifier_attribution_oracle.mk \
  BUILD=build/flash-v6-verifier-attribution -j8 flash-v6-verifier-attribution-oracle
build/flash-v6-verifier-attribution/flash-v6-verifier-attribution-oracle --help
```

Every host/core object is fresh in this private directory, and the oracle asserts `sizeof(CommandTiming)==200`. The private BatchVerify snapshot retains its class name/friendship and adds only diagnostic route capture plus a metadata getter. Eight reversible transformations reproduce the original production bytes; no production/default/checkpoint files change.

The frozen v6 fixture contains all10 independently retokenized sample0 contexts128/2048 and width1 lane0/width4 lanes0–3. `v6-environment.json` supplies34 runtime flags and both verified saved-store directories. The wrapper strips inherited Flash settings and adds the existing expert-ID capture flag. It defaults to a CPU dry-run; Root alone uses `--run` after stopping the owned model service.

Whole matrix, then per-dispatch attribution:

```sh
.venv/bin/python dev/benchmarks/flash_v6_verifier_attribution_run.py \
  --mode normal --report build/release/flash/v6-verifier-attribution-normal.json --run
.venv/bin/python dev/benchmarks/flash_v6_verifier_attribution_run.py \
  --mode stage --report build/release/flash/v6-verifier-attribution-stage.json --run
```

A narrow joint long-context case:

```sh
.venv/bin/python dev/benchmarks/flash_v6_verifier_attribution_run.py \
  --mode stage --context 2048 --physical-rows 16 --executor joint \
  --report build/release/flash/v6-verify-ctx2048-joint-R16-stage.json --run
```

Singleton rows4/8/16 mean one lane with4/8/16 real incoming tokens. Joint physical rows4/8/16 mean B1/B2/B4 x4 real rows per lane. The production joint API does not support8/16 per lane. All runs use target capacity8192, prefill arena2048 and verify tape16, source/head saved-only residency, warmup1/repeat1 per case by default. Geometry can be narrowed with the wrapper's options.

Artifacts:

- `REPORT.invocation.json`: exact environment/control settings, original and private source hashes, binary/library/provenance hashes and ABI.
- `REPORT.commands.jsonl`: exact prompt/input/prediction tokens, true state/head offsets, accepted prefixes, last target feature hashes, optional active mutable-plane hashes, completed target GPU/wall/caller time and normal host subphases.
- `REPORT.routes.jsonl`: all48 layers' exact row-major expert IDs, I64 hashes, unique IDs, multiplicities and canonical route membership for potential logical grouping.
- `REPORT.trace.jsonl`: full-command profiling metadata/timestamps plus normalized small-window MoE, dense, vocabulary, GDN, attention/QSA, shared expert, PLE, HC, copies and greedy attribution.

Small verification windows use gathered projections; actual blocked-MoE bucket jobs are0. Derived expert groups are labeled logical possibilities rather than submitted GPU jobs. Vocabulary classification recognizes the INT8 producer or actual output-binding extent.

Capturing exact routes adds48 existing word-copy dispatches per target graph. Default active-state hashing reads Shared state before the timed call and may change cache residency; `--state-hashes 0` omits those reads. These are diagnostic graphs, not HTTP scores. `normal` keeps counter profiling off; `command` retains the original encoder shape with metadata; `stage` creates an encoder per dispatch and `dispatch` adds timestamp barriers. Sample status/validity and incomplete/truncated metadata remain explicit. GPU/memory-query/wait/callback durations can overlap; scheduled callback arrival is not an actual GPU scheduling timestamp.

The sampled windows are the initial speculative cycles at the exact frozen prompt contexts. Source/body/head parameters, prompts, target features and real proposals are preserved for reproducibility. Use the matched HTTP harness and ordinary v6 service for throughput qualification.
