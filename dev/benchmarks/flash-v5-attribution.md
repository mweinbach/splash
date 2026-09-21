# Local v5 target-trunk attribution

This private oracle uses the frozen v5 environment (30 runtime flags and both verified saved-store directories), original affine package, and independently retokenized matched 2K prompts. It builds all host objects with the current `CommandTiming.host` ABI. Nothing changes the normal launcher/defaults/checkpoints.

Build on the CPU:

```sh
make -f Makefile -f dev/flash.mk \
  -f dev/benchmarks/flash_v5_attribution_oracle.mk \
  BUILD=build/flash-v5-attribution -j8 flash-v5-attribution-oracle
build/flash-v5-attribution/flash-v5-attribution-oracle --help
```

The prepared fixture directory is `build/flash-v5-attribution/fixture`. `fixture-provenance.json` links exact source/profile/layout/store/plan identities. The preparer imported only CPU tokenizer/metadata code; it independently reproduced all 2048 token IDs in the width1 lane0 and width4 lanes0–3 frozen prompts. The frozen plan SHA is `cb3747ba6e77cbfe2d74af976163efc92bf3300f4d1a98c72c4da88bfd6badb4`.

The coordinator alone runs actual Metal inference, after stopping its owned model server. The wrapper defaults to a CPU dry-run; `--run` executes the private oracle. It strips inherited Flash settings, applies the exact32 frozen settings, and saves the invocation/binary/library/source hashes alongside the report. Each report name must be fresh.

```sh
.venv/bin/python dev/benchmarks/flash_v5_attribution_run.py \
  --mode normal --lanes 1 \
  --report build/release/flash/v5-attribution-b1-normal.json --run
.venv/bin/python dev/benchmarks/flash_v5_attribution_run.py \
  --mode stage --lanes 1 \
  --report build/release/flash/v5-attribution-b1-stage.json --run
.venv/bin/python dev/benchmarks/flash_v5_attribution_run.py \
  --mode normal --lanes 4 \
  --report build/release/flash/v5-attribution-b4-normal.json --run
.venv/bin/python dev/benchmarks/flash_v5_attribution_run.py \
  --mode stage --lanes 4 \
  --report build/release/flash/v5-attribution-b4-stage.json --run
```

Default geometry is capacity8192, rows2048 per real lane, target verification arena16, one unprofiled warmup trial, one measured trial, and no decode. Four-lane mode uses four different exact frozen prompts and `FlashBatchPrefill`; singleton mode uses `FlashForward`. Saved-only residency uses the existing source-owned saved tensors/selected INT8 backing. Original checkpoint/PLE buffers are excluded from that residency lease. Optional `--decode-steps N` profiles sequential autoregressive continuations; the HTTP scheduler and trained MTP head are absent.

`REPORT.commands.jsonl` contains completed command GPU/wall/caller durations and normal host subphases, even in normal mode. Preparation is outside command wall. The four memory queries, submission-return interval, ticket wait, and callback latencies can overlap GPU execution and each other. Missing or incomplete asynchronous memory spans retain zero sample counts. Scheduled callback latency is callback arrival, not a GPU schedule timestamp.

`REPORT.trace.jsonl` contains optional full-command metadata/timestamps and normalized family attribution for GDN, QSA, MoE, shared expert, dense, PLE, HC, embedding, copies, and greedy. Shared expert classification uses the frozen source order: dense projection/activation after expert down and before canonical combine. Each complete graph validates one MoE route per model layer. Router projection before routing belongs to dense. The report identifies the model's36 GDN/12 QSA layers separately from repeated QSA chunk dispatches.

`normal` leaves counter/encoder profiling off. `command` collects existing full-command metadata with unchanged encoder boundaries. `stage` introduces an encoder per dispatch; `dispatch` inserts timestamp barriers. Counter-family durations/fractions explain the perturbed graph only. They are never HTTP throughput evidence, and gaps between command GPU time and summed sampled dispatch time remain explicit. Unsupported/truncated/missing counter data stays visible through the existing profile status and sample validity fields. Use the surrounding matched HTTP harness for performance conclusions.
