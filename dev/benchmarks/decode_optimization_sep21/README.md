The private worker combines the sealed exact SG8 bulk prefill and Full512 I8
target with the existing direct gathered MPP producer for small physical rows.
The gathered projection shader and target bucket shader are unchanged. The
trained MTP graph is unchanged. Every original bucketed Store method remains
present as the same-binary flag0 control and the larger-row route.

```sh
.venv/bin/python dev/benchmarks/decode_optimization_sep21/overlay.py
make -f Makefile -f dev/benchmarks/decode_optimization_sep21/worker.mk \
  -j8 decode-sep21-worker-cpu
.venv/bin/python dev/benchmarks/decode_optimization_sep21/witness.py \
  build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1
```

GPU/model work is performed by Root only. Use the existing exact-bulk Full512
environment, plus strict `SPLASH_FLASH_ALLROWS_GATHERED_MPP=0/1`. Flag0 keeps the
original target numerical derivative. Flag1 includes the gathered policy and
the route cap in its numerical derivative. Optional strict
`SPLASH_FLASH_ALLROWS_GATHERED_MPP_MAX_ROWS=1/2/4/8/16` defaults to4. Both flag
and cap are frozen at Store construction and rechecked by its route methods.
The status fields expose the enabled flag, cap and both producer/route policies.
Existing dedicated gathered Store graph counters prove branch construction;
they do not assert GPU completion.

Existing one-layer synthetic paired reports found exact old-MPP F32 dots,
scaled F32, BF16 projections, activated rows and full-chain down bytes at
rows1/2/4/16. GPU medians old/gathered were0.498/0.199ms at1 row,
0.550/0.222ms at2 mixed rows and0.509/0.272ms at4 repeated rows. At16 repeated
rows, the gathered0.737ms regressed against the bucketed0.529ms. Therefore4
is the conservative experiment cap. These reports did not contain real
normalized decode activations or qualify the complete service. The existing
baseline and gathered F64 strict failures stay recorded as failures; exact
old-MPP reproduction is the intended arithmetic compatibility axis.

The component/build CPU tests and source witness passed. Real generation,
acceptance, state/rollback, malformed diagnostics, service quality, lifecycle,
prefill and decode timing remain Root's qualification work. Use a fresh output
path for another source composition; the generator refuses existing outputs.

The singleton verifier diagnostic links the completed private worker's host
objects and the same normal core ABI. Build/help/CPU checks create no backend:

```sh
make -f Makefile -f dev/benchmarks/decode_optimization_sep21/verify_attribution.mk \
  -j4 decode-sep21-verifier-attribution-cpu
build/sep21-gathered-verifier-attribution-v1/verify-attribution --help
```

Root invokes GPU mode with the existing Full512/exact-bulk environment and
either gathered flag0 or1, in separate processes:

```sh
build/sep21-gathered-verifier-attribution-v1/verify-attribution --gpu \
  build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1/splash.metallib \
  PACKAGE PROMPT2048_JSON NEW_REPORT_JSON
```

Optional `--incoming FOUR_TOKENS_JSON` admits an exact four-token proposal
fixture. Otherwise the oracle derives four ordinary greedy continuation IDs
once on an untimed disposable target state; these are not trained MTP proposals.
Each measured trial then prefills a fresh state to exactly2048, profiles only
the four-row target verification, commits all4 without another GPU command,
and validates a healthy next-token continuation. It records prompt/artifact/
derivative provenance, prefill and verification logits, hidden and continuation
hashes, predictions, family/pipeline timings and full raw profiling metadata.
`DECODE_SEP21_VERIFY_MODE=normal/command/stage/dispatch` defaults tostage;
`DECODE_SEP21_VERIFY_WARMUP` defaults to1 and `DECODE_SEP21_VERIFY_REPEATS` to1.
All counter timing is diagnostic; stage and dispatch modes alter scheduling.
The optional existing expert-ID capture flag adds48 word-copy dispatches, and
the report makes that condition explicit.
