This private prefill candidate fuses the cached shared-expert gate and up
whole-K matmuls with canonical compiled BF16 SwiGLU. It shares direct device
BF16 input A across two distinct FP32 destination fragments, rounds both dots
to BF16, applies the compiled sigmoid's BF16 exp/add/div/sub boundaries, rounds
the SiLU product, and rounds the final up product. The separate scalar shared
expert gate and its precise unary sigmoid remain outside this experiment.

The current production gate/up cache route is M16/N64 because each weight
matrix has N640,K2560. The candidate preserves that dynamic-whole-K multiply
descriptor and four SIMD groups. Down uses the same M32/N128 cache projection
in both graphs. Complete rows therefore reduce the full shared chain from
four dispatches to two and avoid writing/reading two BF16 gate/up planes.
Weights use their existing immutable BF16 cache bits; no extra weight cache or
checkpoint conversion is required. The original matrices are Q8/G128.

Only rows256..8192 are accepted. A final incomplete row tile uses literal
qualified vector projections and canonical pointwise activation. All immutable
weights, input/output, diagnostic and tail scratch extents/overlaps are checked
before graph mutation. The private API has no model lookup, CPU dot product or
GPU submission. Timing uses the untapped kernel; an independent untimed
variant writes both rounded dot planes for qualification.

The original `build/flash-shared-expert-fused` host link used stale pre-v5
objects and must not be run. The replacement `build/flash-shared-expert-fused-v2`
links fresh `build/flash-default-v5` objects. Host object and ABI-header SHA256
provenance is embedded in each report, objects older than the ABI header are
refused, and nonfinite/at-most1ns GPU or wall timings fail qualification. This
guards against the observed `CommandTiming` return-layout change from16 to200
bytes. Earlier LUT experiments compiled before that header change remain
separate historical artifacts.

```sh
make -f Makefile -f dev/benchmarks/flash_shared_expert_fused_oracle.mk \
  BUILD=build/flash-shared-expert-fused-v2 flash-shared-expert-fused-oracle
.venv/bin/python -m unittest dev/tests/flash/test_flash_shared_fused_contract.py
build/flash-shared-expert-fused-v2/flash-shared-expert-fused-oracle --cpu-self-test
```

Compilation and CPU self-tests create no Metal device/backend and submit no
GPU work. Root runs GPU qualification serially after stopping the service:

```sh
FLASH_SHARED_FUSED_ROWS=256,257,2048,2049,8191,8192 \
FLASH_SHARED_FUSED_PAIRS=6 \
build/flash-shared-expert-fused-v2/flash-shared-expert-fused-oracle \
  build/flash-shared-expert-fused-v2/flash-shared-fused.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/shared-expert-fused-layer0-m16n64.json
```

Select layer47 with
`FLASH_SHARED_FUSED_PREFIX=language_model.model.layers.47.mlp.shared_expert`.
`FLASH_SHARED_FUSED_INPUT` accepts an exact raw BF16[rows,2560] input fixture;
one extent should then be selected. Otherwise input is explicitly declared
deterministic synthetic BF16 using real source weights. `M=16|32` and
`N=64|128` select separately compiled geometries; nondefault descriptors still
compare against the actual production M16/N64 reference and require full
numerical qualification rather than inheriting the canonical descriptor claim.

The oracle checks all4,915,200 cached coefficients against independent original
Q8 reconstruction, every rounded gate/up dot, every activated/down BF16 byte,
input/weight hashes, sticky diagnostics and output/input canaries. Invalid
row counts, immutable-input aliasing, output aliasing, dtype and absent tail
scratch must reject before modifying an empty graph. NaN,+Inf,-Inf input cases
run outside timing by default; use `FLASH_SHARED_FUSED_INVALID=0` only for a
later timing repeat after qualification. Metal fast-exp halfway behavior must
be checked on GPU; CPU libm is not its oracle. Matched full shared-chain
commands alternate order after warmup. A primitive gain needs full-model HTTP
qualification before becoming a production route.

The production opt-in is `SPLASH_FLASH_SHARED_EXPERT_FUSED=1`, requiring
`SPLASH_FLASH_DENSE_CACHE=1`. Its default remains off. Both `FlashForward` and
`FlashBatchPrefill` call one bridge owned by the frozen source trunk, share its
existing BF16 operands, and borrow existing gate/up workspace for tails. The
new route marker is
`shared-expert-cached-bf16-whole-k-mpp-f32accum-bf16-dots-compiled-bf16-swiglu-m32n128-min256-tail-m16n64-v1`.
No new workspace allocation or slot is added. MTP/verification/decode rows
below256 retain their previous routes. The production remainder uses canonical
M16/N64 for a remaining16-row block plus vector rows below16; this preserves
the production traversal for272/287/8191 rather than treating all31 possible
remaining rows as one vector window.

Fresh worker/library and batch-prefill oracle are built at
`build/flash-shared-expert-runtime-v1`. The production primitive oracle links
these fresh objects and the production shader, retaining the frozen private
M32/N128 dot witness solely for untimed checks:

```sh
make -f Makefile BUILD=build/flash-shared-expert-runtime-v1 flash-next \
  build/flash-shared-expert-runtime-v1/flash-batch-prefill-oracle
make -f Makefile -f dev/benchmarks/flash_shared_expert_production_oracle.mk \
  BUILD=build/flash-shared-expert-production-oracle-v1 \
  flash-shared-expert-production-oracle

FLASH_SHARED_FUSED_ROWS=256,257,271,272,287,2048,2049,2064,8191,8192 \
FLASH_SHARED_FUSED_PREFIX=language_model.model.layers.47.mlp.shared_expert \
FLASH_SHARED_FUSED_PAIRS=6 \
build/flash-shared-expert-production-oracle-v1/flash-shared-expert-production-oracle \
  build/flash-shared-expert-production-oracle-v1/flash-shared-production.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/shared-expert-production-layer47-tails.json
```

Root runs that GPU command serially. The production primitive keeps the same
complete-chain and immutable/diagnostic checks and mandatory canonical M16/N64
reference. Its runtime main descriptor is fixedM32/N128. The existing v2
private primitive artifacts and their historical GPU results are unchanged.
The standalone policy/ABI test covers96,118 CPU checks in optimized and
ASan/UBSan builds; this is separate from GPU and full-model qualification.
