The v3 worker retains the existing F32 dense cache, BF16 prefill cache, Full512
target and gathered decode routes. It adds a private numerical alternative for
known main target projections. Default0 preserves all existing projection
branches and the numerical derivative seed.

```sh
.venv/bin/python dev/benchmarks/decode_optimization_sep21/bf16_overlay.py
make -f Makefile -f dev/benchmarks/decode_optimization_sep21/bf16_worker.mk \
  -j8 decode-sep21-bf16-worker-cpu
.venv/bin/python dev/benchmarks/decode_optimization_sep21/bf16_status_v3.py
make -f Makefile -f dev/benchmarks/decode_optimization_sep21/bf16_status_v3.mk \
  -j4 decode-bf16-v3-worker-cpu
.venv/bin/python dev/benchmarks/decode_optimization_sep21/bf16_witness.py \
  build/prefill4k-qsa-bulk-gathered-bf16-sep21-v2
```

Root uses the current combined gathered/exact-bulk environment, keeping
`SPLASH_FLASH_DENSE_CACHE=1`, `SPLASH_FLASH_FLOAT_DENSE_CACHE=1`,
`SPLASH_FLASH_DENSE_SMALL_ROWS=0`, F32 HC-up and the vocabulary policies.
The new strict selector is `SPLASH_FLASH_DECODE_BF16_DENSE=0/1`.
Optional strict selectors are:

- `SPLASH_FLASH_DECODE_BF16_DENSE_SCOPE=all_cached/existing_f32_route`, default all_cached.
- `SPLASH_FLASH_DECODE_BF16_DENSE_ROLES=all/attention/shared/ple/hc_up_small`, default all.

The selector admits only physical rows1/2/4/8/16 and exact known source shapes.
all_cached uses the existing rounded BF16 coefficient cache for matched roles.
existing_f32_route additionally requires the baseline generic F32 route to be
eligible for that role/row; its identity binds the frozen selective-F32 flag.
With the current selective policy, rows1/2 do not select body BF16 under this
scope. Unknown roles, format shapes and row counts retain their old routes.
Matching small main prefills/suffixes also select it; full2048 SG8 prefill stays
on its unchanged route.

hc_up_small selects only main HC up K320→N10240 at1/2 physical rows, through
the source trunk's cached-up callback. It writes existing raw-up scratch with
the BF16 small-row producer and uses unchanged `addHCMix`. That mix keeps
precise BF16 sigmoid, BF16 products, ordered stream addition and division
boundaries. Fused down, injection, trained MTP and HC-up at4/8/16 remain
unchanged. The ordinary batch-forward fused-HC helper bypasses the callback,
so this extension currently covers singleton autoregressive1/2 and eligible
joint verifier callbacks; ordinary B2 fused HC remains raw. The other dense
role selectors do cover ordinary batches through `trunk.batchProject`.

Flag/scope/roles/selective choice are frozen and rechecked. Flag1's target
derivative and route/status metadata identify cached BF16 coefficient precision
and the selected role policy. Dedicated graph-call/row counters describe graph
construction, not completed GPU work. The optional BF16 pad arena adds exactly
1MiB and is independently admitted and excluded from external destinations.
Existing kernels and cache helper sources remain unchanged.

V3 corrects the v2 status placement: changing graph-call/row counters live in
the separate top-level `decode_bf16_dense_route_counters` object. Immutable
identity retains only enabled/policy configuration. The status transform's
CPU source guard rejects mutable counters inside identity, including the
original v2 regression. Only Worker source/object changes in v3;46 hash-matched
non-Worker objects and the byte-identical metallib are reused. The benchmark's
full immutable identity comparison stays unchanged.

The model's raw F32 coefficients and cached BF16 coefficients differ; MPP
reduction also differs from raw QMV. Compilation, CPU tests, source seals and
layout/stage review do not qualify greedy outputs, MTP acceptance, state,
service quality or throughput. Root performs those GPU/model checks. The
matching diagnostic oracle is
`build/sep21-bf16-verifier-attribution-v2/verify-attribution`; use a fixed
`--incoming FOUR_TOKENS_JSON` for identical proposal inputs across flag0/1 if
the alternative changes its ordinary greedy seed continuation.
