# Exact QSA prefix coalescing — prepared and dismissed for low impact

Normal production sources/defaults remain untouched. The private host helper
`coalesced.hpp/.cpp` compiles; no coalesced GPU command has been submitted.

## Measured opportunity

The fresh current-profile 2K coding stage trace
`build/release/flash/prefill4k-current-stage.json.trace.jsonl` assigns target
prefix work across all 12 QSA layers/192 original 128-row windows:

| Existing phase | Sampled GPU time |
| --- | ---: |
| `flash_qsa_fast_prepare` | 2.911206 ms |
| `flash_qsa_pool_rope_bf16` | 0.993093 ms |
| `flash_qsa_select_blocks` | 0.749085 ms |
| Total candidate work | **4.653384 ms** |

Target prefill is 845.24 ms and teacher priming 121.17 ms in this diagnostic
trace. Coalescing these phases alone leaves the expensive original attention
kernels unchanged. The direct work opportunity is less than 0.5% of current
prefill, before considering the additional 30 MiB scratch writes/reads. Dispatch
gaps or changed cache locality would need separate measurements, but there is
no evidence suggesting a material gain. This path is dismissed for the 4K tok/s
objective, following the parent task's small-overhead dismissal option.

## Prepared implementation and explicit gates

The dedicated workspace reserves exactly 30 MiB at 2048 rows: full prepared
queries 24 MiB, prepared index queries 2 MiB, chronological selected IDs 4 MiB.
Its constructor documents engine reservation before allocation; no constructor
has been called by this experiment.

`denseCoalescedEligible` requires a positive bounded state capacity, 256..2048
rows, and the **entire append** to end at or before token 2048 and within state
capacity. Sparse appends are rejected. The normal 128-row attention workspace
is mandatory. The helper builds every authoritative current chunk graph before
mutating the caller's graph, preserving each chunk's route, partition count,
attention shape, reducer and parameters. Only preparation, completed-block
pooling and dense chronological selection become three bulk dispatches.

Full input/bulk planes must be large enough and disjoint; bulk planes are also
checked against persistent caches, partition scratch, diagnostics, supplied
positions, and immutable norm tensors. Norm dtype/convention/epsilon/theta and
pool parameters come from the authoritative current prefix.

For dense queries, `flash_qsa_select_blocks` fills only IDs below
`floor((begin+row+1)/4)`. It does not read block scores or pooled keys. Thus pooled
future blocks created by bulk preparation are excluded from earlier queries.
The original attention token mapping retains the causal incomplete raw tail,
and all original attention dispatches/parameters remain unchanged. This is a
source/geometry argument, **not GPU parity evidence**.

## Compile/CPU result and remaining qualification

```sh
make -j3 -f dev/benchmarks/prefill4k_attention/Makefile coalesced-cpu-test
```

**125,852,840 pure CPU geometry/causal-selection checks passed**, including
64-bit overflow/boundary comparisons and explicit future-block/raw-tail
exclusion. Zero MetalBackend constructions or GPU submissions occurred.

Full output/prepared-query/cache byte parity and later sparse append parity
were deliberately **not executed** after the measured opportunity proved
small. This helper must not be integrated or called generation-qualified
without those checks. If revisited, required GPU fixtures include appends ending
at 2048, unaligned starts, mixed norm dtypes/conventions, supplied positions,
guarded cache tails, and subsequent 1/3/7/128-row sparse appends. Compare every
cache plane (keys, values, raw keys, pooled keys, positions), every output byte,
all guards and sticky diagnostics against the current chunked path.
