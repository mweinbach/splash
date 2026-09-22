# Exact attention staging screen

Normal production sources/defaults are unchanged. Both private candidates
retain M32 temporal query tiles, original 64-token score/softmax/reduction banks,
current 128-row chunk route, and current staged gate/reducer. No model is loaded
by these standalone component screens.

`query_reuse.metal` stages the entire BF16 query once per threadgroup. It keeps
the 64 query cells overwritten by the existing alpha union in one register per
participating thread, restores them before QK, and adds no barriers. Its source
shared allocation is exactly 32 KiB versus the existing 20 KiB. **46,898,176 pure
CPU query-bank/alias/strided-operand checks passed** before GPU execution.

`values_n128.metal` retains original K64 reduction banks and widens only PV
output-column tiles from 64 to 128. It halves PV calls/barrier pairs, with a 28 KiB
source shared allocation. Output-column computation remains independent.

The strict oracle requires **zero changed bytes** for every BF16 output and
all F32 partition statistics/value intermediates, alongside deterministic
repeats, guards, inactive partitions, immutable input/cache SHA256, and sampled
independent CPU attention checks. Both candidates passed these gates at 128 rows
and dense causal starts 512,1024,1920.

## Results

| Begin | Current paired M32 | Query reuse | Speedup |
| --- | ---: | ---: | ---: |
| 512 | 0.534 ms | 0.598 ms | 0.893x |
| 1024 | 0.611 ms | 0.700 ms | 0.873x |
| 1920 | 0.978 ms | 1.142 ms | 0.857x |

| Begin | Current paired M32 | PV N128 | Speedup |
| --- | ---: | ---: | ---: |
| 512 | 0.393 ms | 0.492 ms | 0.798x |
| 1024 | 0.616 ms | 0.813 ms | 0.757x |
| 1920 | 1.005 ms | 1.365 ms | 0.736x |

All GPU work ran serially in root's exclusive slot. The paired controls belong
to each screen; compare within the same row. Larger scratch/strided operand
loads or occupancy may explain the slowdown, but no hardware counter establishes
the cause. Neither candidate is promoted or full-service qualified.

## Reproduce

```sh
make -j3 -f dev/benchmarks/prefill4k_attention/Makefile staging query-reuse-cpu-test

SPLASH_FLASH_QSA_ROW_TILES=1 PREFILL4K_ATTENTION_QUERY_REUSE=1 \
PREFILL4K_ATTENTION_TILES=32 FLASH_QSA_MPP_ROWS=128 \
FLASH_QSA_MPP_BEGIN=512,1024,1920 FLASH_QSA_MPP_REPEATS=7 \
build/prefill4k-attention/oracle \
  build/prefill4k-attention/staging.metallib \
  build/prefill4k-attention/query-reuse-screen.json

SPLASH_FLASH_QSA_ROW_TILES=1 PREFILL4K_ATTENTION_VALUES_N128=1 \
PREFILL4K_ATTENTION_TILES=32 FLASH_QSA_MPP_ROWS=128 \
FLASH_QSA_MPP_BEGIN=512,1024,1920 FLASH_QSA_MPP_REPEATS=7 \
build/prefill4k-attention/oracle \
  build/prefill4k-attention/staging.metallib \
  build/prefill4k-attention/values-n128-screen.json
```

Do not enable both staging flags; their combined source scratch exceeds the
32 KiB device limit. These isolated reports use 4 maximum partitions, whereas
the service allocates 32 to support its tiny-row decode routes; active prefill
partitions remain 4 in both. Any later performance qualification must include
the service layout and actual request lifecycle.

The original exact coalesced prefix prototype is separately documented in
[coalesced.md](coalesced.md). Its current-profile direct work opportunity alone
is only 4.65 ms/request, so that prefix-only screen was dismissed. The bulk
prototype below also coalesces attention and reduction and was GPU-qualified
independently.

## Exact temporal SG8 and strict register PV

The temporal SG8 variant keeps M32 query/head grouping, the original four
partitions, F32 probabilities, and reduction/gate behavior. It uses 256 threads
instead of 128. At dense causal starts 512, 1024, and 1920, its paired speedups
were **1.254×, 1.286×, and 1.272×**. Every BF16 output word (786,432 per case),
active F32 partition statistic, and active F32 numerator matched the production
M32 control exactly. Input/cache immutability, output/scratch guards, and
inactive partition checks passed.

Strict register PV v2 also matched those output and F32 intermediate bytes,
after replacing assumed cooperative-tensor coordinates with the API's
`get_multidimensional_index`. It measured **0.703×, 0.644×, and 0.617×** at the
same three starts: throughput fell 30–38%, or GPU duration rose 42–62%. It is not
a performance candidate. These results do not establish a cause such as spills
or occupancy without hardware-counter evidence.

Reports: [SG8 temporal screen](../../../build/release/flash/prefill4k-attention-sg8-screen-v1.json)
and [strict register PV v2](../../../build/release/flash/prefill4k-attention-register-pv-strict-screen-v2.json).

## Exact bulk 2048-row graph

`bulk.cpp` admits only a fresh dense causal append with `begin == 0` and
`rows == 2048`. It prepares all queries, pools cache/index data, and selects
chronological blocks once, then preserves the ordinary 16-window attention
policy: the first 128 rows have one partition, subsequent windows have four,
and temporal M32 begins at 512. `bulk_attention.metal` v2 retains safe attention
math and the original fast scalar reducer math; preserving that distinction
was necessary for exact final BF16 bytes.

The exact SG4 bulk graph has one 7,936-threadgroup attention dispatch and one
bulk reducer, reducing the ordinary 80 dispatches to five. The SG8 composition
splits attention into an unchanged early online dispatch (3,328 threadgroups,
128 threads each) and a temporal SG8 dispatch (4,608 threadgroups, 256 threads
each), for six total dispatches. Both retain the original F32 partial layout
and reducer. Their private full-row workspace adds **234,356,736 bytes
(223.5 MiB)** and must enter memory admission before allocation.

| Graph | Paired control | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| Exact bulk v2 | 9.979 ms | 8.005 ms | 1.247× |
| Exact bulk plus temporal SG8 | 9.945 ms | 6.866 ms | 1.449× |

Both standalone synthetic full-QSA reports completed. They matched all
**12,582,912 BF16 output words**, every active F32 partial, prepared
queries/index/selection, and all five QSA cache planes including the four
future sparse appends. Projected inputs and norms remained unchanged. This is
component qualification only; neither report contains model generation,
whole-model continuation/rollback, inclusive prefill tok/s, or HTTP lifecycle
qualification.

The direct bulk BF16-probability alternative measured 9.930 → 8.023 ms
(1.238×). It casts tile-local unnormalized probabilities, changes partition
behavior, and changed **3,250,932 of 12,582,912 BF16 output words** with relative
L2 **0.001953**, despite exact prepared data/cache/future-append checks. Its
timing offers no added benefit over exact bulk v2, so it is not promoted.

Reports: [exact bulk v2](../../../build/release/flash/prefill4k-attention-bulk-exact-v2.json),
[combined bulk SG8](../../../build/release/flash/prefill4k-attention-bulk-sg8-v1.json),
and [direct BF16 probabilities](../../../build/release/flash/prefill4k-attention-bulk-direct-bf16pv-v1.json).

Run each command serially with the benchmark coordinator's GPU slot. The SG8
library replaces the original bulk attention AIR with the SG8 AIR; linking
both would duplicate the original symbols.

```sh
SPLASH_FLASH_QSA_ROW_TILES=1 \
build/prefill4k-attention/bulk-oracle \
  build/prefill4k-attention/bulk-v2.metallib \
  build/prefill4k-attention/bulk-exact-fresh.json

SPLASH_FLASH_QSA_ROW_TILES=1 PREFILL4K_BULK_SG8=1 \
build/prefill4k-attention/bulk-oracle \
  build/prefill4k-attention/bulk-sg8.metallib \
  build/prefill4k-attention/bulk-sg8-fresh.json
```
