# Flash prefill attention and GDN screen

This private benchmark focuses on actual large-row prompt prefill. Production
sources, defaults, and model residency have not been changed by this screen.
All GPU execution must be serialized with the main benchmark agent.

## Source evidence

`runtime/flash/FlashForward.cpp` caps the shared QSA workspace at 128 rows.
At a 2048-row prefill its QSA loop submits 16 complete
prepare/pool/select/attention/reduce sequences per QSA layer. Query, key, value,
and index projection matmuls use the full 2048 rows before that loop.
`runtime/metal/abi/FlashQSARowTiles.h` admits temporal M32 only for dense causal
windows beginning 512..1920, at most128 rows, and exactly 4 partitions.
`runtime/metal/kernels/shared/flash_qsa_row_tiles.metal` flattens real temporal
rows and 12 query heads per KV head. Its 32-row matrix tile stages each common
K/V bank once. M16 andM64 candidates retain the same dot-product chunks,
probability dtype, partition reducer, staged BF16 gate, and per-row causality.
Their common staging window and matrix schedule change numerical reduction
boundaries, so component equivalence is qualified numerically rather than
asserted bit exact.

For singleton large row GDN, Forward selects staged V16/T16 with 512 threads.
The production V32/T32/SG8 ILP route currently only admits 2..4 lanes with 512..2048
rows; its recurrence carries 4 independent value dimensions per SIMD group.
The existing private driver also admits a singleton, enabling a direct exact
comparison without changing production policy.

The old `build/release/flash/gdn-ilp-layout-prefill-timing.json` contains
implausible 3e-314 durations from an ABI mismatch and is not usable performance
evidence. The fresh private GDN driver requires matching timing ABI 200 bytes
and rejects finite durations outside 1e-9..600seconds.

## Compile and CPU qualification

```sh
make -j6 -f dev/benchmarks/prefill4k_attention/Makefile
build/prefill4k-attention/oracle --cpu-self-test
make -j6 -f dev/benchmarks/flash_gdn_batch_ilp_oracle.mk \
  GDN_BATCH_BUILD=build/prefill4k-attention-gdn \
  GDN_BATCH_BASELINE_BUILD=build/flash-next
build/prefill4k-attention-gdn/flash-gdn-batch-ilp-oracle --cpu-only
```

All backend and host objects are freshly compiled. CPU results before GPU work:
QSA 66,249 checks passed; GDN 1,186 checks passed; zero GPU submissions.

## Serial GPU screen

```sh
SPLASH_FLASH_QSA_ROW_TILES=1 FLASH_QSA_MPP_ROWS=128 \
FLASH_QSA_MPP_BEGIN=512,1024,1920 FLASH_QSA_MPP_REPEATS=5 \
build/prefill4k-attention/oracle \
  build/prefill4k-attention/attention.metallib \
  build/prefill4k-attention/screen.json

GDN_BATCH_ILP_ROWS=2048 GDN_BATCH_ILP_SLOTS=1 GDN_BATCH_ILP_COLD=0,1 \
GDN_BATCH_ILP_EXTREMES=0 GDN_BATCH_ILP_PATTERNS=0 GDN_BATCH_ILP_PAIRS=7 \
build/prefill4k-attention-gdn/flash-gdn-batch-ilp-oracle \
  build/prefill4k-attention-gdn/splash.metallib \
  build/prefill4k-attention-gdn/screen.json
```

QSA compares each output with the current temporal M32 control, an independent
sampled CPU global-softmax reference, and the canonical BF16 probability route.
It checks immutable input/cache hashes, output and scratch guards, inactive
partitions, deterministic repeats, and paired AB/BA timings. GDN checks every
intermediate, output, F32 recurrent-state and convolution-history byte including
padding, guards and inactive rows; two different input sequences continue from
the carried state before timing. GDN complete-graph and recurrence-only results
are separate.

A useful primitive result still requires matched full-model uncached HTTP
prefill timings, continuation output/state validation, cancellation/deadline
recovery, and final unloaded-model verification before promotion.

## Exact GDN register prefetch

`gdn_prefetch.metal` changes private V32/T16 and V32/T32/SG8 recurrence operand
loads only. The next token's q/k, four values, decay and beta enter independent
registers before the current token's SIMD reductions and state updates. Scalar
arithmetic and reduction order remain unchanged; FMA/reassociation are disabled.
The ~14 extra live operand registers can reduce occupancy or cause spills.

It compiled cleanly with the same Metal 4.1/O3/Werror options to
`build/prefill4k-attention-gdn/prefetch.air`. The private GDN pipeline names and
ABI are retained, so linking that AIR in place of `flash_gdn_batch_ilp.air`
allows the same fresh oracle and exact two-sequence continuation/guard checks:

```sh
GDN_BATCH_ILP_ROWS=2048 GDN_BATCH_ILP_SLOTS=1 GDN_BATCH_ILP_COLD=0,1 \
GDN_BATCH_ILP_EXTREMES=0 GDN_BATCH_ILP_PATTERNS=0 GDN_BATCH_ILP_PAIRS=7 \
build/prefill4k-attention-gdn/flash-gdn-batch-ilp-oracle \
  build/prefill4k-attention-gdn/prefetch.metallib \
  build/prefill4k-attention-gdn/prefetch-screen.json
```

## Measured September 21 screen

All GPU work ran serially in the root-granted slot; no full model was loaded.
Sources/defaults remain unchanged. Summary: `build/prefill4k-attention/summary.json`.

| Candidate | Component result | Accuracy |
| --- | --- | --- |
| QSA M16 | Within about ±2% of M32 | Relative L2 2–4e-6; 9–17 changed BF16 cells/786,432; guards/input-cache SHA passed |
| QSA M48 | 4–6% slower | Same bounded numerical qualification; guards/input-cache SHA passed |
| QSA M64 | 14–22% slower | Relative L2 4–10e-6; 24–37 changed BF16 cells/786,432; guards/input-cache SHA passed |
| GDN V32/T16/SG8 | Recurrence 2.05–2.06 ms versus current 1.92 ms | Zero changed bytes over two changed carried sequences, all state/history/guards |
| GDN V32/T32/SG8 | Recurrence 2.01 ms versus current 1.92 ms | Same exact continuation/guards |
| GDN register prefetch | Recurrence 2.21 ms versus current 1.92 ms | Same exact continuation/guards |
| Existing staged GDN V8/T16 or T32 | Complete graph 6–7% slower | Intermediate/output/F32 state/history bytes exact |
| Existing staged GDN V16/T32 | Complete graph 2.076 ms versus 2.130 ms, 2.57% faster | Intermediate/output/F32 state/history bytes exact |

The small V16/T32 primitive gain implies roughly 0.2% total prefill on the fresh
2K attribution, so it is not a meaningful default promotion yet. Current
profile attribution is target 845 ms plus teacher prime 121 ms; QSA contributes
about 132 ms including prime and GDN 77 ms. The original Q4 MoE miss path and
teacher prime computation remain the higher-impact paths toward 4K tok/s.

Additional exact [staging variants](staging.md) passed full intermediate/output
byte equality but reduced throughput 11–38%. The original
[coalesced prefix](coalesced.md) alone was dismissed because its direct work
opportunity is only 4.65 ms/request. The later bulk graph also coalesces attention
and reduction, and has separate measured results below. Normal production
remains untouched.

## Measured attention follow-up

These reports use synthetic QSA inputs and paired GPU timings on the M5 Ultra.
They establish component performance and continuation/cache parity; they do not
measure model generation, inclusive prefill tok/s, or HTTP lifecycle behavior.
Each control belongs to its own paired screen, so small differences between
reports are not an additional measured gain.

| Candidate | Paired GPU result | Qualification |
| --- | --- | --- |
| Temporal M32/SG8, starts 512/1024/1920, 128 rows | 1.254× / 1.286× / 1.272× versus current M32 | All 786,432 BF16 output words per case and all active F32 partition statistics/numerators exact |
| Exact bulk QSA v2, fresh 2048 rows | 9.979 → 8.005 ms, 1.247×; 80 → 5 dispatches | All 12,582,912 BF16 output words, F32 partials, prepared queries/index/selection, all five cache planes, and four future sparse appends exact |
| Exact bulk QSA with temporal SG8 | 9.945 → 6.866 ms, 1.449×; 80 → 6 dispatches | Same full-QSA exact output/partial/cache/future-append checks |
| Strict register PV v2, starts 512/1024/1920 | 0.703× / 0.644× / 0.617×; throughput 30–38% lower | BF16 output words and active F32 statistics/numerators exact; GPU time 42–62% higher |
| Bulk direct tile-local BF16 probabilities | 9.930 → 8.023 ms, 1.238×; 80 → 4 dispatches | 3,250,932 changed BF16 words/12,582,912, relative L2 0.001953; prepared data and cache/future-append parity passed |

The BF16-probability alternative does not improve on the exact bulk v2 timing
and changes numerical behavior. Strict register PV also offers no performance
reason for promotion. The combined exact SG8 bulk graph is the useful component
candidate; matched whole-model prefill, output/state continuation, memory
admission, and service lifecycle qualification remain pending.

Reports: [temporal SG8](../../../build/release/flash/prefill4k-attention-sg8-screen-v1.json),
[exact bulk v2](../../../build/release/flash/prefill4k-attention-bulk-exact-v2.json),
[combined bulk SG8](../../../build/release/flash/prefill4k-attention-bulk-sg8-v1.json),
[strict register PV v2](../../../build/release/flash/prefill4k-attention-register-pv-strict-screen-v2.json),
and [bulk BF16 probabilities](../../../build/release/flash/prefill4k-attention-bulk-direct-bf16pv-v1.json).

The private full-model source overlays are built separately from production:

| Build | Source and coefficient identity | CPU qualification |
| --- | --- | --- |
| `build/prefill4k-qsa-bulk-allrows-full512` | Sealed `prefill4k-allrows-full512` control; its certified private I8 derivative is unchanged | 116 source hashes, reversible Forward edit, nine compiled admission cases, worker self-test |
| `build/prefill4k-qsa-bulk-top64-control` | Fresh normal-source control without the bulk helper | Fresh build and worker self-test |
| `build/prefill4k-qsa-bulk-top64` | Same fresh normal-source control plus the bulk helper; original model identity is unchanged | 112 source hashes, reversible Forward edit, six compiled admission cases, original 2048-row cap and unsupported 8192-row rejection, worker self-test |

`bulk_runtime_overlay.py` and `normal_runtime_overlay.py` create fresh source
snapshots without reading weight payloads. Compile with `Makefile`,
`dev/benchmarks/prefill4k_wide.mk`, and `bulk_runtime.mk`, setting
`PREFILL4K_WIDE_BUILD` to the new build and `PREFILL4K_WIDE_FULLCACHE=1` only
for the private Full512 source. `bulk_runtime_witness.py` checks the source
handoff, seals the four runtime artifact hashes, and exercises compiled
admission; each candidate stores its report in `runtime-cpu-v1.json`.

Both new flags default off. Enable `SPLASH_FLASH_QSA_BULK_PREFILL=1` and
`SPLASH_FLASH_QSA_BULK_PREFILL_SG8=1` alongside existing QSA F32/MPP/ROW_TILES
policies to select the combined route. Eligibility is limited to the main
non-verification call at begin 0 with exactly 2048 rows. Other calls retain
the original 128-row loop verbatim. The planner admits 234,356,736 extra bytes
(223.5 MiB) before construction; the allocation ledger and five-plane sum
are checked, and external destination guards cover every new plane. SG8
does not add another allocation. Whole-model output/state and HTTP lifecycle
qualification remain separate from these CPU checks.
Graph geometry, scratch requirements, and reproduction are in
[staging.md](staging.md).
