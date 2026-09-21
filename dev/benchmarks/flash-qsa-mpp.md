The isolated QSA MPP candidates reuse the qualified prepared BF16 queries,
key/value caches, and chronological selected-block IDs. They replace attention
only. Indexer scores, highest-ID cutoff ties, the incomplete causal tail,
normalization, RoPE, cache append, and rollback remain in the existing QSA
prefix.

The materialized route computes all twelve query heads sharing one KV head in
a padded sixteen-row matrix tile. QK uses BF16 operands with F32 accumulation,
followed by global stable F32 softmax and MPP PV. It supports either a BF16
probability boundary or F32 probabilities with mixed F32/BF16 PV operands.
Its score tile is 64 or 128 selected tokens, and it submits three dispatches.

The fused online route keeps QK, stable softmax weights, and PV in threadgroup
memory. Each group owns twelve query heads and one token partition. It uses
F32 probabilities, BF16 cache values, F32 accumulation, and the qualified
partition reducer and staged BF16 sigmoid/output gate. It submits two
dispatches and reuses `FlashQSAFastWorkspace`; it adds no global score sheet.
Partitions 1, 2, 4, and 8 are supported. Its grouped-query organization follows
[oMLX's sparse GQA attention](https://github.com/jundot/omlx/blob/14194fe74bab38b89c144bd89656fbedca641d14/omlx/custom_kernels/glm_moe_dsa/csrc/kernels/steel_qwen4_qsa_sparse_gqa.h),
using Metal Performance Primitives for the matrix operations.

These are explicit numerical alternatives because the MPP matrix reduction
order differs from the control's SIMD/scalar dot products. They require
standalone numerical qualification and full-model coherence and throughput
qualification before adoption. Compilation and the CPU self-test alone are
not GPU correctness or performance evidence.

Build the isolated oracle:

```sh
make -j8 -f Makefile -f dev/benchmarks/flash_qsa_mpp_oracle.mk \
  BUILD=build/flash-qsa-mpp build/flash-qsa-mpp/flash-qsa-mpp-oracle
build/flash-qsa-mpp/flash-qsa-mpp-oracle --cpu-self-test
```

Root owns GPU execution. A small paired timing/numerical screen at the final
128-row chunk of a 2,048-token prompt is:

```sh
FLASH_QSA_MPP_ROWS=128 FLASH_QSA_MPP_BEGIN=1920 \
FLASH_QSA_MPP_PATTERNS=latest FLASH_QSA_MPP_REPEATS=3 \
FLASH_QSA_MPP_MATERIALIZED=0 FLASH_QSA_MPP_ONLINE=1 \
FLASH_QSA_MPP_ONLINE_PARTITIONS=4 \
build/flash-qsa-mpp/flash-qsa-mpp-oracle \
  build/flash-qsa-mpp/splash.metallib REPORT_JSON
```

The oracle compares every candidate with partitioned F32 and canonical BF16
controls, checks guarded outputs and immutable-input SHA256, and samples an
independent CPU attention reference. It times paired AB/BA commands after
warmup. Use a separate shader-validation run for numerical/safety screening;
collect timing with validation disabled and no concurrent GPU work.

By default it covers rows 1/32/128 and begin positions
0/128/1024/1920/2048/8192, including latest/strided/cutoff-tie sparse selections.
Environment controls include `FLASH_QSA_MPP_ROWS`, `FLASH_QSA_MPP_BEGIN`,
`FLASH_QSA_MPP_PATTERNS`, `FLASH_QSA_MPP_REPEATS`, `FLASH_QSA_MPP_WARMUP`,
`FLASH_QSA_MPP_TILES`, `FLASH_QSA_MPP_F32_PROBABILITIES`,
`FLASH_QSA_MPP_ONLINE_PARTITIONS`, `FLASH_QSA_MPP_MATERIALIZED`, and
`FLASH_QSA_MPP_ONLINE`. A failing or absent report is not a pass.
