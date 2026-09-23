# Flash-Next decode/prefill optimization (September 22)

Self-contained build of the optimized Flash-Next worker used by the local
`m5-ultra-flash-next-v16` profile (v14 and v15 were the first two passes). The base is the
qualified v13 worker (`build/R5-integer-currentQ4-fixed4-sep22-worker-v2`),
whose source previously existed only under `build/`. `worker/` holds that full
source, the prebuilt AIR inputs of its library (original link order in
`air-order.txt`), and the changes below. Every core object rebuilds from
source; the 80 AIR inputs are binary.

```sh
make -C dev/benchmarks/flash_opt_sep22/worker -j12
make -C dev/benchmarks/flash_opt_sep22/worker install DEST=$PWD/build/flash-opt-sep22-v16
```

The v16 profile pins `build/flash-opt-sep22-v16/splash-flash` and its metallib
by SHA-256; rebuild and re-pin `LOCAL_PROFILE_V16` in `install/launcher.py` after
changing anything here. The environment is identical to v13-v15.

## Results (M5 Ultra, 256 GB)

Ten fresh prompts × 256 tokens (`tools/quality.py --decode-only`), which also
exercises uncached PLE rows. Decode speed is cycle time divided by accepted
tokens per cycle; acceptance depends on each prompt's greedy path, which any
numerical change in prefill can move, so compare cycle times first.

| | v13 | v14 | v15 | v16 |
| --- | ---: | ---: | ---: | ---: |
| Aggregate decode, tok/s | 51.5 | 102.5 | 118.9 | 116.3 |
| Accepted tokens per cycle | 3.31 | 3.36 | 3.38 | 3.29 |
| Cycle wall time | 64 ms | 32.7 ms | 28.3 ms | 28.3 ms |
| Verify GPU per cycle | 37.1 ms | 23.7 ms | 23.1 ms | 23.0 ms |

v16 changes short-prompt prefill numerics, which moves these ten paths. On 60
other fresh prompts v15 and v16 accept 2.908 and 2.903 tokens per cycle (paired
difference -0.6 tok/s, t = -0.6); cycle time is unchanged.

Canonical 2,048-token coding prompt, 256 output tokens, greedy, MTP depth 4,
one warmup + three trials (`dev/benchmarks/splash_tuning_sep21.py`), and
seeded 2,048-token lanes for widths 2 and 4:

| | v13 | v14 | v15 | v16 |
| --- | ---: | ---: | ---: | ---: |
| Prefill, tok/s | 4,014 | 4,097 | 4,196 | 4,208 |
| Canonical decode, tok/s | 62.0 | 75.0 | 124.2* | 84.6 |
| Two lanes, aggregate decode | crash | 99.3 | 111.5 | 117.1 |
| Four lanes, aggregate decode | crash | 121.2 | 137.0 | 143.9 |
| Two / four lanes, prefill | crash | 2,862 / 2,955 | 2,925 / 3,007 | 2,935 / 3,015 |

\* v15's prefill summation order put the canonical prompt on a different
greedy path with higher acceptance (70 cycles instead of 100). v14 and v16
follow the same 100-cycle path, so 75.0 → 84.6 is the like-for-like gain.

Fresh short prompts, one output token, median native prefill wall time
(`tools/prefill_probe.py`; token counts include the chat template):

| Prompt tokens | v15 | v16 |
| ---: | ---: | ---: |
| 49 | 113 ms | 95 ms |
| 111 | 169 ms | 161 ms |
| 206 | 204 ms | 205 ms |
| 396 | 265 ms | 265 ms |

Frozen 22-case semantic suite: v13 20/22, v14 19/22, v15 20/22, v16 20/22 (the
same two arithmetic cases fail in every build).

## First pass (v14)

Decode and verification (MTP draft depth 4 verifies 5 rows per cycle):

- **Multi-row quantized GEMV** (`kernels/opt_qmv.metal`, `OptQmv.cpp`). Every
  4/5/6/8-bit affine projection with up to 5 rows reads each weight once for all
  rows, with vectorized code loads and split-K for long inputs. It replaces the
  per-row raw GEMVs, the F32 dequantized "float dense" caches (4 bytes per
  weight) and BF16 small-row caches on these paths.
- **Hyper-connection kernels**: fused down projection (10240→320, SiLU) plus
  the four injection gates in one dispatch, and up projection fused with the
  sigmoid-gated stream mix. HC-down went from 48-72 µs to ~8 µs per call.
- **Router and routing**: multi-row BF16 router GEMV and a one-simdgroup top-10
  route kernel (bit-identical to `flash_moe_route`).
- **Draft head**: greedy single-row MTP drafts project only the leading 98,304
  vocabulary rows (99.8% of generated tokens), falling back to the full head
  when the step's input token is outside that range. The target verifier still
  judges every token. `SPLASH_OPT_DRAFT_VOCAB=0` restores the full head.
- **Attention**: 5-12 row verification windows use 32 QSA partitions.
- **PLE SSD streaming**: row misses are read concurrently (15.9 → 1.2 ms/cycle).
- **Prefill**: few-output projections (N ≤ 64) over long inputs use one
  row-blocked GEMV; MoE bucket prefix scans run in one simdgroup.
- **Batching**: fixed a crash on every multi-lane request in v13 (the batched
  GDN path rejected the FMA prefill recurrence the profile enables).

## Second pass (v15)

The cycle profile showed the GPU idle about 20% of every decode cycle, waiting
on host work. Host-side changes (all in `worker/source`):

- `FlashInt8ExpertStore::immutableDisjoint` compared each expert output against
  all 96 store buffers through Objective-C `contents()` calls on every graph
  build; the ranges are now captured once. `MetalBuffer::contents()` caches the
  CPU address, the backend caches pipelines by name without building an
  NSString per dispatch, and binding checks use a bit mask instead of two hash
  sets per dispatch. R5 setup validation runs once instead of per graph.
- The pre-command status republish is throttled to one full snapshot per
  25 ms (`SPLASH_OPT_STATUS_THROTTLE_MS`); loop and completion boundaries still
  publish, so counters at request safe points stay exact. PLE store statistics
  are a snapshot and never wait behind row I/O.
- Streamed PLE rows for each known token prefix are read on a background queue
  while the head fold and draft steps run (`SPLASH_OPT_PLE_PREFETCH=0` off).
- The verify graph is built while the head fold executes on the GPU
  (`FlashForward::prepareVerify`/`verifyPrepared`; `SPLASH_OPT_PREPARE_VERIFY=0`).
- The middle draft steps run as one command with GPU greedy-token feedback;
  the last step stays separate so its PLE rows load meanwhile
  (`SPLASH_OPT_DRAFT_CHAIN=0`).
- Released request states are pooled and reinitialized (zeroing only the cache
  prefix a request could have written) instead of reallocated: state setup
  27 → <1 ms and about 7 ms less first-command scheduling per request
  (`SPLASH_OPT_STATE_POOL=0`).

Kernels:

- `kernels/opt_qsa_mpp.metal`: few-row selected-block attention with the query
  tile staged once and K/V staged with 16-byte loads and one token lookup per
  row. Identical tile contents and math; 147 → 102 µs per layer at 2K context
  (`SPLASH_OPT_QSA_KERNEL=0`).
- `kernels/opt_gdn_lazy.metal`: lazy-rollback GDN verification with every
  token's convolution, norms and gates computed in parallel before a
  barrier-free recurrence; identical per-value arithmetic, 48 → 35 µs per layer
  (`SPLASH_OPT_GDN_KERNEL=0`).
- `kernels/opt_mppq.metal`: batched-verification projections (6-16 rows, 4-bit
  G64 and 8-bit, N > 1024) on the matrix units reading the packed codes
  directly, 3-4× faster than 5-row SIMD chunks (`SPLASH_OPT_MPPQ=0`).
- Few-output prefill projections choose the split by shape (16-way split-K for
  N ≤ 4 at K ≥ 8192; four 4-output groups with 2-way split-K for N > 8).

## Third pass (v16)

- **8-bit repack of 5/6-bit dense codes** (`OptQmv::repackCodes`, built at
  startup; `SPLASH_OPT_REPACK=0` off). GDN qkv/z/out and attention q/o
  projections with N > 1024 get a lossless uint8 copy of their codes (same
  scales and biases; +1.6 GB, reserved in the startup memory plan), so every
  large projection of a batched verification window runs on the matrix units.
  Two/four-lane aggregate decode +5%. `bench/repack_test.mm` checks it.
- **Short prefill** (21-128 rows; `SPLASH_OPT_PREFILL_QMV=0` off,
  `SPLASH_OPT_PREFILL_QMV_ROWS` caps it): projections use 16-row matrix-unit
  chunks plus a 5-row SIMD tail, and HC uses the chunked fused kernels, instead
  of the dense cache's two 16-row tiles plus a per-row tail kernel (HC down was
  310 + 164 µs per call at 35 rows). 256 rows measured slower than the cache,
  so the default cap is 128. Few-output tall projections split K 16 ways.
- **Fused shared-expert SwiGLU** (`opt_qmv_swiglu`, `SPLASH_OPT_FUSED_SWIGLU=0`
  off): gate, up and SiLU in one dispatch for <= 20 rows, bit-identical. It
  removes 96 dispatches per verify but saves only ~0.04 ms of GPU time (about
  0.4 µs per removed small dispatch), so the other MoE glue fusions
  (route+plan, poison+prepare) were not pursued.

Short-prompt prefill is now dominated by the INT8 expert reads: a 35-row
window touches ~250 experts per layer, 34 of its 84 ms GPU time.

Diagnostics: `SPLASH_OPT_TIMERS=1` prints host phase timers to stderr;
`SPLASH_OPT_PROFILE=<jsonl>` records per-dispatch GPU timestamps.

## Tried and not adopted

- **Original Q4 experts** (`SPLASH_OPT_MOE=1`): verify drops to ~20.9 ms per
  cycle and the 113 GB INT8 store is not resident, but prefill is ~11% slower:
  the staged Q4 expert GEMM runs 4.95 ms/layer against 3.8 for INT8. Warp-
  specialized staging, BK=128 chunks and direct 4-bit per-group matrix runs did
  not close the gap (`bench/moe_exp.metal`, `bench/gemm_bench.*`). Keeping both
  expert sets resident (`SPLASH_OPT_MOE=2`) reaches ~211 GB, raises system
  memory pressure and collapses prefill to ~1.2K tok/s; do not use it on 256 GB.
- **Matrix-unit GEMV for ≤5-row verification**: 4-bit direct per-group runs were
  1.3× faster in isolation, 8-bit slower; staged dequantization was 1.5-2×
  slower. At 5 rows the SIMD kernel is ALU-bound at ~10 ops per weight and does
  not depend on bit width (`bench/mppq.*`, `bench/mpps.*`).
- **Tall projections on the matrix units** (pre-dequantized FP16 or staged):
  slower than the SIMD tall kernel (~270 GB/s of activation reads).
- **W8A8 experts**: int8×int8 runs at ~220 TOPS (`bench/mpp_peak.*`), but the
  September 21 notes record that W8A8 expert prefill lowered draft acceptance.

## Layout

- `worker/`: source, Makefile, AIR inputs, new kernels
- `tools/`: quality + decode suite, benchmark comparison, dispatch profile
  analysis, CPU sample aggregation
- `bench/`: standalone kernel benchmarks and equivalence tests
