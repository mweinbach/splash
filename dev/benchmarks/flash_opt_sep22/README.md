# Flash-Next decode/prefill optimization (September 22)

Self-contained build of the optimized Flash-Next worker used by the local
`m5-ultra-flash-next-v18` profile, the only pinned build. It was developed in
five passes (v14-v18; v17 and v18 are the megakernel passes). The earlier
builds and profiles were removed; the per-pass sections below record what each
pass changed and what it measured at the time. The base is the
qualified v13 worker (`build/R5-integer-currentQ4-fixed4-sep22-worker-v2`),
whose source previously existed only under `build/`. `worker/` holds that full
source, the prebuilt AIR inputs of its library (original link order in
`air-order.txt`), and the changes below. Every core object rebuilds from
source; the 80 AIR inputs are binary.

```sh
make -C dev/benchmarks/flash_opt_sep22/worker -j12
make -C dev/benchmarks/flash_opt_sep22/worker install DEST=$PWD/build/flash-opt-sep22-v18
```

The profile pins `build/flash-opt-sep22-v18/splash-flash` and its metallib by
SHA-256; rebuild and re-pin `LOCAL_PROFILE_V18` in `install/launcher.py` after
changing anything here. Its environment is the v13 flag envelope plus the flags
listed under "Fourth pass".

## Current results (v18, M5 Ultra, 256 GB)

Against the v13 baseline this worker started from and the last INT8-store
build (v16); details are in the pass sections:

| | v13 | v16 | v18 |
| --- | ---: | ---: | ---: |
| Ten fresh prompts, aggregate decode | 51.5 tok/s | 116.3 tok/s | 169.8 tok/s |
| MTP cycle wall time | 64 ms | 28.3 ms | 19.6 ms |
| Canonical 2,048 / 256 prefill | 4,014 tok/s | 4,208 tok/s | 4,165-4,258 tok/s |
| Canonical 2,048 / 256 decode | 62.0 tok/s | 84.6 tok/s | 180.5 tok/s |
| Two / four seeded lanes, aggregate decode | crash | 117.1 / 143.9 tok/s | 161.6 / 196.8 tok/s |
| 49 / 396-token prompt, prefill wall time | | 95-98 / 265-268 ms | 78 / 241 ms |
| Semantic suite | 20/22 | 20/22 | 20/22 |

## Results by pass (v13-v16)

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

## Fourth pass (v17): megakernel phases

v17 runs the original Q4 experts instead of the INT8 store and rebuilds the
few-row decode path and the prefill expert GEMMs around the M5 matrix units.
Its profile is v16's environment plus `SPLASH_OPT_MOE=1 SPLASH_OPT_MOE_TILE=64
SPLASH_MK_MOE=1 SPLASH_MK_DENSE=1 SPLASH_MK_CONCURRENT=1 SPLASH_MK_QMV=1
SPLASH_MK_PREFILL=1 SPLASH_MK_MOE_TILED=1`.

| Ten fresh prompts x 256 tokens | v16 | Q4 experts, Sep 23 start | v17 |
| --- | ---: | ---: | ---: |
| Aggregate decode, tok/s | 115.9 | 145.2 | 165.7 |
| Accepted tokens per cycle | 3.29 | 3.31 | 3.28 |
| Cycle wall time | 28.4 ms | 22.9 ms | 19.8 ms |
| Verify GPU per cycle | 22.9 ms | 17.5 ms | 14.9 ms |

| Canonical 2,048 / 256, MTP 4 | v16 | Q4 experts, Sep 23 start | v17 |
| --- | ---: | ---: | ---: |
| Prefill, tok/s | 4,224 | 3,935 | 4,297 |
| Decode, tok/s | 85.9 (101 cycles) | 159.7 (69 cycles) | 179.4 (69 cycles) |

Concurrent requests and short prompts are where v17 loses (same-day runs; the
batched rows give every lane the same 2,048-token coding prompt, and short
prompts are `tools/prefill_probe.py` median wall time):

| | v16 | v17 |
| --- | ---: | ---: |
| Two lanes, aggregate decode | 153.2 tok/s | 138.2 tok/s |
| Four lanes, aggregate decode | 220.3 tok/s | 159.1 tok/s |
| Two / four lanes, prefill | 2,914 / 3,010 tok/s | 3,094 / 3,223 tok/s |
| 49 / 111-token prompt | 98.0 / 163.5 ms | 97.4 / 163.2 ms |
| 206 / 396-token prompt | 206.5 / 268.3 ms | 213.8 / 283.9 ms |

Batched verification does not use the tile kernels (they cover one request's
2..8-row window), so its experts run on the generic blocked Q4 kernels and each
batched step is 4-8% slower than v16's INT8 store (41.6 vs 39.8 ms at two lanes,
60.2 vs 56.0 ms at four). The lanes also accept fewer drafts per step (2.39 vs
3.08 at four lanes). With one shared prompt all lanes follow a single greedy
path, so acceptance moves with the numerics; v18's tiled batched path accepts
at v16's rate again. v18 ports the tiled kernels to 16-row batched windows.

`SPLASH_OPT_MOE_TILE=64` used to crash the worker on any 256-1,023-row prefill
(a short prompt or the last chunk of a long one), because 64-row MoE jobs need
at least 1,024 rows. Those windows now use 32-row jobs.

The Sep 23 start column is the morning's fused MoE phases (`SPLASH_MK_MOE`,
`SPLASH_MK_DENSE`, `SPLASH_MK_CONCURRENT`) on the Q4 experts. Its decode numbers
come from that build; its prefill number is this build with
`SPLASH_MK_PREFILL2=0` (the morning's pipelined expert kernels) and 64-row
tiles. v17 follows the same 69-cycle path on the canonical prompt, so
159.7 -> 179.4 is like-for-like. The frozen semantic
suite passes 20/22 with the same two arithmetic failures as every build. The
worker is ready in 26-40 s instead of about 2 min 20 s because the INT8 store
is no longer mapped and hashed. All numbers were taken while macOS media
analysis kept the load average at 10-55, so treat small deltas as noise.

Hardware measurements that shaped the design (`dev/megakernel/probe`):

- DRAM reads peak at about 1.2 TB/s. A dependent dispatch costs 2-3 us of
  ramp, while a software grid barrier in a persistent kernel costs 6-23 us and
  some variants lost coherence. Each "megakernel" is therefore a short chain
  of fused, full-bandwidth dispatches per layer, not one persistent kernel.
- The system-level cache keeps about 96 MB warm and serves it at 1.45 TB/s.
- BF16 matrix products peak near 130 TFLOPS. A 16 x 32 x 64 product per
  simdgroup runs at full rate with any of bf16/uint8/int4 right operands, so
  5-row verification pays nothing extra for its 16-row matrix tile.

Decode (2..8-row verification windows):

- **Tiled dense GEMV** (`kernels/mk_dense.metal`, `kernels/mk_tiles.h`,
  `SPLASH_MK_QMV=1`). Startup writes a lossless copy of each wide projection
  (N >= 1024) as 32-column x 64-code tiles in the lane order of the uint8
  right-operand cooperative tensor, plus scales and biases transposed to
  [K/G][N]. Each lane fills the operand with two to four 16-byte loads and a
  few integer ops (5/6-bit codes are stored as nibble plus high-bit planes),
  prefetches the next tile, and applies the scale/bias epilogue with two
  8-byte loads. At 5 rows: 4-bit 10240x2560 22.3 -> 16.9 us (`opt_mppq`),
  5-bit 2560x6144 20.4 -> 13.3 us (`opt_qmv`), 6-bit 10240x2560 24.2 us against
  29.8 for the best SIMD variant, within about 10% of a pure read kernel with
  the same barrier. One-row drafts keep `opt_qmv`, which is
  already at the memory floor.
- **Grouped input projections** (`mk_mpt_multi`). GDN qkv/z/a/b and attention
  q/k/v/index each run as one dispatch over a segment table, which removes the
  latency-bound a/b and k/v/index dispatches (verify 16.61 -> 16.31 ms).
- **Tiled experts** (`mk_moe_gate_up_t`, `mk_moe_down_t`, `SPLASH_MK_MOE_TILED=1`).
  Startup converts every layer's Q4 experts into the same tile format with
  per-expert transposed parameters (+70 GiB). At 35 unique experts per 5-row
  window, gate/up goes from 89 to 76 us and down from 40 to 33 us; verify
  drops from 16.3 to 14.9 ms per cycle. The tiles and the original experts are
  both in the saved residency set. Without the tiles there, every verify
  command waited about 1.1 ms for residency. Without the originals there, the
  tiles evicted the file-backed originals and 2K prefill fell to 700-1,900 tok/s.
- **No INT8 store with Q4 experts.** With `SPLASH_OPT_MOE=1` the store is never
  referenced, so `FlashForward` no longer maps or hashes it and the startup plan
  no longer reserves its 113 GiB. The batched trunks stop requiring it.

Prefill:

- **Row-block expert GEMMs** (`kernels/mk_pf2.metal`, `SPLASH_MK_PREFILL=1` with
  `SPLASH_OPT_MOE_TILE=64`). Each of the eight simdgroups owns a 16-row block
  and half the tile's features, so row blocks past a bucket's end issue no
  matrix work (a 2,048-token window averages 40 routes per expert). Weights
  and activations are dequantized/staged into double-buffered threadgroup
  memory one 32-code chunk ahead. The outputs are bit-identical to
  `mk_moe_prefill_*_pipe`: gate/up is 11% faster on uniform routing and 19%
  on Zipf-0.8 routing, down 17% and 24% (`dev/megakernel/moe/pf2_bench`).
  `SPLASH_MK_PREFILL2=0` restores the pipelined kernels.

Tried in this pass and not adopted:

- Tiled hyper-connections (`kernels/mk_hc.metal`, `SPLASH_MK_HC=1`, off): split-K
  tiled down plus stream-interleaved tiled up with the mix in registers. Output
  matches bit for bit, but down takes 16.7 us against 9.6 for `opt_hc_down` and
  up 16.5 against 11.0. These 2 MB shapes are latency-bound, and the SIMD
  kernels issue their loads sooner.
- Deeper tile prefetch (2-5 tiles in flight per simdgroup): slower everywhere,
  most likely register pressure.
- Expert loads from the MLX layout without a tile copy, by per-lane gathers or
  by row loads plus shuffles: 87-99 us for gate/up against 74-76 us tiled.
- A single-buffered prefill stage to fit two threadgroups per core: no gain.
- SIMD few-row GEMV on a planar 5/6-bit layout: ALU-bound near 530 GB/s at 5 rows.
- MTP depth 5 with the faster verify: 151 tok/s against 154-157 at depth 4.
- GPU-resident PLE tables: the loader ties PLE SSD streaming to its target
  omission rules, so this needs loader work rather than a flag.

Where the remaining v17 cycle goes (19.8 ms): verify GPU 14.9 ms (routed
experts about 5.4 ms, hyper-connections about 2 ms, GDN core about 1.5 ms,
the verification head 0.6 ms), draft GPU 2.3 ms, and about 2.6 ms of host
time: 0.6 ms preparing and encoding roughly 900 dispatches, 0.4 ms staging
PLE rows, 0.9 ms of gaps inside the chained draft commands and 0.3 ms of
prefix restore. For prefill, the GDN recurrence (65 ms of about 480 ms at 2K)
still runs as a per-token scan; a chunked form is the next large item.
Converting the prefill and batched expert paths to tiles would let v17 drop
the 68 GB original expert mapping.

## Fifth pass (v18): tiled batched verification and short prefill

v18 keeps v17's environment and moves the two paths
where v17 lost to v16 onto the expert and dense tiles:

- **16-row tiled kernels for batched verification.** The worker batches up to
  four lanes of four rows (draft depth 3), so a joint verification window is at
  most 16 rows. `mk_moe_gate_up_t16`, `mk_moe_down_t16` and `mk_mpt16_*` give
  each simdgroup a 16 x 32 destination built from two 8-row blocks of the
  same weight tile, so every row shares one weight read. `FlashBatchVerify`
  sends its experts through the megakernel MoE phases (routing plan widened to
  16 rows) and `addQmv` picks the 16-row dense tiles for 9-16 rows.
- **Tiled buckets for short prefill.** Prefill chunks under 1,024 rows ran on
  the generic blocked Q4 M16/M32 kernels at about 380 GB/s effective. They now
  run `mk_moe_bucket_gate_up` / `mk_moe_bucket_down` on the expert tiles, one
  16-row job per threadgroup, driven by the blocked path's bucket and job
  tables. Chunks of 1,024 rows or more keep the v17 row-block GEMMs.
- The megakernel MoE scratch is now part of the single-request and joint
  verifier workspace plans; the joint verifier refused to start without it.

Seeded 2,048-token lanes (`--lane-variation seeded`, two trials), aggregate
decode tok/s:

| | v16 | v17 | v18 |
| --- | ---: | ---: | ---: |
| Coding, two lanes | 117.3 | 126.6 | 161.6 |
| Coding, four lanes | 144.2 | 149.5 | 196.8 |
| Prose, two lanes | 99.4 | 115.3 | 148.9 |
| Prose, four lanes | 137.9 | 142.1 | 186.5 |
| Coding prefill, two / four lanes | 2,889 / 2,986 | 3,066 / 3,194 | 3,058 / 3,193 |

The same prompt on every lane (three trials):

| | v16 | v17 | v18 |
| --- | ---: | ---: | ---: |
| Two lanes, aggregate decode | 153.2 | 138.2 | 197.5 |
| Four lanes, aggregate decode | 220.3 | 159.1 | 267.7 |
| Batched step, two / four lanes | 39.8 / 56.0 ms | 41.6 / 60.2 ms | 31.2 / 45.5 ms |

Fresh prompts, one output token, median native prefill wall time:

| Prompt tokens | v16 | v17 | v18 |
| ---: | ---: | ---: | ---: |
| 49 | 98.0 ms | 97.4 ms | 78.0 ms |
| 111 | 163.5 ms | 163.2 ms | 134.6 ms |
| 206 | 206.5 ms | 213.8 ms | 178.9 ms |
| 396 | 268.3 ms | 283.9 ms | 241.0 ms |
| 760 | 453.6 ms | | 427.0 ms |
| 1,495 | 661.7 ms | | 648.3 ms |
| 2,956 | 1,118 ms | | 1,097 ms |
| 5,905 | 2,169 ms | | 2,163 ms |

Single stream is v17's path: 169.8 tok/s on the ten-prompt decode suite
(19.6 ms cycle, 3.32 tokens per cycle), 180.5 tok/s canonical decode, and
4,165-4,258 tok/s canonical prefill over three runs (v16 4,208-4,224). The
semantic suite passes 20/22 with the usual two arithmetic failures. Through
`./splash serve` the worker is ready in about 40 s. Memory is v17's: the Q4
experts and their tile copy stay resident, and 1,024-row-plus prefill still
reads the original experts, so the 68 GB mapping remains.

## Tried and not adopted

- **Original Q4 experts** (`SPLASH_OPT_MOE=1`; adopted by v17 with the row-block
  prefill kernels): verify drops to ~20.9 ms per
  cycle and the 113 GB INT8 store is not resident, but prefill is ~11% slower:
  the staged Q4 expert GEMM runs 4.95 ms/layer against 3.8 for INT8. Warp-
  specialized staging, BK=128 chunks and direct 4-bit per-group matrix runs did
  not close the gap (`bench/moe_exp.metal`, `bench/gemm_bench.*`). Keeping both
  expert sets resident (`SPLASH_OPT_MOE=2`) reaches ~211 GB, raises system
  memory pressure and collapses prefill to ~1.2K tok/s; do not use it on 256 GB.
- **Matrix-unit GEMV for ≤5-row verification** (solved in v17 by the tile
  layout, transposed scales and cooperative operands): 4-bit direct per-group runs were
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
- `../../megakernel/`: v17 hardware probes (`probe/`), few-row GEMV and HC
  benchmarks (`gemv/build.sh`, `gemv/build_hc.sh`) and decode/prefill expert
  benchmarks (`moe/build2.sh`, `moe/build_pf2.sh`)
