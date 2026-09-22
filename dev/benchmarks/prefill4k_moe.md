# Flash prefill MoE experiments toward 4K tokens/s

These experiments derive private source copies from the current qualified
Direct-A and selected INT8 expert store. They do not change the runtime,
profile, weights, existing prototypes, or current build products.

The historical frozen stage attribution assigns about 62% of early prefill
GPU time to MoE, with about 20% in dense projections. This is a reason to
experiment, not a current uninstrumented throughput claim.

```sh
make -f dev/benchmarks/prefill4k_moe.mk \
  BUILD=build/prefill4k-moe-expanded -j4 all
build/prefill4k-moe-expanded/oracle --cpu-self-test
```

Compilation and the CPU self-test submit no GPU work. The source generator
checks every substitution's expected occurrence count and fails on source
contract drift. Root runs all GPU work serially.

The current-profile control uses the saved top64 INT8 store and exact original
Q4 miss kernels, rather than comparing a new store against all-Q4 arithmetic.
The private comparison requires every live activation, canonical down, and
combined BF16 output bit to match this control. It also checks independent
stable buckets/jobs, unequal route ownership, sticky diagnostics, allocation
canaries, source and store immutability, graph admission rejections, and replay
after the original store/source owners have been destroyed.

| Candidate | Change | Arithmetic and allocation contract |
|---|---|---|
| `production`, tile64 | Explicit M64 at 2K instead of the current M32 row policy | Existing production M64 kernels and GPU jobs; changed descriptor/tile still needs exact output qualification |
| `static` | INT8 hit inputs expose static M rows rather than dynamic valid-row extents | Existing +63 globally initialized guard rows; valid-row epilogue/route masks remain unchanged |
| `paired` | Q4 misses stage two consecutive K64 weight tiles together | Original K64 descriptor and MAC order; barriers halved, gate/up threadgroup B storage rises to32 KiB |
| `expanded` | GPU-expand each active Q4 miss expert once, then read BF16 B directly in the original K64 loop | Original F32 dequantization and once-rounded BF16 coefficient bits; identical K64 MAC order; no per-job weight staging/barriers |
| `sg8`, tile32 | M32 uses eight SIMD groups/256 threads for hits and misses | Original coefficient/activation/K64 arithmetic; execution geometry still needs exact output qualification |

All candidates preserve the saved INT8 coefficients, FP32 row scales, original
activations, original checkpoint operands, BF16 SwiGLU stages, canonical route
order, and combine kernel. No cross-die placement or traffic counter is claimed.

The expanded candidate's three BF16 planes have5,033,164,800 logical bytes.
Its MemoryGovernor reservation includes three separately rounded64-byte
canaries, totaling5,033,214,976 bytes. Conversion skips cached and inactive
experts, but storage remains bounded for all512 original IDs. Every measured
candidate command includes all three expansion dispatches again; conversion
is never hidden outside the timing interval. Scratch construction and CPU
checks are outside the GPU interval. A production integration would admit one
reusable set of planes before allocation and reuse it across layers/commands.

Example serial GPU screen:

```sh
SPLASH_FLASH_MOE_DIRECT_A=1 \
FLASH_INT8_STORE_ROWS=2048 FLASH_INT8_STORE_TILE=32 \
FLASH_INT8_STORE_LAYERS=0 FLASH_INT8_STORE_PAIRS=4 \
PREFILL4K_MOE_VARIANT=expanded PREFILL4K_MOE_TILE=32 \
build/prefill4k-moe-expanded/oracle \
  build/prefill4k-moe-expanded/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  install/local-models/Flash-Next-int8-experts-top64-v1 \
  build/release/flash/prefill4k-moe-expanded-r2048.json
```

The default suite runs hit-concentrated, hit-spread, miss-only, mixed, and
spread-all routing. `FLASH_INT8_STORE_PATTERN=miss-only` isolates a changed Q4
miss route; `FLASH_INT8_STORE_PATTERN=hit-spread` isolates static hit tails.
Layer lists accept0,24,47. Matching captured BF16 hidden and I64 top10 IDs can
be supplied through `FLASH_INT8_STORE_INPUT` and
`FLASH_INT8_STORE_ROUTE_IDS`. Valid larger geometry accepts rows8192/tile64;
partial-row shader validation can use rows129/tile32 for same-tile variants.

`build/prefill4k-moe/` is the earlier frozen static/paired/M64 artifact and
remains separate from the expanded build. No primitive performance result or
service speedup has been measured by this subtask yet. The latest private
`build/prefill4k-moe-sg8/` artifact contains every candidate including SG8;
earlier build products remain frozen.
