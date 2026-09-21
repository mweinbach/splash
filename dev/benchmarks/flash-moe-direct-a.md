The direct-device-activation route removes threadgroup A staging from Q4x8
expert prefill. It retains the existing M8/M16/M32/M64, N64, K64 matmul
descriptor, SIMD-group count, ascending K64 accumulation, original FP32
coefficient reconstruction, BF16 matrix operands, and BF16 SwiGLU/down/combine
boundaries. Gate/up still share two accumulators and stage their BF16 weights.

Each matrix row is independent. A final expert job can therefore read dummy
activation rows belonging to the next expert without changing valid results.
The epilog masks rows at the expert end before any output or route-map access.
Jobs begin at arbitrary packed offsets; rounding route capacity to M is not
sufficient. Both packed planes expose 63 extra rows globally, adding 322,560
input bytes and 80,640 activation bytes. The usual 16 KiB planner accounts for
409,600 additional bytes. Padding and inactive rows are initialized on every
GPU replay.

The pack producer performs the original sticky scan of every original input
row, including rows with excluded expert IDs. It also moves the old staged-A
nonfinite-to-zero substitution into the packed copy. Down prepares its operand
once, preserving every finite BF16 bit and replacing valid NaN/Inf with zero
and the same sticky flag. The production route prepares down in place; no
second large activation allocation is needed. Original inputs, IDs, weights,
scales, biases and canonical route order are unchanged by this route.

`SPLASH_FLASH_MOE_DIRECT_A=1` opts in and requires
`SPLASH_FLASH_MOE_Q4X8=1`. The flag accepts only 0/1 and freezes at first use.
Existing M64 selection remains restricted to the existing qualifying row
policy. Flag 0 preserves previous allocation extents and dispatches.
`flashMoEBlockedWorkspacePlannedBytes` independently rounds all ten blocked
allocations; Worker and BatchPrefill use that subtotal before construction.

The installed MLX primary implementation
`mlx/backend/metal/kernels/quantized_nax.h:qmm_t_nax_tgp_impl` similarly reads
activation tiles directly from device memory while staging quantized weights.
The shared helper here is
`runtime/metal/kernels/common/flash_moe_direct_a_common.h`; a separately
validated expert-rank predicate can reuse it for a saved-INT8 cache miss.

Root's private out-of-place prototype passed byte comparisons of every live
activation, canonical down, combined result, CPU bucket/job expectations,
independent sticky diagnostics, and output canaries. Reported medians measure
the complete GPU bucket-pack/gate/up/SwiGLU/down/scatter/combine chain:

| Source | Rows / tile M | Pattern | Q4x8 ms | Direct A ms |
|---|---:|---|---:|---:|
| Layer 0 | 2048 / 32 | spread | 15.012 | 10.875 |
| Layer 0 | 2048 / 32 | concentrated | 10.767 | 7.127 |
| Layer 0 | 8192 / 64 | spread | 39.546 | 25.522 |
| Layer 0 | 8192 / 64 | concentrated | 35.089 | 22.051 |
| Layer 24 | 512 / 16 | spread | 5.904 | 4.837 |
| Layer 47 | 2048 / 32 | spread | 14.917 | 10.580 |

Three negative fixtures passed: NaN/Inf inputs, excluded routes containing
nonfinite original input, and duplicate IDs. A 129-row M32 tail case passed
with Metal Shader Validation. These primitive results do not establish HTTP
throughput or production in-place qualification.

The frozen early worker is `build/flash-direct-a-runtime/splash-flash`, with
its adjacent `splash.metallib` and `private-freeze.json`. Its Forward object
matches the preceding saved-operand v4 worker. Compilation and the native
44-check CPU self-test passed; no GPU work was submitted by this subtask.
The separate CPU policy suite passed 49 checks each with the flag disabled and
enabled, plus strict flag/dependency rejection cases.

Root can qualify the separate in-place primitive without modifying the frozen
worker or original checkpoint:

```sh
FLASH_MOE_DIRECT_A_INPLACE=1 FLASH_MOE_DIRECT_A_ROWS=2048 \
FLASH_MOE_DIRECT_A_TILE=32 FLASH_MOE_DIRECT_A_PAIRS=4 \
build/flash-moe-direct-a/flash-moe-direct-a-oracle-inplace_v2 \
  build/flash-moe-direct-a/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/moe-direct-a-inplace-v2.json
```

`FLASH_MOE_DIRECT_A_NEGATIVES=1` selects the three negative cases.
`ROWS=8192 TILE=64` tests the larger physical batch, and `PREFIX` selects
another original layer. `PHASE=gate_up` or `down` isolates the modified phase.
The private oracle keeps source-owned parameter storage alive and checks
padded operand capacities, exact source-chain identity, geometry and aliases.
