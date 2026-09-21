The private HC-down experiment retains original FP32 coefficient bits and all
BF16 projection/division/activation/injection stages at verifier rows4/8/16.
It never demotes weights to BF16 and never changes production or checkpoints.

The cached literal candidate assigns one SIMD to each output and reuses a
coefficient across all real rows. Every row still visits `k=lane+32*j`, uses
contract-off FP32 products/additions and the original hardware SIMD sum. Raw
dots round to BF16, divide by4 and round again, then use compiled-fast SiLU.
The four injection dots retain their original packed format, including mixed
Q4-down/Q5-injection sources, and use precise unary sigmoid followed by BF16
`2*sigmoid`. A cloned untimed literal witness exposes FP32 and BF16 raw dots.

Modes1/2 use M8/N32 or M8/N64 mixed BF16-input/FP32-weight whole-K MPP.
Modes3/4 split K into4/8 partitions with FP32 partials and an ascending FP32
fold before the first BF16 cast. These are explicit reduction alternatives.
The standalone canonical post witness checks candidate activation/injection
against its own rounded raw dots. Strict raw/activation relative L2 stays at
1e-4; changed BF16 cells receive independent double midpoint annotations, but
those annotations cannot override `accuracy_pass:false` or exit status1.

Root's completed screens rule out these current candidates:

| Screen | Cases | Accuracy | Candidate throughput / control |
|---|---:|---|---:|
| Cached literal, actual Q4/Q5/Q6/Q8/mixed/final sources, R4/8/16 | 18 | All raw FP32/BF16 dots, activation and injection byte-exact | 0.451–0.875× |
| MPP modes1–4, Q4/Q5/Q6/mixed sources, R16 | 16 | 12 pass;4 strict failures | 0.291–0.816× |

Q5 whole-K modes fail raw relative L2 at0.00017730005. Mixed layer19 split4/8
modes fail activation relative L2 at0.00012754805. These failures remain
failures despite diagnostic midpoint compatibility. Both screens passed input
and original/cache weight preservation, canaries, sticky diagnostics, and the
candidate-own canonical post checks. Inputs are explicitly declared synthetic
normalized BF16; real-input capture was unnecessary after every candidate lost.

The likely performance constraints follow from geometry rather than measured
bandwidth or register counters. The down matrix has K10240,N320. Its F32
weight stream is13,107,200 bytes versus1,638,400/2,048,000/2,457,600 packed
Q4/Q5/Q6 bytes before SF/bias metadata. Cached literal row reuse reduces R16's
1,296 control groups to81 and increases live per-lane accumulators to16.
Whole-K M8/N32 launches only10/20 groups at R4/8 versus R16; N64 launches5/10.
Split-K enlarges the grid but adds partial/fold work. The observed loss is
conclusive for these implementations; cache bandwidth, reduced wave occupancy
and register pressure remain explanations, not counter-proven bottlenecks.

Frozen artifacts are `build/flash-hc-down-f32-v1/flash-hc-down-f32-oracle` and
`flash-hc-down-f32.metallib`, linked with fresh v6 host objects and the current
200-byte `CommandTiming` ABI. Compilation and CPU self-test create no backend
or Metal device. The independent CPU contract audit passes19 tests. Original
FP32 cache coefficients are checked exhaustively at GPU-screen startup:
19,660,800 values in the literal screen and13,107,200 in the MPP screen.

```sh
FLASH_HC_DOWN_F32_MODES=0 FLASH_HC_DOWN_F32_ROWS=4,8,16 \
FLASH_HC_DOWN_F32_PAIRS=4 \
build/flash-hc-down-f32-v1/flash-hc-down-f32-oracle \
  build/flash-hc-down-f32-v1/flash-hc-down-f32.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/hc-down-f32-literal-screen.json
```

That command documents the completed Root-only screen; no additional GPU runs
are warranted without a new structural candidate. `FLASH_HC_DOWN_F32_PREFIXES`
selects comma-separated HC role prefixes, `MODES` selects0..4, `ROWS` selects
4/8/16, and `INPUT` accepts captured BF16[rows,10240]. No production flag or
default was introduced. The compressed literal down route remains preferred.
