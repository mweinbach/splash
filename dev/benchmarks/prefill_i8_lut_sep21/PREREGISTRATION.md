# Exact I8 lookup compression component

This is an isolated one-layer component. It creates no full compressed store,
changes no model/production source/defaults, and preserves the original saved
Full512 I8 coefficients and positive F32 output-row scales. Only Root executes
packing, hashing of model operands, GPU qualification and GPU timing.

Each original Q4/G64 group uses 32 bytes of original Q4 IDs and 16 bytes of signed
I8 lookup entries. The lookup must reconstruct every saved I8 coefficient byte
exactly. Multiple Q4 IDs may legitimately share one I8 value; consistency of
the Q4-ID→I8 function and full byte reconstruction are required. Unused LUT
entries are zero. This reduces coefficient storage from 64 to 48 bytes/group.

The candidate uses unchanged device BF16 activations, M32N64 jobs, ascending
fixed K loops, F32 accumulation, original late F32 row scaling, BF16 projection
rounding and the original compiled BF16 SwiGLU boundaries. Original stable
bucket packing, canonical scatter order and route combination are retained.

| Variants | SG | Descriptor K | B type | Shared B bytes | Gate/up K blocks | Down K blocks |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1–2 | 2 | 128 | BF16/I8 | 16/8 KiB | 20 | 5 |
| 3–4 | 4 | 128 | BF16/I8 | 16/8 KiB | 20 | 5 |
| 5–6 | 2 | 256 | BF16/I8 | 32/16 KiB | 10 | 3 |
| 7–8 | 4 | 256 | BF16/I8 | 32/16 KiB | 10 | 3 |

The SG2/SG4 cooperative-register input proposal is not executable with this
SDK: MPP has a static assertion requiring a single SIMD group for cooperative
input tensors. The supported candidates cooperatively fill one threadgroup B
tile, reuse it sequentially for gate/up, and synchronize before and after each
MPP operation. I8→BF16 conversion is exact. K256 down has a final bounded K128
device-A extent and zero-filled remaining B entries.

Fidelity references are fixed before any GPU observation:

- Current best uncompressed SG2K128 is the raw F32/scaled F32/full BF16 reference
  for SG2K128 LUT candidates.
- Other SG/K candidates require raw F32 and scaled F32 equality to a matched
  uncompressed baseline using the same B type, descriptor, staging and K loop.
  They also require complete BF16 chain equality to current best SG2K128.
- Whole SG4 is a performance and honestly reported numerical contrast.
  Different descriptor results are not required to be universally raw-F32 equal.
- All variants report raw F32 dots, scaled F32 and actual GPU BF16 projection
  boundaries against all relevant controls, plus every activated, scattered
  and combined BF16 word. A fidelity failure receives no qualified timing.

Primitive fixture rows are 2048 with all 10 selections. Synthetic inputs receive
true per-row RMS normalization followed by BF16 rounding; spread and hot route
patterns use independent CPU bucket/job references and unequal route weights.
Raw caller inputs and IDs are accepted together. Synthetic fixtures are not
claimed to be actual model activations.

Timing begins only after source/lookup certification, strict fidelity, sticky
diagnostics and all canaries pass. Every compared arm accrues at least 150 ms
GPU warm-up. Balanced candidate/control order applies to complete chains and
separate gate/down scopes. There are no CPU operand touches, hashes, diagnostics
reads, canary scans or poison writes between warm and timed submissions.
GPU timing metadata and host vector bookkeeping are allowed. All outputs,
diagnostics and guards are checked again after the timing scope.

Malformed IDs/ranks/jobs, excluded -128 LUT values and source/store geometry
must reject or diagnose through bounded guarded cases. Loader and packer checks
run before GPU timings. Neither source reconstruction nor successful compile
establishes GPU parity, whole-model quality or speed.

The storage saving is 25%; a 100 ms or model prefill improvement is not promised.
Full sidecar and whole-model integration require a meaningful component win
and Root's qualification.
