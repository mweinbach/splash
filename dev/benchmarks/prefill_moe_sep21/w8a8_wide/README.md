This isolated W8A8 geometry screen retains the original numerical
activation quantization and compares three variants:

| Variant | Producer | Native job rows | SIMD groups | Gate/down grid X |
| --- | --- | --- | --- | --- |
| 1 | Existing M32N64 baseline | 32 | 2 | 10 / 40 |
| 2 | Wide M32N128 | 32 | 4 | 5 / 20 |
| 3 | Wide M16N128 | 16 | 2 | 5 / 20 |

Variant 1 uses `prefill_moe_sep21_w8a8_` with suffix `m32_n64_sg2`.
Wide gate/up and down/scatter producers use
`prefill_moe_sep21_w8a8_wide_`, with suffixes `m32_n128_sg4` and
`m16_n128_sg2`; audits add `_audit`. Whole dynamic-K I8×I8 dots,
explicit F32 activation/weight scale operations and original BF16
projection, SwiGLU and scatter boundaries remain unchanged.

Original M32 baseline jobs are preserved. M16 uses independently checked
native bucket jobs with `ceil(routes/16)+511` capacity. At 2048 physical
rows, its capacity is 1791 and fits the existing 3071-job scratch
allocation. Widths 640 and 2560 divide exactly by 128. There are no extra
hit lists. Each wide candidate reports complete-chain timing comparisons
against both the original native I8-coefficient/BF16-activation float-dot
control with matching M16/M32 jobs and the best existing W8A8 M32N64 SG2
baseline.

The original gate_t256 and down_t128 GPU quantizers remain in the base
W8A8 AIR. The new wide AIR defines only producers and audits. Both
quantization commands are included in timed W8A8 chains. Every active
quantized I8 and F32 row scale receives an independent CPU check before
timing. Mandatory samples certify exact raw I32 dots, scaled F32 dots
and F64 quantization envelopes against original BF16 activations,
original I8 coefficients and original F32 coefficient row scales.
Audits and CPU certification remain outside complete-chain timing.

Synthetic hidden rows use true RMS normalization with final BF16
rounding. Actual input row RMS minimum and maximum are reported. Raw
BF16 hidden/I64 route fixtures are accepted together with `--input` and
`--ids`. Guards require relative L2 ≤0.05 and cosine ≥0.9985 for complete
activated, scattered-down and combined BF16 output; full BF16 equality
is also reported. `--strict` requires equality before timing. Rejected
candidates retain a failure report with no timing samples. Every
variant remains a numerical alternative, with semantic quality, model
quality and MTP acceptance unqualified.

The oracle maps one certified 2,524,446,720-byte Full512 I8 layer plus
its 16KiB rank buffer and reserves the original 1GiB native scratch
plan. It reuses the original W8A8 workspace: gate/down I8 activations
and F32 scales over `rows*10+63` rows, plus six scaled-F32/raw-I32
audit buffers over `rows*10` rows. All ten allocations have 64-byte
guards and 16KiB rounding. A separate governor admission precedes
allocation, protecting at least 16GiB or 10% of host RAM. The runner
derives exact workspace bytes for the chosen row extent. Physical rows
are bounded to 1024–2048; the full model and production 48-layer store
constructor are excluded.

Build and CPU self-tests read no model payloads and create no Metal
device:

```sh
make -f dev/benchmarks/prefill_moe_sep21/w8a8_wide/Makefile -j4 cpu-self-test
```

The build generates the original W8A8 shader into the new build's
`baseline.metal`, verifies it matches the frozen original source, and
compiles `baseline.air`. It separately generates and compiles wide
`candidate.air`, then links both with the original parent native AIR
inventory. Original W8A8 sources and root build artifacts stay intact.
`all` only builds; `cpu-self-test` also runs `--cpu-self-test`.

Prepare all three variants without executing GPU work:

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/w8a8_wide/run.py \
  --rows 2048 --pairs 4 --pattern spread-all \
  --report build/prefill-moe-sep21-w8a8-wide/spread-all-new.json
```

Add `--run` only in the exclusive root GPU window. `--variant 1` through
`--variant 3` selects one variant; omission screens all three. Patterns
are `hit-concentrated`, `hit-spread` and `spread-all`. The runner records
source, binary/metallib and manifest hashes plus command, controls and
comparison scope, and copies actual RMS statistics from GPU report
metadata. Preparation reads no input, IDs or model payloads.
