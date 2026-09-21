# Private trained MTP projection experiment

This experiment changes proposal projections only. It clones the existing
single and joint trained head into a private C++ namespace and links the
unchanged production target implementation. No production route, checkpoint,
launcher default, GDN state, or PLE state changes.

The current saved operand store has 509 BF16 and 508 F32 entries, with no
`mtp.*` entries. The trained head therefore creates its own 16 BF16 matrices.
Adding all trained F32 matrices would require 340 MiB; the default experiment
selects only `mtp.fc_hidden`, requiring 25 MiB plus diagnostics and padding.

Each head call already submits one graph, and joint calls already combine
independent lanes. Draft tokens depend on earlier proposal hidden/logits.
The useful matrix opportunity is the four-stream `fc_hidden` projection:

| Real pairs per lane | Physical projection rows, C1 | Physical rows, C4 |
|---:|---:|---:|
| 1 | 4 | 16 |
| 4 | 16 | 64 |
| 8 | 32 | 128 |

Two private modes are available with `FLASH_MTP_FLOAT_CANDIDATE=1`:

| `FLASH_MTP_FLOAT_KERNEL` | Proposal implementation |
|---|---|
| `f32` (default) | Original affine coefficients reconstructed once as exact F32; BF16-input/F32-coefficient MPP with BF16 output, in chunks of at most 16 physical rows |
| `qmv` | Existing contiguous Q4/G64 vector kernel with F32 activation sums; no extra coefficient cache |

The QMV mode is restricted to the original Q4/G64 `[2560,2560]` trained input
projections. The default role is `mtp.fc_hidden`. `FLASH_MTP_FLOAT_PREFIXES`
can select trained dense roles for the F32 mode; vocabulary, embedding tables,
expert banks, and target matrices are excluded. Physical rows must be at least
four and each lane's real fold at most eight. Long priming, one-vector body
projections, literal HC fusion, and the existing vocabulary route remain on
their prior policy.

The F32 coefficient operand does not round weights to BF16. MPP reduction and
the QMV group factorization are declared numerical alternatives. Proposal
changes can alter acceptance; the unchanged target must verify every output.
Primitive or head timing does not establish end-to-end HTTP improvement.

Build without GPU work:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  dev/benchmarks/build_flash_mtp_float_candidate.py \
  --production-build build/flash-default-v5
```

Use a complete fresh host build: the active `build/flash-next` executable and
metallib were copied separately from its older objects. The builder rejects
that object directory and records every linked production object SHA256,
the current timing header SHA256, and the metallib SHA256. Timings must be
finite and greater than one nanosecond before a speed comparison is reported.

Only the root agent runs GPU jobs. First compare proposal cost and state
continuation with the private single/joint oracle under the current local
profile environment:

```sh
FLASH_MTP_FLOAT_CANDIDATE=1 FLASH_MTP_FLOAT_KERNEL=f32 \
build/flash-mtp-float-candidate/flash-mtp-float-candidate-oracle \
  build/flash-mtp-float-candidate/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/mtp-private-f32-proposals.json
```

Repeat with `qmv` and a separate report. The oracle compares real folds at
C1/C4 and 1/4/8 pairs, full hidden/logit outputs, finite values, proposal
argmax agreement, and paired timing. Synthetic fixtures are explicitly
labelled; they do not substitute for actual model features or target parity.

Then test all 128 target outputs against the unchanged AR route:

```sh
FLASH_MTP_FLOAT_CANDIDATE=1 FLASH_MTP_FLOAT_KERNEL=f32 \
FLASH_MTP_PROBE_MAX_TOKENS=128 FLASH_MTP_PROBE_DEPTH=15 \
FLASH_MTP_PROBE_RESIDENT=0 FLASH_MTP_PROBE_MODE=paired \
build/flash-mtp-float-candidate/flash-mtp-float-probe \
  build/flash-mtp-float-candidate/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 PROMPT_TOKENS_JSON \
  build/release/flash/mtp-private-f32-target-parity.json
```

The probe records proposal head cost, accepted prefixes, target tokens, and
private proposal policy. Its timing scope includes standalone host sampling
and differs from the normal service; only a subsequent matched HTTP run can
justify a default change. No GPU screen or promotion is claimed by compilation.
