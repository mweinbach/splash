This private screen compares existing cached BF16 coefficient producers with
narrow whole-K MPP tiles. It does not change production or the v3 worker.

```sh
make -f dev/benchmarks/decode_bf16_narrow_sep21/Makefile -j4 cpu
build/sep21-bf16-narrow-role-screen-v2/oracle --help
```

Root runs one GPU job at a time after stopping owned model services:

```sh
SPLASH_FLASH_QMV_F32=1 SPLASH_FLASH_FLOAT_DENSE_SELECTIVE=1 \
SPLASH_FLASH_QSA_OUT_F32_N32=1 \
build/sep21-bf16-narrow-role-screen-v2/oracle --gpu \
  build/sep21-bf16-narrow-role-screen-v2/splash.metallib \
  build/dense-i8-decode-sep21/actual-role-fixtures.json \
  NEW_REPORT_JSON --case 0 --rows 4 --pairs 9
```

Case indices0..6 cover GDN QKV/Z/output, PLE value, QSA Q/K/index. Each fixture
uses the firstR rows of an attested2048-token actual prefill capture. These are
not live decode states. The initial fixture has no actual QSA output activation
capture; metadata for an additional source-attested role can be supplied later.
HC, trained MTP and vocabulary are excluded from this initial screen.

Rows1/4/8 use M8; row16 uses M16. Each invocation tests N32/N64 ×SG1/2/4,
the stock BF16 N128 SG4 control for largeN or N64 SG4 for smallN, active original
quantized projection, and an F32 cached control when its selected role policy
admits it. `FLOAT_DENSE_SELECTIVE=1` matches the current selective production
profile; value0 selects its corresponding general M8/M16N64 F32 policy and the
report identifies this choice. N32 QSA output control follows its existing flag.

GPU mode maps only the selected verified BF16/F32 sidecars and copies only its
original affine source tensor spans. It verifies every saved F32 coefficient
against independent source unpacking and every BF16 coefficient against exact
rounding of that F32 value. Inputs/weights retain immutable hashes and file
stat witnesses. Governor admission covers selected mappings, raw spans and
bounded scratch; no complete model is constructed.

Every candidate has an untimed same-descriptor rawF32 tap. Full output BF16
bytes must equal the actual stock BF16 producer; its probe must also reproduce
shipping output. Every BF16 dot is independently accumulated with compensated
F64, with conservative F32 absolute bounds and separate strict-sensitive BF16/
sign/exceptional counts. Exact stock parity, finite/sign/absolute bounds,
sticky diagnostics, guarded views, exact positive-zero padding and poisoned
scratch replay gate candidate timing. Existing baseline strict F64 failures
remain failed on that independent axis, and cannot become strict numerical or
model qualification merely because a new producer reproduces them.

Timings include original row padding plus projection. Probes, F64 work,
coefficient checks and hashes are excluded. At least150ms cumulative GPU work
on the same prepared commands precedes timing, with warm GPU/wall spans and
command counts recorded. No model-buffer reads or resets occur from warmup
through the end of all timed submissions; sticky/guard/output checks then run.
Qualified routes are compacted before rotating, and requested cycles round up
to complete compact rotations so every route occupies every timing position
equally. The report records requested and actual balanced cycles separately.
V1 artifact/source is preserved as earlier methodology evidence.

Pass/exit0 means at least one candidate passed exact old-BF16 compatibility and
timed guard/replay checks. It does not establish raw F32 coefficient parity,
greedy outputs, speculative acceptance, state continuity, service quality or
decode throughput. Root owns those later GPU/model checks. CPU compilation,
self-tests and independent source review access no GPU or model payloads.
