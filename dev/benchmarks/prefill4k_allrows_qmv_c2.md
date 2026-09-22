# Optional C2 and the frozen one-layer primitive oracle

C1 remains the default. The optional follow-on module
`prefill4k_allrows_qmv_c2.py` composes after C1's Store transform. Stage C1's
`extra_files()` then C2's `extra_files()`; the latter augments the shared header
and adds a second shader. The existing C1 files, kernel definitions, defaults
and C1 numerical identity are unchanged.

Set `SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV=1` and strict
`SPLASH_FLASH_ALLROWS_I8_GATHERED_QMV_COLUMNS=1/2`. Unset columns means1.
Columns2 requires the base QMV flag1. Both flags freeze at Store construction;
`gatheredQMVEnabled()` rechecks them and `gatheredQMVColumns()` returns the frozen
selection. Call `gathered_i8_qmv::requestedColumns()` before backend creation
in a composed Worker to reject invalid configuration before a device probe.

C2 reuses the BF16 activation for two neighboring output columns. Each output
retains C1's lane32 K order, separate F32 accumulator/reduction, late F32 row
scale and BF16 sigmoid/product stages. It halves the QMV grids to
`{80,rows,10}` gate/up and `{320,rows,10}` down, with128 threads:4,000 CTAs/row.
It is separately identified by:

```text
private-gathered-signed-i8-bf16-input-f32-lane32-strided-dot-late-row-scale-bf16-dots-swiglu-sg4-c2-rows1to16-v1
```

The component at `build/prefill4k-allrows-qmv-c2-component` compiled C1/C2
shaders, the combined metallib and transformed Store. Source/CPU checks passed,
including64 paired output columns atK640/2560 with four late scales, cancellation,
signed zero and sanitization. Those are CPU byte expectations, not GPU evidence.

## Root-run one-layer oracle

The actual native oracle is built at
`build/prefill4k-allrows-qmv-one-layer/oracle`, with its combined
`splash.metallib`. Build/CPU self-test creates no backend or model mapping:

```sh
make -f dev/benchmarks/prefill4k_allrows_qmv_oracle.mk -j4 qmv-one-layer-cpu
```

Only Root runs GPU mode:

```sh
build/prefill4k-allrows-qmv-one-layer/oracle --gpu \
  build/prefill4k-allrows-qmv-one-layer/splash.metallib \
  /absolute/path/to/certified/Full512-store 0 2 /absolute/path/to/report.json \
  --pairs 3
```

Arguments are library, certified store, layer0..47, rows1..16 and report.
Optional `--hidden` is an exact-sized contiguous LE BF16 `[rows,2560]` file;
`--ids` is contiguous LE I64 `[rows,10]`. Each read is bounded to2MiB.
Optional `--pairs` accepts1..9. Unique valid IDs are mandatory for this finite
primitive gate; malformed fixtures need explicit separate stage diagnostic
qualification. Caller-provided capture paths are reported as unattested until
paired source provenance is supplied independently.

The default hidden fixture is synthetic normalized BF16; IDs alternate unique
concentrated/spread sets. Existing historical captures provide real IDs but
synthetic hidden values. No genuine paired normalized MoE decode fixture was
found. Consequently this oracle never declares actual normalized decode
performance, whole-model quality, rollback/state qualification or promotion.

The private Full512 loader validates bounded metadata and the certified store
identity, then **GPU mode only** maps and scans one layer. The exact base/rank
ledger is2,524,463,104bytes. Only one readonly2,524,446,720-byte payload is mapped;
the three variants share that base's codes/scales and the same rank buffer.
No48-layer Store or original FlashWeights is constructed. Root-time validation
checks whole/plane SHA256, exact offsets, canonical Full512 IDs, zero padding,
excluded-128 and positive finite F32 scales. Admission reserves the one-layer
ledger plus128MiB scratch, retains it through all allocations, and checks actual
allocation growth before committing. Every writable/input view is guarded on
both ends, inputs remain immutable, and final payload/rank hashes are checked.
Graph-bound views retain the readonly mapping.

The oracle first runs the existing MPP M16 gate/up+SwiGLU+down chain including
the real bucket prelude and down preparation. Its sanitized activation is
unpacked canonically. Untimed projection probes then capture F32 raw dot,
F32 scaled value and rounded BF16 projection for gate, up and down, using
identical projection inputs for MPP/C1/C2. The MPP tap uses the same whole-K
M16/N64/SG4 descriptor with dynamic valid bucket rows, while the timed control
executes the existing store kernels directly. Each producer's shipping fused
activation is checked exactly against qualified `addSiLUMultiply` on its tapped
BF16 gate/up projections. C1/C2 raw F32 dots, scaled F32 projections, BF16
projections, activated values and full-chain down outputs must match bytewise.

## Numerical gate frozen before any GPU run

`prefill4k_allrows_qmv_reference.hpp` provides the independent per-dot CPU
reference. It uses compensated F64 summation, with an explicit conservative
F64 uncertainty term. The reference is a compensated approximation, not a
claim of mathematical exactness for arbitrary inputs. Each BF16×signed-I8
product normally has at most15 significant bits and is exactly representable
in F32; product rounding, overflow and subnormal cases are nevertheless recorded.

For QMV, the absolute F32 dot bound is product-rounding error plus
`gamma(ceil(K/32)+31) * sumAbs(products)`, plus the F64 uncertainty. The31
allows an opaque SIMD reduction tree. The MPP control uses a **separate**
conservative `gamma(K)` report rather than inheriting QMV's lane proof.
The late-scale bound includes the dot bound multiplied by the actual positive
F32 row scale and the final F32 multiplication's rounding. The report records
per-dot absolute-bound failures, F32/F64 differences, every BF16 mismatch/ULP,
sign mismatch, negative-zero mismatch and exceptional classification.

A dot is strict-sensitive when its absolute reference dot or scaled value is
within32 times its absolute bound, or its scaled value is below F32 normal
range. Such dots require exact independently rounded BF16 bits and sign/zero
agreement. A broad norm bound cannot override this gate. The analytical CPU
fixture `2^100 + 1 - 2^100` is recovered as1 by the reference; a candidate0
passes its huge absolute bound but **fails** strict BF16 qualification. All
nonfinite-input/product-overflow/product-subnormal/scaled-overflow/scaled-
subnormal fixtures remain exceptional and never automatically qualify.
CPU tests also reject wrong signed zero despite zero absolute error.

The first oracle version requires the frozen regular finite primitive and
stage gates for **all three** variants. Any failure writes a detailed report
and exits2 without timing. Exit1 means setup/validation failure. Only after
all gates pass does it warm all three shipping chains and alternate matched
timed replays. MPP timings include bucket histogram/prefix/pack/jobs, fused
gate/up, down preparation and scatter; C1/C2 timings contain two gathered
dispatches. Probes, metadata/hash scans and CPU comparisons stay outside
timings. Source-backed synthetic results cannot substitute for a paired real
normalized activation test or final model/state/rollback/service qualification.

## Capture still required before promotion

A genuine paired capture belongs immediately after target
`addRoute(graph, router, ids, route, diag, rows,512,10)` in the private
`FlashForward.cpp`, where original `mixed` BF16 and IDs coexist. Capture via
`flash_forward_copy_words` graph copies into admitted, executor-owned buffers;
read and hash them only after command completion. Bind capture metadata to
layer/begin/rows, source layout, target derivative, routes/producer policy and
post-completion bytes. This initial oracle deliberately does not claim that
such a capture has occurred.

The subtree prepared and CPU-built these files without GPU execution, model
loads or model payload reads. Root should unload all test models/processes
after primitive and service experiments.

## Explicit diagnostic timing of a baseline strict failure

Root's first paired GPU run passed C1/C2's frozen projection bounds, strict
BF16/sign checks and bytewise stage comparisons, but the existing MPP control
had one sensitive down-projection BF16 failure. That original result remains
a failure. The v1 source, executable and witness are preserved under
`build/prefill4k-allrows-qmv-one-layer/frozen-v1`; the original Root report is
left untouched.

The optional bare flag `--allow-baseline-strict-failure` enables a **separate
diagnostic timing policy**, default off. It requires both candidates to pass
all frozen numerical checks, and the baseline to pass finite, sign, absolute
bound and nonexceptional checks. All producer stages, C1/C2 byte comparisons,
sticky diagnostics, canaries and immutability checks remain mandatory.
It does not change reference formulas, tolerances, sensitive classification,
failure records or the `regular_finite_primitive_pass` checker.

With an otherwise eligible baseline strict failure, the report still contains
`primitive_pass:false`, `baseline_strict_pass:false`, and the original failed
projection metrics. It separately records `timing_allowed:true`,
`diagnostic_timing_used:true`, and `diagnostic_timing_pass:true`, with a policy
description saying baseline timing does not assert strict F64/BF16 parity.
Exit0 then means the explicit diagnostic comparison succeeded; it does not
mean that the baseline passed primitive qualification. Without the option,
the same result still exits2 and leaves timing samples empty.

Use a new report path when rerunning with this flag. Append it to the original
GPU CLI without a value:

```sh
--allow-baseline-strict-failure
```

CPU self-test verifies default rejection, explicit eligible diagnostic replay,
rejection of candidate failures, baseline bound/sign/exceptional failures and
common stage/guard failures, and unchanged baseline failure counts after policy
evaluation. This follow-up was CPU-built without GPU execution or hashing.
