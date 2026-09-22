Isolated exact MoE pointwise candidates. This directory does not change the
runtime, the saved local profile, expert payloads, or phase dispatch policy.

Build and device-free checks:

```sh
make -f dev/benchmarks/moe_pointwise_sep21/Makefile -j 4 all cpu-self-test
```

Synthetic GPU qualification and rotated timing (run with the inference worker
quiescent so timing is not contaminated by model execution):

```sh
build/moe-pointwise-sep21/oracle \
  build/moe-pointwise-sep21/pointwise.metallib \
  /tmp/moe-pointwise-sep21.json --gpu
```

The combine baseline is the current production `flash_moe.metal`. Four
candidates share row-invariant ID/duplicate/weight validation and the precise
BF16 shared sigmoid: once per SIMD using lane 0, per-slot SIMD validation, or
once per 256-column CTA, or once per whole-row CTA. The whole-row variant uses
grid `{1, rows, 1}` and each lane iterates columns `tid + 256 * i`; it is an
additional bulk-prefill candidate, with much less parallelism for narrow row
counts. All retain the source BF16 arithmetic and partial slot order
`[(0,8),(1,9),2,3,4,5,6,7]`. Partial-column lanes execute every SIMD collective
and CTA barrier before returning. Column-local finite checks still run when
row metadata is malformed, preserving diagnostic bit combinations.

The frozen poison baseline body has the production validation and grid.
Candidates dispatch `{1, rows * selections, 1}` instead of
`{10, rows * selections, 1}`, with either 256 or 32 threads. They check the
inverse map once per route and write 2560 canonical `0x7fc0` BF16 values only
for an excluded route. The 256-thread variant shares the flag through one CTA
barrier; the 32-thread variant broadcasts within its single SIMD group.

Qualification compares every BF16 output bit, the complete sticky diagnostic
word, every immutable input byte, and both buffer canaries. Cases include rows
1/4/16/2048, 8192 rows at narrow width, widths
1/31/32/33/255/256/257/2559/2560, K1 through K10, excessive dispatch groups,
malformed IDs/weights/nonfinite data, finite overflow and cancellation,
invalid geometry, invalid combine thread counts, two sticky seeds, poison
valid/excluded/mixed maps, ignored poison metadata fields, and map mutation
replay. Timing uses identical input/output addresses for each variant, three
warmup cycles, then eight matched cycles in a rotated/reversed order. No model
files are read; production performance and whole-model qualification remain
separate work.

Private whole-worker composition over the sealed gathered-MPP plus exact bulk
SG8 snapshot:

```sh
.venv/bin/python dev/benchmarks/moe_pointwise_sep21/worker_overlay.py
make -f dev/benchmarks/moe_pointwise_sep21/worker.mk -j 6 all cpu-self-test
.venv/bin/python dev/benchmarks/moe_pointwise_sep21/worker_witness.py \
  --output /tmp/moe-pointwise-worker-cpu-witness.json
```

The private worker is `build/moe-pointwise-sep21-worker-v1/splash-flash`.
`SPLASH_FLASH_MOE_POINTWISE_SEP21` accepts exactly `0` or `1`; unset means `0`.
The worker freezes this setting before reading paths or metadata and before
creating its backend. Flag `0` uses the original dispatches. Flag `1` selects
SIMD-slot combine for physical rows1..16 and whole-row CTA combine for rows
256..8192, both only at W2560/E512/K10. Other combine geometries use the
original kernel. Route32 poison applies only at rows256..8192; smaller rows
retain the original poison dispatch. Three host poison sites are composed.
The existing buffer bindings, validation, workspace planning, and numerical
derivative remain unchanged. Kernel-route status adds an exact-pointwise
execution marker only when enabled. The private attribution classifier treats
all combine variants as the same MoE boundary.

Batch teacher v2 composition, retaining its exact M128 dense-prefill and reused
bulk-QSA arena:

```sh
.venv/bin/python dev/benchmarks/moe_pointwise_sep21/batch_worker_prepare.py
make -f dev/benchmarks/moe_pointwise_sep21/batch_worker.mk -j 8 all cpu-self-test
.venv/bin/python dev/benchmarks/moe_pointwise_sep21/batch_worker_witness.py \
  --output /tmp/batch-pointwise-fma-source-witness.json
```

This creates `build/prefill4k-batch-pointwise-fma-sep21-v1/splash-flash`.
Pointwise and `SPLASH_FLASH_GDN_PREFILL_FMA_SEP21` are independent flags; both
default to `0`. FMA `1` requires `SPLASH_FLASH_GDN_STAGED=1` and applies only to
staged prefill rows64..2048, lanes1..32. Enabled FMA binds the numerical
derivative to its qualified source/policy; disabled FMA retains the base
derivative. Original v2 teacher owner/batch API objects come from its
authenticated build audit. The original 68-AIR shader command is reconstructed
and must produce the identical qualified v2 control metallib bytes before the
two candidate AIRs are added. No SG2 tail producer is included. Combined model
fidelity and performance require a separate root GPU run.
