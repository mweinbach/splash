# September 21 private expert prefill experiments

The current Full512 + bulk SG8 2K prefill attribution spends about 266 ms in
MoE, including 149 ms gate/up and 63 ms down. This experiment targets the
matmul path while preserving persisted signed I8 bytes, F32 row scales,
BF16 input/projection/SwiGLU boundaries, native job ownership and scatter.
Root owns all model/GPU executions. Every build and CPU check below submits
no GPU work and reads no model payload.

## Native M64 bucket jobs at 2K

The accepted native job policy selects M32 at 2K and M64 at 4K+. Full512
2K workloads route about 40 rows per expert, so M32 commonly reads each
expert's weights twice. The separate `native_m64_worker.py` snapshot lowers
the native M64 threshold to 2K, retains the original single bucket
prefix/emission pair, and uses the original M64 SG8 INT8 hit shaders.
This differs from the older hit-only M64 experiment, which kept M32 source
jobs and added two compact-hit-list dispatches.

`build/prefill-moe-sep21-native-m64-worker-v2` preserves the original-target-
omitted Full512 loader and the combined gathered decode + exact bulk QSA
route. Only frozen MoE row policy and its explicit rows2048plus marker change.
All 119 handed-off source files are checked and copied. Existing 63-row
Direct-A padding and blocked workspace admission stay unchanged.

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/native_m64_worker.py
make -f dev/benchmarks/prefill_moe_sep21/native_m64_worker.mk \
  BUILD=build/prefill-moe-sep21-native-m64-worker-v2 -j2 all cpu-self-test
```

Use the same normal attribution/HTTP flags as the combined base build.
The smaller [native M32/M64 one-layer oracle](native_m64/README.md) maps one
2,524,446,720-byte Full512 layer and compares complete chains in one process.
Root must measure full output equality and actual model throughput before
promoting the M64 threshold change.

## Low SIMD counts and register operands

MPP accepts register input operands only for `execution_simdgroup`, and
restricts register/register N and K to 16 or 32. The private M32N64 expert
kernels therefore compare these variants:

| Variant | Operation | SIMD groups | Details |
| --- | --- | --- | --- |
| 1 | Whole K from device memory | 1 | Dynamic row bounds |
| 2 | Whole K from device memory | 2 | Dynamic row bounds |
| 3 | Whole K from device memory | 1 | Static full-row tensor extents |
| 4 | Whole K from device memory | 2 | Static full-row tensor extents |
| 5 | Fixed K64 from device memory | 1 | F32 accumulate, static full rows |
| 6 | Fixed K128 from device memory | 1 | F32 accumulate, static full rows |
| 7 | Fixed K128 from device memory | 2 | F32 accumulate, static full rows |
| 8 | Register BF16 operands | 1 | M32N32K32; K64 loop grouping |
| 9 | Register BF16 operands | 1 | M32N32K32; K128 loop grouping |
| 10 | Register BF16 operands | 1 | M16N32K32 row parts; K64 grouping |
| 11 | Register BF16 operands | 1 | M16N32K32 row parts; K128 grouping |

Register kernels convert each unchanged signed I8 code to BF16 exactly in
registers. K64/K128 suffixes describe loop unrolling of 2/4 K32 substeps,
not different accelerator reduction sizes. N64 output uses sequential N32
halves; only gate and up accumulators are live together, and A is reused
for the paired gate/up suboperations. M16parts additionally sequences row
partitions to reduce register pressure. Fixed K and changed scopes may alter
reduction order, so numerical equivalence remains a measurement.

The original-target-omitted worker supports all eleven routes through
`SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT=1..11` for native M32 jobs. M16/M64
jobs retain their original kernels; gathered decode remains unchanged.
Unset or `0` preserves the original route and numerical derivative identity.
Active selection is bound into `identity.kernel_routes`,
`identity.target_numerical_derivative_sha256`, `target_prefill_moe_variant`
and `target_prefill_moe_policy`. Status identifies exact gate/down kernel
names and actual register shapes. Construction rejects malformed flags
before creating the Metal backend.

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/worker_overlay.py
make -f dev/benchmarks/prefill_moe_sep21/worker.mk -j2 all cpu-self-test
```

The frozen output is `build/prefill-moe-sep21-worker-v2`. Parent uses normal
attribution/HTTP controls plus the selected variant. Every variant uses
original M32 jobs, parameters, route coverage, allocation and scatter.
There are no new prefix/emission dispatches or coefficient caches.
Full model output/quality and throughput qualification remain pending.

An older complete-source primitive remains available for partial stores:

```sh
make -f dev/benchmarks/prefill_moe_sep21/Makefile -j2 all gathered cpu-self-test
.venv/bin/python dev/benchmarks/prefill_moe_sep21/run.py \
  --variant 1 --pattern hit-spread --strict \
  --report build/release/flash/sep21-prefill-moe-sg1-private-root.json
```

Omit `--run` to write provenance only. The original-source primitive explicitly
rejects Full512 to avoid duplicating both the whole source model and whole
Full512 sidecar. The bounded one-layer and omitted-target worker paths above
provide Full512 qualification. Synthetic inputs are normalized BF16 fixtures;
raw input/route-ID pairs and actual-code cancellation fixtures are supported.
Measured changed BF16 results are labeled numerical alternatives and require
independent model quality qualification. Complete-chain timings exclude
constructor/checksum scans, CPU comparisons and final hashes.

## Gathered decode scope controls

`gathered_sg1.metal` and `gathered_sg2.metal` use the same gathered M16N64
validRows1 ABI and original signed-I8/scale/BF16 boundaries. They preserve
rank/duplicate validation and nonfinite fallback, while reducing the requested
threads and thread-strided scan/output loops from 128 to 32/64. The independent
gathered-stage oracle integrates these sources for exact raw/scaled/BF16 checks.
They are not selected by the prefill worker flag.
