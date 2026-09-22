# Prepared chunked GDN source experiment

`prepared.metal` derives from `candidate.metal` and shares its F32 carried-state,
relative-alpha-product, triangular-solve and audit formulas. The primary mathematical
reference is [FLA gated delta rule naive implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py).
This source has been compiled, but has not been numerically qualified or benchmarked.

The preparation stage computes raw BF16-source `K K^T` and `Q K^T` into F32
coefficients once for every key head and chunk. The state stage loads those matrices,
applies its value-head beta and relative-alpha masks in threadgroup memory, then
projects, solves, emits output and updates F32 state as in the original experiment.
Preparation reads Q and K directly from the projected device rows, whose stride is
10240 BF16 elements. Q begins at row offset 0 and K at 2048. Both input tensors retain
dynamic row extents and use `slice(0, 0)` so partial chunks never declare padded
source rows valid.

## Dispatch and storage

Run the preparation stage before the corresponding state stage, with an ordering
barrier appropriate to the command encoder. Both stages use 128 threads per group.

| Stage | Entry | Threadgroup grid |
| --- | --- | --- |
| Preparation, Time 16 | `private_gdn_chunk_prepare_t16` | `{16, ceil(rows/16), lanes}` |
| Preparation, Time 32 | `private_gdn_chunk_prepare_t32` | `{16, ceil(rows/32), lanes}` |
| State, V16/T16 | `private_gdn_chunk_prepared_v16_t16` | `{48, 8, lanes}` |
| State, V16/T32 | `private_gdn_chunk_prepared_v16_t32` | `{48, 8, lanes}` |
| State, V32/T16 | `private_gdn_chunk_prepared_v32_t16` | `{48, 4, lanes}` |
| Register state, V32/T16 | `private_gdn_chunk_prepared_register_v32_t16` | `{48, 4, lanes}` |
| Register state, V32/T32 | `private_gdn_chunk_prepared_register_v32_t32` | `{48, 4, lanes}` |

The coefficient layout is F32 `[lane, chunk, keyHead, matrix, token, previous]`,
where matrix 0 is raw Gram and matrix 1 is raw QK, and both final dimensions have
size Time. Allocate
`lanes * ceil(rows/Time) * 16 * 2 * Time * Time * sizeof(float)` bytes. For 2048
rows this is 4 MiB per lane at Time 16 and 8 MiB per lane at Time 32. Coefficient
slots outside a partial chunk are initialized to zero by preparation.

| Preparation buffer | Binding |
| --- | --- |
| BF16 projected mixed rows | 0 |
| F32 coefficient scratch, writable | 1 |
| Atomic diagnostics | 2 |
| `FlashGDNParams` | 3 |

State buffers 0 through 6 retain the candidate ABI: mixed rows, decay, beta,
recurrent state, BF16 output, diagnostics and parameters. F32 coefficients are read
at buffer 10. Standard state tiles use SG4 matrix products; register state tiles
retain F32 carried state and use SG1 projections and updates across four value
partitions, matching `candidate.metal`.

Audit entries are `private_gdn_chunk_prepared_audit_v16_t16`,
`private_gdn_chunk_prepared_audit_v16_t32`,
`private_gdn_chunk_prepared_audit_v32_t16`,
`private_gdn_chunk_prepared_register_audit_v32_t16` and
`private_gdn_chunk_prepared_register_audit_v32_t32`. They restrict the head grid to
head 0, and use the candidate compact head-0 buffers: history at 7, solved delta at
8, and F32 output at 9. Preparation can still use the complete key-head grid.

## Source-only verification

Compiled successfully with:

```sh
xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror \
  -Iruntime -mmacosx-version-min=27.0 \
  -c dev/benchmarks/gdn_chunk_sep21/prepared.metal \
  -o build/gdn-chunk-sep21/prepared.air
```

No driver, production runtime, model payload or GPU execution changes are included.
Numerical qualification must compare audit outputs and F32 state, including partial
chunks, nonzero carried state, gated edge cases and continued sequences before any
performance result is treated as valid.
