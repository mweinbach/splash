Private actual-input BF16 Steel screen. This adapter calls the unmodified official
MLX `gemm_loop` pinned to commit `2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99`.
Vendored source provenance, exact small-file hashes, and MIT license are recorded
in `source_manifest.json` and `steel_vendor/LICENSE`.

Compile/link only:

```sh
make -f dev/benchmarks/prefill4k_dense/steel.mk steel-all
```

The separate `build/prefill4k-steel/steel_adapter.air` can also be linked with the existing
production and private-screen AIR files into one oracle metallib.

| Kernel | Output tile | SIMD layout | Dispatch threads |
|---|---|---|---|
| `prefill4k_steel_bf16_m128_n64_bk512_wm4_wn2` | M128N64 | WM4/WN2, SG8 | 256,1,1 |
| `prefill4k_steel_bf16_m64_n128_bk512_wm2_wn4` | M64N128 | WM2/WN4, SG8 | 256,1,1 |

Both kernels use the existing Splash buffer ABI without function constants:

| Index | Binding |
|---:|---|
| 0 | Read-only native BF16 input, contiguous M×K |
| 1 | Read-only native BF16 weights, contiguous N×K |
| 2 | Native BF16 output, rows with stride `params.output_size` |
| 3 | Sticky atomic UInt32 diagnostics: mask0x2 geometry, mask0x4 nonfinite |
| 4 | Existing 32-byte `FlashDenseCacheParams` |

Use exactly 2048 rows. Set `input_size=K`, `output_size=N`, `output_begin` and
`output_count` to the requested column interval, and `tile_rows`/`tile_outputs`
to the kernel's M/N tile. All operands and output capacity must cover the original
matrix extents. K must be 32-aligned and output_count 64-aligned.

The grid uses `rowTiles=2048/BM`, `columnTiles=ceil(output_count/BN)`, and the same
Splash `FlashDenseTraversalGrid` mapping. `reserved=0/1/2/3/4` means column-fast,
row-fast, swizzle2, swizzle4, swizzle8. For the M64N128 kernel, do one ceil-rounded
column dispatch for N320; do not change its tail params to `tile_outputs=64`.
Official safe N loads/store clamp the final 64-column tail.

The adapter uses NT layout: A's and B's physical row strides are both K. Each
SIMD group computes SM32×SN32. The official loop processes BK512 groups with SK32
register loads and keeps one F32 NAX destination. Non-BK512 K tails select the
official safe K path, so K320 and K640 are supported without padding. The final
store converts to BF16 once.

The official NAX MMA descriptors permit relaxed precision. This is changed
reduction arithmetic relative to Splash's strict whole-K MPP path. Full output
differences, sampled scalar comparisons, guards, and immutability checks remain
necessary; compilation alone does not establish numerical equivalence or speed.
No physical die or memory affinity is requested.
