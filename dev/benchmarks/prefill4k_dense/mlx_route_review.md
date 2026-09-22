CPU-only route review. No GPU workload or full weight hash was run for this
review. The frozen installed reference uses MLX0.32.2 and original coefficients,
but its own report marks canonical Transformer quality unqualified: installed
Qwen3.5 GDN forward does not invoke the vendored Qwen4 normalization override.
The installed implementation's throughput is therefore a useful comparison,
with a remaining semantic qualification requirement.

At a fresh2048-token prefix, the installed QSA indexer produces no sparse mask
once its raw index state is stored. The attention module then invokes fast dense
causal SDPA. Both official v0.32.2 source and the pinned Ultra commit select the
wide-head split NAX kernel for D256: BQ64/BK32, WM4/WN2, eight SIMD groups.
Scores/probabilities and accumulators stay F32; each SIMD pair owns opposite
128-dimension halves, exchanges QK partials, and keeps Q in registers.

For2048 queries and24 heads that source layout launches768 CTAs per layer.
The current Top256 diagnostic trace launches7936 main attention CTAs per layer:
the first512 rows use per-query online groups, followed by temporal M32 groups,
each with partitions and128-row scheduling windows. The attention owner is
already coalescing these launches without changing their arithmetic.

Native temporal M32 uses strict K64 operations, four QK partial sums, F32
softmax/PV and four256-dimension-slice numerator accumulators. MLX's register
NAX, split QK reduction, exp2 scaling/softmax and relaxed precision differ; they
are not a proven byte-equivalent replacement for this route.

The new private `qsa_sg8_rows.metal` keeps M32 windows, masks, partition order,
strict K64 descriptors and F32 probability boundaries. It changes only tensor
SIMD ownership and staging:256 threads,256-word staging increments, one32-head
softmax/stat pass. Source-level CPU enumeration confirms every head/lane,
staging slot and statistic writer has exactly the original ownership set.
Each thread owns half as many numerator elements. GPU numerical and timing
qualification is still required because intrinsic reduction ownership changed.

The native expanded expert rows and installed MLX-LM sorting both produce20480
BF16 rows at2K/top10 (100MiB). MLX's source sorted RHS gather-qmm route selects
BM32/BN64/BK64/SG4 for an average40 rows/expert. It processes expert runs inside
640 global row tiles. Native uses stable per-expert jobs with conservative1151
job capacity and separate guarded hit/miss passes. Exact active job counts and
instruction costs require instrumentation; the different grid counts alone
do not establish less matrix work.

A bounded next MoE comparison is original Q4-miss projection math through pinned
register NAX, retaining existing stable buckets, packed DirectA, fused gate/up,
nonlinear BF16 boundaries and final scatter. This isolates matrix execution
from bucket changes and from the separate INT8 coefficient approximation.
Earlier stage preprocessing totals about19.4ms, while Q4 misses dominate the
projection budget. Source findings guide a private experiment; they do not
identify the precise kernel selected inside the installed binary.

Sources:

- Installed compatibility attention: `/Applications/oMLX.app/Contents/Resources/omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py`
- [Official SDPA dispatch](https://github.com/ml-explore/mlx/blob/2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99/mlx/backend/metal/scaled_dot_product_attention.cpp)
- [Official split-head NAX kernel](https://github.com/ml-explore/mlx/blob/2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h)
- [Official sorted quantized dispatch](https://github.com/ml-explore/mlx/blob/2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99/mlx/backend/metal/quantized.cpp)
