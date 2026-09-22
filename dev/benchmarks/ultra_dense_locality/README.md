This isolated M5 Ultra experiment changes BF16 dense MPP matrix geometry and
threadgroup traversal. It does not request die or memory placement. Production
kernels and the native Splash MetalBackend are compiled without edits alongside
the candidates.

Build and CPU mapping/BF16 checks (these submit no GPU work):

```sh
make -f dev/benchmarks/ultra_dense_locality/Makefile -j 6 all cpu-self-test
```

Only run with an otherwise idle GPU; root coordinates the GPU slot:

```sh
build/ultra-dense-locality/oracle \
  --rows 4,8,16,512,2048 \
  --shapes 6144x2560,2560x6144,2560x2560,2560x320,320x10240 \
  --samples 7 --repeat 4 \
  --out build/ultra-dense-locality/results-v1.json
```

The geometry notation is KxN. The prefill baseline follows FlashForward's tile
policy: M32N128 for rows>=128 and N>=1024, otherwise M16N64. At rows4/8 the
baseline includes production input padding and the M8N128 or M8N64 tile.
Alternative geometries include M64N128 and M128N64 with 8 SIMD groups; traversal
variants include column-fast, row-fast, and MLX-style row blocks of2/4/8. Tiny
rows also compare the production SIMD vector kernel and M8/M16 geometries.

Each variant must preserve guards and diagnostics, and changing traversal within
the same geometry must produce exactly equal BF16 words. A full output comparison
against the production baseline and a sampled double-precision CPU dot-product
oracle are recorded. Changed geometry may change rounding: its differences are
reported and are not evidence of greedy-output correctness. Timings contain7
rotated-order samples, with4 repetitions in each native command, and report both
GPU and wall time per projection. Tiny-row timings include production padding.
The shader boundary test exercises a partial output interval, sticky diagnostics,
and malformed parameters. No production defaults are modified by this harness.

The operands are deterministic synthetic BF16 values at the runtime shapes.
Full-model HTTP output and latency qualification must follow any promotion.
