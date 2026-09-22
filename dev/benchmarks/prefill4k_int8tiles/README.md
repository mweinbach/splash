# Cached INT8 hit-only tile and precision experiments

This isolated oracle compares the same persisted Top256 signed INT8 bytes,
F32 row scales, BF16 activations, and BF16 output boundaries. Q4 misses retain
their original job lists, row tiles, kernels, and coefficient arithmetic.
New GPU prefix/emission dispatches create compact cached-hit-only M32/M64/M128
jobs; their cost is included in every complete-chain timed command.

M64 uses eight SIMD groups, M128 has eight- and sixteen-group options. A
precision-only M32/four-group option isolates MPP `relaxed_precision=true`.
The installed SDK names the final bool before matmul mode as that precision
control. The relaxed source changes only this bool and helper/kernel names.
BF16 inputs and integer weights are exactly representable, but the precision
policy can still change accelerator accumulation/conversion, so equivalence
must be measured.

```sh
make -f dev/benchmarks/prefill4k_int8tiles/Makefile -j2 all
build/prefill4k-int8tiles/oracle --cpu-self-test
.venv/bin/python dev/benchmarks/prefill4k_int8tiles/run.py \
  --tile 128 --sg 8 --pattern hit-spread \
  --report build/release/flash/prefill4k-int8hit-m128-sg8.json --run
```

Root owns the model/GPU slot; compilation and the CPU self-test submit no GPU
work. Omit `--run` to write provenance only. `--tile 64 --sg 8`,
`--tile 128 --sg 16`, and `--tile 32 --sg 4 --relaxed` provide focused variants.
`--pattern mixed` exercises the unchanged miss path alongside changed hits.
`--edge` constructs exactly cancelling products from two actual persisted
gate codes and adds a tiny BF16 residual, alternating signs across rows.

The oracle admits three separately rounded guarded job buffers through the
real MemoryGovernor. It independently checks all new GPU offsets, active and
inactive job records, route coverage, and canaries against a CPU bucket walk.
All activation, canonical down/scatter, and combined BF16 bytes must match
the current INT8 control before and after timing. Source/store checksums,
negative graph-construction rejections, sticky diagnostics, and replay after
source/store owners are destroyed are retained. Dynamic valid-row tensors
avoid new +127-row activation padding for M128. Pipeline thread/static-memory
limits are checked before Root runs the GPU.

No whole-model speedup follows from a primitive win. Report scope is complete
expert-chain timing, with constructor/checksum scans excluded and explicit
tile/precision/cancellation fields. Coefficients are not requantized. Any
changed BF16 result fails the current strict gate and remains a numerical
experiment requiring independent correctness and quality qualification.
