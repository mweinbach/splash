# Original Q4 group-affine expert primitive

This private numerical alternative uses the unchanged original unsigned Q4
codes, BF16 source scales/biases, and BF16 activations. Sorted M32 jobs remain
on the GPU. For each ascending G64 group it computes a code dot, applies
`F32(scale) * code_dot + F32(bias) * F32(input_sum)`, and accumulates in F32.
Gate/up share the input sum and preserve compiled BF16 SwiGLU boundaries.
Down preserves canonical route scatter and BF16 combine.

Current blocked Q4 instead rounds every reconstructed coefficient to BF16
before matrix multiplication. These boundaries and affine association differ;
this candidate must not be described as model/bit parity.

Gate/up use legal original row-major `uint4b_format` device tensors, with
1,280-byte packed row stride. Original down's 320-byte packed row stride does
not satisfy the format tensor's 128-byte row alignment rule. Down therefore
decodes unsigned nibbles into a register-owned U8 cooperative operand with a
single-SIMD operation and 32-thread dispatch. No BF16 coefficient conversion,
threadgroup weight staging, or staging barrier exists in either producer.
Partial A rows use dynamic slices, and epilogs mask invalid rows.

```sh
make -f dev/benchmarks/prefill4k_q4coded/Makefile -j2 all
build/prefill4k-q4coded/oracle --cpu-self-test
.venv/bin/python dev/benchmarks/prefill4k_q4coded/run.py \
  --pattern miss-only --pairs 4 \
  --report build/release/flash/prefill4k-q4coded-miss-only-layer0.json --run
```

Root owns model/GPU timing. Compilation and the CPU self-test submit no GPU
work. Omit `--run` for bounded provenance only. The default source is the
current Top256 persisted control; only Q4 miss producers change. `--all-experts`
removes existing INT8 hits and applies original Q4 group arithmetic to every
valid route. `--pattern mixed/spread-all`, `--layer 24/47`, and finite `--edge`
inputs give focused alternatives. The ALL mode still uses Top256 metadata and
does not claim the separate Full512 source-omission runtime is compatible.

Both GPU sum dispatches are included in complete-chain GPU/wall timing.
Constructor checksums/mapping are performed once and excluded. Separate audit
kernels capture F32 gate/up/down linear results after timing; full BF16
activation/down/combine hashes must match the timed fast path. NaN sentinels
prove sampled linear cells were written, and exact sample cardinality is
required. Independent byte extraction and FP64 certificates cover up to
16 active changed experts per plane, first/last packed rows, and six columns.
This sampling scope is explicit; fused nonlinear/whole-model quality remains
separate. Job/bucket ownership, source/store checksums, sticky diagnostics,
canaries, and retained mapping replay are also checked.

[Precision audit](precision.md) derives an absolute envelope with coefficient
boundary error, F32 sums/dots/affine correction/accumulation, and FP64 diagnostic
rounding. Positive bound operations round upward. Two exact CPU cancellation
fixtures demonstrate expected differences: a BF16 coefficient boundary changes
0 to -2^-9, and affine epilog cancellation changes -2^-24 to 0 despite exact
coefficient agreement. CPU envelope/coefficient audits passed; actual MPP
behavior, shader tails, full source-model continuation and quality still need
Root's GPU qualification.

All sources/artifacts are new private files. Production/defaults, original
checkpoint, and qualified Top64/Top128 stores remain unchanged by this task.
