Private final dense prefill screen. It does not edit production and makes no
performance claim until Root executes the oracle.

BF16 candidates are whole-K device operands at M256N32 SG4/8 and M256N64 SG8.
HALF candidates use numerically converted cached original BF16 weights and a
BF16-to-HALF GPU input converter included once per projection repetition. All
candidates accumulate to F32 and cast the final output to BF16. M128N128 SG8
was already measured by `actual-activations-screen-v1.json` and was slower or
equal, so it is omitted.

Build and freeze without GPU or model-payload reads:

```
make -f dev/benchmarks/dense_final_sep21/Makefile -j4 all cpu-self-test preview freeze
```

Root-only first screen, all seven captured projection roles with M256:

```
build/dense-final-sep21/oracle --run --include-m256 --samples 10 --repeat 4 --out build/dense-final-sep21/results-v1.json
```

Optional `--projection-filter in_proj_qkv` narrows captured roles. Optional
`--bf16-only` tests the independent BF16 M256 lever with selected/original BF16
controls; HALF candidates are omitted. Default tests only the HALF M128 lever.
`--include-m256` combines both. Optional
`--all-traversal` tests all five traversal modes for M256. Without it, M256 uses
the selected BF16 policy's traversal. `--preview` reads manifest metadata only,
and `--cpu-self-test` exhaustively validates finite BF16/IEEE HALF conversion
against native CPU `_Float16` and finite HALF round trips.

The oracle records the full input/weight HALF conversion census, overflow,
underflow, source and destination subnormals, GPU-versus-scalar conversion bits,
full BF16 output differences and hashes, and sampled original-operands FP64
errors. Conversion overflow/nonfinite values remove HALF variants and retain
BF16. Any remaining nonexact conversion/output is explicitly labeled a numerical
alternative; an exact certificate requires exact operands and full output.

Each variant has its own guarded BF16 output and sticky diagnostic buffer.
Qualification verifies repeated complete output before timing. Timings use
warmed balanced AB/BA pairs against selected BF16, with candidate order rotated
each round. No CPU operand/output/guard access occurs in the timing block.
Afterward every complete output is checked again, alongside guards, diagnostics,
GPU conversion and immutable operand SHA256. The report contains every paired
reference/candidate sample and GPU/wall medians.

Component evidence does not establish full-model generation or service quality.
