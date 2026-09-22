This private oracle screens eleven M32N64 gate/up and down/scatter candidates
against the original native M32N64 control over one shared, certified Full512
layer. Control and candidates use identical original M32 buckets, jobs,
parameters, route fixtures, preparation and combine dispatches. Only the two
producer pipelines and their SIMD-group counts change. It adds no hit-only job
lists, prefix dispatches or gathered shaders.

The inventory is memory whole/static at SG1/SG2, memory fixed K64 at SG1 and
K128 at SG1/SG2, register K64/K128 at SG1, and register K64/K128 at SG1 with
M16 row parts. Register variants convert original signed I8 codes exactly to
BF16 in registers, retain the F32 post-dot row scales and BF16 SwiGLU
boundaries, and use sequential N32 halves for each logical N64 job.

The native adapter retains the frozen store oracle's BF16 hidden fixture,
original I64 route patterns, unequal route weights, independent bucket
reference and complete-output comparisons. It uses the existing one-layer
loader rather than the production 48-layer store constructor or full Q4
model. Full512 has no Q4 misses. Only one 2,524,446,720-byte payload and its
rank map are mapped. Admission reserves another 1GiB for both guarded scratch
sets, fixtures and transient guarded replacements, protecting at least 16GiB
or 10% of host RAM. Physical rows are restricted to 1024–2048.

Strict C++ and Metal 4.1 compilation plus CPU checks read no model payloads and
create no Metal device:

```sh
make -f dev/benchmarks/prefill_moe_sep21/one_layer/Makefile -j4 cpu-self-test
```

The metallib links the original parent AIR inventory plus
`build/prefill-moe-sep21/memory.air` and `register.air`. It excludes the private
gathered AIRs. `all` only builds; `cpu-self-test` also runs `--cpu-self-test`.

Prepare an invocation for all eleven variants without executing GPU work:

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/one_layer/run.py \
  --rows 2048 --pairs 4 --pattern spread-all --strict \
  --report build/prefill-moe-sep21-one-layer/spread-all-new.json
```

Add `--run` only in the exclusive root GPU window. `--variant 1` through
`--variant 11` selects one candidate; omission screens all eleven. Patterns
are `hit-concentrated`, `hit-spread` and `spread-all`. Raw BF16 hidden/I64 IDs
are accepted together with `--input` and `--ids`. Synthetic hidden inputs use
the inherited divisor 512 by default; `--normalized` sets
`PREFILL_MOE_SEP21_NORMALIZED=1` and uses divisor 74. Normalization cannot be
combined with raw input.

Before timing, each variant reports equality of every activation,
scattered-down and combined BF16 element, and full-output errors against
relative L2 ≤0.001 and cosine ≥0.999999. Guard-passing numerical alternatives
may be timed unless `--strict` requires complete BF16 equality. A rejected
candidate remains in the report with its failure and no timing samples.
Warmed complete-chain GPU and wall timings rotate candidates and alternate
control/candidate order. Synthetic equality and error checks are primitive
qualification, not model-quality qualification.

The runner records code, binary/metallib and manifest hashes plus the exact
command and sanitized controls. Preparing an invocation reads no input, IDs
or model payloads. GPU execution alone maps the admitted single layer; it
never constructs or scans all 48 payloads.
