This private oracle compares native M32N64/SG4 against native M64N64/SG8 over
one shared, certified Full512 layer. Both graphs independently create original
expert buckets and jobs, then run gate/up, down poison/preparation, down scatter,
and combine. It adds no hit-only job lists or private shaders.

The generated source retains the frozen store oracle's BF16 hidden fixture,
original I64 route patterns, unequal route weights, guards, independent bucket
reference and complete-output comparisons. Native dispatches reproduce the
frozen Full512 store's graph behavior through a small one-layer adapter. It
uses the existing one-layer loader rather than the production 48-layer store
constructor or full Q4 model. Full512 skips Q4 miss dispatches in the frozen
store. Only one 2,524,446,720-byte payload and its rank map are mapped; the
governor reserves another 1GiB for both guarded scratch sets, fixtures and
transient guarded replacements, protecting at least 16GiB or 10% of host RAM.
The oracle restricts physical rows to 1024–2048.

Compile and CPU checks read no payloads and create no Metal device:

```sh
make -f dev/benchmarks/prefill_moe_sep21/native_m64/Makefile -j4 cpu-self-test
```

Prepare a root-controlled GPU invocation without executing it:

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/native_m64/run.py \
  --rows 2048 --pattern spread-all --strict \
  --report build/prefill-moe-sep21-native-m64/spread-all-new.json
```

Add `--run` only in the exclusive root GPU window. Patterns are
`hit-concentrated`, `hit-spread` and `spread-all`. Raw BF16 hidden/I64 IDs are
accepted together with `--input` and `--ids`. Complete-chain GPU and wall times
alternate M32/M64 after warming both. Initial/final independent jobs use
`makeJobs(packed,32)` and `makeJobs(packed,64)` respectively. The report measures
every activation, scattered-down and combined BF16 element and labels differing
results as a numerical alternative. `--strict` rejects any difference before
timing and preserves its error report. Synthetic equality is a primitive check,
not a model-quality qualification.
