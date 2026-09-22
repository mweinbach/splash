This private one-layer oracle compares the shipping M32N64/SG4 INT8 expert
kernels against shipping M16N64/SG4 kernels. Both graphs independently build
the original bucket/job lists, then gate/up, down poison/preparation, down
scatter and combine. The same persisted INT8 coefficients, FP32 scales,
BF16 boundaries, hidden inputs, route IDs and unequal route weights feed both
chains. It adds no coefficient cache, hit prefix, extra job list or shader.

At R2048, `spread-all` routes give exactly forty rows to each of 512 experts.
M32 runs 1,024 jobs and 32,768 padded matrix rows; M16 runs 1,536 jobs and
24,576 padded matrix rows, reducing padded matrix work by 25%. M16's declared
job capacity is ceil(20,480/16)+511 = 1,791. Both independently allocated
scratch sets retain the original minimum-tile-eight capacity of 3,071 jobs.

The frozen store's `allRowsScratch` explicitly accepts M16/M32/M64 at physical
rows up to 8,192; only M64 needs its special wide policy. `allRowsParams` and
`int8_expert_store_job` agree on the M16 capacity. This bounded adapter uses
the existing certified loader for one 2,524,446,720-byte payload, excluding
the production 48-layer constructor and full Q4 model. The governor reserves
another 1 GiB for both guarded scratch sets and fixtures and protects at least
16 GiB or 10% of host RAM. This oracle bounds physical rows to 1,024–2,048.

Compile and run strict CPU checks without creating a Metal device or reading
model payloads:

```sh
make -f dev/benchmarks/prefill_moe_sep21/native_m16/Makefile -j4 cpu-self-test
```

Prepare a root-controlled GPU invocation without executing it:

```sh
.venv/bin/python dev/benchmarks/prefill_moe_sep21/native_m16/run.py \
  --rows 2048 --pattern spread-all --normalized --strict \
  --report build/prefill-moe-sep21-native-m16/spread-all-rms-new.json
```

Default input normalizes each row to true RMS 1 in FP32 before BF16 rounding;
the report records the resulting minimum/maximum row RMS. `--inherited`
reproduces the frozen `/512` fixture and `--divisor74` reproduces the existing
`one_layer` approximate normalized fixture. Raw BF16/I64 inputs are accepted
together with `--input` and `--ids`.

Only root adds `--run` during an exclusive GPU window. Complete activation,
scattered-down and combine BF16 equality is mandatory before timing. Complete
comparison metrics are retained even when equality or numerical guards fail;
rejected cases have no timing samples and a failure report. Timings alternate
warm M32/M16 commands, with CPU checking and payload hashing outside timing.
Synthetic parity is a primitive check and does not qualify model quality.
