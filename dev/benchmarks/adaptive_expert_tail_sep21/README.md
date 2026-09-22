This private, bounded one-layer oracle compares the shipping native M32N64
expert chain with M16 and M8 adaptive tails. Both candidates retain the original
M32 bucket jobs, parameter bytes, grids, 128 producer threads, preparation,
poisoning, packing and combine dispatches. Only the gate/up and down/scatter
pipeline names change. There are no extra classification or prefix dispatches.

Qualification first proves that copied private M32 probe pipelines match every
native-control activation, scattered down value and combined BF16 output. Each
adaptive probe then must match copied M32 raw F32 gate/up dots and canonical
scattered down dots bit for bit, with finite values, and every scaled BF16
gate/up/down value bit for bit. The non-probe complete BF16 chain must also match
before and after timing. Rejected candidates remain in the report without timing
samples. Probes are excluded from timings. This primitive qualification does not
establish model quality.

The default synthetic input is normalized to true RMS 1 separately for each row
before BF16 rounding, reusing the native M16 fixture generator. `--inherited`
selects the historical divisor 512, and `--divisor74` the approximate divisor 74.
Raw BF16 hidden/I64 route IDs require both `--input` and `--ids`. Physical rows
are bounded to 1024–2048. Only one certified 2,524,446,720-byte Full512 layer is
mapped by the existing one-layer loader; the production 48-layer constructor and
full model are excluded. Admission reserves 3GiB beyond the layer for scratch,
guards, fixtures and the two raw/scaled probe sets, preserving at least 16GiB or
10% of host memory.

Build and CPU/source checks create no Metal backend and read no model payloads:

```sh
make -f dev/benchmarks/adaptive_expert_tail_sep21/Makefile -j4 cpu-self-test
```

The metallib links the native one-layer AIR inventory and the private
`adaptive.air`. CPU checks cover normalized fixture RMS, original M32 padding
goldens, malformed parameters/jobs/ranks/offsets and private pipeline names.
The source checker verifies the copied native validator and descriptor policy.

The runner defaults to a dry run. It verifies code provenance, reads only the
store manifest and writes a fresh invocation witness. Preparing an invocation
does not read raw input/IDs or model payloads:

```sh
.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/run.py \
  --rows 2048 --pairs 4 --pattern spread-all --strict \
  --report build/adaptive-expert-tail-sep21/spread-all-new.json
```

Add `--run` only in the exclusive root GPU window. `--variant 1` selects M16
tails, `--variant 2` M8 tails, and omission compares both. Timing pairs must be
even (2–32). Every accepted variant and its matched native control receive at
least 100ms of measured GPU warm work. The timed sweep rotates variant positions
and alternates candidate/control order with balanced counts. Its loop contains
only command submissions and timing bookkeeping; diagnostics, canaries, hashes,
bucket scans, probes and output reads occur before or after it. Final per-variant
replays recover evidence because candidates share one scratch set.

The executable exits with status 2 if any candidate is rejected. The report
retains successful qualification/timing for other candidates in that same run.
An invocation witness records required qualification policy; actual GPU
qualification results appear only in the execution report.

The root qualified both descriptors on the R2048 spread fixture with zero raw
F32, scaled BF16 and complete-chain differences. M16-tail median GPU time was
5.6546 to 5.402 ms. The R2048 concentrated case, which has no tail jobs, was exact
and neutral at 3.59442 to 3.584355 ms. The selected whole-worker candidate uses
M16 tails with the original SG4 whole-K reduction. These are component results;
the whole-model prefill result remains separate.

Prepare and build the sealed pointwise-base composition without GPU execution:

```sh
.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/worker_overlay.py
make -f dev/benchmarks/adaptive_expert_tail_sep21/worker.mk -j4 all
make -f dev/benchmarks/adaptive_expert_tail_sep21/worker.mk cpu-self-test
.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/worker_witness.py --output build/adaptive-expert-tail-sep21-worker-v1/cpu-witness-new.json
```

The private artifact lives in `build/adaptive-expert-tail-sep21-worker-v1`.
`SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SEP21=0` retains the pointwise base's native M32
pipelines; `=1` selects only its native M32 gate/up and down/scatter hit producers.
The first successful parse is frozen before paths, metadata and backend creation.
M16, M64, gathered and miss paths retain their original routes. The enabled route
adds `private-native-m32-jobs-m16-tail-validle16-sg4-exact-f32-bf16-sep21-v1` to
`kernel_routes`; the numerical derivative, workspace planner, buffers, job list,
parameter ABI, launch extents and producer threads are preserved. Pointwise,
bulk and gathered controls retain their independent settings.

The overlay freezes 243 sources and 117 reused host/core/AIR inputs with hashes,
copies the root qualification evidence, and classifies the adaptive producer
pipelines as MoE in the attribution executable. Device-free CPU checks exercise
84 actual pipeline cases, all 245760 legal parameter/launch geometries, flag
freezing across translation units and invalid-value rejection before paths or
backend creation. `cpu-witness-v1.json` records the passing initial qualification.
Optional no-tail commands are in `no-tail-commands.md`; GPU execution is root only.
