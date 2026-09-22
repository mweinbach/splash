# Optional root no-tail qualification

The root already qualified the R2048 concentrated case: exact raw F32,
scaled BF16 and complete BF16-chain PASS; M16-tail median 3.59442 to
3.584355 ms, neutral. Existing evidence:
`build/release/flash/sep21-adaptive-tail-r2k-hot-v1.json`.

The commands below are optional independent confirmations. Only the root may
run them during its exclusive GPU window. Each command uses a fresh report
and invocation witness. Do not reuse a path that already exists.

```sh
.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/run.py --rows 2048 --pattern hit-concentrated --variant 1 --pairs 16 --report build/release/flash/sep21-adaptive-tail-r2k-hot-root-confirm-v2.json --run
```

This creates 20480 routes on ten experts, 2048 routes per expert, and 640
active original M32 jobs. Every job has 32 valid rows, so the adaptive branch
executes the original M32 descriptor throughout. The declared M32 job
capacity remains 1151 and the producer launch remains 128 threads. This is
the control for branch/dispatch overhead without smaller tail descriptors.

```sh
.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/run.py --rows 1024 --pattern spread-all --variant 1 --pairs 16 --report build/release/flash/sep21-adaptive-tail-r1k-spread-root-no-smaller-tail-v1.json --run
```

The R1024 spread fixture creates 10240 routes across all 512 experts,
exactly 20 routes per expert, and 512 active original M32 jobs. Every job
has 20 valid rows, so none selects the M16 descriptor. The declared M32 job
capacity remains 831; original parameters, grids, buffers and 128 threads
are retained. It complements the full-job concentrated control with masked
M32 jobs that still have too many valid rows for the smaller descriptor.

Omit `--run` to prepare provenance without executing GPU work. Source and
artifact hashes and store manifest metadata are read during this dry run;
payloads and raw fixtures are not read. The component oracle executes one
readonly Full512 layer when run, and is not a whole-model prefill result.

For either executed report, require an overall passing sweep, copied-M32
probe qualification, zero raw-F32/scaled-BF16/full-chain differences,
finite probes, unchanged sticky diagnostics and guards, and final replay
equality. Timing uses at least 100 ms GPU warm work for the candidate and
matched native control, balanced alternating pairs, and no CPU buffer
reads inside the timed loop. Inspect per-pair GPU times and compare both
means and medians before interpreting a sub-percent delta as an improvement.

# Whole-worker CPU checks

`worker_policy_cpu.cpp` needs only these four hooks, defined in the copied
`FlashInt8ExpertStore.mm`, in `namespace splash::flash`, after its anonymous
namespace closes:

```cpp
std::string adaptiveExpertTailPipelineForCPU(const char *phase, uint32_t tileRows, bool enabled);
std::string adaptiveExpertTailPipelineRuntimeForCPU(const char *phase, uint32_t tileRows);
FlashInt8ExpertStoreParams adaptiveExpertTailParamsForCPU(uint32_t rows, uint32_t selections, uint32_t tileRows);
uint32_t adaptiveExpertTailLaunchForCPU(const FlashInt8ExpertStoreParams &params);
```

The explicit pipeline hook must call the same helper the actual runtime
wrapper uses; the runtime pipeline hook must call the actual frozen
`pipeline()`. Parameter and launch hooks must call actual `allRowsParams()`
and `allRowsLaunch()`. Link the CPU executable against the sealed host
objects without `FlashWorker.o`; it creates no backend or model. Run no
arguments and each of `--freeze0`, `--freeze1`, `--missing0`, `--retry0`,
`--retry1` as separate processes.

The checks pin the real parser, all original pipeline names, only-M32-hit
substitution, unchanged M16/M64/miss names, the 32-byte parameter ABI and
actual job capacity/launch across all 245760 legal geometries, and
first-successful-use policy freezing shared with the actual Store
translation unit. They pin the distinct enabled runtime identity marker.
The overlay source witness must also check that
`FlashForward::kernelRoutes()` includes `marker(requested())`; calling that
method dynamically requires a model and is outside this CPU-only test.
