Private worker composition for the root-qualified dense W8A8 numerical alternative.
The default parent is the exact pointwise, SG2/K128 tail, optional staged GDN FMA
worker. `--base` and `--base-make` can select another sealed pointwise Full512
parent. Preparation, compilation and CPU witnessing do not create a backend or
cache, read model/captured operands, or execute a GPU command.

```
python3 dev/benchmarks/dense_w8a8_sep21/worker_overlay.py
make -f dev/benchmarks/dense_w8a8_sep21/worker.mk -j4 all cpu-self-test
python3 build/dense-w8a8-sep21-worker-v3/machinery/worker_witness.py \
  --output build/dense-w8a8-sep21-worker-v3/cpu-witness-v1.json
```

Use a fresh output for every freeze; existing frozen workers are immutable.
The freezer verifies every parent `files[]` hash and root-qualified standalone
compiler input/artifact. It queries the actual parent executable/metallib Make
prerequisites, retains all actual rebuilt/reused/core objects except Forward and
Worker, and retains every actual AIR once, including the parent's own AIR.
Every inherited source/header, available parent-owned `.d`, role report, Make
rule and new private helper is hashed. Rebuilt dense Forward, Worker and cache
objects use private source/runtime includes only; no public header is changed.
The worker also archives its generator, transform, Make, witness, standalone
qualified inputs and artifacts. Reverification can run its archived
`machinery/worker_witness.py` and remains independent of later repository
machinery edits.

`SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21` accepts exactly `0` or `1`, defaults to
`0`, and freezes before path, metadata or backend access. Flag 1 selects SG4
M128/N64 W8A8 only for singleton main, nonverification, exactly R2048 QKV, Z and
QSA Q projections. Flag 0 retains inherited projection graphs. Both flags plan
and construct the same private coefficient and activation resources at model
startup when maximum rows are at least 2048, so paired comparisons share one
fixed cache/quantization identity and resource policy. The scope also requires
the inherited BF16 dense cache. Decode, verification and batch calls retain
their inherited graphs; the failed GDN output component is excluded.

The offline fixed resource plan is 84 roles, 168 immutable coefficient bases,
1,890,975,744 separately 16-KiB-rounded coefficient bytes and 12,615,680 activation
workspace bytes. Startup adds these resources before the trunk MemoryGovernor
reservation. Actual allocations are bounded by their plans; immutable getters
validate and census complete bases for residency. Resource witnessing exercises
metadata guards and geometry only; it never fits coefficients or constructs
these resources.

The kernel source is pinned to SHA256
`32806c52af23cee704b4e64864fa6befc3b38f81fc097f05f5d8d3c5fc1df555`.
Qualified QKV/Z/QSA-Q component reports and their full output/quantization error
metrics are frozen with the worker. These are component evidence. Full-model
quality, MTP acceptance and throughput require the root's single GPU driver;
they remain pending after a passing CPU/source witness.

The composed resident hybrid parent is also supported:

```
python3 dev/benchmarks/dense_w8a8_sep21/worker_overlay.py \
  --base build/hybrid-sg2-tail-fma-sep21-worker-v1 \
  --base-make dev/benchmarks/adaptive_expert_tail_sep21/combined/hybrid.mk \
  --output build/dense-w8a8-hybrid-sg2-tail-fma-sep21-worker-v2
```

Its immutable union becomes 916 complete coefficient bases / 204,143,722,496
bytes at maximum rows ≥2048, appending only the audited 168 new coefficient
bases. The original 25 expert owners, their getter, and the existing host
headroom policy retain their inherited source and guard. Activation resources
stay outside the immutable union. Compile and witness with the corresponding
`BUILD=` and `--build` values; no hybrid GPU run belongs to this workflow.
