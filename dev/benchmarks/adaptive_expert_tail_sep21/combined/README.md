This component isolates adaptive M16 math within original M32 jobs on the
existing fixed-K128, SG2 expert path. Both matched paths use 64 threads,
ordered K128 F32 accumulation, original late F32 row scales, native BF16
projection/SwiGLU boundaries, and canonical BF16 down scatter.

The timing control is the existing variant7 pair:

- `prefill_moe_sep21_memory_fixed_gate_up_m32_n64_k128_sg2`
- `prefill_moe_sep21_memory_fixed_down_scatter_m32_n64_k128_sg2`

The candidate uses private `adaptive_expert_tail_sg2k128_sep21_*_m16_tail`
pipelines. Each producer calls the unchanged M32/SG2 job validator once and
chooses an M16 descriptor only when that original job has at most 16 valid
rows. All other jobs use M32. Both paths retain `Static=true`: full descriptor
tiles use original static K128 tensor slices and incomplete tiles retain the
original dynamic slices. No native M16 jobs, extra dispatches, prefix scans,
hit lists, SG4 kernels, M8 kernels, split-K reductions, or FMA composition are
introduced.

At 2048 input rows with all 512 experts receiving 40 routes, the original job
inventory remains 1024 active M32 jobs with 1151 parameter capacity. The
control executes 32768 descriptor rows; the candidate executes 24576. Scratch
allocation remains the inherited bounded allocation, and both graphs retain
the original buffer roles, parameter bytes, grids and ownership.

Build and CPU/source qualification:

```sh
make -f dev/benchmarks/adaptive_expert_tail_sep21/combined/Makefile cpu-self-test -j4
```

`generate_oracle.py` derives the established adaptive-tail oracle and changes
both scratch graphs from native M32/SG4 names to the existing variant7 names
and shared SG2 launch before any submission. The native SG4 graph is inspected
as an ABI source; it is never submitted or timed. `variant7.air` is separately
compiled from the frozen SG2 worker candidate source, so private copies cannot
silently become the baseline. The combined private shader exposes eight
non-probe/probe control/candidate kernels and omits existing baseline exports.

The strict GPU screen requires the following untimed gates before warm-up or
timings:

1. Execute the existing non-probe variant7 baseline and require a finite full
   BF16 chain with original buckets and clean diagnostic/canary checks.
2. Execute the private M32 probe and require bit-exact full-chain BF16 equality
   against that baseline, plus finite raw-F32 and scaled-BF16 probes.
3. Execute adaptive M16 probes and require bit-exact raw-F32 gate/up/down,
   scaled-BF16 gate/up/down, and complete BF16 activation/down/combine parity
   against the qualified control.
4. Execute the non-probe adaptive candidate and require complete BF16 equality.

An arithmetic difference is reported as a numerical alternative and receives
no matched timings. Accepted candidates and their own matched controls each
receive at least 100 ms measured GPU warm work. Timed pairs alternate order;
the timed region contains submissions only and no CPU buffer reads. Final
canary/diagnostic/bucket checks and poisoned-output replays are untimed.

Prepare a provenance witness without any GPU work or model payload reads:

```sh
.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/combined/run.py \
  --rows 2048 --pairs 8 --pattern spread-all \
  --report build/adaptive-expert-tail-sg2k128-sep21/spread-dry-v1.json
```

Root exclusively owns GPU qualification. Use fresh report paths:

```sh
.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/combined/run.py \
  --rows 2048 --pairs 8 --pattern spread-all \
  --report build/adaptive-expert-tail-sg2k128-sep21/spread-root-v1.json --run

.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/combined/run.py \
  --rows 2048 --pairs 8 --pattern hit-concentrated \
  --report build/adaptive-expert-tail-sg2k128-sep21/hot-root-v1.json --run
```

The default synthetic hidden fixture normalizes every F32 row to RMS 1 before
BF16 rounding; reports include observed BF16 row RMS bounds. `--inherited` and
`--divisor74` select earlier synthetic policies. Paired `--input` BF16 hidden
and `--ids` I64 route files support caller fixtures; invocation provenance does
not read or hash their contents. `--strict` is mandatory behavior even when
the flag is omitted.

Source checks are embedded in `run.py --source-check`. They check source and
generator hashes, unchanged original validator bytes, eight pipeline contracts,
SG2/K128/static descriptor policy, original job validation calls, both native
graph adaptations and independent job/tail ownership goldens. Invocation
witnesses additionally hash the oracle, metallib, baseline AIR, generators,
loader, runner and manifests. A GPU run loads only one certified Full512
layer, retains a 3 GiB scratch reservation plus host reserve, and avoids the
production full-store constructor. The model-quality claim remains false.

This is component qualification only. No production code or whole-worker
overlay is built until the root screen demonstrates a component win.
## Resident hybrid composition

The root's combined component runs passed exact raw F32, scaled BF16 and full
BF16 equality against existing SG2/K128. Spread median GPU time was 5.373125 to
5.238475 ms; concentrated no-tail time was 3.24844 to 3.225125 ms.

The composed worker is `build/hybrid-sg2-tail-fma-sep21-worker-v1`. It selects the
qualified SG2/K128 and adaptive M16 math only inside hybrid v4's existing I8 main
nonverification R2048/M32 prefill branch. Original Q4 decode/verification, native
other-row I8 paths, strong original tensor aliases, fixed-R4 coefficients, and
expert-only residency remain sealed. ALLROWS target/gathered flags stay disabled.

```sh
.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/combined/hybrid_overlay.py
make -f dev/benchmarks/adaptive_expert_tail_sep21/combined/hybrid.mk -j5 all
make -f dev/benchmarks/adaptive_expert_tail_sep21/combined/hybrid.mk cpu-self-test
.venv/bin/python dev/benchmarks/adaptive_expert_tail_sep21/combined/hybrid_witness.py --output build/hybrid-sg2-tail-fma-sep21-worker-v1/cpu-witness-new.json
```

Independent controls are `SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT=0|7`,
`SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21=0|1`, and optional
`SPLASH_FLASH_GDN_PREFILL_FMA_SEP21=0|1`. FMA flag1 requires `GDN_STAGED=1`.
SG2=0, tail=0, FMA=0 preserves v4 execution. SG2=7/tail=0 selects the existing
variant7 math; SG2=7/tail=1 selects adaptive M16. Tail alone has no effect with
SG2 disabled. The exact tail certificate and FMA's changed-rounding certificate
remain separate; FMA was observed to pass20/22 inherited task cases and this
combined worker still needs its own root model run.

The base hybrid numerical identity remains
`1a00dd45649f14de4ad48bafa32ff67f1207aaf27b201de01bc6641a210134e2`.
FMA adds its existing numerical policy and kernel SHA to that base before SHA256;
its enabled identity is
`c9fc21162d0b27d39e9291e1846af3f0a77f524e17798b7c50163af8d98c86f8`.
SG2/tail controls add kernel route markers without changing the phase identity.

The initial CPU/source witness passed:245 sources and116 effective compiler
inputs;245760 real job-capacity cases;688296 eligibility cases;all independent
freeze/retry modes;frozen FMA identity/source suite;full worker self-test. The
closure filters the ancestor weights object and retains v4's correct
`FlashWeights.o`, restores three omitted v2 AIRs, and includes original SG2 and
native I8 controls. Original Q4 methods, nine tensor aliases, fixed-R4 caches,
weights getter and resident union compare byte-for-byte. Registration remains
25 expert owners/69363302400 bytes and748 total owners/202252746752 bytes, with
the unchanged2GiB host margin. No new GPU backing or workspace is added.

The previous pure-I8 composed worker remains at
`build/adaptive-expert-tail-sg2k128-fma-sep21-worker-v1`, independently frozen.
