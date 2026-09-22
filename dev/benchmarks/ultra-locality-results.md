# M5 Ultra locality and decoding experiments — September 21

Measured on Apple M5 Ultra (80 GPU cores), 256 GiB RAM, with the installed
`local/Qwen3.8-Flash-Next-oQ4e-mtp` package. All GPU work ran serially. Native
Metal executables were used; Splash does not inherit MLX routing automatically.
The motivation was [MLX's M5 Ultra matmul tuning](https://github.com/ml-explore/mlx/commit/2d27ab05fb7dcda69bb3c57abd74c0b3bc9a5a99).
No experiment assigns threadgroups or buffers to a particular die, and no
inter-die traffic counter was collected. Locality explanations are hypotheses.

## Changes retained

- The local profile keeps MTP draft depth 3 and 64 MiB SSD PLE row caching.
- `SPLASH_FLASH_QSA_OUT_F32_N32=1` now enables the existing exact Q5/Q6/G64
  main QSA output projection tile. Other roles and quantizations keep their
  original routes. Explicit environment overrides remain authoritative.
- Status timing percentiles cache their sorted window until the next sample,
  preserving the previous percentile values, all publications and safe points.
  The prior four sorts cost about 99 microseconds per publication at the full
  4,096-sample window; savings are smaller in fresh sessions.

Three warmed singleton samples per task used exactly 2,048 input tokens,
256 completion tokens, temperature 0, reasoning off, and zero prefix reuse.
Initial per-task warmup was excluded. Streaming decode uses the server's exact
post-first-emission token count and interval, rather than end-to-end throughput.

| Task | Preserved baseline decode | Status cache + QSA tile | Observed change |
| --- | ---: | ---: | ---: |
| Coding | 72.47 tok/s | 72.70 tok/s | +0.33% |
| Counting integers | 96.49 tok/s | 97.14 tok/s | +0.67% |
| Operational prose | 58.67 tok/s | 59.08 tok/s | +0.70% |

These are small observed differences, without a statistical significance claim.
Uncached 2K prefill remained roughly 2,100 tok/s. Every repeated output hash
matched the preserved baseline. Reports are
`build/release/flash/ultra-locality-{baseline,status-qsa}.json`.

## Experiments kept opt-in or isolated

| Experiment | Component observation | Decision |
| --- | --- | --- |
| Whole-K tile orientation and traversal | Selected unchanged geometries gained about 9–26% GPU time with byte-exact BF16 outputs | `SPLASH_FLASH_DENSE_TRAVERSAL=1` is opt-in; same-build HTTP changes were within roughly ±0.3%, so it is not a default |
| BF16 split-K | Small matrices improved 2.25–2.31×; optimized M64 prefill slowed | Isolated; small BF16 path does not match most current verifier routes |
| Original F32 coefficient split-K | Actual Q5 QSA output improved 1.91× at 4 rows, including padding and reduction | Isolated; five BF16 output cells changed, and 16-row tests had near-zero numerical failures |
| Depth-3 GPU head chaining | Exact feature/state continuation; head API improved 2.8–4.6%, GPU time worsened about 2% | Isolated; estimated decode benefit is only 0.3–0.4%, without service integration |
| Quantized expert dispatch traversal | Four-row gains at most about 0.75% | Isolated; insufficient gain |
| GPU expert-sorted jobs | Exact scatter and outputs, but roughly 17–21% slower in overlap tests | Isolated; sorting overhead outweighed reuse |
| Resident PLE table | About 29.8 GiB more native peak allocation, no warmed throughput gain | SSD streaming remains the default |

Traversal timings rotated variant order and compared full output bytes. The
production traversal bridge passed guards and full BF16 parity. Source F32
split-K independently checked every exported affine coefficient and every
changed cell against double-precision dots. Its broad FP32 error-bound pass is
separate from strict BF16 fidelity: cancellation fixtures lose small residuals
even on the original whole-K path. It must not be called generation-qualified.

Matched same-build traversal reports are
`build/release/flash/ultra-locality-traversal-{qsa,off-qsa}.json`.
The preserved baseline and resident comparisons are also retained. Their plans,
actual completion counts, output hashes, binary/metallib hashes, native deltas,
memory audit, and final idle status are recorded in each report.

Primitive sources and reproducible commands are in
[dense locality](ultra_dense_locality/README.md),
[split-K](ultra_splitk.md), [draft chaining](ultra_draft_chain.md), and
[quantized locality](ultra_quantized_locality/README.md). Local results are under
`build/ultra-dense-locality/`, `build/ultra-splitk/`, and `build/release/`.

## Validation and bottleneck

The current service qualification plan contains **eight cases**: arithmetic,
structured output with system/developer messages, four concurrent requests,
tool call, tool continuation, cancellation, deadline, and recovery arithmetic.
Both baseline and combined candidate passed. All nine non-control normalized
response comparisons matched; time-dependent cancellation/deadline output is
not part of that equality claim. Reports:
`build/release/flash/ultra-locality-{baseline,candidate}-quality-quality.json`.

Production-policy CPU checks, source/header permutation review, exact percentile
append/eviction parity, source coefficient checks, and shader-validated split-K
padding/tail guards passed. The profiles' historical review copies retain their
old settings. The preserved benchmark runtime remains available under
`build/ultra-locality-baseline/` for subsequent paired work.

Final launcher/profile tests: 100 passed, one skipped. The normal
`build/flash-next` runtime was rebuilt, and an ordinary `./splash serve` launch
selected depth 3, QSA tile enabled, traversal off, and SSD streaming without
feature overrides. Two full 256-token coding responses matched the original
golden hash and exercised the QSA route. This smoke's single warmed request
observed about 2,099 tok/s prefill and 72.7 tok/s native decode; it is not an
additional benchmark median. See
`build/release/flash/ultra-locality-normal-default-proof.json`.
The service and native model processes exited, and both test ports closed.

A warm coding command trace assigns approximately **84.85% of decode GPU time
to target verification**. Head/verification/restore command boundary overhead
was about 117 ms versus 3,245 ms GPU work. This limits the gains available from
CPU submission improvements and points future work toward the actual tiny-row
target projection and state-update kernels. Trace and independent review:
`build/release/flash/ultra-decode-cpu-audit.json` and
`build/release/flash/ultra-locality-independent-review.json`.
