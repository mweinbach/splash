# Private all-row Full512 target with original GPU tensor omission

This candidate keeps the original checkpoint on disk and keeps all source
identity, tensor dtype/shape, quantization, range/padding and norm checks. Exactly
432 target switch tensors /144 projections are disk-only metadata. Trained MTP
weights and methods are unchanged. Ordinary GPU tensor/projection access fails
closed for omitted records; geometry-only lookup uses `projectionMetadata`.

The loader requires the explicit private flag, PLE SSD mode, locked source
manifest and certified Full512 manifest before creating original GPU windows.
Canonical range partitioning removes 67,947,724,800 target bytes, reducing original
GPU resources from 28 windows /74,317,889,536 bytes to 7 windows /6,370,164,736
bytes. This proves omitted resource construction; physical/wired memory savings
remain unmeasured until root's actual startup and requests.

All four target executors use the same Full512 store: singleton Forward,
BatchForward, BatchVerify and BatchPrefill. Rows below256 use the existing M16
INT8 producer, with Y=min(rows*selections,job_capacity). Large rows keep the
existing M16/M32/M64 policy. There is no CPU count readback. Stable jobs each
consume at least one valid route, so active jobs are bounded by route count.
The hit shader rejects a count above that bound and missing Full512 ranks.

Store construction and hit graph validation hold no original target MetalBuffer
copies and construct no Q4 validation graphs. Metadata-only geometry, Shared
buffer extents, alias protection, valid-route poisoning and padded prepare-down
remain explicit. Arithmetic is unchanged from the existing INT8 hit producer:
BF16 activations × signed INT8 codes accumulate in F32; one stored F32 row scale
applies after the dot, followed by the existing BF16 boundaries/nonlinearity.
Original small-row target Q4 arithmetic is replaced, so this is a new numerical
derivative; model quality, state equivalence and MTP acceptance require testing.

Expected `identity.target_numerical_derivative_sha256`:

```
2e858faa201554642a443d48d38b7302159fe1a811c028a895454cc8ce5c073d
```

The checked original `loaded_model_layout_sha256` remains unchanged. The numerical
derivative binds that fingerprint, the Full512 store manifest, all-row producer
policy and original-trained-MTP marker. Prefix caching remains disabled.

Compile and CPU checks (no GPU):

```sh
.venv/bin/python dev/benchmarks/prefill4k_allrows_overlay.py
make -f Makefile -f dev/benchmarks/prefill4k_wide.mk \
  BUILD=build/flash-next SPLASH_PRECISION=hybrid \
  PREFILL4K_WIDE_BUILD=build/prefill4k-allrows-full512 \
  PREFILL4K_WIDE_FULLCACHE=1 prefill4k-wide -j 2
build/prefill4k-allrows-full512/splash-flash --cpu-self-test
.venv/bin/python dev/benchmarks/prefill4k_allrows_loader_cpu.py
.venv/bin/python dev/benchmarks/prefill4k_allrows_routes.py --cpu-self-test
.venv/bin/python dev/benchmarks/prefill4k_allrows_witness.py \
  --output build/release/flash/CHOOSE-FRESH-allrows-source-witness.json
```

The source witness checks every original/copy/transformation hash, omitted strong
buffer ownership, no original validation graphs/readback, untouched MTP, early
pre-backend policy validation, GPU guards and2,700 independently generated active
job partitions. Worker startup without policy or with the certified Top256 store
was refused before backend creation. These are CPU evidence only.

At capacity16384/rows8192/depth3, the complete static planner totals
160,785,563,648 bytes. Four eligible target/head states give163,273,441,280 bytes,
leaving84,116,674,970 inside the247,390,116,250 engine bound before driver margin.
Actual constructor deltas and every subsequent state reservation retain the
MemoryGovernor engine/host-pressure checks. Source memory plan is at
`build/release/flash/prefill4k-allrows-full512-memory-plan-v2.json`.

Root-only HTTP launch, after the current GPU job has finished:

```sh
.venv/bin/python dev/benchmarks/prefill4k_allrows_http.py \
  --rows 2048 --port 8011 \
  --admission-report build/release/flash/CHOOSE-FRESH-allrows-admission.json \
  --witness build/release/flash/CHOOSE-FRESH-allrows-launch.json --run
```

The helper is dry-run by default. It resolves the current qualified profile,
then selects the certified Full512 store, private all-row flag, teacher cache-only
priming, depth3, PLE SSD, idle/original residency off and qualified projected dense tiles on.
For wide comparisons set rows4096 or8192, preserving all other flags.

General `persisted_experts.graph_counters` count all target row sizes. Their
`large_row_*` counterparts use physical rows>=256 and exclude tiny prefill,
decode and verifier graph construction. Both clearly count graph construction,
not completed hardware work. The22-case quality runner uses the latter for
main-prefill coverage and retains general differences as small-row evidence.

Root must still qualify full output budgets, frozen22 task outcomes, small-row
primitive guards/canaries, fixed-prefix singleton-versus-verifier state/logits,
retained-prefix rollback, real joint lanes, long-prefill cancellation/deadline,
allocation refusal, actual host/wired footprint and unload/reload. Existing
`expert-oracle` in this build is retained for CPU self-test only: its original-Q4
GPU reference paths intentionally cannot access omitted target resources.

## Root's first measured private proof

The first actual run completed four uncached2K/256 requests (one startup warmup,
three warmed repeats), every request HTTP200 with256 output tokens. Warmed
native-counter medians were **2976.02 prefill tok/s** and **60.55 decode tok/s**;
per-request stream decode was approximately60.46. These are distinct timing
scopes. The current qualified v11 coding baseline decoded around71.7 tok/s.
Accepted/proposed drafts increased from approximately0.667 to0.738, which did
not offset the small-row INT8 route cost. Keep this numerical derivative private.

The final run, including the22 semantic checks, ended healthy and idle with
26/26 requests completed, zero request/Metal errors and zero denied reservations.
Native allocation was158,211,801,088 bytes; host availability88,418,041,856 exceeded
the27,487,790,694 reserve and growth remained allowed. The actual original GPU
ledger matched6,370,164,736 bytes. This establishes the resource-construction and
host-pressure benefit; it does not measure per-die placement or physical traffic.

The22 frozen tasks had full request/cache coverage and20 passes. The same two
arithmetic tasks still failed, so the individual report has `valid=false`.
The independently regraded comparison againstTop64's19/22 has `valid=true` and
zero new task regressions; `python_merge_counts` was resolved. This comparison
does not establish universal numerical parity or general model quality.

Evidence:

- `build/release/flash/ultra-locality-prefill4k-allrows-full512-first.json`
- `build/release/flash/prefill4k-allrows-admission-first.json`
- `build/release/flash/prefill4k-semantic-prefill4k-allrows-full512-first.json`
- `build/release/flash/prefill4k-semantic-top64-vs-allrows-full512-v1.json`

The2049-token nested JSON case consumed[2048,1], establishing one actual short
prefill tail. All observed decode/prefill widths wereB1; cancellation, memory
refusal, real joint lanes and reload were not exercised. Old Top64 wide-state
and raw-cache-disabled deep-prefix oracles do not carry over to this derivative.
An unchanged wide-state wrapper forcesTop64 and cannot qualify this candidate;
two separate Full512 stores also duplicate approximately242GB of payloads.
Use one immutable store with independent request/workspace ownership for the
next same-derivative state/rollback oracle, and measure lifecycle separately.

Root's subsequent eight-case HTTP lifecycle report completed with `pass=true`,
`coverage_complete=true`, and no case errors:
`build/release/flash/ultra-locality-prefill4k-allrows-full512-service-quality.json`.
It exercised actualB3/B4 prefill and decode widths; final counters show11 submitted,
9 completed,2 cancelled,0 failed. Peak allocation was159,921,668,096 bytes and
the service returned idle. This supersedes the initial run's missing joint/
cancellation/recovery coverage. Same-derivative retained-prefix state/numerical
consistency and the new QMV producer remain separate qualification gates. Root
reported automatic unload and no remaining model PIDs/ports after this run.

## C2 scalar reducer rejected after model testing

Root's same-build three-warm2K/256 comparison kept prefill effectively unchanged:
2969.61 MPP control versus2967.87 C2 tok/s. Streaming decode increased only from
60.30 to61.83 tok/s (approximately2.5%), despite the1.73× one-layer diagnostic
speedup. Accepted/proposed drafts fell from0.7384 to0.6576 and the stable output
hash changed. The frozen C2 suite passed19/22 versus MPP's20/22; it introduced a
`python_merge_counts` regression through a forbidden builtin call. The two
original arithmetic failures persisted. **C2 is not promoted and the prepared
C2+bulk build is not a qualified product candidate.**

Preserved evidence:

- `build/release/flash/ultra-locality-prefill4k-c2-control.json`
- `build/release/flash/ultra-locality-prefill4k-c2-on.json`
- `build/release/flash/prefill4k-semantic-allrows-mpp-vs-c2-v1.json`

The next private experiment keeps the old M16/N64/dynamic-K multiply descriptor
and executionSG4, but gathers each canonical token/selection directly into one
valid-row tensor. It keeps the same code/scale/rank buffers, F32 late row scale
and BF16 stages while removing the512-expert bucket prelude and down preparation.
Descriptor equality is a hypothesis about the dot tree, not proof: validRows1
versus the old packed multirow bucket and logical row relocation may still
change opaque MPP implementation behavior. Require exact old-MPP raw dot,
late-scaled F32, BF16, activation/down and diagnostic parity before model testing.
