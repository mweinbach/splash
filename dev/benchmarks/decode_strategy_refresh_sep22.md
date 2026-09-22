# Decoder source refresh, September 22

CPU/source/JSON-metadata work only. No model/operand/capture payload was opened,
hashed, exported, or submitted by this research task. Root owns every GPU run.
The qualified parent is `build/mtp-teacher-bulk-ab-qsa-sep21-worker-v5`.

The concrete next component is **two-row reuse of original packed Q4 weights
inside the existing raw MLX F32XSUM QMV tile**, initially only physical R4,
K2560/N10240/Q4/G64. Root approved its CPU component preparation after the
source and budget audit below. There is no worker integration authorization or
present speed claim. A later Q5/Q6 expansion is a separate decision.

CPU component preparation completed at `build/raw-q4-rowpair-sep22-component-v5`:
303142 CPU checks,289 pinned source files, unchanged Core4 and authenticated
native QMV AIR; no device/model/operand/capture payload access. Original/control
transitive SSA-erased opcode censuses117/117 and candidate/probe50/50 match,
without a dependency-tree/register-equivalence claim. Native fast-path float
allocas total80 bytes, candidate168 bytes; final hardware spill/occupancy is
unmeasured. V5 corrects post-timing parity by saving actual measured outputs
before probe replay. Root's independent CPU source/artifact review passed in
`build/release/flash/sep22-raw-q4-rowpair-v5-Root-independent-CPU-review-v1.json`
(SHA65cc8deb39478c77d0fed5450a476f3aad6d6f91511b3906f65c29c2748dd92c).
`READY-CPU-reviewed.json` and `root-synthetic-command-reviewed-v2.json` bind that
receipt and Root's artifact pins. The runner is prepared and not executed;
exclusive Root GPU release remains pending. Only bounded synthetic GPU scope is
approved. No worker integration, current input/state qualification or larger
format expansion is made.

## Primary timing audit

The primary source is
`build/release/flash/sep21-gathered-verifier-stage-v1.json` and its
`.trace.jsonl`, two complete R4/begin2048 target-verifier samples. The incoming
IDs are real ordinary greedy continuation, **not current trained MTP proposals**.
Stage-per-dispatch profiling changes encoder boundaries. These are diagnostic
budgets, not the present normal-service timing of a candidate.

The classifier in `decode_optimization_sep21/verify_attribution.mm` classifies
`flash_affine*`, `flash_dense*` and `flash_float_dense*` as dense unless the
shared-expert interval or full vocabulary output binding applies. QSA output
projection is dense; `flash_hc*` is HC. In particular, HC input-padding dispatches
are named `flash_float_dense_small_rows_pad` and belong to the dense family.
Family subtraction does not isolate coefficients or projections reliably.

All numbers below are milliseconds of actual recorded dispatch intervals.

| Cached F32 multiply pipeline | Calls | Sample 0 | Sample 1 |
|---|---:|---:|---:|
| `flash_hc_up_f32_mpp_m8_n32_s4` | 97 | 2.580327 | 2.575914 |
| `flash_float_dense_small_rows_m8_n64` | 14 | 1.568043 | 1.639247 |
| `flash_float_dense_small_rows_m16_n64` | 2 | 0.250832 | 0.252291 |
| `flash_qsa_out_f32_n32_m8_n32_s4` | 5 | 0.520834 | 0.522125 |
| **Total 118 cached F32 multiplies** | **118** | **4.920036** | **4.989577** |

The median of the two totals is 4.9548065 ms. All 118 pad dispatches add
0.565085/0.566211 ms; a weight-layout change does not remove them. Dense-family
totals are 11.208539/11.246036 ms, HC-family totals 8.161035/8.174082 ms. Neither
family total is the cached-F32 weight budget.

Weight binding 1 is 13,107,200 bytes for every one of the 97 HC-up entries.
Source `flash_hc_up_f32_mpp.metal` views one original F32[N10240,K320] matrix as
four chronological streams: four M8N32 whole-K operations with output width2560,
not one N10240 output operation. Other F32 binding sizes are six 104,857,600-byte
QKV maps, one 26,214,400-byte PLE-value map, seven 125,829,120-byte QSA-q maps,
and seven 62,914,560-byte QSA-output maps. Their sum is exactly 3,247,964,160
bytes: HC 1,271,398,400 plus other dense 1,976,565,760. This independently matches
`sep21-decode-current-route-roofs-v2.json` without a payload read.

The earlier candidate, an exact physical F32 output-tile repack, is unclosed:
the inspected traversal/tile/K-loop/LUT screens retain row-major F32 B storage.
Nevertheless it has only the 4.9548065 ms multiply budget. With the current
guard-only report's 150 accepted drafts/105 cycles and 52.66825649069135 native
tokens/s, the observed average cycle is 46.1107238095 ms. Hypothetically doubling
**all** these F32 multiplies yields 55.6586434 tokens/s; removing them entirely
yields 59.0090463 tokens/s. This combines an older perturbed component budget
with current normal timing and is only a conditional sensitivity. It does not
justify a 2 GB sidecar or a peak-decode claim. At most, a single 13.1 MB HC-up
repack would be an optional cheap probe; it is not the selected larger path.

## Raw projections actually present

The same primary trace exposes these raw QMV shapes. Counts and input/output
binding extents, not family-name subtraction, establish the rows and shapes.

| Raw pipeline/source geometry K/N | Calls | Sample 0 | Sample 1 |
|---|---:|---:|---:|
| Q4/G64 2560/10240: GDN QKV plus PLE key | 27 | 1.359415 | 1.345415 |
| Q5/G128 6144/2560: GDN output | 36 | 1.242414 | 1.242209 |
| Q5/G128 2560/6144: GDN Z | 26 | 0.927291 | 0.918751 |
| Q5/G128 2560/48: GDN a/b | 51 | 0.848245 | 0.846586 |
| Q6/G64 2560/48: GDN a/b | 21 | 0.486876 | 0.483539 |
| Q6/G64 2560/6144: GDN Z | 10 | 0.424707 | 0.417374 |
| Q4/G64 2560/12288: QSA q | 5 | 0.296916 | 0.296750 |
| Q5/G64 2560/10240: GDN QKV | 4 | 0.242583 | 0.242000 |
| Q4/G64 2560/512: QSA k/v | 10 | 0.159126 | 0.159832 |
| Q4/G64 6144/2560: QSA output | 5 | 0.147875 | 0.148834 |
| Q4/G64 2560/640: QSA index | 4 | 0.063625 | 0.063460 |

Large output N>=2048 covers 113 calls and 4.641201/4.611333 ms, median
4.626267 ms. The two-row idea is useful only if a substantial fraction of this
larger domain eventually qualifies. A hypothetical 2x improvement across that
whole domain yields approximately55.45 tokens/s; it is not a 240 tokens/s path.
The first N10240/Q4-only component covers only 1.352415 ms of the older trace.

Other large raw budgets are 97 HC-down/injection calls (sum of Q4/5/6/8 entries:
4.882252/4.891624 ms) and 48 BF16 routers at 1.350711/1.349829 ms. Shared Q8/G128
gate/up is 96 calls at 1.238917/1.240375 ms, with shared down 48 calls at
0.566334/0.564332 ms. Shared gate is another 48 Q8/G64 N1 calls at
0.893502/0.893581 ms. These entries remain distinct from the Full512 I8 expert
producer and its prefill paired-gate experiments.

## Selected first component: raw Q4 two-row reuse

Shipping math is `runtime/metal/kernels/shared/flash_affine_qmv_f32.metal`;
`FlashAffine.cpp::route` chooses it only for measured dense Q4/Q5/Q6 geometry.
At Q4/K2560, FastBlock512 is exact: each lane loads 16 contiguous BF16 values,
produces the original F32 prescaled `x_thread[16]` and activation sum, calls the
original Q4 `qdot` for four output columns, adds each chunk result in the original
K512 chronology, then performs `simd_sum` and one BF16 cast. Two SIMD groups
produce eight columns per64-thread CTA. R4/N10240 uses grid1280x4x1.

The candidate pairs physical rows0/1 or2/3. It retains two separate original
load-vector results and eight independent result scalars per SIMD. For each
output column/K512 chunk, read the same BF16 scale/bias once, read four original
packed ushort values once, and share only their integer masks between the two
row-specific qdots. Each row still has its own zero-initialized qdot accumulator,
the same four chronological parenthesized Q4 sums, `scale*accum + xsum*bias`,
the same per-chunk `result+=`, the same SIMD sum and the same BF16 boundary.
No K partition, factor redistribution, tensor-coefficient cache or numerical
threshold changes. Grid becomes1280x2x1 with the same64-thread CTAs.

Source-requested packed weights, scales/biases and integer masks halve. BF16
activation loads and all row-specific floating arithmetic remain. Normal
workspace, backing, graph dispatch count and model state allocation are unchanged.
The first source matrix is original layer1 QKV, default source Q4/G64, with
13,107,200 packed bytes and two819,200-byte parameter planes: 14,745,600 bytes.
The component may copy only these selected source spans into guarded buffers;
it must not instantiate/load the whole model merely to test this projection.

The larger grid is why this is distinct from the closed HC-down row-reuse
branch: HC N320 reduces324 CTAs at R4 to162 for reuse2, whereas this first raw
projection reduces5120 to2560. This observation does not prove occupancy or
speed. Two x-vector/result sets increase live registers; actual final machine
spills/occupancy remain unknown. The component must expose any compiler `alloca`
or changed F32 intrinsic/attribute sequence, and all GPU proof must precede timing.

### Before GPU timing

- Freeze the parent source/ABI/Core4-object and actual original QMV AIR hashes.
  Use the actual native Metal4.1/O3/fast-math recipe, not a new fno-fast policy.
  The selected single-role primitive uses a scoped library containing eight
  immutable original QMV entries plus three private entries; it is not a Worker
  library. The original native library identity is separately authenticated.
  Never replace a shipping entry or infer whole-library/Worker equivalence.
- Build timed candidate and untimed original/candidate raw-F32 taps from the
  same literal bodies. The taps add only a raw output argument and a store after
  the corresponding value is computed. No independently recomputed dot can
  establish production register equality. Record nonempty transitive ordered
  floating LLVM opcode/intrinsic/attribute censuses. Their SSA-erased equality
  does not prove the operand dependency tree; preserve unnormalized LLVM for
  review and require actual GPU register/output proof separately.
- Both tapped BF16 witnesses must reproduce their respective untapped timed
  output, and the control must reproduce the immutable original entry. Require
  full raw-F32 bit equality and BF16 bit equality for the declared cases before
  timing, plus the original per-row L2<=1e-4/cosine>=.999999 gate and exact zero
  reference norms. Preserve independent sampled F64 diagnostics; no new boundary
  exception or broad absolute-bound override.
- Cover identical/different/permuted rows, zeros/signed zeros, cancellation,
  high magnitudes, source scale/bias signs/zero, BF16/F32 subnormal behavior,
  NaN/Inf/sticky diagnostics, changed immutable inputs, output/scratch canaries,
  nontrivial original row strides and selected-view offsets. Test malformed
  params, short extents/aliases and partial/overlarge grids before graph mutation.
  GPU exceptional cases compare output bits/diagnostics and do not manufacture
  finite quality scores.
- Root alone reads selected source spans or captured input. If no certified
  current layer1 BF16[R4,2560] projection-input capture exists, label deterministic
  component inputs synthetic. An older prefill capture is not a current MTP
  register input. Before any whole integration, Root must capture actual current
  producer inputs and repeat raw-F32/production-BF16 proof in that context.

### Timing and decision

Reserve every selected span, input, result, diagnostic, tap and guard allocation
through MemoryGovernor before construction; report native actual charges and
clean teardown. No persistent sidecar or additional weight cache is created.
Use exactly the shipping original entry and the untapped candidate for timing
over the complete four-row projection. Warm each for at least150ms actual GPU
execution, then collect18 balanced AB/BA pairs on the same addresses, with no
CPU buffer reads/hashes/diagnostic accesses from warmup through the last sample.
Retain all GPU/wall values and timing positions. This first component should stop
if exactness fails or a substantial gain is absent. No service rate is inherited.

Later integration would initially be singleton target Verify/R4 only and only
on authenticated raw Q4/K2560/N10240 roles rejected by the existing selective
F32 tile. Prefill/tails, R1, MTP seed/head and batch routes keep their existing
math. Default-cache membership, including212 null-policy prefixes, must remain
unchanged. Actual whole-trunk cache/tape/logit/hidden/greedy/future/prefix/abort
proof and fresh normal1warm/3trial plus original22 semantics are prerequisites
for a worker performance claim. General Q5/Q6 or wider batches are unapproved.

## Closure checks

Targeted `rg` over benchmark source/docs found no existing **raw main dense**
QMV two-row coefficient reuse retaining the original F32XSUM/qdot tree. Existing
qmv-one-layer/C1/C2 kernels are signed-I8 **expert** software dots with another
reduction, not this producer. Ordinary R1 expert vector is known and not repeated.
The rejected expert R4 N16 route34 strict-L2 failure remains rejected.

HC-down SG1 passed original-dot exactness but was neutral .98–1.035x
(`sep21-tuning-progress.md`, checkpoint02:05). HC packed rowreuse subsequently
passed24 exact R4/R8 cases but was slower in all cases
(`hc_down_rowreuse_sep21/README.md`). It is closed. Dense all-role/narrow BF16,
whole-K/tile/K-loop/traversal, paired prefill I8, compressed I8/BF16 threadgroup
LUT, and GDN fallback variants remain closed or separately unqualified as their
current reports state; no result here relaxes them.

Only this proposal and new component source are owned by this task. Existing
production and private actor files are not edited.

## Current measured and qualified scope (September22)

Root completed the one-role current-input component proof with92 gates:
production rawF32/BF16 exactness, fault/guard cases, at least150ms GPU warmup
per variant,18 balanced timing pairs and clean teardown. On the captured
ordinary target Verify/R4 layer1 QKV input, original GPU time was
.0417500268668s and rowpair .0301250256598s:1.3859x,27.8% less component GPU
time. This input is a real continuation from the current disposable2K state;
it is not an MTP-proposal input. The report is
`build/release/flash/sep22-raw-q4-rowpair-layer1-current-v1.json`.

The narrow worker uses the identical qualified4a67 AIR, appended to the
authenticated current library after reproducing parent06cc exactly. Scope is
only26 main GDN QKV rawQ4/G64/K2560/N10240 NULL-tile roles on singleton
Verify/R4. PLE and selected F32-cache entries are excluded. Root's fresh whole
proof passed26 frames/54 replays,10,699,174,139 compared bytes and9891 planes,
with exactly130 calls/520 rows over five R4 trials, unchanged bundle240/960 and
HC485/1940/485 counters, owned-buffer guards and backend/governor teardown.
Prefill and future R1/R4/R8 recorded no extra raw-rowpair calls. The immutable
worker/library/source identities are663663/754028/162b01; exact hashes and the
actual report digest are bound by
`build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2/Root-rawQ4-native-qualified.json`.

The original22 semantic grader and normal matched same-binary flag0/1 benchmark
are still required. A separately sealed private Python adapter retains the
original active guard/compact/HC/base gates and adds this fresh receipt,
source/AIR/flag marker and26-times-actual-H3-cycle counter checks. Component
speed does not establish normal throughput or semantic qualification. The
older26-role budget is approximately1.30ms of a37.18ms target cycle, so this
initial scope has under1% expected whole-cycle sensitivity. The larger decode
goal remains open.
