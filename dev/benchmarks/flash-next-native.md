# Native Flash-Next developer route

Splash runs the existing local `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` checkpoint through a native Metal worker, with autoregressive decoding, trained greedy MTP and real shared-weight decode batches of up to four requests. Import, serving and qualification use local files. The original checkpoint remains untouched.

The supported checkpoint is `Qwen4ExpForConditionalGeneration` / `qwen4_exp`: hidden width 2,560, 48 layers, four hyperconnection streams and low-rank width 320. It has 36 Gated DeltaNet layers, 12 Qwen Sparse Attention layers, 512 routed experts/top-10 selection and intermediate width 640. QSA pools four-token blocks and selects up to 512 complete blocks beyond its 2,048-token budget, plus the causal incomplete tail. Mandatory PLE at zero-based layer 1 uses 16 hashed bigram/trigram heads and 128 embedding shards. The final HC mixer reduces four streams to the untied vocabulary head.

The loader preserves UInt32 packed 4/5/6/8-bit weights, BF16 scales/biases and stored I64 PLE arrays, with groups of 32/64/128 per projection. Qwen4 RMSNorm uses audited `OnePlusWeight`; GDN gated norms use direct gamma. PLE retains its stored shared scale, `0.00019931793212890625`. The trained one-layer MTP head is implemented; mapped vision coefficients still have no execution path.

**Import locally.** The standard-library Python importer writes a separate aligned derivative, preserves tensor/tokenizer/config bytes, hashes and verifies sources and payloads, then publishes atomically. The bundle contains 21 payloads and 3,748 tensors, totaling 106,320,429,056 aligned payload bytes. Allow approximately 99 GiB of additional disk space.

```sh
python3 dev/tools/import_flash_next.py \
  "$HOME/.omlx/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp" \
  --alias Flash-Next-oQ4e-mtp-v1
```

An existing valid bundle is verified/reused; append `--verify-only` for an explicit source audit. The derivative is `install/local-models/Flash-Next-oQ4e-mtp-v1`, schema `splash-local-qwen4-affine-v1`, with 16 KiB alignment, `manifest.json` and exact-byte `manifest.sha256`. Native startup checks metadata, layouts, padding, PLE parameters and norm anchors. Optional C++ `verifyPayloadHashes=true` performs a full payload scan. Source identity: `ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e`.

**Build and serve.** From the repository root:

```sh
make -j12 BUILD=build/flash-next \
  flash-next build/flash-next/flash-forward-oracle

build/flash-next/splash-flash --cpu-self-test

./splash serve \
  --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --local-package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --port 8011 --max-context auto --max-memory auto
```

This machine's `.splash-build.mk` selects `hybrid`, Metal 4.1 and macOS 27.0. Flash reads the original mixed-affine checkpoint and can use separate persisted BF16/F32 dense operands and selected signed INT8 expert weights. Compiler selection does not itself convert weights. An explicit `SPLASH_PRECISION=q4` override selects the optional Metal 4.0/macOS 26.4 compatibility build. `BUILD` isolates outputs and configuration stamps. The CPU self-test exercises host sampling/protocol handling without loading weights or running GPU work.

The launcher selects `build/flash-next/splash-flash`, which speaks framed native protocol v5. `server/server.py` supplies HTTP, local tokenizer/chat formatting, streaming and constraints. Remote code is disabled. Ctrl+C stops the foreground frontend and its owned worker. A manual frontend can select the executable with `--binary` and the same `--local-package`.

The local `.splash-local-profile.json` applies only to the exact source identity and `qwen4_exp` architecture on an `Apple M5 Ultra` with at least 192 GiB RAM. Qualification used 80 GPU cores and 256 GiB RAM. It supplies runtime defaults; explicit environment values, including `0`, take precedence. Missing/mismatched profiles retain ordinary defaults. The profile changes no compiler settings or other model routes. The active v6 profile has 34 static flags and two metadata-gated optional saved-store paths. The four new kernel policies reuse existing weights and preserve state precision; [v6 audit details](flash-v6-kernel-round.md) retain the small gains and failed private synthetic proposal bounds. [Saved-format details](flash-saved-formats.md) describe their layouts, validation and measured behavior.

| Runtime setting | Qualified profile behavior |
| --- | --- |
| `SPLASH_FLASH_QMV_F32=1` | Qualified activation-prescale/XSUM F32 QMV for selected dense Q4/Q5/Q6 projections |
| `SPLASH_FLASH_FUSE_HC=1` | HC Down/UpMix and adjacent InjectNorm fusion; norm reuse stops before PLE |
| `SPLASH_FLASH_FUSE_GDN=1` | Persistent decode/verification; real batches bind each request's state directly, avoiding state pack/scatter copies |
| `SPLASH_FLASH_QSA_F32=1` | Fused preparation and four-partition online attention with F32 probabilities |
| `SPLASH_FLASH_DENSE_CACHE=1` | Immutable derived BF16 dense operands, approximately 8 GiB for the trunk plus 170 MiB for the MTP head |
| `SPLASH_FLASH_BLOCKED_MOE=1` | Bucketed MPP expert computation for prefill windows of at least 256 rows |
| `SPLASH_FLASH_PREFILL_ROWS=2048` | Up to 2,048 trunk rows; ordered causal QSA chunks remain at most 128 rows |
| `SPLASH_FLASH_BATCH=1` | Actual B2–B4 shared-weight decode; B1 retains the singleton path |
| `SPLASH_FLASH_MTP=1` | Trained MTP for greedy, temperature-zero, unmasked requests with output budgets greater than one |
| `SPLASH_FLASH_EXPERT_QMV=1` | Qualified contiguous Q4 expert vector projections for small windows |
| `SPLASH_FLASH_GDN_STAGED=1` | Staged prefill recurrence with exact state/BF16 boundaries |
| `SPLASH_FLASH_QSA_MPP=1` | Grouped-query attention with measured context/window partition choices |
| `SPLASH_FLASH_MTP_QSA_F32=1`, `SPLASH_FLASH_MTP_QSA_MPP=1` | The same qualified attention policy in the trained head |
| `SPLASH_FLASH_FLOAT_DENSE_CACHE=1`, `SPLASH_FLASH_FLOAT_DENSE_SELECTIVE=1` | Original F32 coefficient operands for measured small-window winners; raw fallback for other shapes |
| `SPLASH_FLASH_MOE_Q4X8=1` | Vectorized aligned Q4 staging with unchanged blocked-MoE arithmetic |
| `SPLASH_FLASH_MOE_M64=1` | M64N64/256-thread expert tiles for physical cohorts of at least 4,096 rows without an expert cache |
| `SPLASH_FLASH_MTP_DRAFT_DEPTH=15` | Singleton maximum 15, output-budget bounded; joint maximum remains three |
| `SPLASH_FLASH_BATCH_MTP=1` | Shared trained-head drafting and target verification with independent retained prefixes |
| `SPLASH_FLASH_BATCH_PREFILL=1`, `SPLASH_FLASH_BATCH_PREFILL_ROWS=2048` | Real main-trunk prefill cohorts, up to four 2K lanes |
| `SPLASH_FLASH_BATCH_MTP_PREFILL=1` | Shared prompt priming for compatible real trained-head pair chunks |
| `SPLASH_FLASH_PLE_LOOKUP_FUSED=1`, `SPLASH_FLASH_PLE_POST_FUSED=1` | Exact GPU n-gram lookup/history and PLE post/injection fusion |
| `SPLASH_FLASH_GPU_GREEDY=1` | Compact exact token selection from inline GPU argmax |
| `SPLASH_FLASH_INT8_HEAD=1` | Original byte-addressable UINT8 vocabulary codes, BF16 activations and F32 group correction |
| `SPLASH_FLASH_MOE_DIRECT_A=1` | Device activation operands in blocked MoE, with sanitization and bounded tail padding |
| `SPLASH_FLASH_QSA_ROW_TILES=1` | Temporal M32 query tiling for qualified dense causal windows, retaining F32 probabilities |
| `SPLASH_FLASH_SAVED_OPERANDS_RESIDENT=1` | Metal API request for verified saved allocations; physical pinning remains unverified |
| `SPLASH_FLASH_SHARED_EXPERT_FUSED=1` | Exact rounded BF16 shared gate/up/SwiGLU fusion for qualified prefill windows |
| `SPLASH_FLASH_DENSE_M64_OUT=1` | Role-gated M64N128 for N2,560/K6,144 attention outputs; existing routes for other roles |
| `SPLASH_FLASH_GDN_BATCH_ILP=1` | Batched prefill register/geometry change, retaining independent FP32 recurrent state |
| `SPLASH_FLASH_MTP_QMV_F32=1` | Original-Q4/G64 FC-hidden proposal projection; target verification and prompt priming retain their policies |


Optional `SPLASH_FLASH_OPERAND_STORE` maps 509 BF16 and 508 F32 dense matrices. Optional `SPLASH_FLASH_INT8_EXPERT_STORE` maps the 48-layer × 64-selected-expert signed INT8 derivative. These paths are gated separately from the 34 static flags. Eligible blocked prefill uses cached experts; misses retain original Q4. Small-window/decode routes and trained MTP weights retain their existing policies.


The v6 profile retains up to 15 singleton drafts, bounded by remaining output budget, and a joint cap of three drafts. Committed head folds use ordered chunks of at most eight real rows. Exact target verification accepts the matching greedy prefix, restores its GDN/PLE/QSA state and emits the target correction on mismatch. Three drafts do not guarantee three accepted tokens. Eligible requests retain trained MTP in real joint head/target batches; sampling and masked requests use AR. MTP prompt priming retains ordered hidden/token-pair chunks of at most 128 rows. The singleton override is fixed; oMLX uses an adaptive maximum of three.

Up to four requests hold active state. Admission rechecks frames queued during state allocation. Prefilling peers share real uniform windows, up to 2,048 rows per lane in the v6 profile, with independent request state. Ready decoders share matrix/router/expert work in one graph. Prefill choices are 32/64/128/256/512/1024/2048, default 128 without the profile. Status exposes actual B1–B4 widths and MTP counters.

Auto context is 8,192 tokens; the architecture ceiling is 262,144. Prompt plus output budget must fit. Auto memory is physical RAM minus `max(16 GiB, 10% of RAM)`. MemoryGovernor reservations precede trunk, MTP-head and batch workspace construction, including optional dense caches/blocked-MoE scratch; actual allocations are checked against planned sizes. Request-state allocation is also reserved before construction. Dense-cache startup runs conversion graphs when no saved artifact is selected; saved operands skip those graphs; unused pipelines and OS file pages can still affect first-request latency.

The route supports text, streaming, seeded top-k/top-p sampling (native top-k limit 32), token masks, tools, cancellation and deadlines. External DFlash, vision/image/video execution, cross-request prefix reuse and persistent state snapshots remain disabled. Large PLE tables remain mapped and demand-paged. The v5 residency request covers verified saved operands, excluding the original checkpoint and PLE. Unused vision coefficients count toward mapped memory.

**Run the native AR oracle.** Create a local token-ID array:

```sh
.venv/bin/python - <<'PYTOKENS'
import json
from pathlib import Path
from transformers import AutoTokenizer

package = Path("install/local-models/Flash-Next-oQ4e-mtp-v1")
tokenizer = AutoTokenizer.from_pretrained(
    package, local_files_only=True, trust_remote_code=False
)
tokens = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Return only the result of 13 + 3."}],
    tokenize=True, add_generation_prompt=True, enable_thinking=False
)
if hasattr(tokens, "keys"):
    tokens = tokens["input_ids"]
Path("build/flash-next/prompt.json").write_text(json.dumps(tokens))
PYTOKENS

FLASH_ORACLE_PROFILE=off FLASH_ORACLE_MAX_TOKENS=32 \
  build/flash-next/flash-forward-oracle \
  build/flash-next/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/flash-next/prompt.json build/flash-next/oracle.json
```

The oracle uses AR and 128-row prefill chunks; it does not apply launcher defaults. Prefix its invocation with desired `SPLASH_FLASH_*` settings for optimized AR routes. Output limits are 1–256, default 32, with early EOS termination. `pass:true` means finite logits, clean diagnostics and completed generation, not automatic MLX comparison or answer-quality evaluation. Command `gpu_seconds`/`wall_seconds` exclude loading, CPU sampling and report I/O.

`FLASH_ORACLE_PROFILE=command` writes ordinary command/host metadata to `REPORT_JSON.trace.jsonl`; `stage` adds dispatch timestamps and changes encoder boundaries. Stage timings are diagnostic. `FLASH_ORACLE_RESIDENT=1` remains an optional residency experiment, without a second weight copy or guaranteed physical pinning.

The math tag is `native-flash-f32affine-mlx-bf16-reductions-fast-silu-precise-sigmoid-moe-idtie-qsa-highid-v3`; `kernel_routes` separately identifies arithmetic/cache choices. QMV and QSA F32 routes are qualified numerical alternatives, not claims of bit-identical arithmetic. Primitive GPU oracles cover projections, GDN prefix rollback, QSA, HC BF16 staging, PLE and MoE. Bounded 32-token control/optimized AR generations match the installed MLX reference and each other; this is not universal equivalence.

**Measured performance.** The final normal `./splash` launcher uses local v6 defaults. Each cohort has exact 128/2,048-token prompts, greedy reasoning-off requests, zero cached tokens, all 128 requested output tokens per request, warmup and two measured waves. Hardware is Apple M5 Ultra with 80 GPU cores and 256 GiB RAM, Metal 4.1/macOS 27. Plan SHA256 is `cb3747ba6e77cbfe2d74af976163efc92bf3300f4d1a98c72c4da88bfd6badb4`.

| Prompt tokens / concurrency | Same-build v5 OFF control | Final normal v6 median | Fresh installed oMLX, adaptive MTP max 3 |
| --- | ---: | ---: | ---: |
| 128 / C1 | 100.353 | 100.282 | 89.710 |
| 128 / C4 | 152.037 | 157.127 | 112.084 |
| 2,048 / C1 | 64.359 | 66.391 | 61.995 |
| 2,048 / C4 | 78.226 | 79.676 | 77.993 |

Values are **complete-wave HTTP output tokens/s**, including prefill, queueing and transport. C4 aggregates four independent requests. Versus the matched OFF control, final v6 changes short single by approximately −0.07%, short C4 +3.35%, long single +3.16%, and long C4 +1.85%. These are small measured gains, not a broad breakthrough. Two preceding ON runs repeated smaller concurrency/long gains and a short-singleton tradeoff; retain those samples separately. Two samples per run and this deterministic workload do not establish universal speed or general model quality.

Raw reports: [final normal v6](../../build/release/flash/splash-normal-default-v6-http-performance.json), [same-build OFF](../../build/release/flash/splash-next-same-build-v5-control-http-performance.json), [fresh installed oMLX](../../build/release/flash/omlx-mtp3-saved-formats-refresh-http-performance.json). [v6 kernel-round details](flash-v6-kernel-round.md) retain both ON runs, per-head timing, disabled experiments and private synthetic bound failures. The historical [normal v5 run](../../build/release/flash/splash-normal-default-v5-http-performance.json) remains preserved.

The normal v6 launcher passed [22 bounded quality/protocol checks](../../build/release/flash/default-v6-quality-qualification.json) and [111 CPU checks](../../build/release/flash/local-profile-v6-cpu-qualification.json). All 21 completed benchmark generations match the same-build OFF control exactly. Its [fresh final idle proof](../../build/release/flash/default-v6-final-idle-status.json) matches engine 550771527669187: 49 submitted = 47 completed + two deliberate cancellations, zero failed, no active/in-flight request, healthy Metal, normal observed pressure, and allocation-ledger peak 155,472,838,656 bytes.

Per-head GPU time is approximately 9% lower than matched OFF, while cycle and budget-end phase mix also change. Private QMV and F32 proposal screens still pass zero of 17 unchanged 1% hidden/state bound cases. Their JSON omits selected-prefix provenance; keep those failures as unresolved numerical evidence. Exact verified target outputs and the bounded service checks do not assert universal proposal hidden/state equivalence. Target verification checks greedy candidates using the full target, and no additional weights or state precision demotion are introduced by v6.

[Saved-format details](flash-saved-formats.md) describe the unchanged v5 weight derivatives and observed Metal API registration; physical pinning remains unverified. Original checkpoint and oMLX preferences remain preserved. The [oMLX source](https://github.com/jundot/omlx) was inspected read-only; no GitHub updates were made.

The v2 routes add original-coefficient F32 cache selection by measured role/format/row count, direct shared-weight joint MTP, 16-row exact rollback support, context-dependent grouped attention, staged GDN prefill and vectorized Q4 expert staging. Exact CPU NEON argmax preserves finite-logit checks, signed-zero ties and lower-token-ID ties. Earlier BF16 expert caches of 32 or 64 selected experts per layer regressed matched HTTP throughput and remain inactive; v5 instead uses the separately qualified signed INT8 derivative. N128 and hybrid expert tile candidates also remain experiments. Standalone primitive speedups are not substitutes for the HTTP table.

[Round 7 HTTP qualification](../../build/release/flash/http-round7-m64-simd-qualification.json) passed all eight cases. [A real four-request joint-MTP disconnect test](../../build/release/flash/http-round7-joint-disconnect-qualification.json) cancelled one lane, preserved three full-budget survivors, reached idle and recorded one dropped joint member. [Deep verification](../../build/release/flash/deep15-prefix-state-qualification.json) passed all 64 retained-prefix/phase cases with exact recurrent, PLE and valid QSA state plus two future continuations. The [8,192-row prefill oracle](../../build/release/flash/batch-prefill-8192-m64-qualification.json) checked over one billion output/state elements, capacity tails and greedy continuations. These are bounded qualifications.

The historical local v2 profile had 23 defaults. The normal launcher passed [all eight qualification cases](../../build/release/flash/http-default-v2-qualification.json) and 37 focused CPU tests. [A long four-request deadline test](../../build/release/flash/http-round9-long-cohort-deadline-qualification.json) dropped the expiring lane and grouped priming for three full-budget survivors. Original-coefficient F32 decode MPP passed accuracy but was slower; BF16 decode MPP was slower and exceeded its accuracy bound. They remain private and inactive. The profile merger preserves explicit choices and disables only implied dependent defaults when their parent is explicitly disabled. Compiler settings remain Metal 4.1/macOS 27.0. Original model bytes and oMLX preferences remain preserved.

Separate warm **GPU decode command** measurements rose from approximately 20.74 tokens/s in [the original qualified route](../../build/release/flash/native-qualified-final.json) to approximately 43.0 in [the fused online route](../../build/release/flash/native-online-fused32.json). The same 32-token prompt's GPU prefill fell from 502.426 ms to 143.628 ms with [the dense cache](../../build/release/flash/native-dense-cache32.json). These AR command metrics exclude HTTP, sampling, admission and loading; they must not replace the table above.

The original v1 normal launcher passed [all eight HTTP cases](../../build/release/flash/http-default-profile-qualification.json): arithmetic streaming, instruction/JSON-schema handling, four concurrent JSON requests, tool call/continuation, disconnect cancellation, deadline and recovery. Its saved status records valid memory audits and real B2/B4 commands. The final build's CPU self-test and all 37 focused launcher/profile/frontend CPU tests pass. Flash serves on port 8011; the 27B service on port 8000 and isolated reference services are stopped. [The preceding direct-state qualifier](../../build/release/flash/http-direct-state-final-qualification.json) is also preserved. Repeat against an already running local server:

```sh
.venv/bin/python -B dev/benchmarks/qualify_flash_http.py \
  --base-url http://127.0.0.1:8011 \
  --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --output build/release/flash/http-qualification-repeat.json
```

Entry points: [FlashWeights](../../runtime/flash/FlashWeights.hpp), [FlashForward](../../runtime/flash/FlashForward.hpp), [FlashBatchForward](../../runtime/flash/FlashBatchForward.hpp), [FlashMTP](../../runtime/flash/FlashMTP.hpp), [FlashWorker](../../runtime/flash/FlashWorker.mm), [launcher](../../install/launcher.py), [build rules](../flash.mk) and [AR oracle](flash_forward_oracle.mm).

The historical v4 profile had 27 defaults; v5 retains those and adds three saved-format routing/residency flags. Exact n-gram I64 hashing, direct128-shard lookup and ordered history update reduce the former18-dispatch lookup to2; PLE post/injection fusion reduces7 to3, retaining every BF16 reduction and state boundary. Inline GPU argmax returns compact exact token records, while lifecycle checks and tiny accepted-prefix calculations remain on CPU.

`SPLASH_FLASH_INT8_HEAD=1` supplies original UINT8 Q8 code bytes directly to the BF16-activation matrix operator at real rows2..16, with F32 group-scale/bias correction and BF16 output. R1 and the body retain their existing routes. The canonical packed Q8 layout is already byte-addressable, so no copied code cache or conversion command is needed. This eliminates roughly2.54GB of expanded F32 vocabulary operands. Group-factored reduction is a qualified numerical alternative, not a universal bit-equivalence claim. Original checkpoint files stay unchanged.

[Previous v4 GPU/precision summary](../../build/release/flash/gpu-precision-final-summary.json) records the active artifacts, original source identity, memory, bounded numerical proofs and matched HTTP results. The final normal launcher passed [all8 HTTP cases](../../build/release/flash/http-default-v4-zero-copy-qualification.json),37 focusedCPU tests and its CPU protocol self-test.

`SPLASH_FLASH_GPU_PREFILL_COPY=1` remains optional: its producer-command copy passed byte/ownership/alias guards and full-model continuation checks, but matched HTTP did not show a clear improvement. BF16 attention probabilities and dense INT8 body projections remain private experiments after marginal speed or strict-accuracy failures.
