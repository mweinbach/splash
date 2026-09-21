Private original trained-head attribution

This fixture profiles the checkpoint's original trained MTP head with actual target premixer features. It does not change production math, backend behavior, routing defaults, source weights, or the local profile. Root alone executes GPU inference. Building, `--help`, wrapper dry-run, and the analyzer's tests are CPU-only.

Frozen v2 artifact: `build/flash-trained-head-profile-v8-v2/flash-trained-head-profile-v8-oracle` with adjacent `splash.metallib`. All 44 host objects were compiled freshly into that private directory with `sizeof(CommandTiming)==200`, hybrid Metal 4.1 configuration `26ff33bbeeaa8d04`. `frozen-build-source-manifest.json` records source hashes and artifact hashes. `frozen-environment.json` contains the accepted v7's 36 flags plus both qualified saved-store directories. It uses the existing v6 token fixtures; those exact benchmark tokens have unchanged contexts and the original fixture provenance remains explicit.

An R1 proposal consumes the actual final target premixer feature and target anchor after teacher-priming the original head with every real adjacent prompt pair. R4/R8 committed folds consume the original target's real greedy continuation: each row holds an actual previous target premixer feature and its next committed target token. They use `FlashMTPLogits::Last`, matching a final committed chunk before drafting. They represent full genuine committed chunks; they do not assert that every requested speculative window achieves that acceptance count. B4×R1 uses four independent real benchmark lanes, their own correctly primed QSA state, and the production joint head's cached vocabulary from the target.

The measured original graph gains no diagnostic dispatches. No state hashes run before its timed call. After completion, full BF16 premixer hidden and vocabulary output hashes plus greedy predictions are recorded. For command/stage/dispatch modes an uninstrumented original replay runs first, logical length is truncated to its original offset, and the sampled replay must produce identical full output hashes and exact greedy predictions. This normal replay precedes the sampled call and therefore warms its operands; its duration is diagnostic, not a matched HTTP baseline. Original QSA truncation is explicitly designed to replace stale token rows and any newly completed compression block.

The command profile classifies exact first two non-embedding projection roles as `fc_embedding` and `fc_hidden`, excludes padding dispatches, identifies vocabulary output by its exact full BF16 row extent, and splits HC/QSA/MoE/shared/dense/greedy/other families. Stage mode adds an encoder boundary per dispatch, while dispatch mode adds counter barriers. Neither is an uninstrumented performance baseline; unsupported sampling has no fallback. The report keeps full-command GPU duration separately from the sum of timed dispatch intervals.

First Root GPU invocation:

```sh
.venv/bin/python -B dev/benchmarks/flash_trained_head_profile_v8_run.py \
  --mode stage --context 2048 --phase proposal --rows 1 \
  --warmup 1 --repeats 1 \
  --report build/release/flash/trained-head-v8-v2-2048-r1-stage.json --run
```

Use `--phase committed_fold --rows 4|8` or `--phase joint_proposal --rows 1` for the other bounded cases. Omitting phase/rows/context runs all eight geometry/context cases. Every report path must be fresh; the wrapper clears inherited experimental Flash environment settings and records a complete invocation sidecar. Only use `--run` after the coordinator has serialized all GPU work.

Build command:

```sh
make -j8 -f Makefile -f dev/benchmarks/flash_trained_head_profile_v8_oracle.mk \
  BUILD=build/flash-trained-head-profile-v8-v2 SPLASH_PRECISION=hybrid \
  flash-trained-head-profile-v8-oracle
```

Do not rebuild the frozen artifact after GPU qualification; select a fresh BUILD directory for subsequent edits.

The initial v1 GPU R1 fixture passed full hidden/vocabulary/greedy normal-versus-sampled parity, but its family summary used default six-digit stream precision while the raw ProfilingJson numbers retained full precision. The strict CPU analyzer correctly marked that report invalid. V1 artifacts/report are retained. V2 fixes summary serialization at max_digits10 and further splits original attention and router projection roles by their exact source binding extents; the analyzer tolerance is unchanged.

Root's v2 R1 proposal at context2048 completed and passed full normal-versus-sampled hidden/vocabulary output hashes and greedy predictions. The strict CPU analyzer reports `valid=true`, `attribution_complete=true`, with40/40 valid dispatch timestamps and exact family reconciliation. The recorded true anchor was16 and head prediction198; hidden SHA `7b1b696301f5822282f2ae549d984d05f306170300faf34e8d0e43b3f2eca138`, vocabulary SHA `827be23faf2306e11e6ca62f2292e41a9a2094d6a9d099dd96c52fe76d591e1b`.

Measured original sampled command:1.874583ms GPU; sum of timed dispatches1.700706ms. Its preceding normal replay was1.619ms, illustrating scheduling perturbation rather than a performance improvement. The largest family is the original raw-Q8 vocabulary projection at0.771750ms (45.38% of timed intervals,41.17% of the sampled command). HC totals0.222707ms; QSA0.134166ms; original attention q projection0.128500ms; MoE0.097582ms; attention o0.078042ms; shared expert0.057541ms. `fc_hidden` is0.017667ms after the accepted QMV route, so its next optimization has a much smaller available budget than vocabulary or HC.

Evidence: `build/release/flash/trained-head-v8-v2-2048-r1-stage.json` and `...json.trace.jsonl`; independent CPU summary `build/release/flash/trained-head-v8-v2-2048-r1-stage-summary.json`. One initial-cycle sample does not establish a steady or HTTP throughput gain. Remaining geometry/context cases are prepared but only become GPU-qualified when Root runs them.

Root's v2 joint B4×R1 at context2048 also passed full normal-versus-sampled output parity and strict CPU attribution:57/57 valid timestamps. Normal replay was3.550ms, sampled command4.0935ms, timed dispatch sum4.130797ms; the signed −0.037297ms remainder is retained. The BF16 cached vocabulary matrix dispatch consumed1.596084ms (38.64% of timed intervals). Attention q projection0.517500ms, o0.254209ms, HC0.321125ms, and MoE0.158917ms follow.

QSA totaled0.890177ms, but this contains a single first-lane reduce outlier0.374583ms. The other three reductions with identical dispatch geometry were0.015167/0.015792/0.015500ms. All four online MPP partitions were stable0.099791/0.100417/0.100875/0.100625ms. Do not infer0.421ms of intrinsic reduction math from that aggregate; repeat uninstrumented and sampled measurements are needed to separate dependency/scheduling/cache effects. Independent CPU summary: `build/release/flash/trained-head-v8-v2-2048-b4r1-stage-summary-cpu-agent.json`.

Private real R1 vocabulary-input export is a separate build: `build/flash-trained-head-input-export-v8/flash-trained-head-input-export-v8-oracle`. Its private copied head source adds only a diagnostic getter for completed shared `Scratch::Mixed` BF16[1,2560]. The original head's final HC mixer has already normalized/mixed this input before its raw-Q8 vocabulary projection. The fixture writes this exact input after command completion and before any next head call, with no GPU dispatch added. Snapshot headers are included by every freshly compiled host object; production head sources and API remain unchanged.

```sh
.venv/bin/python -B dev/benchmarks/flash_trained_head_input_export_v8_run.py \
  --mode normal --context 2048 --phase proposal --rows 1 --warmup 0 --repeats 1 \
  --report build/release/flash/trained-head-input-export-v8-ctx2048.json \
  --capture-directory build/release/flash/trained-head-real-inputs-v8 --run
```

That Root invocation writes exact5120-byte `head-length2048-rows1.bf16` and same-prefix JSON provenance containing input hash, original vocabulary/hidden output hashes, prompt hash, real target feature hash, and anchor/greedy tokens. Both directory and report must be fresh. Export fixture input construction and head math are unchanged; its optional diagnostic file I/O is outside the measured head call.

Root's context2048 real R1 input export completed. Independent CPU qualification passed9 checks: exact5120 bytes, all2560 finite BF16 words, metadata hash, zero diagnostic GPU dispatches, and original full hidden/vocabulary/target-feature hashes plus greedy198 identical to the separately qualified original v2 head fixture. Captured input SHA `b06f531a8f2b14c1464db67869d484e42fe84d2e45cad936700984376b4f518f`. Validation artifact: `build/release/flash/trained-head-real-input-v8-cpu-validation.json`. This is a real post-final-mixer vocabulary input for the independent original-Q8 R1 candidate screen.

The private joint original-Q8 unsigned-code MPP screen is frozen separately at `build/flash-trained-head-joint-q8-v8-v2` (44 fresh host objects, unchanged original production metallib). Root ran the context2048 B4×R1 initial screen with six restored normal AB/BA pairs. It **failed the unchanged per-lane full-vocabulary L2≤1e-4 gate**: each-lane relative L2 was0.0020188364–0.0020585102. Full premixer BF16, complete five-plane QSA buffers, and GPU/independent-CPU greedy IDs were exact in all six pairs. Report status remains `accepted=false`, `numerical_rejection=true`, exit2; no other context or future continuation was run.

The rejected numerical alternative had median command GPU3.482313ms original versus2.770104ms candidate (20.45% lower duration), and median head API3.888687ms versus3.181146ms. These are **unqualified timings** and do not support default promotion or a service-speed claim. The original route consumes BF16-rounded dequantized cached vocabulary weights; unchanged original-code MPP instead factors F32 scale/bias outside each code dot, producing different coefficients/rounding. The strict numerical gate correctly distinguishes them despite the same observed greedy IDs.

Evidence: `build/release/flash/joint-head-q8-v8-ctx2048-screen.json`; independent CPU rejection/timing summary `build/release/flash/joint-head-q8-v8-ctx2048-rejection-cpu-summary.json`. A future coded-Q8 route can reconstruct precisely the original BF16 coefficient words into cooperative/threadgroup blocks, then use BF16 matrix operands. It must separately qualify accumulation order against the current dynamic-K2560 MPP operation; exact weight reconstruction alone is insufficient to declare matching outputs.
