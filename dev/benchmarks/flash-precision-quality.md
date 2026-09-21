# Whole-model checks for persisted precision derivatives

`flash_precision_quality.py` freezes deterministic prompts and checks task
outcomes through an existing native Flash HTTP service. It never starts,
restarts, builds, or reconfigures a service. Only the root coordinator runs
`measure`, serializing the original and converted model policies.

The CPU-frozen plan is
`build/release/flash/persisted-int8-quality-plan-v1.json`, with content hash
`a4135cc4b5d8bf7c13e22c5dfdad68ece414b3dd69bdfded63a77e007aefb466`.
The same plan must be used for both policies. It records the original
checkpoint identity, complete token IDs and their little-endian hash, and
request-body hashes excluding only the runtime model name. The plan and each
report are fresh artifacts; the tool refuses to overwrite them.

The plan includes the existing eight HTTP protocol cases unchanged, then 14
supplementary cases. The supplements issue 17 requests: four arithmetic
questions, factual water-cycle prose, unconstrained and schema-constrained
JSON, Python code, three 2K-context retrieval positions, four concurrent
retrieval lanes, a forced tool call, and a tool continuation. Every
supplementary prompt has exactly 512 or 2,048 tokens. These prompts exercise
the large-row prefill policy; the original protocol cases retain their short
prompts and cancellation/deadline behavior.

Prompt generation uses the real frontend's CPU message/tool/schema
normalization. It includes the system instruction injected for JSON schemas,
coalesces initial system messages, converts tool-history argument strings to
objects, and suppresses template tools for `tool_choice=none`. Retrieval
needles appear at approximately 2.3%, 50%, and 97.7% of the user-content token
range, so the final-position case does not accidentally become an early
needle followed by padding.

```sh
.venv/bin/python dev/benchmarks/flash_precision_quality.py measure \
  --plan build/release/flash/persisted-int8-quality-plan-v1.json \
  --label qualified-v4 \
  --base-url http://127.0.0.1:8011 \
  --output build/release/flash/persisted-int8-quality-baseline-v1.json \
  --run-root-gpu

# Root stops the baseline and starts the converted policy, then runs the same
# command with a candidate label and a new output path. --required-route can
# additionally demand a specific substring in the actual kernel route.

.venv/bin/python dev/benchmarks/flash_precision_quality.py compare \
  --baseline build/release/flash/persisted-int8-quality-baseline-v1.json \
  --candidate build/release/flash/persisted-int8-quality-candidate-v1.json \
  --output build/release/flash/persisted-int8-quality-pair-v1.json
```

The original checkpoint source must match the plan. Native source/kernel
identity remains stable within each run. Each supplementary request checks
HTTP/SSE completion, token accounting and frozen prompt rows, zero cached
tokens, reasoning disabled, and coherent terminal completion. Arithmetic,
retrieval, JSON values/types, and tool arguments have exact task answers.
Four concurrent retrieval lanes have distinct expected codes. Python output
must parse as a single pure function and pass held-out empty, duplicate,
negative, and larger inputs in a separate Python process with a one-second
CPU limit and a two-second parent timeout. Code checking permits only a
narrow subset of Python; it is a fixture evaluator, not a general security
sandbox.

The factual-prose checker requires the requested water-cycle concepts and a
reasonable word count. A human must read its recorded outputs for factual
contradictions and coherence. These finite fixtures are not a general quality
evaluation. Pair reports flag newly failed tasks and retain both policies'
answers. Exact text equality is reported separately and is not required for
a numerical alternative when both outputs satisfy their task.

Current native HTTP offers no usable token log-probabilities, and the normal
forward oracle writes greedy IDs rather than logits. The plan/report explicitly
mark teacher-forced likelihood/logit metrics unavailable; they must not be
inferred from output similarity. Quality-run timing is observational and must
not replace the matched full-budget HTTP performance benchmark.

For a bounded first screen, `--skip-protocol` and repeated `--case-id` can
select supplementary cases. Reports explicitly record skipped cases and
`full_plan_coverage=false`; a partial screen is not a full qualification.

CPU-only checker validation:

```sh
.venv/bin/python -m unittest dev.tests.flash.test_flash_precision_quality
```
