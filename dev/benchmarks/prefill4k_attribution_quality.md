# Long-prompt expert-inventory semantic qualification

The frozen plan `build/release/flash/prefill4k-semantic-plan-v1.json` contains
22 supplementary semantic tasks, separate from the existing eight-case protocol
and lifecycle suite. Its SHA256 is
`a28041a2c487191a94aa9a375030b1a7b4294f313a5fc4674dbebaf9347d8aac`.

Tasks cover four exact arithmetic answers, four extraction/logic outcomes,
four strict JSON values/types, five Python functions with deterministic
held-out vectors, and five exact-copy tasks. Copy payloads occur early or in
the middle of 2K context, in the middle of 4K, and early/late in 8K. Every
prompt is actually rendered with the local tokenizer and real frontend
message/schema normalization. Eighteen prompts have exactly2,048 input tokens,
one2,049, one4,096 and two8,192:59,393 real prompt tokens overall. All requests
exercise large-row prefill. The2,049 case also includes a one-row raw tail
under the normal2,048 chunk policy.

Bodies, full token IDs/U32LE hashes, request-body hashes, answer properties and
budgets are fixed for every inventory. Temperature0, reasoningoff, seed0,
MTPdepth3, teacher-cache-only on, PLESSD on, idlemaintenance off, zero prefix
reuse and one outstanding request are common. The8K budgets require8,288
context tokens; use16,384 consistently. A report records actual chunk/context
settings and root-supplied runtime/metallib hashes; comparison requires them
identical across Top64, Top128, Top256 and Full512. The CPU auditor regrades saved
responses against the frozen answer properties and recomputes policy gates,
cache coverage and native terminal deltas from saved snapshots; passing report
summary booleans cannot replace those checks.

The runner contacts an existing dedicated server only. It never starts,
builds, changes or stops a server. Root serializes models/measurements and
verifies unload between stores. The runner checks the exact source fingerprint,
manifest/plan hashes, inventory count, mapped bytes and numerical-alternative
marker. It requires fresh idle/healthy status and exactly one submitted/completed
native request per task. Prompt usage and native main input counts must match
the frozen rows; actual completion counts/finish/usage are saved. Generation
text/hash differences are diagnostics, not task regressions.

```sh
.venv/bin/python dev/benchmarks/prefill4k_attribution_quality.py measure \
  --plan build/release/flash/prefill4k-semantic-plan-v1.json \
  --runtime-build build/prefill4k-fullcache \
  --expert-store install/local-models/Flash-Next-int8-experts-top64-v1 \
  --inventory 64 --base-url http://127.0.0.1:8011 --label top64-current \
  --output build/release/flash/prefill4k-semantic-top64.json --run-root-gpu
```

Repeat on the same private runtime with the verified Top128 store, the
certified Top256 subset at
`build/prefill4k-fullcache-artifacts/int8-experts-frequency256-v1`, and then
`build/prefill4k-fullcache-artifacts/int8-experts-all512-v1`, inventory512.
Use fresh report paths and identical unrelated flags.

```sh
.venv/bin/python dev/benchmarks/prefill4k_attribution_quality.py compare \
  --reports build/release/flash/prefill4k-semantic-top64.json \
            build/release/flash/prefill4k-semantic-top128.json \
            build/release/flash/prefill4k-semantic-top256.json \
            build/release/flash/prefill4k-semantic-full512.json \
  --output build/release/flash/prefill4k-semantic-comparison.json
```

## Actual cache use and completion

The private overlay exposes
`persisted_experts.graph_counters.{gate_up_graph_calls,gate_up_graph_rows,
down_graph_calls,down_graph_rows,encoded_hit_dispatches,encoded_miss_dispatches,
full_inventory_graph_calls}`. Scope is explicitly graph construction, not GPU
completion; no inputs or tokens are collected by these counters. The runner
computes physical large-row main windows from the actual scheduler policy.
Every window requires48 gate and48 down constructions with matching real rows,
and96 encoded hit dispatches. Full512 requires no encoded miss dispatches and
96 full-inventory phase calls per large window. Top64/128/256 still encode96 guarded
miss dispatches. Successful native completion and a fresh idle state prove
command completion separately. A root-run completed-command trace should also
verify the Full512 graph binds no Q4 miss pipelines/raw miss operands.

Missing counters fail full qualification. `--allow-inferred-cache-coverage` is
an explicit screen only; unavailable actual counters keep final validity false.
An explicit `--case-id` subset is also incomplete. Under2,048-row chunks the
whole plan has30 main commands,29 large windows and1 tiny tail, producing
1,392 gate and1,392 down constructions and2,784 hit dispatches. Teacher calls
are checked per MTP-eligible request against all actual adjacent prompt pairs;
constrained JSON requests retain their original AR cohort.

## Strict outcome grading

Arithmetic/extraction/copy must equal the complete expected trimmed answer;
containing the right number somewhere is insufficient. JSON rejects duplicate
keys, extra keys, wrong values and wrong types, including bool/integer and
float/integer substitutions. Case/order/punctuation and internal newlines
matter for copy tasks. The five functions are sum_even, clamp_values,
prefix_sums, longest_run and merge_counts. Hidden vectors cover empty inputs,
duplicates, negatives, boundaries and larger bounded lists.

Pure-function tests require unchanged input arguments. clamp_values must return
a fresh list, including empty or already-clamped inputs.

Generated Python is accepted only as one requested plain function, optionally
inside one Python code fence. AST validation rejects imports, attributes,
dunders, annotations/decorators/defaults, helper functions, arbitrary calls and
operations outside a narrow pure-function subset. A separate isolated
`-I -S` Python process receives JSON stdin, minimal builtins/environment and
closed inherited descriptors. CPU1second, wall2seconds, bounded input/output,
file/core0 and FD16 restrictions apply. On macOS large reserved virtual address
space requires virtual-headroom AS/DATA limits; a native resident-memory
watchdog checks128MiB every10ms and may overshoot between samples. The evaluator
is for restricted fixtures, not general code execution or a complete OS sandbox.
Children/pipes are killed/reaped/closed on exceptional paths.

CPU verification:

```sh
.venv/bin/python -m unittest dev.tests.flash.test_prefill4k_semantic_quality \
  dev.tests.flash.test_prefill4k_attribution_quality_python
```

Ten semantic/cache-coverage tests and18 restricted Python evaluator tests
passed. The frozen plan is CPU-only; no model generation has yet qualified
these22 tasks. Coefficient/F64 certificates, GPU numerical bounds, memory
admission, the eight protocol/lifecycle cases and real performance controls
remain independent requirements. This finite task suite supplies no general
quality guarantee or teacher-forced likelihood metric.

## Comparison with preexisting baseline failures

Comparison `valid=true` means complete, independently audited evidence.
`full_plan_coverage`, `all_tasks_pass` and `no_new_task_regressions` are separate
outcomes. A complete baseline with genuine task failures can still be compared;
its score and failure IDs remain explicit. The comparison identifies new
regressions, persisting baseline failures and resolved baseline failures. Exit
zero does not establish that every task passed. Task grading is unchanged.

The first same-build Top64 generation run is19/22: signed arithmetic produced
-695 instead of-691, inventory arithmetic134 instead of96, and merge_counts
used .get despite the frozen no-attributes constraint. The other four Python
functions passed. CPU regrading reproduced those failures without model reruns
and found no coverage/protocol errors. No attribute rewrite or checker
relaxation is justified by this explicit prompt.

## Private all-row Full512 numerical target

The separate private all-row runtime changes target decode/verifier and tiny
prefill arithmetic as well as large prefill; it is an explicit numerical
model derivative. It leaves the trained MTP implementation intact. Use the
same frozen tasks and common sampling settings, but identify the target
explicitly with `--target-derivative-sha256` and its private runtime path.
Current expected derivative identity is
`2e858faa201554642a443d48d38b7302159fe1a811c028a895454cc8ce5c073d`.
Original source/layout, Full512 manifest, exact original-GPU omission geometry,
and all-row identity markers are separately checked.

When `identity.target_all_rows_full512` is true, the seven
`large_row_*` graph counters establish the same48-layer large-prefill coverage
as ordinary runs. General graph counters are retained, and general-minus-large
records additional small target projections: decode/verifier and tiny prefill
tails. These remain construction counts; completion and idle prove successful
request execution separately. Neither general counts nor construction counts
are described as physical GPU instructions.

```sh
.venv/bin/python dev/benchmarks/prefill4k_attribution_quality.py measure \
  --plan build/release/flash/prefill4k-semantic-plan-v1.json \
  --runtime-build build/prefill4k-allrows-full512 \
  --expert-store build/prefill4k-fullcache-artifacts/int8-experts-all512-v1 \
  --inventory 512 \
  --target-derivative-sha256 2e858faa201554642a443d48d38b7302159fe1a811c028a895454cc8ce5c073d \
  --base-url http://127.0.0.1:8011 --label allrows-full512 \
  --output build/release/flash/prefill4k-semantic-allrows-full512.json \
  --run-root-gpu
```

Comparison across this intentional different binary requires explicit
`--allow-runtime-change`; common execution policy must still match. The report
exposes the binary change and derivative identities. Default comparisons still
require identical runtime hashes. Task grading, existing failures, source error
certification and no-new-regression checks remain separate and unchanged.
