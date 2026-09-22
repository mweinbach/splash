# Live tuning checkpoint, September 22 11:30 UTC

Goal remains active: canonical uncached B1 prefill above 4K and substantially faster standard/MTP decode, including B2/B4.

Root GPU is idle. Session 47362 exited before loading the model because the paired quality launcher supplied both `--local-package` and an external `--tokenizer`. Server argparse requires the bundled package tokenizer. The V3 failed attempt, summary and log are preserved. The quality owner is preparing a fresh V4 launcher and CPU regression check; no task results or candidate batch performance are qualified yet.

Best qualified singleton: Q4 rowpair, median prefill 4002.1633 tok/s (one of three trials below 4K), native decode 53.2260 tok/s. Composite HC+guard provides all three prefill trials above 4K, median 4012.2604, native decode 52.6275. Canonical original22 outputs match their respective controls with no new regressions.

Batch two-pass alternative has actual B2/B4 intended-reference full-state and three-step AR proof, receipt 23f15fc2…; original old-SG8 EVERYROW comparison remains failed at 116/2048 rows. Fresh standard and MTP task comparisons are required before mode-specific service timing. The sealed service plan is `build/batch-twoPass-native-service-plan-sep22-v2/CPU_READY.json` (c54d75b0…).

Finite-summary component is closed: 478 checks passed but the inclusive chain was 2.1155 times slower. A separate CPU component is authorized to move duplicate-ID validation onto thread 0, preserving floating math, native MPP, scan, geometry and diagnostic meaning. No worker integration or GPU run yet.

Root alone runs the GPU; independent source reviews proceed in parallel. No production promotion, tracked-source changes, or memory writes authorized by this checkpoint.

11:39 UTC: Root session 72403 is active on fresh paired quality suite V5. CPU_READY c2441607…, command 5aa9d550…, config cde206f7…, independent review 6461a1d2… all checked; 32 Root pins match. Summary is `build/release/flash/sep22-batch-twoPass-original22-paired-standard-mtp3-B4-B2-suite-v5.json`. All pinned LIVE task sources and copied/runtime sources are frozen until terminal. V3 failure and unused V4 are retained.

11:45 UTC: Old standard B4/B2 completed all22 plus mixed tasks, evidence valid and 20/22 passes per lane; worker unloaded RC0 without SIGKILL. New standard B4 is running with matching outcomes and no native coverage errors so far. Remaining mode comparisons are pending.

Queued after 72403: existing adaptive MTP0..3 on the identical qualified Q4 worker, `build/mtp-adaptive-Q4-sep22-root-v3/run-root-adaptive.sh`. Command 15f14d0c…, CPU_READY 45a2fee7…, independent review 36c8d2d9…. Private driver intentionally omits forced draft depth3 and selects ADAPTIVE1, preserving all benchmark bodies/budgets/grading, actual H3 composite coverage and mixed-depth policy checks. No kernel rebuild or runtime result yet. Adaptive depth0 includes committed head folding and is not head-disabled standard decode.

Source-only next plan: cache full host scratch/immutable admission for unchanged exact strong MetalBuffer views, with full guard fallback for any different views and fresh per-graph token. No implementation authorized yet. Physical RHS layout, axis permutation, Q5 composition and finite-summary losing probes remain closed.

11:48 UTC: Quality 72403 terminal, all22 at both standard widths complete. OLD20/22 versus NEW19/22; every lane on `python_merge_counts` newly emits `result.get`, violating the explicit no-attributes prompt. Both formal comparisons are evidence-valid but no-new-regressions false. Both workers unloaded RC0 without SIGKILL. No standard grade receipt or candidate performance qualification; MTP mode was not reached. Source/threshold/grader untouched. Separate MTP-only paired campaign CPU preparation is authorized because that mode remains unmeasured; it must preserve all22 and cannot inherit standard quality.

Root sole GPU session 97585: existing Q4 adaptive MTP V3 trial launched after clean 72403 teardown. All87 pins verified. Adaptive adapter/program/commands are frozen until terminal.

11:50 UTC: Adaptive 97585 terminal/unloaded RC0. Three actual native rates 52.1102 / 59.9730 / 42.5549 tok/s; median 52.1102, below qualified fixed3 53.2260. Prefill median 4000.4561, one trial 3995.10. Normal source coverage valid, but semantic helper failed before first saved task with `ModuleNotFoundError: dev.benchmarks.flash_http_performance`; task quality unqualified. Executed V3 sources were archived with exact pins before a bounded loader regression fix. No whole adaptive rerun authorized absent a distinct reason.

11:52 UTC: Root sole GPU session 29021 is running MTP3-only paired batch quality, `build/batch-twoPass-original22-mtp3-only-quality-sep22-root-v1`. CPU_READY cfbc04a5…, command b330d7b7…, config 39eea8f3…, independent review b8737ae4… all checked with Root source pins. Fresh summary `build/release/flash/sep22-batch-twoPass-original22-paired-mtp3-only-B4-B2-suite-v1.json`. All pinned task/live/helper/copy/runtime sources frozen. Standard remains 19/22 failed; this explicitly separate mode must establish its own full22 comparison.

Host guard work: sorted immutable96 interval index PLAN independently approved, portable header + extracted-old decision parity and CPU metadata microbenchmark only. No Store/Worker integration yet. Null/zero/overflow/non-indexable source/query keeps full literal96 guard fallback.

11:57 UTC: MTP-only 29021 old B4 full22=19/22 (merge_counts attribute failure), old B2 full22=20/22. Both evidence-valid with native/head/source coverage, same worker unloaded RC0/SIGKILL false. New MTP worker is loading. No actual candidate grade or speed yet.

Duplicate-TID0 component is compiled and frozen for Root after 29021: `build/expert-r1-duplicate-tid0-sep22-component-v1/run-root-component.sh`. Registration cd5fa0e9…, CPU_READY 9afb5ab4…, exe 00e9b8e3…, library 94599914…. Root verified 661 listed (364 unique) source/object/code-artifact pins. Actual original a0cd native control, changed only duplicate-ID thread0 predicate placement; final compiled independent receipt is pending. No numerical/worker/performance qualification inherited.

12:01 UTC: MTP-only 29021 terminal/clean unload. B4 paired evidence-valid/no-new-regression true: 19/22 old and new. B2 evidence-valid/no-new-regression false: old20/new19 due prohibited `.get`. No generic MTP2+4 grade. Controlled B4-only normal benchmark authorized with distinct strict schema and source/grade/width checks; other modes/widths remain rejected and runtime is not promoted. Actual Root small receipt `sep22-batch-twoPass-MTP3-B4-only-Root-grade-admission-v1.json`, SHA f5c725e4…. B4 comparison SHA060ae978…, B2 comparison cd923bb1…. Full preexisting both-width service gates remain unchanged.

12:03 UTC: Duplicate component Root99532 DONE numericalPASS413checks/51guards, raw/scaled F32/BF16 exact, CTA census100/400, >=150ms warm each/18 balanced pairs, Gov0/owner0/backenddestroyed. GPU old0.0712916371 -> new0.0725000282ms: 1.69% slower; wall0.2394785 ->0.26475ms. CLOSED: no whole standard integration or promotion. Actual report SHA e696b15a…c3d80. Root GPU idle while B4-only service wrapper and interval-index CPU gates finish.
