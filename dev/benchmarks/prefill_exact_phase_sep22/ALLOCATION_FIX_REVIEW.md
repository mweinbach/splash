# Comparator reservation correction

Root's comparator-v2 stopped after constructing the phase trunk and before any byte comparison because its arena guard detected an insufficient diagnostic plan. The completed teacher export remains valid: 28 frames, 14 checkpoints, 3,355,376,229 spilled bytes, with the parent backend destroyed.

Three independent CPU source audits identified the same omitted category: `dense_w8a8_sep21::Cache::plannedBytes()`, exactly **1,890,975,744 bytes** for 84 projections and 168 separately rounded code/scale bases. The QKV category contributes 945,487,872 bytes, Z 567,410,688, and QSA-Q 378,077,184. Forward construction charges Cache and Workspace in its measured allocation ledger; `workspacePlannedBytes()` charges only the 12,615,680-byte activation Workspace. The production Worker already adds Cache separately at its startup planner. The diagnostic now mirrors that term using `requiresCache(maximumRows)`, including when kernel selection is disabled. Sealed Forward/Worker planning and arithmetic are unchanged.

The parent figures reconcile exactly:

| Quantity | Bytes |
|---|---:|
| Old diagnostic reservation | 146,398,183,424 |
| Missing coefficient cache | 1,890,975,744 |
| Corrected reservation for those parent figures | 148,289,159,168 |
| Measured parent workspace plus one state | 148,264,442,124 |
| Conservative margin remaining | 24,717,044 |

This explains the former failure; it is not an assumed phase allocation result. F32 counts remain fully charged: the phase cache contains 296 tensors / 12,097,945,600 payload bytes, including all 178 transient-only maps. F32/BF16 residency pruning does not remove their allocation charges. QSA bulk/two-pass and the W8 activation workspace were already counted; HC/FMA add no allocations.

The final fresh comparator is `build/trunkpref-exact-phase-sep22-v4`. It uses the unchanged phase-v3 worker CPU seal `cb44ee7757168b07779a085f65b627897ce6b5e86562bed123658b7984c15a66`, shared metallib `bb09bf88bb53a8b6e9bfc5254068c16942bb0913784ff1c672d810f13c6eb6f0`, and its own 283-source / 53-object closure. Oracle SHA is `a6e571b6ba44cdf274bc5297a643a9c32d8e252445b2f9e9a64ad867207e3d65`; command SHA is `2158a2b6e9a1772e6eb29c700aa511b0c8f44961ad24d72d6b858a6bf3dfc34f`.

The corrected native diagnostic records each planned category and the initial model ledger, target delta, Forward workspace, state delta, total live ledger, cache census, shortfall/slack, and governor snapshots. It publishes reservation/target/state checkpoints before assertions and includes the same breakdown on failure. Six guards retain category, reservation and state bounds. Backend lifetime ordering now records completed destruction accurately during exceptional unwinding. The in-flight reservation snapshot is labeled separately from committed/current snapshots.

Independent reviewers confirmed the new equality guard between target delta and Forward's workspace is valid for this sealed constructor: pre-counter PLE initialization and immutable weight getters only retain existing handles and CPU metadata; they allocate no Metal buffers. They also confirmed the correction charges the coefficient cache once and preserves all snapshot frames, labels, schedules and common policy. All 38 shared numeric Prefill flags still match the exporter, so no export repeat is needed. No budget or arena guard is bypassed.

CPU checks pass, including an independent geometry-based W8 census and its constructor-capacity gate, streaming byte gates, state formula and unchanged spill bound. Actual phase byte preservation remains a Root-owned run. Trained-head and Worker-residency proof remain explicitly outside TRUNKPREF scope.

Reviewers: `/root/decode_kernel/pref_oracle_snapshot_review` and its script/constructor audit, plus `/root/decode_kernel/bf16_dependencies`. They performed no edits, GPU calls, model/input payload reads or hashes.
