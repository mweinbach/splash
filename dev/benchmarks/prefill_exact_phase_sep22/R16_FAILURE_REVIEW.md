# R16 prefill preservation failure

The accounting-fixed comparator found a genuine tensor mismatch in `tail-r16.output.bin` at byte 153 after all earlier prescribed checkpoints passed. Its native failure report exists at `build/release/flash/sep22-trunkpref-compare-v3.json.failure.json`; no success report is expected after an exception. It records `tail-r16`, completed backend destruction, all six allocation guards true, zero denied reservations and 24,717,044 bytes of reservation slack. At failure, 24/28 unique frames and all 12 repeated-body frames had completed. R16 state and future output/state were not yet compared.

The offset follows directly from the literal frame format, without reading tensor payload:

```text
16-byte magic + 8-byte metadata length + 83-byte metadata
+ 8-byte plane count + 8-byte hidden label length + 6-byte label
+ 24-byte dtype/live/physical declarations = 153 bytes
```

Thus the mismatch is hidden payload byte zero, BF16 word zero, before vocabulary or greedy records. The header-only parser uses unbuffered FileIO, bounded reads and payload seeks; its synthetic trap checks this location without any model data access.

Two independent source audits identified the changed route. Teacher's 508-map F32 cache contains generic dense prefixes even when `flashFloatDenseSmallRowsPolicy()` rejects their tile. The selective branch therefore executes raw `addAffine()` and returns. Phase's 296-map union omits those rejected prefixes. At rows below 16, absent prefixes still reach raw affine because BF16 small-row mode is off. At R16 they instead reach `denseCache && rows >= 16`, changing coefficients and reduction to cached BF16 whole-K. All 96 shared-expert gate/up projections are concrete affected roles: they are in the parent default cache, absent from the selective policy table, and use generic projection at R16.

The phase actor's final worker-v5 retains the 296-map backing and original 118-owner F32 residency subset, while preserving the parent's 508-prefix selector membership for explicit Prefill. Qualified tiles use the same cached producer and require backing; rejected tiles retain the original raw affine producer. The context also propagates through generic HC fallback calls. Decode and verification context stay separate. Worker arithmetic helpers, dtypes, GPU coefficient allocations and metallib are unchanged; actual byte preservation still requires Root qualification.

The final comparator is `build/trunkpref-exact-phase-sep22-v6`, pinned to worker CPU seal `6b307d3e48f918f890ba27a6039d5ae1a84c958f890e899f73c2e9c7533e7326`, oracle `be15196ec31e7845482cf7c3df5a3e20b489a3d947708184186c84cd54bdcb05`, shared metallib `bb09bf88bb53a8b6e9bfc5254068c16942bb0913784ff1c672d810f13c6eb6f0`, and command `55a7db69504feab93c7141d95c5f5f2d444d6d174de8762070fddb3e37221717`. It reuses the unchanged parent export and targets fresh report `sep22-trunkpref-compare-v4.json`.

Native diagnostics now retain current frame/plane/section, exact offsets and word width, completed/pending labels and counters, allocation/governor telemetry, and teardown state across unwinding. Root-runtime differing bytes are captured only by the running oracle. Early argument/role errors, generic native exceptions and publication errors have reporting paths; the launcher covers preflight errors, failed launches, nonzero exits and signals. CPU checks pass for multi-chunk last-byte, metadata and extent context, frame byte comparison, the independent W8 census and the unchanged spill bound.

Source/framing reviewers and builders performed no GPU calls or model/input/export tensor payload reads or hashes. Trained-head, Worker-residency, decode and performance proof remain outside this TRUNKPREF oracle.
