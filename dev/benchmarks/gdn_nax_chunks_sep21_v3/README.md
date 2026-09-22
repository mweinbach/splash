# Private guarded GDN W/U T32 matrix experiment v3

This source/build is isolated from sealed T16 v1. Root measured T16 SG8
1.63283 ms versus native 1.8972 ms in `sep21-gdn-wy-bench-r2k-v1.jsonl`.
T32 SG8 v2 subsequently measured 1.18065 ms versus native 1.89627 ms in Root's
`sep21-gdn-wy-t32-bench-r2k-v2.jsonl`, including preparation and GPU reset.
V3 adds full-head snapshot/restoration/native replay; its costs and fallback
frequency remain unmeasured. Sealed v1 and v2 remain unchanged.

CPU-only preparation by subagents. Root owns all GPU execution. No model payload
or production runtime is edited. This is a numerical alternative and is not
qualified or promoted.

The earlier `gdn_chunk_sep21` kernels already use MPP matrix products for Gram,
state projections, attention, and state updates. This candidate changes the
remaining value-dependent triangular dependency: preparation forms
`F=(I+L)^-1`, `W=F diag(beta*prefix) K`, and `U=F diag(beta) V` once per value
head/chunk. The state phase computes `D=U-W*S0^T`, emits
`diag(prefix)*Q*S0^T + score*D`, and carries
`prefix_end*S0 + D^T*end_key`. Every relative decay product is formed directly;
no prefix quotient is used.

Source q/k/v/beta and output stay BF16. Decay, inverse, transforms, scores,
projections, and recurrent state stay F32. Every MPP descriptor disables relaxed
precision. The API compilation proof is in [sdk_probe/README.md](sdk_probe/README.md).
It does not establish internal instruction format or runtime utilization.

T32, V32, SG4 and SG8 are compiled. At 2048 rows, one lane, preparation uses
163,971,072 bytes (156.375 MiB) of temporary device workspace. Declared
threadgroup storage is 12,800 bytes in preparation and 8,516 in SG8 application;
actual pipeline resource use must be checked by Root before timing. The added
snapshot is 3,145,728 bytes and flags are 192 logical bytes at B1. The complete
added arena is 167,116,992 logical bytes / 167,133,184 bytes after separate
16-KiB rounding of coefficient, snapshot, and flag allocations.

`frozen/` contains the captured canonical kernel, ABI, and independent CPU
fixture/reference sources. `make -f dev/benchmarks/gdn_nax_chunks_sep21_v3/Makefile`
only compiles. Default/help and `--cpu-layout` create no Metal device.

CPU checks cover 46 cases from 23 fixtures with T16 and T32, independent F64
recurrence, continued sequences, zero decay, beta zero/one, coordinate
orientation, tails, cancellation, and F32/F64 prefix underflow. F64 maximum
absolute error is 8.8818e-16; strict/padding checks pass. The original F32 field
gate remains relative RMS <=1e-4 and max absolute <=5e-4*max(1, reference peak).
Cancellation delta fails that gate (T16 2.436e-4, T32 2.137e-4), and extreme
incoming-state/prefix range loss is separately reported. The previous direct
triangular-solve conditioning certificate does not automatically certify W/U
association. The CPU reference therefore exits 2 and does not claim complete
F32 numerical qualification.

Root GPU commands, serialized after the current worker is fully unloaded:

```sh
build/gdn-nax-chunks-sep21-v3/metal-oracle-wy --resources
build/gdn-nax-chunks-sep21-v3/metal-oracle-wy --quality --lanes 1
build/gdn-nax-chunks-sep21-v3/metal-oracle-wy --bench --rows 2048 --lanes 1
```

Use fresh JSONL destinations and inspect all fixture records when `--quality`
exits 2. No gate is silently loosened. A component timing does not establish
whole-model prefill throughput or quality. The benchmark includes preparation,
the preparation/application buffer barrier, and a GPU seed-state copy used by
both control and candidate. It warms with at least 150 ms of GPU work and ten
balanced ABBA/BAAB pairs; it performs no CPU tensor accesses between warm/timed
commands. Input hashes, immutable seed hashes, diagnostics, and buffer canaries
are checked after timing.

The GPU quality mode additionally checks the actual timed SG4 and SG8 pipelines
across all 48 heterogeneous value heads (16 distinct Q/K trios), with a 67-row
tail and 19-row continuation. Both per-head prepared coefficients and complete
carried state use the unchanged F32 gate. BF16 output separately uses registered
RMS <=.003 and max absolute <=.004*max(1, reference peak); this accounts for its
exposed BF16 rounding and does not relax any F32 gate. NaN output sentinels detect
unwritten fragments. Identical lane fixtures are repeated, and every lane is
checked; the quality report states that scope.

`seal.py --output FRESH.json` records all private sources, frozen inputs,
compiled artifacts, and CPU reports. It refuses to overwrite a witness and
explicitly records the failed F32 qualification and absence of GPU/model proof.

`source_proof.py --output FRESH.json` is CPU-only. It checks literal native
source closure, native reduction/contraction configuration, actual-zero /
nonzero-prefix / post-zero relative-product distinctions, FTZ operand cases,
snapshot/workspace/flag bounds, and the exact added arena plan. It establishes
source-level replay feasibility; actual GPU byte equality remains unproven.

V3 uses sticky per-(lane,value head) range/conditioning flags. The range scan
uses actual decay/beta values, raw bits for genuine zeros and subnormals, and a
normal-range boundary margin. Real zeros reset an independent nonzero-segment
scan so later relative-product underflow is covered. Out-of-domain or nonfinite
gates select native replay. The Cauchy/L2 projection selector uses MSL's permitted
RTZ worst-case u=2^-23 and rejects unrepresentable norm terms and finite
saturation. It remains a selector heuristic: W/U transform and incoming-history
errors are not certified by it. No old conditioning certificate is reused.

Fallback restores the full selected head from its immutable incoming state and
overwrites every full-row output with the literal captured native recurrence.
Head-0 audit replay also overwrites full F32 history/delta/pre-output tapes while
preserving other heads. Speculative nonfinite WY values become replay reasons;
committed native diagnostics are retained. Raw F64 gate failures are never
relabeled as passes: reference equivalence is a separate proof, and the raw
tiny-W failure remains visible even when that transform is unused by replay.
See [fallback_review.md](fallback_review.md) and [worker_plan.md](worker_plan.md).
