# Private GDN W/U matrix experiment

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

T16, V32, SG4 and SG8 are compiled. At 2048 rows, one lane, preparation uses
157,679,616 bytes (150.375 MiB) of temporary device workspace. Declared
threadgroup storage is 3,264 bytes in preparation and 4,096 in application;
actual pipeline resource use must be checked by Root before timing.

`frozen/` contains the captured canonical kernel, ABI, and independent CPU
fixture/reference sources. `make -f dev/benchmarks/gdn_nax_chunks_sep21/Makefile`
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
build/gdn-nax-chunks-sep21/metal-oracle-wy --resources
build/gdn-nax-chunks-sep21/metal-oracle-wy --quality --lanes 1
build/gdn-nax-chunks-sep21/metal-oracle-wy --bench --rows 2048 --lanes 1
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
