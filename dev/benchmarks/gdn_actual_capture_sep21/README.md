This private wrapper includes the sealed v6 oracle without editing it and uses
its exact SG8 shaders, dispatch, guarded coefficient arena, and full-head frozen
native replay. It adds an explicit `--actual-capture MANIFEST` mode only. Default
help and `--cpu-contract MANIFEST` create no Metal device and never open tensors.
The CPU preparation agent is not authorized to run `--actual-capture`.

Root's explicit capture mode loads the nine manifest-listed tensors, checks all
byte extents and SHA256 values, and hashes the sealed v6 metallib. The frozen
original native must reproduce recorded cold final state and BF16 recurrence
exactly before the candidate runs. The final gated RMS and maximum-absolute
tolerances remain F32 1e-4/5e-4 and BF16 .003/.004, identical to v6. Unflagged WY
is not claimed byte-exact. Flagged heads must remain exactly native from the
same immutable incoming state. Canaries, immutable inputs/seeds/snapshots and
sticky diagnostics are checked. The sticky diagnostic rerun also requires the
same cold state, output and reason bits exactly.

The capture contains one real layer0 input segment. Carried screens repeat that
same captured input segment from the recorded native final state and from the
candidate's own cold final state. Each candidate is compared with a native
recurrence from that identical carried seed. These are actual-input stability
screens, not a claim about a different captured continuation segment. No final
Z/norm/output fusion is dispatched here; their content identities are verified
for complete capture provenance only.

Cold and native-carried timing each perform at least 150 ms GPU warmup followed
by 10 pairs, exactly five ABBA and five BAAB. Every timed command includes GPU
seed reset, snapshot, preparation, WY, flagged restore and original-native
replay. Tensor inspection and hashing happen outside the timed loop. Timing is
collected even when the unchanged numerical gate fails, so Root can assess
actual guard frequency and decide against integration without losing cost
evidence. The wrapper exits 2 for numerical qualification failure, 1 for
provenance/runtime failures, 0 for success. It never claims whole-model quality
or whole-model prefill throughput. The selector remains a heuristic, not a
whole-history error certificate.

Compile with `make -f dev/benchmarks/gdn_actual_capture_sep21/Makefile -j2`.
Root's exact GPU command is in
`build/gdn-actual-capture-sep21-v1/root-actual-capture-command.txt`.
