Private Root-only actual capture wrapper over the **unchanged sealed local
component**. Default help and `--cpu-contract MANIFEST` do not open captured
binary tensors or create a Metal device. Only Root is authorized to execute
`--actual-capture MANIFEST`.

The explicit Root mode verifies all nine actual layer0 manifest tensor extents
and SHA256 identities and the sealed local metallib SHA. Original V16/Time16
native must reproduce recorded cold final F32 state and BF16 recurrence exactly
before the candidate runs. Captured Z/norm/final output identities are checked
for provenance but their fusion is outside this component.

Candidate quality is compared with original native for cold, identical
native-carried seed, and future trajectory from candidate's own cold carry
against native's own cold carry. Each screen repeats the same real captured
input segment; it is not a newly captured continuation. Original F32 1e-4 RMS /
5e-4 peak scale and BF16 .003 RMS / .004 peak scale gates remain unchanged.

For every quality screen, reuse the sealed full-head probe tape contract to
record each actual incoming hybrid seed. Candidate and probe must match state,
BF16 output and decision words exactly. Each native-selected T32/V32 tile is
then compared bitwise with original V16/Time16 native on the raw captured chunk
from that identical hybrid incoming F32 seed. This scoped native-chunk proof is
separate from whole-sequence quality: no full-sequence bit identity or
whole-history error certificate is claimed. The 201 MB proof seed tape is not
used in the timed benchmark. Every buffer's canaries, immutable source/reset
seed hashes, and immutable tape are checked. A sticky diagnostic sentinel repeat
also requires identical cold state/output/decision words.

Selection reporting uses the sealed helper's actual word validation and reason
counts, then exposes compact counts per `(chunk,value_tile)` across all 48
heads, plus static range chunk/head count and actual native tile-row steps.
It does not convert them to the old whole-head flag policy.

Cold and identical-native-carried benchmarks use original V16/Time16 as the
baseline. Each includes at least 150 ms GPU warmup and 10 balanced pairs,
exactly five ABBA and five BAAB. GPU seed reset, preparation, local checks,
deferred commit and local native fallback are included. Input copies, hashes,
CPU inspection and probe/native proof dispatches occur outside timed loops.
Timing is retained even on unchanged numerical-gate failure. Exit 0 means
component quality passes, 2 means its numerical/proof gate fails, and 1 means a
provenance/runtime failure. No whole-model performance/quality claim is made.

Compile: `make -f dev/benchmarks/gdn_tile_actual_capture_sep21/Makefile -j2`.
CPU seal: `.venv/bin/python -B dev/benchmarks/gdn_tile_actual_capture_sep21/seal.py`.
Root command: `zsh build/gdn-tile-actual-capture-sep21-v1/root-actual-capture-command.txt`.
