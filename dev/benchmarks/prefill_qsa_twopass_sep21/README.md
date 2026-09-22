# Private fresh2K global QSA matrix screen

This numerical alternative replaces attention only for a non-verification,
fresh 2048-row append. The original bulk prefix retains Q/K/index norm and
RoPE, all five persistent cache planes, pooling, and chronological completed
blocks with the incomplete causal tail. Its ordinary append workspace keeps
the original 32-partition allocation and four later sparse append checks.
No production source/default, decoder, model payload, or trained MTP changes.

The new graph packs prepared Q per KV head, computes whole-K BF16 QK in
M128N64K256 tiles, masks every future dense score to negative infinity,
computes global causal F32 softmax in place, and runs whole-K F32P/BF16V PV
in M128N64K2048 tiles. It then casts attention once to BF16 and runs the
original staged BF16 gate. Packed-V mode reuses the first2MiB of the dead
25MiB Q-pack buffer after QK completes. Every pack and unpack cost is timed.
The active chronological selected-block identity check remains in Q packing.

The extra arena is478,150,656 bytes; including existing prepared planes,
the candidate workspace is509,607,936 bytes. A real MemoryGovernor reserves
2GiB before fixture allocation, keeps the production10% host reserve, and
checks actual backend allocation peak against the reservation. External
guards cover all new planes, prepared planes, both five-plane caches, and
outputs. Source inputs/norms are hashed outside timed work. Physical K/V
cache padding beyond token2048 is NaN poisoned and remains unchanged.

Whole-K QK/PV and global softmax change numerical reduction boundaries.
Before any GPU run, the registered gates are raw source relative-L2<=2e-5,
BF16 source relative-L2<=1e-4, source cosine>=.99999999; sampled F64 QK
error<=2e-5+2e-6*abs(reference), sampled raw attention
error<=2e-6+2e-5*abs(reference), and probability sum error<=2e-6.
Baseline raw attention is checked against the same F64 envelope.
Sign flips above the absolute error floor fail. Source BF16 gating and an
independent staged-gate F64 sample are checked separately. There is no
bit-parity or full-model quality claim for changed attention arithmetic.

The GPU oracle includes finite-future causal anchors, active selection
corruption and ignored inactive cells, exact exponential-underflow/tie/
cancellation anchors, nonfinite fresh-V numeric rejection, and malformed
shader geometry without destination writes. Valid prefill requires every
fresh source row to be finite, as enforced by the unchanged prefix.
Partial output parity for an invalid nonfinite fresh row is not claimed.
Cancellation/deadline recovery of a future integrated worker and full
22-case model semantics remain separate integration requirements.

Build and seal submit no GPU commands:

```sh
.venv/bin/python -B dev/benchmarks/prefill_qsa_twopass_sep21/prepare.py
make -j6 -f dev/benchmarks/prefill_qsa_twopass_sep21/Makefile
.venv/bin/python -B dev/benchmarks/prefill_qsa_twopass_sep21/seal.py
```

Root owns serialized GPU execution. Without `--run`, this only verifies the
seal and prints the command. Run direct and packed modes sequentially with
fresh reports:

```sh
.venv/bin/python -B dev/benchmarks/prefill_qsa_twopass_sep21/run.py --pairs 8 --report FRESH.json --run
.venv/bin/python -B dev/benchmarks/prefill_qsa_twopass_sep21/run.py --pairs 8 --packed-v --report FRESH_PACKED.json --run
```

Warmup submits at least150ms of GPU work and at least8 full graph pairs.
Measured counts are a multiple of4 with AB/BA/BA/AB order. No CPU tensor
access occurs during warmup or measured pairs. Timing ABI is200 bytes.
Performance evidence is inclusive one-layer synthetic QSA, separate from
whole-model uncached2048/256 HTTP throughput and correctness.
