# Exact BF16 expert operands with current Direct-A fallback

This private primitive explores whole-K native BF16 matmul for the original
hot expert operands. It never requantizes to INT8 or changes the checkpoint.
The existing governed `FlashExpertDenseCache` constructor converts selected
Q4/G64 coefficients with separate F32 multiply/add and one BF16 rounding. The
oracle independently checks every stored coefficient bit before and after
all requests.

The old cache is insufficient for today's profile: it uses whole-K only for
full jobs, sends cached tails to staged Q4x8, and assumes the second down
dispatch is the producer. Current Direct-A inserted a preparation dispatch
there, making that fixed-index memcpy unsafe. This primitive uses only its
constructor/immutable operands. A new private bridge locates and validates
the current producers by name/ABI, preserves excluded-route poisoning and
Direct-A down sanitization, then adds whole-K BF16 hits and the original
Direct-A Q4 misses. Dynamic row extents permit all live cached tails.

Cached BF16 storage preserves source coefficient bits. Whole-K changes the
MPP reduction descriptor and therefore requires separate output qualification.
The Root wrapper requires byte equality of all activation, canonical down,
and combined BF16 outputs before and after timing. The older numerical
1% relative-L2/.9999 cosine guard remains diagnostic, not the promotion gate.
Source and immutable cache bit checks, CPU buckets/jobs, sticky diagnostics,
canaries, and alternating complete-chain GPU/wall measurements are retained.

```sh
make -f dev/benchmarks/prefill4k_bf16cache/Makefile -j2 all
build/prefill4k-bf16cache/oracle --cpu-self-test
.venv/bin/python dev/benchmarks/prefill4k_bf16cache/run.py \
  --hot 128 --layer 0 --rows 2048 --tile 32 --pairs 4 \
  --report build/release/flash/prefill4k-bf16cache-hot128-layer0.json --run
```

Compilation and CPU self-test submit no GPU work. Omit `--run` to write only
provenance. Root serializes model/GPU work. The wrapper removes inherited
feature flags and picks actual frequency-ranked hot IDs from the same verified
Top64/Top128 plans used elsewhere. It forces PLE-SSD source loading and current
Direct-A policy. `--hot 64`, `--layer 24/47`, and `--pattern mixed/spread/all-cold`
allow focused source-qualified comparisons. Rows up to 8192 and explicit M64
are supported by the new shader/bridge; the old fixed-index class methods are
never called.

Constructor conversion and independent coefficient scans are excluded from
per-request timings because the cache is immutable across requests; the report
records initialization GPU time and actual/planned allocation bytes. Layer 0
Hot128 has 1,258,291,200 logical coefficient bytes; 48 layers have 60,397,977,600.
Real full-model admission would also include separately rounded canaries and
rank/selection/diagnostic arrays. No heavy cache construction/store generation
or GPU run was submitted by this subtask.

Old layer 0 Hot128 R2048/M32 results were byte-exact on synthetic activations:
all-hot 10.768→3.981 ms, mixed 14.068→12.211 ms, spread 15.883→14.830 ms.
They used earlier Q4x8/full-job policy and do not qualify this primitive.
Old depth 15 HTTP hot32/hot64 variants regressed and remained inactive; actual
whole-model quality/lifecycle and throughput validation are still required.
