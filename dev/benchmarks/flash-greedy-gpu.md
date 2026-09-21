The isolated GPU controller preserves the finite BF16 greedy and verification
contracts in `FlashGreedy.hpp` and `FlashMTPWindow.hpp`. It is a prototype; the
worker's qualified CPU selection remains the production route.

Each vocabulary row is split into 2,048-token partitions. A 256-thread group
scans each partition using integer BF16 ranks, maximizes rank, minimizes token
ID on ties, and ORs all nonfinite flags. Both signed zeros have the same rank.
Subnormals remain distinct, independently of GPU floating-point modes. The
second dispatch reduces partition records with one SIMD32 group per real row.
For verification, that dispatch also finds the matching draft prefix and
produces the exact retained count, output IDs, and EOS/quota result per lane.

The operator supports 1–16 real rows or 1–4 uniform verification lanes totaling
at most 16 rows. It never invents padded verifier inputs. Every logit in every
real row is validated, including rows after a mismatch or quota boundary.
Inactive lanes retain zero. Active lanes with nonfinite logits, invalid input
IDs, or a zero output budget retain zero and report an error. The caller must
reject errors before model state commit/restore. Cancellation, generation
cookies, deadlines, and transport remain CPU responsibilities before commit
or emission. The GPU result is a proposal, not a model-state mutation.

Build in a private directory:

```sh
make -f Makefile -f dev/benchmarks/flash_greedy_gpu_oracle.mk \
  BUILD=build/flash-greedy-gpu \
  build/flash-greedy-gpu/flash-greedy-gpu-oracle -j8
build/flash-greedy-gpu/flash-greedy-gpu-oracle --cpu-self-test
build/flash-greedy-gpu/flash-greedy-gpu-oracle --cpu-benchmark
```

The CPU self-test covers all 65,536 BF16 encodings, rank ordering against zero
and adjacent finite values, signed-zero ties, and all bounded mismatch/quota
combinations. It submits zero GPU commands. Optimized and ASan/UBSan executions
passed 263,006 checks. Host and Metal 4.1 compilation passed.

The CPU-only benchmark places every winner at the final vocabulary ID, forcing
the tie-preserving second scan through the complete row. Observed medians were
31.89/61.12/132.30/254.75/519.19 µs for 1/2/4/8/16 rows in
`build/flash-greedy-gpu/cpu-neon-winner-last.json`. These are selection timings,
not model throughput; earlier winners can make the CPU second scan cheaper.

Root alone runs the GPU oracle, after stopping other GPU workloads:

```sh
build/flash-greedy-gpu/flash-greedy-gpu-oracle \
  build/flash-greedy-gpu/flash-greedy-gpu.metallib \
  build/release/flash/greedy-gpu-qualification.json
```

The full GPU oracle checks guarded source/stride/scratch/result buffers,
independent scalar and NEON argmax, every BF16 NaN/Inf payload/sign at eight
SIMD/threadgroup/partition/tail positions, 8,160 prefix combinations, four
real B4 scenarios, and fail-before-encoding host guards. Prefix combinations
use valid synthetic producer partials to isolate integer prefix math; the B4
cases exercise the complete vocabulary-to-prefix pipeline. No GPU execution
has been performed by the authoring agent.

`FLASH_GREEDY_GPU_BENCH_ONLY=1` skips GPU qualification and measures selection
only. The timing report distinguishes CPU NEON scan time, a standalone GPU
command's GPU/wall time, and sixteen reductions sharing one command. Sharing
one command exposes approximate marginal dispatch/scan cost. It does not prove
whole-model speedup or eliminate the draft synchronization boundary. The CPU
benchmark prevents loop-invariant scan elimination with a compiler memory
barrier. The output is 16 bytes per selected row or 144 bytes per verification
lane, rather than a 496,640-byte vocabulary row.

The next fusion target is the vocabulary projection epilog. Rank the exact
BF16 value the current projection already writes, reduce tile-local
rank/token/error records, and run final selection plus acceptance in a single
small dispatch. Ranking the F32 accumulator would change BF16 ties. Keep the
existing coefficient route, accumulation, BF16 cast, and diagnostics. A
bounded GPU-token-input/graph-building head API is then needed to chain
draft → selection → embedding without an intervening CPU token upload.
The standalone two-dispatch path may lose to CPU NEON for one row; measurement
must determine that before routing changes.
