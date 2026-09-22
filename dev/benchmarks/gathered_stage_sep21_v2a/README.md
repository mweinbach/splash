# Private gathered MPP finite-input staging screen

`../prefill4k_allrows_gathered_stage.metal` replaces every producer CTA's
activation scan and static threadgroup fallback with one canonical GPU
sanitize/copy dispatch per hidden/intermediate phase. These private kernels
retain device BF16 × signed-I8, whole dynamic K, M16/N64, multiply mode, SG4,
late F32 row scaling, and the compiled BF16 SwiGLU boundaries. Neither
production sources nor the existing gathered shaders are modified.

Hidden staging scans `[R,2560]` even when a route ID is excluded. Intermediate
staging scans only valid ID/rank routes in `[R,10,640]`; excluded routes write
scratch positive zero without reading their input. All finite BF16 values are
copied as `ushort` bits, preserving negative zero and subnormals. Every NaN/Inf
becomes positive zero and atomically sets sticky bit4. The unchanged producer
ID/rank and duplicate checks retain invalid gate bit1, invalid down bit5 and
duplicate bit1 behavior. Exact malformed arithmetic parity remains unmeasured
because the original malformed fallback uses a different address space.

The two staging buffers occupy `17,920*R` payload bytes, plus their independent
allocation canaries. They are separate admitted allocations, never aliases of
source operands or outputs. Staging adds `40*R` CTAs to `500*R` producer CTAs.
It reduces source BF16 values scanned from `512,000*R` to `8,960*R`; whole-K
coefficient reads are unchanged. No producer reserves static threadgroup
activation storage or executes a scan barrier.

This v2a oracle loads one shared readonly layer once and compares five chains:
old bucketed MPP, staged gathered SG4, unchanged local-scan gathered SG4, and
unchanged local-scan SG1/SG2 from `../prefill_moe_sep21/`. Each alternative has
its own matching raw F32 dot, scaled F32, and BF16 projection tap. All taps use
the same original hidden and canonical baseline intermediate values.

Reports separate exact raw/scaled F32 parity, exact BF16 projection parity,
activated/down/SwiGLU parity, conservative opaque-MPP F64 finite/sign/absolute
bounds, strict F64 results, stale-buffer replay, canaries and input immutability.
Timing requires exact BF16 projection and stage parity with the baseline and
all common checks. Raw/scaled F32 differences remain named numerical
alternatives; those timings cannot claim exact producer identity or qualify
model integration. Strict F64 failures remain visible even when the frozen
conservative bounds pass. No synthetic primitive report qualifies actual
decode activations or full-model quality.

Five full rotating warm cycles immediately precede nine rotating matched
sample cycles by default. Timed staged chains include both sanitize dispatches;
untimed projection taps, CPU reference scans and immutable hashes are excluded.

```sh
make -f dev/benchmarks/gathered_stage_sep21_v2a/oracle.mk -j4 \
  gathered-stage-one-layer-cpu

# Root alone invokes GPU mode; preserve all prior reports by choosing NEWreport.
build/prefill4k-allrows-gathered-stage-one-layer-sep21-v2a/oracle --gpu \
  build/prefill4k-allrows-gathered-stage-one-layer-sep21-v2a/splash.metallib \
  /absolute/Full512-store 0 4 /absolute/NEWreport.json \
  --pattern repeated --pairs 9 --warm-cycles 5
```

The subtree performed warnings-as-errors host/Metal compilation and CPU checks
over all 65,536 BF16 bit patterns. It submitted no GPU work and read no model
payloads. GPU parity, malformed-input behavior and speed are Root's remaining
qualification steps.
