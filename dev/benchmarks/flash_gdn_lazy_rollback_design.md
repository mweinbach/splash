# Lazy GDN verification rollback: source and byte audit

This is a CPU/source design audit. No GPU ran, no model was loaded, and no production implementation was changed. All performance effects below are hypotheses. The proposed route preserves the existing BF16 prepared operands, FP32 decay and FP32 recurrent state; it does not quantize or demote the recurrent state.

## What the current implementation actually captures

Each of the 36 GDN layers has an independent live recurrent state `FP32[48,128(value),128(key)]`, exactly 3,145,728 bytes, and a three-row unconvolved projected-QKV convolution history `BF16[3,10240]`, exactly 61,440 bytes. Allocation rounding is 16 KiB, so one historical convolution state occupies 65,536 bytes; the recurrent state already occupies a whole number of allocation pages.

`runtime/flash/FlashForward.cpp:284–298` allocates `(maximumVerifyRows−1)` historical recurrent and convolution states per layer. `prefixTape()` at lines 321–326 indexes those configured extents. `verificationWorkspaceBytes()` at lines 704–711 includes these tapes plus the three PLE/control allocations. `workspacePlannedBytes()` additionally allows one 65,536-byte shared before-convolution buffer at line 479. The before-convolution buffer is allocated only when fused GDN is enabled and is reused across layers; it is not a 36-layer initial-history tape.

The active v6 profile requests depth 15. `runtime/flash/FlashWorker.mm:2799` constructs the singleton trunk with `maximumVerifyRows=maximumDepth+1`, hence 16. The singleton historical tapes allocate 1,734,082,560 bytes (1,653.75 MiB). `verificationWorkspaceBytes(16)` returns 1,734,311,936 bytes after adding the 16,384-byte PLE token history, 196,608-byte PLE convolution history and 16,384-byte retained-count allocation. The fused before-GDN-convolution scratch makes the actual verification-specific singleton allocation 1,734,377,472 bytes. Ordinary request live state and ordinary projection/GDN scratch are separate allocations and excluded from the tape tables.

At each actual singleton verification call with `R>1`, lines 859–861 explicitly construct `FlashGDNCapture{..., capturedRows=R−1, threadsPerHead=512}`. `runtime/flash/FlashGDNFused.cpp:82–83` selects that explicit count and puts it in the shader ABI. The fused shader `runtime/metal/kernels/shared/flash_gdn_fused.metal:256–260` writes only when `token<capture_rows`. It stores each historical state directly from the FP32 state registers. The full final state remains in the live request state, written once at lines 282–286, and is not duplicated in the tape. `R=1` does not use the singleton verification tapes or capture path.

Convolution prefixes are reconstructed after the fused recurrence from the before-history scratch and projected raw QKV. Each retained prefix `1..R−1` gets exactly three history rows. `flashMTPConvolutionPrefix()` selects old-history suffix plus incoming rows for retained counts 1 or 2, then the last three incoming rows for every larger retained count. The unfused fallback instead consumes rows individually and copies convolution and recurrent state after each nonfinal row; it has the same tight prefix-state byte count.

`runtime/flash/FlashBatchVerify.cpp:172–176` allocates the same tapes for configured maximum lanes and maximum rows. Production creates maximum lanes 4 and maximum rows 4 at `runtime/flash/FlashWorker.mm:2885`, so joint historical tapes allocate 1,387,266,048 bytes (1,323 MiB). `FlashBatchVerifyGDN.cpp:129–130` also passes `capturedRows=R−1` explicitly, while using configured `(maximumRows−1)` as the per-lane tape stride. It packs each lane's live convolution/recurrent state into a reusable layer-local packed workspace, runs one captured graph, reconstructs only `R−1` convolution prefixes and scatters the full state back to each request. Packing and scattering add two reads and two writes of each live state per layer; they are not historical prefixes. There is no equivalent extra per-layer before-history copy because original request convolution history remains unchanged until final scatter.

Both singleton and batch commits skip GPU restore when all retained rows equal the verified row count. Singleton partial commit copies the selected recurrent and convolution prefix, then restores PLE. Joint partial commit restores only nonterminal lanes with `0<retained<R`; a terminal `retained=0` lane is discarded/poisoned and must never acquire restored state. QSA continues to use logical length to exclude provisional cache rows. These lifecycle rules must stay unchanged.

## Proposed retained representation

One lazy record per layer/lane retains the initial recurrent state, initial convolution history and inputs that reproduce the recurrence:

| Field | Type and shape | Tight bytes |
|---|---|---:|
| Initial recurrent state | FP32 `[48,128,128]` | 3,145,728 |
| Initial convolution history | BF16 `[3,10240]` | 61,440 |
| Projected raw QKV | BF16 `[R,10240]` | `20,480 R` |
| Prepared normalized Q/K and convolved V | BF16 `[R,10240]` | `20,480 R` |
| Decay | FP32 `[R,48]` | `192 R` |
| Beta | BF16 `[R,48]` | `96 R` |

The conservative layout independently rounds each of these six fields to 16 KiB per layer/lane. This exposes the actual small-gate allocation overhead rather than silently packing it away. A later arena layout may pack fields, but cannot claim these independently rounded totals if it does.

With `A(x)=ceil(x/16384)*16384`, `S=3145728` and `H=61440`:

```
eager_tight(R,L) = 36 L (R−1) (S+H)
eager_padded(R,L) = 36 L (R−1) (A(S)+A(H))
lazy_tight(R,L) = 36 L (S+H+41248 R)
lazy_padded(R,L) = 36 L (A(S)+A(H)+2 A(20480 R)+A(192 R)+A(96 R))
```

All table quantities are MiB (2^20 bytes), for the entire 36-layer rollback representation. They exclude ordinary live state, ordinary scratch, PLE/QSA state and any guard allocation. R8/R16 with four lanes are hypothetical geometry comparisons; the current joint verifier accepts at most four rows per lane.

| Lanes | Configured rows | Eager tight MiB | Eager padded MiB | Lazy tight MiB | Lazy padded MiB |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.000000 | 0.000 | 111.525513 | 113.625 |
| 1 | 4 | 330.328125 | 330.750 | 115.773926 | 117.000 |
| 1 | 8 | 770.765625 | 771.750 | 121.438477 | 122.625 |
| 1 | 16 | 1651.640625 | 1653.750 | 132.767578 | 133.875 |
| 4 | 1 | 0.000000 | 0.000 | 446.102051 | 454.500 |
| 4 | 4 | 1321.312500 | 1323.000 | 463.095703 | 468.000 |
| 4 | 8 | 3083.062500 | 3087.000 | 485.753906 | 490.500 |
| 4 | 16 | 6606.562500 | 6615.000 | 531.070312 | 535.500 |

R1 lazy values show the requested unconditional formula only. An actual route must keep the existing no-tape R1 bypass: there is no shorter positive prefix to restore, so all candidate fields and capture writes can be omitted. R2 is also a poor candidate: the eager tape is one state, whereas lazy adds that same state plus replay inputs. Consider the lazy route only for R>=3 pending measured results.

The present singleton R16 plus joint L4/R4 tapes total 3,121,348,608 bytes (2.906982421875 GiB). Replacing both with these lazy records would require 631,111,680 bytes (601.875 MiB), a tape allocation reduction of 2,490,236,928 bytes (2,374.875 MiB), before any change to PLE/control/guards. Singleton alone saves 1,519.875 MiB; joint alone saves 855 MiB. These are byte calculations, not throughput measurements.

## Verify-time writes and dispatch choices

Current singleton fused verification adds, relative to ordinary same-R GDN, `(R−1)(S+H)+H` tight bytes of writes per layer/lane: historical recurrence stores, convolution prefix copies and one initial-history scratch copy. Padding is not written. Its 36-layer totals are:

| Actual singleton rows | Current extra writes MiB | Lazy explicit-copy writes MiB | Lazy direct-input extra writes MiB |
|---:|---:|---:|---:|
| 1 | 0.000000 | 0.000000 | 0.000000 |
| 4 | 332.437500 | 115.773926 | 110.109375 |
| 8 | 772.875000 | 121.438477 | 110.109375 |
| 16 | 1653.750000 | 132.767578 | 110.109375 |

“Lazy explicit-copy writes” assumes one initial snapshot plus copies of the raw and prepared inputs/gates into retained storage, hence `S+H+41248R` per layer/lane. It is a conservative easy-to-isolate candidate. Every copy also reads its source: write bytes alone are not total traffic. An initial recurrent snapshot emitted directly from the already-loaded FP32 registers avoids a second state read and a separate copy dispatch.

“Lazy direct-input extra writes” assumes projection writes raw QKV directly into its retained destination and the persistent GDN kernel writes prepared mixed/decay/beta directly into retained destinations. Those row writes already exist in the ordinary route; retaining their allocation across all layers adds no second store. Only the initial FP32 recurrent snapshot and initial convolution history are extra: `S+H` per layer/lane. This requires proving that destination changes leave projection tiling/policy and prepare math identical. It is not implemented or qualified here.

Current singleton fused GDN uses one before-history copy, one persistent captured recurrence, one full convolution carry and `(R−1)+min(2,R−1)` convolution-prefix copies per layer. Total GDN-path dispatches per layer are `3+(R−1)+min(2,R−1)` for R>1: 8 at R4, 12 at R8 and 20 at R16, excluding projections/output projection. A lazy shader that captures the initial state in its existing initial load loop, writes directly into retained prepared storage and retains raw QKV directly can remove all historical-prefix copy dispatches. It still needs initial-history retention and the ordinary convolution carry. The count reduction is a proposed implementation property that must be confirmed in the actual graph.

Joint pack/scatter writes are a separate invariant cost of `2(S+H)` per layer/lane. At L4/R4, current pack/scatter plus prefix writes are 2,202.1875 MiB across 36 layers. Keeping packing/scattering unchanged and adding explicit-copy lazy records yields 1,343.970703125 MiB. Direct-input lazy snapshots with unchanged packing/scattering yield 1,321.3125 MiB. A future shader taking separate live per-request state buffers could remove packing/scattering, but that is an additional independent change requiring its own ownership/alias/cancellation and byte-exact qualification.

Joint GDN adds `2L` pack dispatches, two fused GDN/carry dispatches, `L((R−1)+min(2,R−1))` convolution prefix dispatches and `2L` scatter dispatches. At L4/R4 this is 38 per layer. Existing ordinary single-lane kernels may already have different dispatch structures, so the appropriate timing control is the existing joint helper at the same lane/row geometry.

## Exact retained-prefix replay

The persistent shader uses a sequential FP32 state matrix for each value head. Its four key components per SIMD32 lane update as follows, preserving the exact order of the four additions and the existing `simd_sum` reduction:

```
for each real retained token in chronological order:
    s[i] = s[i] * float_decay
    memory_partial += s[i] * float(saved_bf16_key[i])  # i=0,1,2,3
    memory = simd_sum(memory_partial)
    delta = (float(saved_bf16_value) - memory) * float(saved_bf16_beta)
    s[i] = s[i] + float(saved_bf16_key[i]) * delta     # i=0,1,2,3
```

Both existing GDN shader files explicitly disable floating-point contraction. Replay must retain that setting, SIMD32 reduction tree, chronological traversal, FP32 arithmetic and identical conversion sites. A mathematically equivalent tensor dot, FMA, parallel scan, BF16/INT8 recurrent state or altered reduction tree is not an exact rollback route.

Only the saved mixed K/V, FP32 decay and BF16 beta are needed to reproduce the state. Saved normalized Q, Z and output norm are not needed for the state-only recurrence; keeping the complete width-10240 mixed row simplifies matching the original prepare ABI and independent checks. Raw QKV plus initial history reproduce the convolution history after a retained prefix using `flashMTPConvolutionPrefix()`. Do not rerun the projections or nonlinear prepare from shortened inputs: changing the actual row count could change dense-cache or tensor tiling policies and recreate different operands.

`retained==R` must do no replay and leave the already-computed full state untouched. `0<retained<R` restores the initial snapshot and replays exactly those retained rows. A private replay shader should read saved operands with their original row/lane stride R, independently of the retained count; the existing `FlashGDNParams.rows` is both loop count and lane row stride, so simply substituting `rows=retained` in a multilane call would read the wrong lane offsets. Ragged joint retained counts therefore need independent counts/masks and immutable input strides. Terminal zero-retained lanes must be skipped rather than read or restored.

A state-only replay shader can omit Q/output dots, recurrent-row stores, Z gating and RMS/output math. Keep the FP32 memory-dot/update statements identical to the qualified kernel and measure the benefit; compiler equivalence still needs zero-byte GPU tests. Reconstruction can also copy the selected three-row convolution prefix in a single per-lane dispatch rather than reproducing every historical prefix at verify time.

Replay adds per rejected layer/lane one initial-state read, one final state write and chronological computation for K retained rows. Its 3 MiB state transfer per layer does not scale with K when state remains in registers. For a whole 36-layer request, initial-state read plus final state write is 216 MiB, excluding any separate initial-state restore copy. Writing the initial snapshot back into live state via a separate raw-copy dispatch would add another 216 MiB read+write; a replay kernel loading directly from the snapshot and writing final live state avoids that copy.

The state-only recurrent update performs five scalar multiplies/adds per state element per token (one decay multiply, one memory multiply/add, one update multiply/add), plus two per-value beta/delta operations and one SIMD32 memory reduction. Across 36 layers this is 142.000128 million scalar arithmetic operations per retained token, counting add and multiply separately and excluding SIMD reduction additions. The omitted Q/output dot would add another two operations per state element plus its reduction. Saved prepare inputs avoid convolution and nonlinear prepare work. Replay does not rerun the 48-layer transformer, expert projections, PLE, QSA or vocabulary head. Retaining those other subsystems' existing rollback behavior is mandatory.

High draft-match rate alone does not determine replay frequency: EOS/quota truncation can produce a shorter retained prefix despite matched draft tokens. A hypothetical 99% full-window acceptance would amortize replay over one in 100 windows, but acceptance, retained-prefix distribution and command cost must be measured from the actual workload. No speedup claim follows from the allocation or byte tables.

## Private qualification gate

Use the current production persistent-capture route as the control at each actual row count. Test R1/2/3/4/5/7/8/9/15/16, lanes 1/2/4 (joint production only rows<=4), cold and nonzero initial state, two carried sequences, every positive retained prefix and terminal lanes. Require zero byte differences in all FP32 recurrent and BF16 convolution states; compare prepared operands, decay and beta exactly, plus guards, diagnostics and source identities. Prove both full-accept no-replay and partial replay followed by at least three ordinary autoregressive continuations. Test independent padded strides, retained masks, pending ownership, missing/expired cancelled lanes, aborts and invalid geometry before graph mutation. These are candidate gates, not completed validation.

Measure control/candidate verify time, full-accept commit, each retained-prefix replay and mixed-lane commit separately, with allocation/page-fault/residency conditions held equal. Report actual capturedRows, retainedRows, bytes written, dispatch count and full-window acceptance per measured cohort. Only then run matched end-to-end HTTP single/concurrent short and long prompts against oMLX. Keep the default unchanged until the production qualification and whole-service comparison justify it.
