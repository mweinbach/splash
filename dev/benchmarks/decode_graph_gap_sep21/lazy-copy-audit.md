# Lazy GDN copy elimination: CPU-only audit

Status: source/JSON inspection only; no GPU execution, model payload reads, or runtime/kernel edits. The candidate is not implemented or qualified. The parent selected HC-down row reuse for the immediate experiment; this records a separate, narrow follow-up.

## Current work and byte counts

The exact gathered/pointwise worker projects QKV into ordinary scratch at `build/moe-pointwise-sep21-worker-v1/source/runtime/flash/FlashForward.cpp:1042-1057`. Lazy `begin()` then saves convolution history and raw QKV with two `flash_gdn_lazy_copy` dispatches, runs persistent verification, and carries history at `.../FlashGDNLazyRollback.cpp:238-255`.

The **initial F32 recurrent snapshot is already fused** into the persistent kernel's initial state load (`runtime/metal/kernels/shared/flash_gdn_lazy_rollback.metal:153-161`). It is not another copy opportunity. Only the small raw-QKV/history copies remain:

| Record, R4/B1 | Bytes per GDN layer | Bytes across 36 layers | MiB across 36 layers |
|---|---:|---:|---:|
| Raw QKV, BF16 `[4,10240]` | 81,920 | 2,949,120 | 2.812500 |
| Initial history, BF16 `[3,10240]` | 61,440 | 2,211,840 | 2.109375 |
| Two copy payloads together | 143,360 | 5,160,960 | 4.921875 |
| Already-fused initial recurrence, F32 `[48,128,128]` | 3,145,728 | 113,246,208 | 108.000000 |

Copy payload is not traffic: the two copies also read their sources, giving 9.843750 MiB logical read/write traffic across 36 layers. The current copy kernel handles one byte per thread (`flash_gdn_lazy_rollback.metal:424-437`): R4/B1 uses 240 history groups plus 320 QKV groups per layer, 20,160 groups / 5,160,960 threads total.

## Proposed exact destination/snapshot fusion

1. Project `in_proj_qkv` directly into that layer's retained `RawQKV` plane. Preserve the existing prefix, real row count, input, weight identity, projection pipeline/tile, diagnostics, and BF16 conversion sites. Projection selection currently depends on prefix/geometry/policy rather than destination identity (`FlashForward.cpp:414-454`). Remove the QKV copy; verification and carry read the retained plane.
2. Replace lazy carry with a private variant that snapshots all three **old** history rows for its channel into `InitialHistory`, then executes the existing carry loop unchanged. Carry already has one unique thread per `(lane, channel)` (`build/moe-pointwise-sep21-worker-v1/source/runtime/metal/kernels/shared/flash_gdn.metal:294-318`). Preserve the separate verify-to-carry dispatch boundary.

This removes 72 copy dispatches. The diagnostic graph would change from 1,363 to 1,291 dispatches; the GDN interval changes from four to two dispatches per layer. Direct QKV output removes 5.625000 MiB logical copy traffic; initial-history retention still requires its original 2.109375 MiB snapshot stores, plus reads. No tape allocation reduction is implied. Direct raw input retention was discussed, but not implemented or qualified, in `dev/benchmarks/flash_gdn_lazy_rollback_design.md:75-77`; the inspected worker still has both copies.

## Exactness and lifecycle contracts

- Add a checked pretrial destination accessor, or equivalent reservation, for `RawQKV`. Only the exact bounded RawQKV view may alias `b.qkv`; do not relax arbitrary overlap checks. `begin()` currently rejects all input/tape aliasing (`FlashGDNLazyRollback.cpp:218-230`). All other inputs, live state, diagnostics, weights, and tape planes must remain disjoint Shared buffers. Preserve guards and padded live-state strides; retained QKV remains tightly row/lane packed.
- Reserve before graph construction, reject pending or out-of-capacity trials before graph mutation, and keep each layer's destination alive and immutable until commit/abort. Ordinary R1 and nonverification paths keep their existing destination and no-tape bypass. Never reuse a tape as next-layer QKV scratch.
- Persistent verify reads convolution history and mutates only the recurrent plane of live GDN state (`flash_gdn_lazy_rollback.metal:72-88,136,240-245`), alongside its existing output/tape writes. Therefore the old history is still intact when ordered carry starts. Each carry thread must snapshot **all three old rows before its first overwrite**; this also preserves R2/R3 behavior. Snapshot addressing uses tight saved-history lane stride 61,440 bytes while live history uses `p.convolution_lane_stride_bytes`. Channels are disjoint, so no inter-thread synchronization is needed for snapshot stores.
- Keep carry after all verify history reads. Folding history mutation into the per-head persistent kernel is unsafe: multiple value-head threadgroups read shared Q/K history and there is no global barrier.
- Keep replay operands, initial recurrence, raw-QKV bytes, history restoration and their conversion/order contracts unchanged (`FlashGDNLazyRollback.cpp:201-214,262-282`; `flash_gdn_lazy_rollback.metal:392-420`). Full acceptance must still build no GPU replay (`FlashForward.cpp:1332-1341`). Abort still discards live state; no snapshot can be reused or promoted from an aborted trial.

## Instrumentation-limited saving ceiling

`build/release/flash/sep21-gathered-verifier-stage-v1.json` reports all 72 lazy copies timed in each R4/B1 sample: **1.316830 ms** and **1.023378 ms**, mean **1.170104 ms**. This is the removable diagnostic-copy-time ceiling, before charging the fused history reads/stores or changed scheduling. Existing carry averages 0.182209 ms and stays present; initial recurrent snapshot and persistent-verify math stay present.

Stage profiling creates and ends one serial encoder per dispatch (`runtime/metal/MetalBackend.mm:2304-2323`); unsampled submission uses one encoder for the graph (`:2324-2356`). Consequently these totals are **not an unsampled speedup prediction or measured >1 ms result**. A >1 ms saving is a narrow hypothesis, not guaranteed by this ceiling. Qualification would require byte-exact tape/live-state/continuation checks, partial-prefix and full-accept lifecycle checks, and a matched unsampled target-verifier comparison.
