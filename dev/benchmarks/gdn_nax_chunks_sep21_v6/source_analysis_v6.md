# V6 restore and native-selector source review

CPU-only review of the actual v6 adapters. No GPU, model payload, v5 source,
sealed worker, or oracle edits were performed. Root reported v5 SG8 zero-flags
1.568604 ms versus unguarded 1.180650 ms; this review does not independently
measure those times or assign the difference to a phase. V6 timing is unmeasured
here. Timing implications below are hypotheses only.

## Exact launch and atomic counts

R2048/B1, H48, T32, V32, SG8; preparation processes 64 chunks/head. Counts are
source thread invocations and atomic-load calls, not compiler instructions,
physical memory transactions, or occupancy measurements. They apply regardless
of final fallback count, because the selector load precedes the uniform skip.

| Phase | V5 groups | V6 groups | V5 launched threads | V6 launched threads |
| --- | ---: | ---: | ---: | ---: |
| Snapshot | 3072 | 3072 | 786,432 | 786,432 |
| Preparation | 3072 | 3072 | 786,432 | 786,432 |
| Apply | 192 | 192 | 49,152 | 49,152 |
| Restore | 3072 | 48 | 786,432 | 12,288 |
| Native replay | 384 | 384 | 196,608 | 196,608 |
| **Five-phase total** | **9792** | **6768** | **2,605,056** | **1,830,912** |

Restore removes 3024 groups and 774,144 launched threads. Both versions still
have five compute dispatches; snapshot, preparation, apply, and replay launch
geometry remain unchanged.

| Flag consumer | V5 atomic loads | V6 atomic loads |
| --- | ---: | ---: |
| Apply initial selector | 192 | 192 |
| Restore selector | 48*64*256 = 786,432 | 48 |
| Replay selector | 48*8*512 = 196,608 | 384 |
| **Total** | **983,232** | **624** |

Issued flag-load bytes fall from 3,932,928 to 2496 against the same 192-byte flag
plane. Snapshot still issues 48 flag stores. Guard OR sites and eligibility
tests are unchanged; at zero final sticky flags they execute zero flag ORs.
Repeated uniform atomic reads may already coalesce in the compiler/hardware,
so these source counts do not imply a proportional latency or bandwidth gain.

Restore adds one explicit threadgroup selector barrier/head and replay adds one
per value tile: 48+384 = **432 group barrier instances**, with **208,896 thread
barrier calls**, independent of fallback count. Invalid groups return uniformly
before these barriers. The existing inter-dispatch buffer barriers remain.

For F selected heads, restore copies **65,536F payload bytes**, issuing
**131,072F state/snapshot read-plus-write bytes**, unchanged from v5. At F=0,
neither adapter reads snapshot words or writes recurrent words. Each selected
restore thread executes exactly **64 loop iterations**; 256 threads cover all
16,384 uint words/head. Snapshot remains 3,145,728 payload bytes saved, issuing
6,291,456 state/snapshot bytes plus 192 flag-store bytes at H48.

Native replay arithmetic, writes, and internal barriers are unchanged. At R2048,
each selected replay group has 128 T16 blocks and 256 explicit recurrence
barrier instances; 8F groups give **2048F** recurrence barrier instances. No
replay recurrence executes at F=0. New `selected[1]` adds 4 declared threadgroup
bytes per adapter group; persistent workspace allocation is unchanged. Actual
pipeline resource padding and occupancy require Root's resource record.

## Head-zero audit counts

The oracle retains all 48 preparation heads even in head-zero audit. Snapshot,
apply, restore, and replay use headLimit=1; therefore preparation must not be
counted as a one-head launch.

| Audit quantity, B1 | V5 | V6 |
| --- | ---: | ---: |
| Five-phase groups | 3212 | 3149 |
| Five-phase launched threads | 824,320 | 808,192 |
| Apply/restore/replay flag loads | 4+16,384+4096 = 20,484 | 4+1+8 = 13 |
| Selector group barrier instances | 0 | 1+8 = 9 |

Preparation's coefficient/flag work for other heads is inherited behavior.
The audit mutable recurrent/output/history/delta/preoutput ownership remains
head zero; the audit replay wrapper retains `group.x != 0` rejection.

## Ownership, stride, and uniformity result

**Source review passes, contingent on the recorded ordered dispatch contract.**

- Restore validates `group.y == 0`, 256 threads, valid lane/head, aligned
  recurrent stride, and the same metadata eligibility before its barrier.
  The corrected host dispatch uses `(headLimit, v5PhaseControl ? 64 : 1, lanes)`.
  Snapshot stays unconditional `(headLimit,64,lanes)`; its original helper and
  entrypoint prefix are byte-identical to v5. A temporary host-edit mismatch was
  reported and the corrected calls were subsequently observed.
- Restore maps mutable state to `batch*recurrent_lane_stride_bytes/4 +
  head*16384 + element` and saved state to `(batch*48+head)*16384 + element`.
  Snapshot lane storage stays compact across all 48 heads, including audit.
  For every valid stride S>=3,145,728 bytes with S divisible by four, the last
  restored byte is `batch*S + head*65,536 + 65,535`. Lane padding is untouched.
  CPU caller extent/overflow validation remains required as before.
- `element=tid+256*n`, tid 0..255 and n 0..63, partitions 0..16,383 exactly:
  no holes, overlap, padding writes, or extra final iteration. `uint` assignment
  preserves original F32 bits, including subnormals, signed zeros and payloads;
  no floating-point conversion or FTZ arithmetic enters the copy.
- Only tid zero loads the stable flag into group-local `selected[1]`. Every
  valid thread reaches the threadgroup-memory barrier before testing it. The
  skip is uniform; selected groups enter the literal recurrence together, so
  its conditional/internal barriers retain their original uniformity.
- The preceding apply-to-restore buffer barrier resolves all flag OR writers.
  Restore does not modify flags, and the restore-to-replay buffer barrier
  resolves restored state writes before replay. Replay changes diagnostics,
  output and recurrence, not flags. No later flag writer may overlap either
  selector phase. No extra end-of-loop shader barrier is needed between
  independent restore words; the retained dispatch boundary handles consumers.
- Native and audit helper/control prefixes are byte-identical to v5. Only their
  adapters replace per-thread flag loads with leader-load/barrier/uniform-skip.
  Candidate/preparation/cache source is byte-identical to v5; original numeric
  operations, eligibility, reasons, sticky diagnostics, allocation extents and
  canary contracts are not changed by these adapters.

Anchors under `dev/benchmarks/gdn_nax_chunks_sep21_v6/`:
`snapshot.metal:4-34` (unchanged snapshot), `:42-55` (restore);
`native_fallback.metal:114-126` (native selector);
`native_fallback_audit.metal:153-165` (head-zero selector).
Oracle phase bindings retain snapshot y64, conditional restore old64/new1, replay
y8, and buffer barriers between snapshot/prep/apply/restore/replay.

Reviewed adapter SHA256 values:

| File | SHA256 |
| --- | --- |
| `snapshot.metal` | `35cecd82752d29c72251655eb66c605642d94493299a3920d33c1e79b560dd1d` |
| `native_fallback.metal` | `fe2eef4e818f6380e005840bc33877590a100422de35067e0e8594471eb98087` |
| `native_fallback_audit.metal` | `f0ad01c2567d8fb2df7f229f9353d6b8b4d9393c0f9fc9bc65c22c89e43deae7` |

**Timing hypothesis only:** fewer source flag loads and restore launches may
reduce zero-flags overhead; the 432 added selector group barriers may offset
some benefit. Selected restore work now runs as 64 iterations/thread instead of
64 separate groups/head, with identical bytes. Its timing effect is unmeasured.
No phase bottleneck, speedup, occupancy improvement, or numerical GPU pass is
inferred from these counts. Root must establish v5/v6 byte equivalence and
measure the unchanged snapshot/prep/apply plus adapted restore/replay together.
