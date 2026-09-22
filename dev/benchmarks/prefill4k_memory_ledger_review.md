# CPU audit: wide singleton plus Full512 memory ledger

This is a read-only source/metadata audit of the current FlashNext profile,
capacity 16384, singleton arena 8192, draft depth 3 / verify rows 4, batch arena
2048 rows per lane, direct-A padding enabled, PLE SSD streaming enabled, and
route capture disabled. No model was loaded and no GPU work was submitted.
The arithmetic below is independently reproduced from source; it is not
measured candidate admission or physical-residency evidence.

## Governing formulas

The automatic physical capacity is 274,877,906,944 bytes. Host reserve is
`max(16 GiB, physical / 10) = 27,487,790,694`; engine limit is
`physical - reserve = 247,390,116,250` bytes. Do not subtract host reserve from
this limit again.

At every actual reservation, `MemoryGovernor` requires:

```
O = max(backend allocatedBytes + sparseResidentBytes, device currentAllocatedSize)
O + outstandingReservationBytes + nextStagePlannedBytes <= engineLimit
hostAvailable - hostReserve - outstandingReservationBytes - nextStagePlannedBytes >= 1 GiB
effectivePressure == Normal
```

Loss of host telemetry is a refusal. Recovery after pressure requires 2 GiB.
The governor is currently constructed after original Flash weights load.
Reservation commit releases the reservation once allocations are represented
in the backend; reservations are not additional backing in the final ledger.

The complete before-mapping plan should sum these categories:

```
P_static = P_originalMappedWeights
         + P_trunk
         + P_sequentialMTPHead
         + P_batchDecode
         + P_batchPrefill + P_ownedBatchPrefillHidden
         + P_batchMTPPriming + P_ownedBatchMTPPrimingInput
         + P_jointVerify
         + P_jointMTPHead + P_ownedJointHidden
         + P_enabledIdleMaintenanceDiagnostics

P_requests(N, E) = N * FlashForward::requestStateBytes(capacity)
                + E * (FlashMTPForward::requestStateBytes(capacity)
                       + singletonPolicy.hiddenBytes())

P_engine_upper = P_static + P_requests + P_driverPipelineAllowance
```

`0 <= E <= N <= 4`. CPU parsing, request objects, PLE caches and transport/HTTP
memory need a separate host ledger; they are not Metal allocation bytes.

The exact trunk expression used by `FlashWorker` is:

```
P_trunk = FlashForward::workspacePlannedBytes(C, M, V)
        + FlashForward::expertCachePlannedBytes(weights)
        + FlashForward::floatDenseCachePlannedBytes(weights)
        + FlashForward::int8HeadPlannedBytes(weights)
        + optional worker online-QSA allowance
        + optional FlashDenseCache::plannedBytes(weights, defaultPrefixes(..., true))
        + optional flashMoEBlockedWorkspacePlannedBytes(M, 10)
```

For the audited flags and geometry, source-derived conservative arithmetic is:

| Trunk term | Planned bytes |
|---|---:|
| Forward fixed scratch upper bound | 3,623,387,136 |
| Full512 payloads plus ranks | 121,174,228,992 |
| F32 dense maps plus diagnostics | 14,391,721,984 |
| BF16 dense maps plus diagnostics | 8,467,267,584 |
| INT8 vocabulary head extra buffers | 32,768 |
| Worker online-QSA allowance | 12,713,984 |
| Blocked MoE scratch | 944,947,200 |
| **Trunk total** | **148,614,299,648** |

The vocabulary head reuses original Q8 codes and float-dense padding. Adding
another 635,699,200-byte code copy would describe a different storage route.
Forward's fixed estimate already includes padding and optional route capture
allowances, lazy verifier rollback, PLE staging, QSA and greedy scratch. Its
constructor's actual `workspaceBytes()` also includes all immutable caches.
Never add saved map bytes again to a measured `workspace_bytes` value.

The other source-derived conservative planner terms, with current flags and
idle maintenance disabled, complete the CPU engine ledger:

| Planned category | Bytes |
|---|---:|
| Original mapped weights | 74,317,889,536 |
| Trunk | 148,614,299,648 |
| Sequential MTP scratch | 241,434,624 |
| Sequential MTP BF16 cache including diagnostics | 178,274,304 |
| Batch decode | 18,857,984 |
| Batch prefill | 4,351,918,080 |
| Owned batch-prefill hidden | 167,772,160 |
| Batch MTP priming | 288,964,608 |
| Owned priming input | 10,485,760 |
| Joint verifier | 522,518,528 |
| Joint MTP head | 17,154,048 |
| Owned joint hidden | 327,680 |
| **Static planner subtotal** | **228,729,896,960** |
| Static plus one eligible request | 229,351,866,368 |
| Static plus four eligible requests | 231,217,774,592 |

The four-request planner leaves 16,172,341,658 inside the engine ceiling before
an explicit driver/pipeline allowance. These are conservative backing bounds
derived independently from source formulas and metadata, not candidate actual
allocations. The generated MTP BF16 cache has sixteen matrices; infer logical
K from source scale columns times the effective per-projection group size,
not by blindly multiplying packed U32 columns by eight for all quantization
bit widths. The native private planner should be the authoritative arithmetic
check and emit every category, rather than treating historical peaks as exact
planning bounds.

## Exact metadata bytes versus upper bounds

Full512 has exactly 48 layer payloads, each 2,524,446,720 aligned bytes:
2,516,582,400 code bytes and 7,864,320 F32-scale bytes. Each layer also needs a
16,384-byte ID/rank buffer. Therefore `totalBytes = 121,173,442,560` and
`plannedBytes = 121,174,228,992`. Payload plane views do not add backing.
Metadata loading validates all shape/offset/length arithmetic, canonical file
names, sorted unique selected IDs, exact readonly file extents, and bound
source identity. It does not validate payload hashes or content; the consuming
constructor validates those before exposing operands and verifies each layer
fits device `maxBufferLengthBytes` and actual allocation delta <= planned.

Production metadata admits only 32/64/128 inventories. Full512 requires the
explicit private combined-overlay inventory guard, shader guard, and miss
dispatch removal; a plan file alone does not enable it.

Original PLE-SSD mapped weight windows remain 74,317,889,536 bytes, including
original target expert and trained MTP banks. Another 32,002,539,520 original
payload bytes are disk-only. Removing large-row Q4 miss dispatches does not
remove original mappings: small-row decode retains them. Original shard/tensor
views must be deduplicated by base allocation in the CPU metadata plan.

Saved target BF16 payloads total 8,467,251,200; saved F32 payloads total
14,391,705,600. Their constructors add individual diagnostic buffers. The
sequential MTP head also owns a generated BF16 cache of trained MTP matrices
and head scratch; the current saved target operand manifest has no `mtp.*`
entries. Plan the MTP dense cache separately from original MTP dimensions.
Batch/joint arenas share immutable coefficient owners and do not allocate
duplicate model caches. Their owned input-feature staging remains separate.

Saved residency references already charged backing; registration union bytes
are diagnostic and must not be added again. Residency is not proof of physical
pinning or per-die placement. Full512 safely fails the current idle-maintenance
qualified-union byte predicate. Explicitly disable maintenance for matched
comparisons rather than assuming the top64 union contract remains valid.

## Reconciliation with retained same-source baseline

`build/release/flash/prefill4k-current-context-baseline.json` has capacity 16384
and singleton arena 2048. Static category counters are:

| Actual baseline category | Bytes |
|---|---:|
| Original mapped weights | 74,317,889,536 |
| Trunk including all caches | 39,390,550,284 |
| Sequential MTP head | 419,708,928 |
| Batch decode | 18,789,056 |
| Joint verifier | 522,481,920 |
| Joint MTP head | 17,133,568 |
| Owned joint hidden | 327,680 |
| Batch prefill | 4,351,833,804 |
| Owned batch-prefill hidden | 167,772,160 |
| Batch MTP priming | 288,964,608 |
| Owned priming input | 10,485,760 |
| **Static category sum** | **119,505,937,304** |

Initial `memory_actual.current_bytes = 119,506,337,792`, 400,488 bytes above
that category sum. The current counter is the maximum of the native allocation
ledger and sampled device allocation counter, so this small remainder must
not be silently discarded or named as a proven allocation category.
Measured peak is 120,129,617,920. Trunk state is 582,959,104 per request;
eligible MTP state plus committed hidden is 39,010,304, giving 621,969,408 per
eligible request. Four eligible requests need 2,487,877,632 state bytes.

For arena 2048 -> 8192, the actual row-allocation increase with capture off is:

| Row-dependent term | Increase bytes |
|---|---:|
| 39 BF16 scratch buffers (sum widths 197859) | 2,431,270,912 |
| Token IDs / decay / expert IDs / ngram IDs | 2,506,752 |
| Blocked MoE buffers and buckets | 708,345,856 |
| Singleton PLE SSD staging | 10,616,832 |
| **Actual row subtotal increase** | **3,152,740,352** |

QSA and vocabulary rows remain capped at 128. Verification geometry stays 4.
The fixed planner additionally admits route capture regardless of active flag;
its increase is 23,592,960, making conservative planned wide increase
3,176,333,312. Per-allocation 16 KiB rounding is required; multiplying the
whole baseline workspace by four would multiply immutable caches incorrectly.

The existing Full512 preflight substitutes the top64 planned store bytes only:

```
120,129,617,920 - 15,147,466,752 + 121,174,228,992 = 226,156,380,160
```

Adding the actual wide-row increase gives 229,309,120,512 bytes, leaving
18,080,995,738 under the engine ceiling. This is a retained-workload peak
extrapolation, not a full planner bound or actual candidate measurement.
The extrapolated static category sum is 228,685,439,896; with four eligible
requests it is 231,173,317,528, before unknown device/pipeline/host overhead.
`minimum_free_bytes` in the existing fullcache preflight is destination disk
space, not host memory availability.

## Host ledger and measurement

The governor estimates host availability with:

```
usedPages = active + inactive + speculative + wired + compressor
          - fileBacked - purgeable
hostAvailable = physical - usedPages * pageSize
```

A CPU-only `vm_stat`/`sysctl` sample at 2026-09-21 16:39:01.466 UTC had page
size 16384; active 3924640, inactive 11567963, speculative 203959, wired 459089,
compressor 177014, file-backed 13651876, purgeable 186041. This gives
2,494,748 used pages / 40,873,951,232 bytes, host available 234,003,955,712,
and 206,516,165,018 above reserve before outstanding reservations. Free pages
were 387617 / 6,350,716,928 bytes; free alone is not the governor estimator.

The estimate credits reclaimable file-backed pages. A blanket comparison of
the entire projected engine allocation ledger to unloaded host headroom is a
conservative all-backing-resident assumption, not the phased runtime host gate.
Actual runtime checks use fresh host availability and the next stage's planned
reservation after earlier mappings/allocations exist. Do not infer physical
residency from mapped file bytes or convert the historical warmed top64 host
sample into a fresh unloaded-startup admission decision.

The PLE host cache counter is 50,331,648 against a 67,108,864 configured budget;
read scratch is bounded to 1,048,576. Wide lookup also owns temporary ID/miss
vectors bounded by 16 * actual singleton rows and segment vectors bounded by
48 * actual singleton rows; these capacities
and C++ allocator overhead are not in the engine Metal sum. Request tokens,
logits, masks, transport frames, command metadata and the Python HTTP process
likewise need measured host/process footprint rather than a fabricated exact
metadata total.

## Counters missing for complete runtime qualification

- `reserved_bytes`, `observed_resident_bytes`, `host_headroom_bytes`, separate
  device current/peak and native simultaneous resident peak in Flash status.
- Aggregate live request-state bytes and per-admission allocation delta versus
  its reservation; request construction currently lacks the startup delta check.
- Startup category reconciliation and a budgeted driver/pipeline allowance;
  lazy pipeline growth is observed after compilation but not individually reserved.
- Host-availability minimum and process resident/physical-footprint peak over
  initialization and requests, including PLE host-vector peaks and HTTP memory.
- Peak-budget validity. `memory_audit.valid` currently checks current <= limit;
  an earlier peak over the limit is not rejected by that boolean.

Source entrypoints: `FlashWorker.mm:3176-3410`, `FlashForward.cpp:486-561`,
`FlashForward.cpp:811-823`, `MemoryGovernor.cpp:15-70,143-207`,
`FlashInt8ExpertStoreMetadata.mm:245-350`, `FlashInt8ExpertStore.mm:130-186`,
`FlashMoEBlocked.cpp:180-195`, `FlashWeights.mm:441-471,808-812`,
`FlashMTP.cpp:126-141,449-505`, and `FlashPLESSDStore.cpp:154-173,322-358`.
