# Guarded WY T32/SG8 worker integration plan

This is a CPU-only source/metadata plan, written before the standalone v3 guard
proof exists. It does not authorize a whole-worker build or GPU execution. Root
must first accept the v3 guard/source certificate and unchanged numerical gates.
No production runtime, parent sealed build, public file, model payload, or root
W/U implementation is modified by this plan.

## Frozen base and evidence

Use `build/dense-w8a8-sep21-worker-v3`, the pure dense-W8 worker, not
`build/dense-w8a8-hybrid-sg2-tail-fma-sep21-worker-v1`. The pure base records
`dense_w8a8_hybrid_parent=false`, inherited route
`private-inherited-dense-w8a8-prefill-sep21-v1`, and parent
`build/adaptive-expert-tail-sg2k128-fma-sep21-worker-v1`.

Its `.config` identity is:

```
26ff33bbeeaa8d04-private-wide-fullcache-v1-qsa-bulk-sg8-v1-pointwise-exact-sep21-v1-fixed-sg2-k128-main-prefill-sep21-v1-exact-sg2-k128-m16-tail-optional-qualified-fma-sep21-v1-dense-w8a8-prefill-sep21-v1
```

Relevant current metadata/source seals, obtained from targeted reads/hashes:

| Relative path inside base | SHA256 |
| --- | --- |
| `overlay-manifest.json` | `6e973f0ceea7d9a425b0c9084335c5c2261418568094d7daef9ec119fb02b1db` |
| `qualification/source-manifest.json` | `2b1c47c480bee071c81d6a9dd5e5215de6665f978b2392930c839079e0f0133b` |
| `cpu-witness-v1.json` | `3ee06d2df4e7868e6fe60654c1865f825f2e66fbeec2e11168f7313581e53609` |
| `link-inputs.mk` | `1447f2d65b18b1f30d70fbb9f6f4309fd8b6dfa418413f82eb96f3be4d137035` |
| `source/runtime/flash/FlashForward.cpp` | `b65aa23b1f299ac985c63b9b460083c4fa0bfdf0e223b0688b3a9bbcfe5d2edf` |
| `source/runtime/flash/FlashWorker.mm` | `e2c1c1ae750529d21928b3cc30015b2d5db5fb45562b57ba48c3b5746347297e` |
| `source/runtime/flash/FlashGDNStaged.cpp` | `80038a094335404022a78c87aae066714cbdcfe74aa8f812181f7174234a7ac4` |
| `source/runtime/metal/kernels/shared/flash_gdn_staged.metal` | `e500cf0563fb84fe6dbb966cab8f38aa0562bcc9b103c5e2f28a212f8e174013` |
| `source/dev/benchmarks/gdn_chunk_sep21/scalar_fma.metal` | `166381991f5444585e7db3634f41432322efdd66bf08a8c402d5fed49e58c699` |

The recorded CPU witness passes and explicitly records no GPU or model/payload/
cache construction. That is recorded component/closure evidence; this planning
pass has not independently rehashed all parent artifact inputs.

## Exact source anchors

All worker anchors below are in the base's frozen `source/` tree:

| Path | Anchor | Required change |
| --- | --- | --- |
| `runtime/flash/FlashGDNStaged.hpp` | lines 23-28, existing helper ABI | Keep the old entrypoint available. Add a separately named WY helper or overload accepting an explicit workspace; do not force decode/ILP consumers to change. |
| `runtime/flash/FlashGDNStaged.cpp` | lines 17-24 native selector, 30-33 FMA override, 34-40 validation/params | WY branch must take priority before the FMA recurrence dispatch. Keep flag-zero behavior and frozen FMA selection byte-for-byte semantically unchanged. |
| `runtime/flash/FlashGDNStaged.cpp` | lines 41-53 prepare/recurrence/output/carry | Replace recurrence only for eligible WY; preserve fused preparation, BF16 recurrence-row boundary, normalization/output, and convolution carry. |
| `runtime/flash/FlashForward.cpp` | lines 145-146 flags, 164-167 shared arena fields | Add a frozen WY flag and one optional owned workspace next to other shared workspaces. |
| `runtime/flash/FlashForward.cpp` | lines 240 and 385-387 allocation delta | Allocate within this measurement interval; verify actual physical extent equals WY planned extent. |
| `runtime/flash/FlashForward.cpp` | lines 350-357 bulk-QSA allocation/admission check | Follow this pattern for explicit WY plane/arena checks. |
| `runtime/flash/FlashForward.cpp` | lines 593-642 `workspacePlannedBytes` | Add the same WY physical planned bytes used by construction, before the worker calls `tryReserve`. |
| `runtime/flash/FlashForward.cpp` | lines 664-668 staged route marker | When WY is enabled, report WY marker instead of the active FMA stage marker. Retain FMA as inherited available fallback scope only; guarded native fallback is non-FMA. |
| `runtime/flash/FlashForward.cpp` | lines 863-888 `batchValidateExternalDestination` | Reject overlap with WY coefficient/snapshot/flag views if Shared. Private backing must retain its storage contract. |
| `runtime/flash/FlashForward.cpp` | lines 1091-1140 verification branch | Leave lazy tickets, capture tapes, prefix copies, and replay/restoration unchanged. |
| `runtime/flash/FlashForward.cpp` | lines 1141-1152 nonverification branch | First try eligible WY. Otherwise execute the frozen staged/FMA/fused/separate chain. |
| `runtime/flash/FlashWorker.mm` | lines 3243-3247 pre-backend flag freeze | Parse/freeze WY strict `0|1` selector here, before paths, metadata, backend, or model. Enabled requires staged=1. |
| `runtime/flash/FlashWorker.mm` | lines 3330-3333 planner, 3383-3389 reserve/constructor/check/commit | WY bytes must already be in Forward's planner. No post-reservation hidden allocation. |
| `runtime/flash/FlashWorker.mm` | lines 2886-2903 identity/status fields | Bind WY numerical identity, kernel source SHA, guard policy SHA, native fallback SHA, effective selection, workspace plan/actual bytes, and scope. |
| `runtime/flash/FlashBatchPrefill.cpp` | lines 145-146 staged marker check | Accept WY marker only if batch WY is in scope. |
| `runtime/flash/FlashBatchPrefill.cpp` | line 279 trunk/batch lock; lines 420-441 ILP/serial staged route | Borrow trunk arena under the existing lock; enabled WY must precede ILP for eligible prefill if the flag promises override across batch prefill. |

`runtime/metal/CommandGraph.hpp:17-19,37-47` binds buffers contiguously and
places the parameter bytes at `buffers.size()`. The v2 oracle apply entrypoint
uses params at 6 and prepared at 10 (`gdn_nax_chunks_sep21_v2/candidate.metal:
281-285`), so it cannot be inserted unmodified into `graph.add`. Use distinct v3
worker entrypoints with contiguous bindings and a captured ABI header. Do not
change CommandGraph globally just to preserve the oracle's sparse indices.

## Arena ledger and lifetime

For T=32, K=V=128, H=48, rows R, lanes B, let C=ceil(R/32). All multiplication
and alignment must be overflow-checked before backend allocation or graph edits.
The coefficient stride is 13,344 F32 elements, 53,376 bytes, per chunk/head:

| Field | F32 elements | Bytes | Byte offset in chunk/head |
| --- | ---: | ---: | ---: |
| W | 4096 | 16,384 | 0 |
| U | 4096 | 16,384 | 16,384 |
| end-key E | 4096 | 16,384 | 32,768 |
| score | 1024 | 4096 | 49,152 |
| prefix | 32 | 128 | 53,248 |

Base layout is `((lane*C + chunk)*48 + head)*53,376`, then the field offset.
Tail rows are zero-padded inside the fixed T32 stride. The endpoint is the last
active token, not the padded final position. Source/CPU layout anchors are
`gdn_nax_chunks_sep21_v2/wy_cpu.hpp:13-39`, candidate `:28-29,49`, and oracle
`:703-744`. V3 must retain this layout or explicitly reseal a changed ABI.

| Device storage | General logical bytes | R2048/B1 logical bytes | R2048/B1 physical bytes under separate 16 KiB rounding |
| --- | ---: | ---: | ---: |
| coefficients W/U/E/score/prefix | B*C*48*53,376 | 163,971,072 | 163,971,072 |
| immutable initialStateSnapshot F32 | B*48*128*128*4 | 3,145,728 | 3,145,728 |
| U32 per-head guard flags | B*48*4 | 192 | 16,384 |
| **added total** | sum above | **167,116,992** | **167,133,184** |

A single packed backing allocation rounded to 16 KiB also totals 167,133,184
bytes at R2048/B1. Snapshot starts at byte 163,971,072; flags start at
167,116,800; 16,192 trailing alignment bytes are unused. All views must be
nonoverlapping and retain their own logical extents. Do not report logical 192B
flags as their full physical allocation if they receive a separate buffer.

At B32, coefficients alone need 5,247,074,304 logical bytes; total logical added
storage is 5,347,743,744 bytes. Do not silently allocate that envelope for the
singleton worker. Singleton Forward provisions B1 and a bounded prefill window
up to min(maximumRows,2048); wider native windows remain on inherited routing
unless independently split/qualified. Generic standalone/helper eligibility is
R64..2048/B1..32 with an explicitly adequate workspace.

Own one arena per Forward, not per request or per GDN layer. All 36 GDN layers
reuse the arena in graph order. Snapshot and flags are refreshed on the GPU
before each eligible layer. Never clear all flags on the CPU while constructing
the 48-layer graph: that would only clear before submission, not between layers.
No host tensor reads or waits enter the warm/timed forward path.

BatchPrefill currently has four separately allocated request states and serial
lane dispatches (`FlashBatchPrefill.cpp:430-441`). Those calls can borrow the B1
Forward arena sequentially under its already-held mutex, as long as batch rows
fit the trunk's provisioned WY row bound. This adds no second batch allocation.
If a new true B-lane dispatch is desired, provision/reserve that B-lane arena
explicitly; separate recurrent state allocations cannot be represented by a
fake stride. No layer-local or request-owned arena may be introduced.

Scratch/diagnostic ledger:

- Existing `FlashGDNBuffers` mixed BF16, decay F32, beta BF16, recurrentRows
  BF16, and output BF16 remain unchanged (`FlashGDN.hpp:43-55`).
- Existing sticky U32 diagnostics is 4 logical bytes, already provisioned as a
  rounded16K Forward plane (`FlashForward.cpp:342`; planner `:608-610`). Reuse it;
  expected guarded fallback must not set the native nonfinite/invalid error bits.
- Head guard flags are the new 192 logical bytes above. OR guard reasons across
  preparation/application phases; never erase an earlier chunk's reason.
- Preparation v3 threadgroup scratch is `3*T*T + 3*T` F32 plus T raw alpha U32
  words = 12,800 declared bytes. Apply V32/T32/SG8 v3 declares 8192 base bytes
  plus state/weight/condition norms and a skip U32 = 8516 declared bytes. These are per resident
  threadgroup resources, not additional persistent GPU buffers. Actual resource
  declarations may grow for the guard; reseal/count them before Root GPU use.
- Standalone audit histories/delta/pre-BF16 audit planes and immutable benchmark
  seed are qualification-only and must not be included in the worker allocation.
  InitialStateSnapshot is the worker's required fallback seed.
- Any new guard diagnostic counters beyond head flags must be listed, rounded,
  reserved, measured, and bound into workspace identity. No implicit scratch.

## Guarded dispatch and original native fallback

Proposed per-layer ordered graph chain:

1. GPU snapshot the entire incoming recurrent state into immutable snapshot and
   GPU clear B*48 guard flags. Preserve convolution state as native preparation
   already requires; preparation does not consume the carry state prematurely.
2. Run inherited `flash_gdn_fused_prepare` with the same BF16 q/k/v and gates,
   F32 decay, norm epsilon, state strides, and diagnostics (`GDNStaged.cpp:41-45`).
3. Guarded WY T32/SG8 preparation fills coefficients and ORs per-head flags.
   Decide based on original BF16/F32 inputs and the captured immutable snapshot,
   not an already-mutated WY state. Padded tails must not cause false guard math.
4. Guarded WY application processes all chunks in order and writes only eligible
   unflagged heads. Guard failure is a route decision. If application can discover
   a new guard failure, it may leave provisional outputs/state; a later separate
   dispatch must fully replace that head before downstream output reads.
5. Separate flagged-head native recurrence restores every flagged head from the
   immutable snapshot, recomputes every supplied row, and writes that head's
   full recurrentRows and final recurrent state using the captured original
   `flash_gdn_staged_v16_t16` algorithm. Unflagged heads return without writes.
6. Run inherited `flash_gdn_output` and `flash_gdn_convolution_carry` once
   (`GDNStaged.cpp:49-53`).

Native fallback must be a captured non-FMA original staged-v16-t16 implementation
with only an entrypoint/head-flag adapter. Capture
`base/source/runtime/metal/kernels/shared/flash_gdn_staged.metal`, whose scalar
recurrence and contraction-off/reassociation-off pragmas are at `:6-7,70-100`,
and whose exact v16/t16 entry is at `:127`. Hash captured source and adapter
separately. Never call the existing `addGDNStagedPrefill` recursively for fallback:
that helper routes to `private_gdn_scalar_fma_v16_t32` when inherited FMA flag is
1 (`base GDNStaged.cpp:30-33`). Do not replace the captured original fallback
with the FMA AIR or a numerical-equivalent variant.

Each head is 65,536 snapshot bytes. Restore/read it with tight snapshot indexing
but retain original caller recurrent lane stride for the mutable destination.
Fallback recomputes the full window from snapshot, not just a failed chunk or
from partially advanced WY state. Every output/state location has one effective
writer after the fallback dispatch; no in-dispatch cross-threadgroup restore or
flag race is acceptable.

The graph is an ordered dispatch list. Current backend uses a serial compute
encoder and tracked buffer resources (`runtime/metal/MetalBackend.mm:2325-2356`)
and has no explicit normal-mode barrier primitive in CommandGraph. The oracle
uses an explicit preparation/application buffer barrier (`v2/metal_oracle.mm:
278-280`). The worker proof must explicitly establish producer/consumer hazard
ordering for snapshot/prepare/apply/fallback/output on this existing backend.
If an explicit barrier is necessary, add a private backend/graph adaptation with
its object in the closure; do not rely on timestamp profiling barriers, because
normal unprofiled execution uses the path at `:2353-2354`.

## Flag precedence, identity, and status

Suggested strict flag: `SPLASH_FLASH_GDN_WY_PREFILL_SEP21_V3=0|1`; missing means
0. Freeze once before backend. Flag 1 requires `SPLASH_FLASH_GDN_STAGED=1`.
Inherited FMA, dense-W8, SG2, tail, pointwise, and QSA selectors stay independent.

For an eligible nonverification staged-prefill rectangle:

| WY | Inherited FMA | Effective recurrence |
| --- | --- | --- |
| 0 | 0 | original frozen staged selector |
| 0 | 1 | frozen scalar FMA bridge |
| 1 | 0 or 1 | guarded WY T32/SG8, per-head captured non-FMA v16/t16 fallback |

Outside R64..2048/B1..32, or outside prefill, execute the frozen selector chain.
Decode, verify, lazy rollback, prefix capture, replay, and pending tapes stay
unchanged. Do not expose WY marker for a physically absent arena or unsupported
row bound. Suggested route marker:
`;private-gdn-prefill-wy-guarded-v32-t32-sg8-r64to2048-lanes1to32-native-v16t16-fallback-sep21-v3`.

Bind numerical identity to the effective active WY route, candidate source SHA,
guard policy source SHA/constants, captured native SHA+adapter SHA, and BF16/F32
layout policy. When WY is enabled, do not claim active FMA numerical identity
merely because the inherited FMA flag is still 1: it has been overridden for WY
eligible prefill. Retain inherited dense/SG2/tail identities and include selected
WY/native fallback markers. Workspace identity additionally binds ABI/plane
offsets, configured maximum rows/lanes, 16K alignment policy, and physical planned
bytes. Kernel source and resource plan are CPU metadata, not model fingerprints.

Add status fields near `FlashWorker.mm:2898-2903`: requested/effective WY,
numerical policy, source/guard/native hashes, scope, configured arena rows/lanes,
logical coefficient/snapshot/flags bytes, physical planned/actual arena bytes,
and `whole_model_qualified=false` until Root completes service qualification.
Any fallback counters read only after the existing command completion; distinct
expected fallback reasons must remain distinguishable from sticky failures.

## Necessary private build closure

Create a new worker overlay/build directory after accepted standalone guard
proof. Copy only frozen source and sealed artifact inputs; never edit the parent
or promote to production runtime. Base machinery `machinery/worker.mk:7-9,28-29`
links OWN + REUSED + CORE; `:43-44` links dense W8 AIR + AIRS.

Rebuild at minimum:

- `FlashForward.o` for arena owner, planner, route marker, eligible branch, and
  overlap checks.
- `FlashWorker.o` for strict early flag freeze and numerical/resource identity.
- A private WY bridge/workspace object if it is not header-only.
- `FlashGDNStaged.o` only if the existing helper is extended/overloaded; remove
  inherited `reused/base/host/FlashGDNStaged.o` from effective link inputs to avoid
  duplicate definitions. Keeping it unchanged and adding a separately named WY
  helper minimizes transitive changes.
- `FlashBatchPrefill.o` if WY is routed in serial batch lanes or its stage marker
  validation needs updating. Remove inherited reused BatchPrefill.o accordingly.
- `FlashGDNBatchILP.o` only if the original helper ABI is changed; retaining its
  original ABI avoids this dependency. Do not change ILP decode/verify behavior.

Retain the sealed `worker_cache.o`, all other inherited REUSED/CORE objects, and
all native/FMA/dense/SG2/tail AIRs unchanged unless a documented adapter requires
replacement. Compile new guarded-WY AIR and original-native flagged adapter AIR;
keep original `flash_gdn_staged.air` as well. Names must be private/distinct.

Base frozen recipe has 45 REUSED, 4 CORE, 72 AIRS; base OWN has Forward, Worker,
worker_cache (total effective 52 host objects with OWN). Relevant reused seals:

| Input | SHA256 |
| --- | --- |
| `reused/base/host/FlashGDNStaged.o` | `64fc82c897f1494f8c8aeb49366fbaaa39cf459e11e9810a9de31a3ee24f6323` |
| `reused/base/reused/sg2-base/reused/base-metal/flash_gdn_staged.air` | `cb6ad7737adc862a20d3c5018612509e234f70d3362014ade7d2d6324795de34` |
| `reused/base/reused/qualified-fma/gdn-prefill-fma.air` | `0323e3024e79ebe0e9d0030680262e325f08e5706bcd00e40aaab3e5bac83de4` |

Use captured C++20/O3/Werror, private source include roots, macOS27 minimum,
ARC, SPLASH_INT8_EXPERIMENT=1; Metal4.1/O3/Werror and matching private ABI include.
Record all regenerated `.d` dependencies and private headers. Include the
immutable base manifests, original/candidate/adapter source seals, build recipe,
compiler inputs, all linked artifacts, and self-contained relative closure in
new overlay and CPU witness. Verify flag-zero route/planner/native source
preservation, strict malformed/dependency flags, geometry+overflow+tight/strided
state checks, physical admission==construction, and the existing CPU self-test.
Do not run a whole worker until standalone v3 guard proof is accepted.
