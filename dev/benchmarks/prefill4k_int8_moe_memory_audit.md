# CPU audit: replacing retained trunk Q4 sources with INT8 expert operands

This is a bounded source/metadata audit. It loads no model, submits no GPU
work, scans no coefficient payloads, and changes no production code/defaults.
Numbers below describe native backing bytes, not measured physical residency,
host availability, or a demonstrated prefill/decode speedup.

## Recommendation

The simplest useful private experiment is Full512 INT8 for **all trunk row
counts**, together with a loader that never maps the replaced original trunk
expert tensors. It removes the original target-expert backing and avoids the
current duplicate Q4/INT8 weight inventory. Keep `mtp.layers.0` on its original
trained Q4 route. First qualify the small-row kernel and the resulting model
policy; then test the memory-saving loader independently.

Full512's saved GPU-interval prefill rate was approximately 2,966 tokens/s,
while host pressure damaged wall time. Source replacement could restore that
wall rate; the audit does not imply it reaches 4,000 tokens/s. Matrix/kernel
tuning and remaining layer costs still matter.

Top256 with cold-only Q4 backing is an alternative if Full512 decode/quality
is unattractive. It has more loader, argument-table, and miss-kernel work.
Removing raw bytes requires replacing hot experts in **small-row decode** as
well; prefill-only hits do not make their raw sources redundant.

## Current ownership and use

`FlashWeights::load` maps complete source shards, or in PLE-SSD mode the
checked non-PLE windows, wraps them with `wrapSharedMemory`, and makes tensor
views. Views retain their complete base allocation; making narrower
`backend.view` objects does not reduce backing or accounting. Original PLE
window geometry is hardcoded and must receive a distinct private layout
policy if experts become disk-only.

`FlashInt8ExpertStore::Impl::Layer` additionally copies the nine source
`FlashTensor` objects and three projection records for every trunk layer.
Those copies retain raw allocations independently of `weights.bases`.
Clearing only the base list cannot release them. Do not construct replaced
raw GPU views in the first place; preserve source geometry/identity as disk
metadata and omit source ownership where the inventory is complete.

The current Full512 overlay only suppresses the final miss dispatches.
`addGateUp` and `addDownScatter` still construct a temporary raw-Q4 graph
using `addMoEBlockedGateUp/DownScatter`, then copy its producer parameters.
Those validators require valid raw GPU buffers even with complete coverage.
Source removal therefore also needs a source-independent blocked-scratch/hit
validator and direct construction of the typed hit parameters. Keep the same
poison/sanitize precursor behavior and complete scratch/alias checks; do not
fake raw GPU buffers or silently bypass validation. Raw disjoint checks must
only run for actual raw operands. Completed tickets and residency leases
also retain complete allocation owners, so rebuilding at startup is clearer
than trying to release raw buffers during a serving session.

The raw routes that must be covered before removing the 48 trunk layers are:

| Consumer | Raw route |
|---|---|
| `FlashForward.cpp:1116` | `rows < 256`, including singleton decode and short prefill/tails |
| `FlashBatchForward.cpp:432` | all batched target decode rows |
| `FlashBatchVerify.cpp:548` | all batched target verification rows |
| `FlashBatchPrefill.cpp:498` | flattened batch rows below 256 |
| `FlashInt8ExpertStore.mm:175` | copied raw miss-source ownership |

`FlashMTP.cpp:598` and `FlashBatchMTPForward.cpp:406` use `mtp.layers.0`,
which is outside the 48-layer target store and must remain mapped.
Shared experts, routers, vocabulary head, GDN/QSA/HC and other operands also
remain available. An immutable-residency list or a graph that omits raw
bindings is not proof that the loader released their allocation.

## Numerical contract

Original scales/biases are **BF16 storage promoted to F32**. The default
selected-expert Q4 specialization reconstructs each coefficient in F32 and
multiplies by the BF16 activation promoted to F32, with the qualified lane-K
and SIMD reduction order. The optional expert-QMV route also uses F32
coefficients, with a separately tagged contiguous-K reduction order.

Saved INT8 codes were requantized from the blocked-prefill reference:

```
w_ref = BF16_RNE(F32_add(F32_mul(q4, source_scale), source_bias))
code, row_scale = symmetric_int8(w_ref)  # signed -127..127, F32 row scale
dot = F32_accumulate(BF16_activation * signed_int8_code)
output = BF16_RNE(F32_mul(dot, row_scale))
```

Using those codes for small decode changes the original F32 coefficient
policy, adds INT8 quantization error, and may change accumulation order.
Exact sidecar byte preservation or the existing coefficient certificate is
not original-model output parity. Cancellation fixtures require absolute
bounds; a relative bound near zero is invalid. New semantic/cache identities
must distinguish this policy so saved prefixes and kernel routes cannot mix.

## Bounded small-row primitive

Add off-default gathered INT8 projection entry points that directly index
the validated expert-ID-to-rank map. Start with separate gate, up and down
outputs and the existing BF16 SwiGLU/combine kernels. Grid dimensions can
mirror gathered Q4: output blocks, logical rows, selections. Each lane
accumulates `float(BF16_activation) * float(signed_code)` in F32, then one
positive finite row scale is applied after reduction and the result is
rounded to BF16. Down input is per selection; gate/up input is per row.
This needs no bucket packing or large-row scratch.

Qualify rows 1, 2, 3, 4, 8, 16, 32, 128 and 255 against an independently
declared candidate reference, including max IDs/ranks, duplicate experts,
invalid-ID/rank rejection, signed zero/subnormals, zero coefficient rows,
alternating-sign cancellation, sticky diagnostics, canaries, disjoint
writables and retained-mapping replay. Then compare same-prompt target-only
and depth-3 speculative generation, first-token acceptance and accepted
prefix lengths. Gate/up fusion or tiny MPP tiles are subsequent performance
variants, not prerequisites for source release.

INT8 code bytes are twice Q4 code bytes. Removing coefficient reconstruction
may help compute, but additional code reads can slow bandwidth-bound decode.
A memory improvement must not be reported as a decode improvement without
balanced request measurements.

## Loader alternatives

**Full512:** original source manifest still validates the complete checkpoint
inventory/identity, but replaced trunk expert tensors get checked disk-only
records with no Metal views. Partition each shard into aligned active
non-PLE/non-replaced windows; keep mixed-shard complements and `mtp` data.
Complete inventory permits omitting both raw miss sources/dispatches.
Plan and reserve the replacement maps plus remaining weights before mapping,
then assert actual allocation deltas and source-layout identity. Never load
both full inventories and release one afterward: that creates an avoidable
transient peak.

**Top256:** a new cold-rank Q4 layout must handle both large-row misses and
small-row cold experts. Exact cold-only repack copies original Q4/BF16 bytes
without reconstruction, with certified source offsets and stable source ID.
It needs the shared-memory slot for the large copy/hash/readback.

A zero-copy alternative uses existing readonly Tier-2 pointer argument
buffers (`MetalBackend.hpp:553`), with source file mappings created only for
cold expert runs. Q4 code stride is 819,200 bytes per expert (50 aligned
16-KiB pages). BF16 scale/bias stride is 51,200 bytes (3.125 pages); initially
retain all parameter planes and exclude only hot code pages. Pointer arrays
may use dense cold ranks; all slots must bind valid buffers because the
backend rejects null/empty resources. A new validated original-ID-to-cold-rank
map selects the pointer. The helper retains indirect allocations and declares
them Read in submitted commands. The miss kernel must replace original flat
expert-stride addressing with the pointer/rank lookup while preserving raw
coefficient and reduction math.

The pointer approach avoids repacking but creates many small MTLBuffers and
indirect resources. Qualify creation limits, measured startup/time/driver
overhead, source ownership and full lifecycle before preferring it over a
small number of compact cold payloads.

The backend deduplicates indirect owners, but every dispatch declares the
entire table's referenced allocation union through `useResources(Read)`,
whether or not actual routes use each expert. At the audited worst of 140
cold runs in one layer, gate/up needs up to 280 code allocations and down
140, before params and boundary coalescing. Submitted tickets retain that
complete union. A compact repack avoids much of this dispatch-time driver
work and should be the reference if the sparse-pointer experiment regresses.

## Source byte arithmetic

The 48 trunk layers have 144 projection planes, each with 512 experts:

| Original target category | Logical bytes |
|---|---:|
| Q4 codes | 60,397,977,600 |
| BF16 scales and biases | 7,549,747,200 |
| All trunk expert source data | 67,947,724,800 |
| Hot256 Q4 codes removable with full parameter planes retained | 30,198,988,800 |
| Hot256 all planes removable with exact cold-only packing | 33,973,862,400 |

The source's existing PLE-SSD GPU maps total 74,317,889,536 bytes. The bounded
metadata audit confirms removing all trunk expert ranges leaves exactly
6,370,164,736 original mapped bytes in seven windows: source shard 1 has
three windows totaling 4,954,587,136 bytes; shard 21 has four windows totaling
1,415,577,600 bytes (including retained MTP operands). Source shard 8 has no
remaining GPU-backed complement. Twelve pure target shards can be omitted
entirely. Source bytes stay on disk.

The metadata checked source manifest SHA256 is
`0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402`;
Top256 manifest SHA256 is
`ed271c58bd52f5914d0a3601321a2c716e24c8713c9a590fccbaee37fe888e10`.
No payload content hash was recomputed in this audit.

The Top256 cold inventory has 6,101 contiguous expert runs across 48 layers,
or 18,303 runs across 144 Q4 code planes. Coalescing touching source ranges
across plane boundaries gives 18,292 code windows. Cold BF16 scale/bias pages
need 262,536 unique 16-KiB pages (4,301,389,824 bytes) in 36,551 coalesced
per-shard windows. That is 526,516,224 bytes beyond the exact cold parameter
logical bytes. Even the code-only slicing variant therefore has substantial
native-buffer metadata overhead; the original source is intentionally packed
into far fewer allocation bases.

After coalescing all touching ranges across tensor boundaries, code-only cold
slicing with full parameter planes has 18,295 expert-source windows totaling
37,748,736,000 bytes. Including the non-MoE complement gives 18,296 windows
and 44,118,900,736 original mapped bytes. Fully sparse-aligned cold parameter
slicing gives 54,782 expert-source windows / 34,500,378,624 bytes; with the
complement this is 54,784 windows / 40,870,543,360 original mapped bytes.
Exact cold repacking plus unchanged complement needs 40,344,027,136 logical
bytes, before small destination alignment/metadata allowances.

The Full512 derived payload is 121,173,442,560 bytes, plus 786,432
rank bytes. The Top256 derived payload is 60,586,721,280 bytes, plus 786,432
rank bytes. New argument/rank metadata and driver overhead require their own
admission allowance.

Original-map plus derived-store payloads alone (excluding rank/argument
metadata, workspaces and request state) compare as follows:

| Operand layout | Bytes |
|---|---:|
| Current Top256 + original PLE-SSD maps | 134,904,610,816 |
| Current Full512 + original PLE-SSD maps | 195,491,332,096 |
| Full512 replacing all raw trunk experts | 127,543,607,296 |
| Top256 + cold-only code slices + full raw params | 104,705,622,016 |
| Top256 + sparse-aligned cold-only Q4/params | 101,457,264,640 |
| Top256 + exact cold-only repack + complement | 100,930,748,416 |

These differences cannot be subtracted from measured `workspace_bytes`
without checking what it already includes. Fresh `MemoryGovernor` host
telemetry, actual native/device peak bytes, terminal HTTP responses and final
idle/unload state remain the authority for sustained use.
