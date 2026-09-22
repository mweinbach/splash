# Preregistered Root actual-projection capture and QSA-only proof

CPU source preparation only. Agents do not run the GPU, load model/capture data,
or label synthetic inputs as real. Root authorized the necessary private source
and owns all capture/export/comparison execution.

Initial capture is lane0 of an actual real B2 or B4 fresh2048 cohort, selected
QSA layer3 (layer7 is an independent later extension). Run the exact restored
parent profile with old batch QSA; do not capture whole singleton Forward,
whose W8 upstream projections differ. Copy after actual q/k/v/index projection
dispatches and before the subsequent lane's QSA prefix consumes/reuses them.
Append existing `flash_forward_copy_words` GPU copies into retained owned Shared
views. Copies execute in the same original graph at the declared position; no
CPU read may be substituted during graph construction.

Capture only these complete BF16 views, retaining q's original gate half:

| Plane | Rows | Width | Exact readable bytes |
| --- | ---: | ---: | ---: |
| qProjection | 2048 | 12288 | 50331648 |
| kProjection | 2048 | 512 | 2097152 |
| vProjection | 2048 | 512 | 2097152 |
| indexProjection | 2048 | 640 | 2621440 |
| total inputs | | | 57147392 |
| optional original QSA BF16 output | 2048 | 6144 | 25165824 |

All allocations and guards are Governor-planned before construction; actual
owner charges, completed submission/healthy status and teardown are recorded.
The optional output copy occurs after the original complete QSA reducer for
this same lane/layer. Input+output is82,313,216 bytes, excluding small norms and
guard charge. Initially keep one layer, comfortably below150 MB. Existing
batch scratch, coefficient pointers, outputs and profile bodies remain literal
except debug copy insertions; capture defaults off and counters stay separate.

The metadata schema is `splash-current-batch-qsa-projection-capture-v1`. It binds
the completed real source/library/operation/profile, actual width/lane/layer,
begin0/capacity, row geometry, all four tensor norm dtype/shape/convention and
payload identities, epsilon/theta, exact file extents, producer graph insertion
point and actual completed native batch widths. Include original state owner
and identity evidence only as local descriptors, never cross-process pointers.
If a real source/cohort/position/completion field is absent, reject the capture
as uncertified. File payload SHA verification is Root-only.

Capture all four actual norm tensors into small owned views using the original
dtype/shape/logical extents, not presumed F32. Preserve each norm's actual
OnePlusWeight or Direct convention. The existing helper uses supported model
epsilon1e-6/theta1e7; validate these fields literally before use. Fresh begin0
cache planes can be recreated using the actual capacity and the original fully
zeroed allocator; do not import stale future/ticket state.

The QSA-only oracle uses one backend and bounded original/cache workspaces,
without constructing a second model graph or calling a whole-model loader.
Load only these Root-selected projections/norms and metadata. Compare three
arms on identical immutable inputs and independently owned fresh cache states:

1. Existing restored legacy batch `addBulkExactQSA(...sg8=true)`.
2. New dedicated batch509,607,936 arena wrapper calling unchanged
   `addTwoPassQSA(...verification=false,packedV=true)`.
3. Existing singleton QSA helper using the same projected inputs, same norm
   tensors/conventions and cache capacity. This is a **QSA-only policy golden**.

Arms2/3 require full BF16 output and five cache-plane bit equality and identical
diagnostics. Arms1/2 retain the original preregistered source per-row relative
L2/cosine and sampled F64 QK/softmax/PV/gate envelope from the sealed packed-V
component, without a new boundary exemption or invented envelope. Prefix norm,
rope/pooling/chronological dense selection and all five cache planes must remain
bitwise exact across all arms; preserve inactive physical cache bytes. Direct-V
variant remains excluded. Report any raw/F32 diagnostic differences honestly.

Use controlled finite and exceptional perturbations of temporary copies while
preserving original captured inputs. NaN/Inf/subnormal/signed-zero/sticky/alias/
short view/bad capacity/future masked entry cases keep their original diagnostic
and no-write semantics. Replay original future QSA cache appends with actual
captured follow-up q/k/v/index inputs when available; synthetic follow-ups are
explicit additional cases. Full-trunk future greedy proof remains separate.

After all quality gates, time the **complete** producer pipeline including
prepare/pool/select, packQ/QK/F32 softmax/packV/PV/unpack, using fixed addresses,
at least150 ms GPU warm per arm and balanced AB/BA pairs. No CPU tensor/hash/
diagnostic/canary accesses occur from warmup through the last timing sample.
Copy/tap instrumentation is excluded from timed arms and its count is separate.
Last measured output witnesses must be captured before any post-timing reset.

This primitive does not prove real B2/B4 whole hidden/logits/greedy, trained head,
Worker cancellation/deadline, all18 batchVerify partitions, original22 model
quality or service rate. Those corresponding current-header proofs and normal
benchmark gates follow independently. Old batch bit-parity golden cannot qualify
the changed attention floating tree, and a whole B1 W8 golden is invalid here.
