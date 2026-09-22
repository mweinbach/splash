Source preparation only. The current parent is the fixed-four trained worker at
`build/R5-integer-currentQ4-fixed4-sep22-worker-v2`; its original library remains
byte-identical. No whole-worker compilation, shader compilation, or GPU work is
authorized by these files. Root must review and pin this source before a fresh
private build.

The v1 source package is preserved verbatim under
`archive/source-v1-8441a6b13ec2f594`. The narrow v2 correction adds a whole
campaign preflight before owner/host/slot allocations. It budgets three full
134-plane physical request snapshots, three full216-arena initialized lazy
snapshots and defined PLE ranges, two complete Verify5 hidden/logit/greedy
outputs, all113 input payloads, five JSON headers capped at1MiB each and4KiB
per possible1180 payload/header files. Undefined local PLE/count tails are
also conservatively reserved without exporting them. Before the first payload
write, actual counts/extents/source capacity/output views/slots must match the
whole plan and its checked cumulative bound below4GiB. Every write then checks
that preflight and the bounded file/header counts. No plane is shortened or
omitted to meet the bound.

The capture point is the existing selective-F32 cache-member null-tile branch,
immediately before its original raw affine producer. Constructor inspection
binds canonical GDN qkv/z/out, QSA q/o and the first PLE key role to the seven
observed shape/quantization combinations. All 113 original quantized views,
strides, cache memberships and null decisions must authenticate before graph
mutation. The actual production graph must contain the corresponding 113
original F32XSUM producers with their exact ABI, input slot0/output slot5,
unused slot4 input dummy and 64-thread grid. No head, seed, prefill, AR or batch
tap is installed.

A typed lexical cookie comes only from the existing genuine Worker target call:
request1/generation1, four real trained proposals, physicalR5, ordinal3. Two
successful genuine R5 target calls precede capture. One original integer copy
dispatch writes each guarded BF16[5,K] slot before that raw producer reads it.
The producer and all FP math remain unchanged. Complete inputs occupy4,362,240
logical bytes in5,046,272 guarded/aligned allocation bytes. A separate16MiB
Governor reservation admits the capture owner. A64MiB host arena is allocated
before sampling, after an extra128MiB host-headroom check; the post-allocation
host snapshot must remain valid and allow growth. Flag0/proof0 adds no buffers,
paths or instrumentation to the original execution.

The source-only native correctness campaign uses the same fresh instrumented
worker in separate capture-off and capture-on processes. Both receive exactly
one native greedy2048/64 request, with the static Root token pins; agents never
read or hash that file. Actual trained head code constructs every proposal.
Three streamed snapshots compare the third pending Verify5 output/state/tapes,
its real committed state/tapes, and the next genuine R5 target output/state/
tapes. Each includes all134 initialized physical request planes and all216
initialized physical lazy arenas, plus defined PLE ranges. Unknown PLE/count
tails are checked locally. Local owner/view identity, capture guards, lazy
guards and input preservation through commit/future are asserted. Request
buffers are not replaced with artificial guarded storage. Streamed spill is
strictly below4GiB per process and bounded64KiB writes avoid a second full-state
host copy. Native writes and Root streamed comparisons disable Darwin file
data caching so the spill does not require another state-sized page-cache
working set. Undefined bytes are never compared across processes.

One readonly friend is added only to the cloned private Forward header. It adds
no fields, API or arithmetic changes. A future build must run the actual50-TU
header census and recompile Forward, Worker and all three Batch Forward-header
consumers, preserving Core4 and the other exact current objects. The source
journal independently restores the complete original bodies literally. The
Root runner checks clean terminal processes, genuine depth-four cycles, cache
MISS/matched0, the unchanged64-token result and all three exact byte snapshots.
This is input capture QA; it provides no kernel speed, model-task, trained-head
numerical oracle, residency or service promotion evidence.
