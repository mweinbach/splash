`SPLASH_FLASH_SAVED_OPERANDS_RESIDENT=1` requests one retained Metal residency
set for verified saved derived operands after all constructors finish and
before requests begin. It accepts only 0/1 and defaults off. This is a
performance experiment; a successful request means GPU accessibility was
requested, not that physical pinning or a speedup was established.

`FlashForward::cachedOperandsOnly()` enumerates saved mapped BF16/F32 dense
entries and the selected INT8 store's derived payload/rank allocations.
`FlashMTPForward::cachedOperandsOnly()` contributes saved mapped trained-head
dense entries. Runtime-generated dense caches are excluded. Original raw
checkpoint tensors, PLE lookup backing, embeddings, original Q8 vocabulary
code views and raw expert miss operands are excluded.

Dense saved entries occupy individually verified derived payload files at
offset zero. Selected INT8 base backing contains converted code/scale planes;
its rank maps are immutable derived CPU metadata. This matters because Metal
requests the entire native base allocation even when supplied a view.

Worker retains a move-only `ResidencyLease` for its lifetime. The existing
backend registration also retains Metal allocations and mmap owners through
shutdown and outstanding ticket consumption. Only one request per backend is
made, containing the final union; there is no second operand backing copy or
ledger charge. Driver residency-set metadata can affect observed device
allocation totals.

The status object `saved_operands_residency` distinguishes configuration from
execution:

- `requested`: the startup flag was enabled.
- `supported`: the backend's Apple-family support condition was met.
- `request_succeeded`: a lease was returned by the backend.
- `active`: that successful lease is held by a healthy running service.
- `requested_view_count` and `requested_view_bytes`: selected views supplied.
- `registered_base_allocation_count` and `registered_base_allocation_bytes`:
  unique complete native base allocations reported by the lease.
- `failure_reason`: unsupported, empty selection or request failure detail.
- `physical_pinning_verified`: always false.

With a Top64 INT8 store and no saved dense environment, selection contains
48 derived payload bases and 48 rank maps. The existing Top64 manifest maps
15,146,680,320 payload bytes; 48 rank allocations add 786,432 requested-view
bytes. Actual registered base byte totals are authoritative after startup.

The frozen combined residency/timing worker is
`build/flash-saved-residency-runtime/splash-flash`, with its adjacent metallib,
source snapshot and `private-freeze.json`. All source was rebuilt against the
new `CommandTiming.host` ABI. The native CPU self-test passed, and seven
malformed residency flags rejected before Metal initialization. Compilation
and these checks submitted no GPU work. Root performs actual off/on service
qualification and matched performance comparison.
