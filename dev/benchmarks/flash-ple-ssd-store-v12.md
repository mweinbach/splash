The optional PLE SSD store reads the checkpoint's unchanged Q4/G32 n-gram
table through readonly positional reads. It never maps the table or creates
GPU buffers for its full contents. Each staged row contains exactly 80 packed
Q4 bytes, ten BF16 scale bytes, and ten BF16 bias bytes. Reconstruction stays
with the GPU execution path; this module does not change any coefficient.

The CPU cache defaults to 64 MiB. Its fixed row slots and open-addressed index
are admitted together inside that budget. A separate 1 MiB scratch buffer
limits each read. Per-call ID, sorted miss and segment metadata is proportional
to the supplied batch, which the execution path must bound to its geometry.
Backward-shift index deletion avoids an accumulating tombstone penalty after
many cache evictions. Reads are sorted by source and byte offset and coalesce
only overlapping or adjacent requested spans, so sparse random accesses do
not cause speculative gap reads. Duplicate requested IDs read each missing row
once and retain the caller's original output order.

macOS F_NOCACHE is enabled on each store-owned descriptor by default. The
reported file-cache-bypass bit reflects whether that policy was actually
applied; it is not a claim that every request reached uncached physical SSD
hardware. The initial implementation intentionally uses the original three
planes, avoiding a duplicate 32 GB row-interleaved conversion. Actual HTTP
tests determine whether a reader pool or derived layout is justified.

Before and after each batch, stat and fstat bind every source to its captured
device, inode, size, modification time, and change time. Opening rejects a
symlink. Source mutation or I/O failures poison the store and throw, including
on warm cache hits. This is a checked immutable-file contract, not a new full
payload SHA256 scan; the loader remains responsible for manifest identity and
its configured digest validation. Invalid IDs, output extents, and overlapping
ID/output spans reject before any output write and do not poison healthy
storage. A failed source/read batch can have partial output bytes, so callers
must never submit its output to the GPU after the thrown error.

The local oMLX reference at
`omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/language.py`
uses whole-file mmap, a seen-page bitmap, and a 48-thread page-prefetch pool.
Splash's store instead has an explicit bounded CPU row cache and no PLE mmap.
The implementations share exact original sparse-row selection; their
performance and residency behavior require separate measurements.

CPU qualification command:

```sh
clang++ -std=c++20 -O1 -g -fsanitize=address,undefined -Wall -Wextra -Werror \
  -Iruntime runtime/flash/FlashPLESSDStore.cpp \
  dev/tests/flash/test_flash_ple_ssd_store.cpp \
  -o build/flash-ple-ssd-store-v12/store-cpu-test
build/flash-ple-ssd-store-v12/store-cpu-test
```

The first completed run passed 10,818,808 byte/accounting checks over 12,000
randomized cache-churn batches across zero-cache and three small cache budgets.
It includes shard boundaries, duplicate output order, invalid IDs, unchanged
output on validation failure, and cold/warm accounting. Independent review
adds padded source strides, warm file mutation/replacement, aliasing, and
native-offset/overflow cases; all 475 independent contract checks passed in
both optimized and ASan/UBSan runs. A separate independent suite passed
45,697,632 checks over 4,800 randomized batches and 400 shared-store concurrent
batches. CPU checks do not qualify GPU execution or HTTP
quality/performance.

`dev/benchmarks/flash_ple_ssd_store_probe.cpp` is a bounded CPU-only checkpoint
probe. Its text spec starts with `sourceCount partCount cacheBytes scratchBytes
noCache idCount`, followed by source `(path byteCount)` records, part
`(rows weightsSource weightsOffset weightsStride scalesSource scalesOffset
scalesStride biasesSource biasesOffset biasesStride)` records, then signed
I64 IDs. It reports two exact cold/warm output checksums, elapsed host time,
read request counts, logical miss bytes, completed bytes and actual cache
bypass. The probe allows at most 65,536 IDs and does not silently escape its
bounds. It has not been run on the actual checkpoint by this agent.
