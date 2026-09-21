Optional SSD n-gram storage preserves the qualified model's outputs and removes about 29.79 GiB from its native allocation footprint. The matched warm service screen shows throughput within 0.71% of the same-build in-memory route across all four workloads. SSD mode remains an explicit option; the accepted 38-flag default profile keeps the table in memory.

All 22 bounded quality/lifecycle cases pass. All 28 normalized records exactly match the accepted v11 service: visible and reasoning text, tool type/name/arguments, finish reason, usage, HTTP status, completion and error fields. Request/tool IDs, timestamp and chunk boundaries are excluded. The factual paragraph coherently describes evaporation, condensation and precipitation and is unchanged. The retained quality snapshot is idle and Metal healthy, with 28 submitted requests, 26 completed, two cancelled and zero failures.

The same integrated binary also passes all 22 cases in the original in-memory mode. Its 28 normalized records exactly match the accepted service and SSD mode, with zero new task regressions; its retained final quality state is idle and healthy. This separately guards the normal default after adding optional storage.

Both matched HTTP runs use the same frozen plan: 128/2048 prompt tokens, concurrency one/four, greedy output with reasoning disabled, 128 completion tokens per request, warmup and two samples per cell. All 21 input bodies and prompt hashes, all normalized generation outputs, draft-acceptance histograms and verifier work match. Both runs have tracing and cross-request KV/prefix reuse disabled. Original model/layout and derived weight identities are unchanged; kernel routes match after substituting only the declared PLE storage marker.

| Prompt tokens | Concurrent requests | In-memory tokens/s | SSD tokens/s | Difference |
| --- | ---: | ---: | ---: | ---: |
| 128 | 1 | 110.619 | 110.188 | -0.389% |
| 128 | 4 | 176.084 | 177.325 | +0.705% |
| 2048 | 1 | 71.160 | 70.848 | -0.439% |
| 2048 | 4 | 86.097 | 86.238 | +0.165% |

These two-sample screens do not establish statistical significance or a speed advantage. Aggregate phase counters include the initial warmup command; raw prefill wall time includes an approximately one-second larger startup submission cost, while aggregate trunk prefill GPU time differs by only 8.60 ms (0.097%). That startup wall difference must not be presented as a steady throughput improvement.

The SSD loader retains 28 native windows totaling 74,317,889,536 bytes from the original checkpoint, omitting only 384 n-gram planes totaling 32,002,539,520 padded bytes (29.8046875 GiB). All 3,748 tensor records and their numerical layout identity remain. The table has no full CPU mapping or native GPU backing. It is read using checked original-file offsets into a shared bounded CPU row cache and executor staging. CPU hashing selects the rows; the GPU's canonical hashes verify those staged IDs before reconstructing the original affine coefficients.

The four GPU row staging owners total 17,776,640 bytes. Singleton replaces the original 3,072-byte fused-PLE pointer owner, and SSD maintenance adds 56 logical diagnostic bytes. The exact logical dense ledger saving is therefore 31,984,765,896 bytes. Rounded native peak backing falls from 152,982,552,576 to 120,997,789,696 bytes, a 31,984,762,880-byte (29.7881 GiB) saving. Logical and rounded allocation extents differ by 3,016 bytes; these are native accounting figures rather than process RSS or guaranteed physical wired memory.

The configured 64 MiB CPU cache admits 50,331,648 bytes of row slots and index storage, plus separate 1,048,576-byte I/O scratch. Temporary batch metadata and unmeasured OS/device caches are outside the native ledger. This explicit host row cache is independent of KV/prefix cache, so a later request can reuse original n-gram rows while still reporting zero cached prompt tokens.

Quality qualification reads 2,033,600 exact payload bytes over 61,005 positional reads, taking 73.481 ms in the recorded host-read scope. The subsequent full HTTP benchmark including warmup requests 423,840 row selections: 385,480 cached hits, 30,436 unique misses and 7,924 duplicate misses. It reads exactly 3,043,600 bytes in 91,305 positional reads, totaling 85.959 ms. The requested-row partitions and byte accounts match exactly in every quality case and benchmark wave; there are no source-validation failures, failed batches, cache evictions or poisoned storage.

Measured benchmark-wave reads take 0–4.83 ms, with 88.73–91.38% cache hits on short prompts and 99.01–100% on long prompts after warmup. This workload shares code context, and the preceding quality pass already warms the row cache. The `F_NOCACHE` file policy is enabled, but pread bytes are not physical SSD traffic and do not establish uncached device latency or bandwidth. The host-read timer covers the store's synchronous read scope, not every CPU staging, hashing or source-validation cost. Native safe-point statuses are captured outside each HTTP wave's clock; some post-wave statuses precede terminal bookkeeping, so separate retained idle snapshots provide lifecycle evidence.

Idle maintenance excludes the SSD table, CPU cache and mutable GPU staging and keeps only 1,141 immutable native owners totaling 112,324,313,088 bytes accessible. Its nine-second idle probe records 17 completed maintenance commands, zero failures and unchanged ownership. Client first content is 393.603 ms after idle versus 387.052 ms warm and 369.075 ms immediately afterward; every probe produces the exact 16-token answer with no KV reuse. System-wide wired-memory snapshots are separately scoped and do not guarantee pinning. Both matched benchmark final snapshots are idle and healthy.

Reproduce the independent CPU-only report from completed artifacts:

```sh
.venv/bin/python -B dev/tools/audit_flash_ple_ssd_service.py \
  --output build/release/flash/ple-ssd-independent-service-audit-v12-r4.json
```

The frozen independent report is `build/release/flash/ple-ssd-independent-service-audit-v12-r3.json`; input report hashes are included. The first audit version's logical-versus-rounded allocation check was superseded by r2, which accounts for the removed 3,072-byte original argument owner; r3 adds the completed in-memory quality guard. This independent service review launches no model, submits no GPU work and changes no runtime source.
