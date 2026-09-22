# Private full target-expert INT8 cache

The current 64-expert-per-layer store reduces only selected large-row prefill
projections. A private 512-expert store removes the target's Q4 miss fallback
for all 48 trunk layers; original small-row decode and trained MTP arithmetic
remain available. This is a larger numerical alternative, not model parity.

The source checkpoint and qualified top64/top128 stores are preserved. Native
runtime/header/shader changes exist only in a frozen private overlay. Full512
inventories are exactly sorted 0..511, so every original expert has a validated
compact rank. The two Q4 miss dispatches are omitted entirely for full512.

```sh
.venv/bin/python dev/benchmarks/prefill4k_fullcache_overlay.py
make -f Makefile -f dev/benchmarks/prefill4k_fullcache.mk \
  BUILD=build/flash-next SPLASH_PRECISION=hybrid -j2 prefill4k-fullcache
make -f Makefile -f dev/benchmarks/prefill4k_fullcache.mk \
  BUILD=build/flash-next SPLASH_PRECISION=hybrid prefill4k-fullcache-cpu
.venv/bin/python dev/benchmarks/prefill4k_fullcache_convert.py --cpu-self-test
.venv/bin/python dev/benchmarks/prefill4k_fullcache_convert.py --inspect-only
```

Compilation and these checks perform no GPU work. Private native metadata
tests passed 147 cases, including full512 sparse-file geometry and strict
metadata negatives. Worker and primitive CPU self-tests passed. Converter
CPU tests include zero rows, BF16 minimum subnormals, exact F32 scales,
coefficient absolute bounds, and a cancellation example that explicitly
changes the result from 0 to approximately -0.007874.

Heavy conversion and checksum/certificate scans must run in Root's shared
memory slot, with GPU measurements paused:

```sh
.venv/bin/python dev/benchmarks/prefill4k_fullcache_convert.py \
  > build/release/flash/prefill4k-fullcache-conversion512.jsonl
```

The new sidecar is under
`build/prefill4k-fullcache-artifacts/int8-experts-all512-v1/`. Conversion uses
eight experts at a time, readonly source mappings, separate F32 multiply/add
followed by BF16 rounding, and the exact existing signed INT8 rowwise policy.
Source shards are checked against original SHA256 hashes before writing.
Every coefficient is independently checked in F64 against
`0.5 * stored_scale + 32 * 2^-24 * row_absmax`; all F32 scale bits and zero-row
rules are checked. Every selected top64 code and F32 scale is compared exactly
with its existing qualified payload. Full expected counters are mandatory:

- 120,795,955,200 coefficients and 94,371,840 rows.
- 15,099,494,400 preserved top64 code bytes.
- 47,185,920 preserved top64 scale bytes.

The certificate, source/plan rechecks, and unchanged preserved-store snapshots
finish before atomic publication of the new sidecar. The default profile never
selects it. GPU/dot/generation qualification remains separate from this
coefficient certificate; cancellation prevents a universal relative dot bound.

The full store contains 121,173,442,560 mapped bytes; 48 rank allocations add
786,432 bytes, for 121,174,228,992 planned bytes. Fresh PLE-SSD/top64 evidence
has 120,129,617,920 peak native bytes. Replacing its 15,147,466,752 planned cache
bytes gives a 226,156,380,160-byte candidate estimate, leaving 21,233,736,090
bytes inside the 247,390,116,250-byte engine budget. Source identity, store SHA,
expert count, mapped bytes, kernel cache identity, and PLE-SSD mode are checked
before using that extrapolation. Real MemoryGovernor admission before mapping,
buffer-limit checks, actual startup/request peaks, and host reserve still need
fresh verification.

The raw target has 432 Q4 source tensors totaling 67,947,724,800 logical bytes.
Twelve original shards contain only target expert data, totaling 62,285,414,400
bytes; full512 target-prefill miss removal means those bases need not be bound
by the MoE graph. Two additional shards mix experts with other model roles.
Original mappings remain registered/accounted for small decode. No physical
pinning, reclaim, or per-die placement is inferred from this resource audit.
Actual dispatch/resource traces are required.

The private Store records successful graph construction under
`persisted_experts.graph_counters`: gate/down calls and physical rows,
encoded hit/miss dispatches, and full-inventory phase calls. Scope is explicitly
"graph construction, not GPU completion". One completed large chunk through
48 layers constructs 48 gate phases and 48 down phases; full512 constructs
96 hit dispatches, zero miss dispatches, and 96 full-inventory phases. Cancelled
or discarded graphs are included, so terminal responses and idle/completed
command evidence remain separate requirements.

The qualified idle-maintenance union assumes top64's exact byte count. Full512
safely fails that union predicate and skips maintenance. Use explicit
`SPLASH_FLASH_IDLE_RESIDENCY_MAINTENANCE=0` for matched warmed comparisons,
then independently qualify an updated private union if post-idle performance
is needed. Preserve PLE-SSD mode; historical PLE-resident ledgers do not fit this
fullcache extrapolation.

[Precision and whole-model validation plan](prefill4k_fullcache_precision.md)
separates changed generations from regressions. Run a fresh same-build top64
baseline and full512 candidate with all 22 meaningful tasks plus long-prefill
cancellation/deadline, chunk boundaries, concurrency, admission failure, and
unload/reload recovery. Inspect prose and final state, rather than reducing
model quality to hashes or an aggregate cosine.

The CPU conversion completed in 753.7 seconds, with no GPU/model work. The new
manifest SHA256 is
`ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1`.
All expected coefficient, row, and exact top64 byte counters passed; source,
plan, and preserved top64 snapshots remained unchanged before publication.
The global coefficient relative L2 is about 0.00847 and maximum absolute error
is 0.0047982. These describe converted coefficients, not generation quality.
See the sidecar's `coefficient-certificate.json` and
`build/release/flash/prefill4k-fullcache-conversion512.jsonl`.

The first full512 HTTP run completed four requests, but warmed prefill fell to
about 739 tokens/s wall time despite about 2,966 tokens/s in GPU intervals.
After GPU use, available host memory fell to 19.76 GB against a 27.49 GB
reserve; growth stopped and 17 reservations were denied. Full512 therefore
remains private and unsuitable for normal interactive use on this measured
configuration. See `prefill4k-fullcache-first-http-analysis.json` under the
retained results directory.

A Top256 alternative was repacked from the certified full store without
requantizing. Its source-ranked plan selects observed count descending with
an expert-ID tie-break from the same verified captures as Top64/Top128, and
contains both smaller inventories. Observed coverage is 94.10%; this is a
capture estimate rather than general workload coverage. The new private store
is `build/prefill4k-fullcache-artifacts/int8-experts-frequency256-v1`, with
60,586,721,280 mapped bytes and 60,587,507,712 planned bytes. Source-layer SHA,
exact copied bytes, independent output plane/whole-file readback, and source
snapshots passed before atomic publication. Manifest SHA256:
`ed271c58bd52f5914d0a3601321a2c716e24c8713c9a590fccbaee37fe888e10`.

The subset helper is `prefill4k_fullcache_subset.py`. Its inspection mode reads
only bounded metadata and file stats; actual copy/hash work requires the shared
memory slot. Source checkpoint and qualified stores remain preserved.
