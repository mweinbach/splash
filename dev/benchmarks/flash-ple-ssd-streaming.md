# SSD streaming for PLE n-gram embeddings

The option keeps the original n-gram table on disk, reads selected rows into
bounded staging, and leaves other model weights in GPU-accessible memory.
It is separate from KV/prefix caching. The qualified local M5 Ultra profile
uses SSD streaming by default in v10. Its39 static settings retain the38 v9
routes and add only `SPLASH_FLASH_PLE_SSD_STREAMING=1`; the native64 MiB row
cache remains implicit. Unknown packages and hardware retain normal defaults.

Usage after the qualified runtime is built:

```sh
./splash serve --model local/Qwen3.8-Flash-Next-oQ4e-mtp \
  --local-package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --port 8011 --max-context 8192 --no-webui
```

Add `--ple-ssd-cache-mb 128` to change the bounded row cache, or
`--no-ple-ssd-streaming` to keep the n-gram table in GPU-accessible memory.
`SPLASH_FLASH_PLE_SSD_STREAMING=0` also opts out; no cache environment setting
is introduced by the profile. For a local package outside the qualified
profile, explicitly add `--ple-ssd-streaming` to enable this route.

Equivalent explicit environment settings are `SPLASH_FLASH_PLE_SSD_STREAMING=1`
and `SPLASH_FLASH_PLE_SSD_CACHE_MB=64`. Cache size is optional,0..1024 MiB; zero
performs uncached application-level row reads. A CLI cache budget requires SSD
streaming in the effective profile/environment, after explicit placement
overrides. CLI arguments apply only to local Flash-Next packages.

The loader replaces21 whole original GPU buffers with28 non-PLE native windows,
preserving3748 tensor metadata records, source/layout identity and raw weight
bytes. All384 PLE table planes are immutable disk metadata; there is no full
table GPU buffer or persistent CPU table mapping. Original non-PLE mappings
occupy74,317,889,536 bytes, versus106,320,429,056 normally. The exclusion is
32,002,539,520 bytes (29.8 GiB), including padding. Logical Q4 table bytes are
32,000,153,600. Original checkpoint and aligned payload files are not modified.

The CPU store uses readonly `pread` and a bounded exact-row LRU. Its64 MiB
budget includes fixed row/index arrays, with a separate1 MiB read scratch
limit. It reads original80-byte Q4/G32 weights and10-byte BF16 scale/bias planes
into100-byte row records, deduplicates IDs, and coalesces adjacent requests
without gap overreads. macOSF_NOCACHE is requested and reported as file-cache
bypass, not a guarantee of physically uncached device I/O. Source replacement,
truncation or read failure rejects the lookup and poisons the store.

Each synchronous executor calculates exact n-gram IDs from known tokens and
completed request-owned two-token histories. The normal GPU hash and history
update remain in the graph; gather checks their IDs against the staged IDs.
Coefficient reconstruction, BF16 cast, and shared BF16 scale boundaries are
unchanged. Post-PLE projections, convolution, masks and speculative-prefix
restoration keep their existing math. Single request/verify, concurrent AR
decode, joint verification and concurrent prefill all support the option.

GPU staging is bounded/admitted independently: singleton2048 rows3,538,944B;
four-lane2048 prefill14,155,776B; jointverify49,152B; concurrentdecode32,768B.
The shared host row cache is separate from these GPU allocation figures.

Idle maintenance adapts to SSD mode: it touches only28 non-PLE originals and
1113 verified derived owners, totaling112,324,313,088 bytes. It never declares
or reads the disk-only table. Raw-mode maintenance retains its existing exact
1134-owner path.

`/status.ple_storage` exposes disk/GPU bytes, table mapping mode, cache budget
and usage, staging bytes, hits/misses, read requests, completed read bytes,
read time and source failures. Read bytes are `pread` payload bytes rather
than measured physical SSD traffic. Use workload-specific HTTP results to
decide whether the memory saving is worth the storage latency.

CPU loader lifetime and all16 shader-validated primitive cases pass. Both SSD
and default-resident modes passed all22 full service quality/lifecycle checks
with zero new task regressions. The matched untraced HTTP plan uses128/2048
prompt tokens,128 output tokens, greedy reasoning off and no KV reuse. Cell
medians, tokens/s:

| Prompt / requests | Resident | SSD64 MiB cache |
| --- | ---: | ---: |
| 128 /1 | 110.619 | 110.188 |
| 128 /4 | 176.084 | 177.325 |
| 2048 /1 | 71.160 | 70.848 |
| 2048 /4 | 86.097 | 86.238 |

Differences are within about1% in this warmed, repetitive corpus; two sequential
samples per cell do not establish statistical significance or a throughput win.
Benchmark-total SSD reads include warmup:30,436 unique row misses,91,305 preads,
3,043,600 payload bytes and85.96 ms host read time. Cache hits cover90.95% of
row requests. OS/device caches were not forcibly cleared; this does not qualify
cold SSD latency on arbitrary prompts.

Retained native peaks are152,982,552,576B resident and120,997,789,696B SSD,
a31,984,762,880B reduction after staging/maintenance overhead. Bounded host
cache+read scratch are51,380,224B separately and are not native GPU allocation.
The SSD idle-maintenance proof returns exact16-token outputs after9 seconds,
first content394 ms versus369 ms immediately, maintaining only non-PLE weights.
Normal CLI opt-in proof completed four concurrent128-output requests and
returned healthy idle, with SSD mode enabled,64 MiB budget and no GPU table
buffers. Normal `flash-next` artifacts are byte-identical to the qualified
integrated runtime. At the end of the original v12 qualification round, the
GPU-resident v9 default was restored and its readiness/smoke passed with SSD
mode off and idle maintenance available. The later v10 profile promotion uses
the same qualified SSD route and preserves the measured resident opt-out.

Evidence:

- `build/release/flash/ple-ssd-loader-raw-runtime-v12.json`
- `build/release/flash/ple-ssd-loader-ssd-runtime-v12.json`
- `build/release/flash/ple-ssd-primitive-shader-validation-v12.json`
- `build/flash-ple-ssd-independent-checkpoint-v12/qualification.json`
- `build/release/flash/ple-ssd-maintenance-launcher-cpu-qualification.json`
- `build/release/flash/ple-ssd-quality-v12.json`
- `build/release/flash/ple-ssd-raw-quality-v12.json`
- `build/release/flash/ple-ssd-ssd-http-performance-v12.json`
- `build/release/flash/ple-ssd-raw-http-performance-v12.json`
- `build/release/flash/ple-ssd-idle-maintenance-proof-v12.json`
- `build/release/flash/ple-ssd-build-identity-v12.json`
- `build/release/flash/ple-ssd-independent-service-audit-v12-r3.json`
- `build/release/flash/ple-ssd-normal-cli-smoke-v12.json`
- `build/release/flash/ple-ssd-normal-cli-final-idle-v12.json`
- `build/release/flash/ple-ssd-restored-default-final-idle-v12.json`
- `build/release/flash/ple-ssd-qualified-summary-v12.json`
