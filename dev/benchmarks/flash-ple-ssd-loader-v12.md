The optional `SPLASH_FLASH_PLE_SSD_STREAMING=1` loader keeps the original
`splash-local-qwen4-affine-v1` package unchanged. It omits only the 128 affine
ngram embedding shard triplets. All other tensor planes, the small ngram hash
constants, the shared scale, and the PLE postprojection weights retain their
original bytes, dtype, and native GPU accessibility.

The 21 original payloads occupy 106,320,429,056 bytes. Their canonical 16 KiB
tensor offsets partition into 28 contiguous non-PLE windows, totaling
74,317,889,536 bytes, and 384 disk-only planes, totaling 32,002,539,520 padded
bytes (32,000,153,600 logical bytes). Six payloads are entirely disk-only. The
first and eighth mixed payloads retain three and twelve windows; the remaining
13 payloads retain one full window each. No PLE embedding page belongs to a
native GPU allocation, an immutable-weight enumeration, or a persistent CPU
mapping. The underlying readonly FD store reads selected original rows on
demand; its bounded row cache does not alter model bytes.

Every tensor record remains present, so the 3,748-record inventory and effective
numerical model fingerprint are identical. Public GPU tensor/projection lookup
rejects omitted table planes. `diskProjection()` exposes checked original file
metadata and affine dimensions without GPU pointers. `pleSSDStore()` shares one
CPU store across model executors. `pleSSDStorageStats()` reports the exact unique
native window and omitted disk-plane counts.

Each retained native window owns its readonly CPU mapping through the existing
Metal allocation lifetime object. Tensor views, moved `FlashWeights`, and
command tickets therefore use the same lifetime behavior as the original
whole-payload loader. The memory ledger charges each native window once,
including padding, and must equal the checked unique mapped byte total. Padding
validation uses bounded `F_NOCACHE` reads in SSD mode; optional payload digest
verification uses a 1 MiB scratch buffer without full-file mapping.

`SPLASH_FLASH_PLE_SSD_CACHE_MB` accepts decimal 0 through 1024, defaults to 64,
and requires SSD streaming to be enabled. Invalid configuration is rejected
before any payload mapping. The legacy checked 13-base text residency selection
is incompatible with sparse SSD windows; the optional maintenance owner must
use the explicit SSD geometry instead.

CPU compilation with the production hybrid flags and `-Werror` passed. The
loader-only backend oracle is built without executing it:

```sh
make -f dev/benchmarks/flash_ple_ssd_loader_oracle.mk \
  BUILD=build/flash-ple-ssd-loader-oracle-v12 SPLASH_PRECISION=hybrid \
  build/flash-ple-ssd-loader-oracle-v12/flash-ple-ssd-loader-oracle
```

Root can run that oracle serially with a production metallib, original package,
and fresh report path under raw and SSD configurations. It creates real native
buffers but submits no GPU commands. It verifies all 128 table parts, unchanged
tensor count/fingerprint, GPU access rejection, native counts and byte ledger,
move/view backing lifetime, and complete native release after the last view.
This document does not claim model-level or HTTP qualification; those are
separate checks performed by Root.
