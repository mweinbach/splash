`flash_ple_ssd_oracle.mm` qualifies the optional SSD PLE gather using synthetic
original Q4/G32 tables. It never loads a real model. All 128 immutable table
parts have 11 rows, 160 channels, checkpoint-shaped I64 hash arrays, BF16
scale/bias, and the checkpoint shared BF16 scale. Padded and packed planes,
nonzero file/native-buffer offsets, signed I64 wraparound, cold history, and
EOS boundaries are covered.

The GPU mode compares resident `addPLENgramIDs` + `addPLEGather` against
`FlashPLESSD::prepare` + `addHashGather`, with exact IDs, final token histories,
BF16 outputs, sticky diagnostics, and output canaries. Geometries are lanes
1/2/4 with rows 1/4/16/2048. The small windows additionally restore zero,
one, partial, and full prefixes using `addPLERestorePrefix`, check the restored
convolution/history against CPU timelines, and compare the next token gather.
Changing GPU hash metadata after preparation must produce NaN only at stale
selections with an index diagnostic. Original NaN biases preserve the numeric
diagnostic. Invalid host tokens cannot change staging or request state, or
reuse a preparation. Truncating the source after a successful preparation
poisons the store and prevents a graph without changing request buffers.

CPU-only qualification constructs no Metal backend and submits no GPU work:

```sh
build/flash-ple-ssd-integrated-v12/flash-ple-ssd-oracle-v2 --cpu-self-test
```

The v2 CPU run passed 1,390,800 checks. Its independent hash uses explicit
timeline indexing, unsigned 128-bit products truncated to I64, and signed
128-bit positive modulo; production uses a running history and native I64
arithmetic. CPU store checks cover 0/4/64 KiB cache budgets, packed/padded
source rows, all hash geometries, and transactional invalid-token history.

Only root may run GPU work, serially with other model/benchmark commands:

```sh
build/flash-ple-ssd-integrated-v12/flash-ple-ssd-oracle-v2 \
  build/flash-ple-ssd-integrated-v12/splash.metallib \
  FRESH_REPORT.json --run-root-gpu
```

Without the explicit final guard, the executable rejects before constructing
the backend. Report timings are single qualification samples and are not a
performance comparison. Physical uncached SSD reads are not claimed.

The original frozen v1 binary remains available as `flash-ple-ssd-oracle`;
its CPU mode passed 1,390,728 checks. V1 compares the GPU hash against the
separate production CPU hash, while v2 uses the independent formulation.
Both retain the same synthetic exactness scenarios and failure guards.
