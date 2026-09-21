The private compacted-job candidate keeps the qualified v5 INT8-hit/Q4-direct-
miss kernels unchanged. One GPU dispatch filters the original canonical stable
jobs by saved expert rank, producing stable hit/miss lists and two count views.
Both gate/up and down reuse that partition. There are no persistent loops,
CPU count readbacks, indirect backend extensions or arithmetic changes.

For `R` routes, tile M and H saved experts, safe static class bounds are
`ceil(R/M)+H−1` and `ceil(R/M)+(512−H)−1`. Their sum removes exactly 512
duplicate expert-slack groups from the original pair of capacity-sized grids.
At 2048 rows, top10, M32 and H64, original capacity 1151 becomes hit 703 and
miss 1087. The original 1151 parameter remains in the frozen producer ABI;
filtered counts and smaller launch dimensions keep every consumer access safe.

The partition uses one 256-thread group and deterministic tiled SIMD prefixes.
It validates complete canonical source job ownership, bounds and rank metadata
before copying, initializes inactive output records to `{UINT32_MAX,0}`, and
publishes counts only after copies finish. Invalid stored rank retains combined
sticky bits 3; malformed input fails closed with zero class counts. The private
host wrapper checks byte extents, exact source identity across all four
producers and disjoint writable outputs before submission.

The oracle compares every live activation, canonical down and combined BF16
byte against v5, independently verifies filtered records/counts and all unused
sentinels, and checks source/sidecar SHA immutability, every output/scratch
canary, retained mapping/parameter lifetime and finite positive timings.
All 41 dependency objects come from fresh `build/flash-default-v5`, with
`sizeof(CommandTiming)==200` and recorded object hashes.

Compilation and CPU checks passed without GPU execution. The partition's
independent CPU suite covered 1,638,412 metadata cases and 540 stable-order
reference cases. Primitive speed and GPU bounds qualification remain for Root.

```sh
FLASH_I8_PARTITION_LAYERS=0 FLASH_I8_PARTITION_PATTERN=mixed \
FLASH_I8_PARTITION_ROWS=2048 FLASH_I8_PARTITION_TILE=32 \
FLASH_I8_PARTITION_PAIRS=4 \
build/flash-int8-job-partition/flash-int8-partitioned-jobs-oracle \
  build/flash-int8-job-partition/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  install/local-models/Flash-Next-int8-experts-top64-v1 \
  build/release/flash/partitioned-jobs-m32-mixed.json
```

Use `LAYERS=47`, `ROWS=8192 TILE=64`, all-hit/all-miss patterns, or matching
captured ID fixtures to vary the class distribution. ID-only input uses
synthetic activation values and is labeled explicitly. The default runs
layers 0/47 and all five patterns with one store constructor and final hash pass.

Previous persistent-grid candidates passed exact-output/lifetime checks but
regressed Root's matched primitive screens. The 64/16 grid changed 7.395 ms to
9.421 ms. Higher-grid and phase-isolated v2 candidates also lost, so they remain
disabled. Compaction is a separate bounded experiment, not a claimed speedup.
Production files, defaults and saved formats are unchanged.
