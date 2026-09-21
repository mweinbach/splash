This private candidate chooses each expert's row tile from its GPU bucket count:
M16 for at most 16 routes, M32 for at most 32, and M64 otherwise. Three GPU job
prefix/emission lists remain sorted by expert and packed-row order. The original
production M16/M32 SG4 and M64 SG8 consumers run verbatim against those lists,
preserving Q4/G64 reconstruction, BF16 boundaries and ascending K64 MACs.

M16/M32 experts each generate at most one job, so those matrix grids cap their
job dimension at 512. The existing ABI still records its conservative capacity,
which the unchanged shader validates. M64 retains its standard launch capacity.
Each private offsets/count/jobs allocation has a 64-byte canary. All job lists
remain on the GPU; no dynamic count is read by the host between dispatches.
Source graph parameters must survive submission; rewritten parameters are
owned by the private candidate. Production files remain frozen and unchanged.

The private artifact links 49 frozen round7 production AIRs and one job-only
candidate AIR. CPU tests check threshold boundaries, independent list coverage,
and conservative launch bounds; compilation and CPU tests submit no GPU work.
The GPU oracle requires every activation/down/combine BF16 byte to match the
fixed M64 production control. It independently checks compact job counts,
offsets, active/inactive records, bucket maps, sticky diagnostics and canaries
before and after alternating matched commands. Extra dispatch/empty-grid costs
may outweigh reduced tensor padding; a GPU result is required.

```sh
build/flash-moe-hybrid/flash-moe-hybrid-oracle --cpu-self-test
build/flash-moe-hybrid/flash-moe-hybrid-oracle --pipeline-metadata \
  build/flash-moe-hybrid/splash.metallib

FLASH_MOE_HYBRID_ROWS=2048 FLASH_MOE_HYBRID_TILE=64 FLASH_MOE_HYBRID_PAIRS=6 \
build/flash-moe-hybrid/flash-moe-hybrid-oracle \
  build/flash-moe-hybrid/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/moe-hybrid-uniform-layer0.json

FLASH_MOE_HYBRID_ROWS=2048 FLASH_MOE_HYBRID_TILE=64 FLASH_MOE_HYBRID_PAIRS=6 \
FLASH_MOE_HYBRID_PREFIX=language_model.model.layers.24.mlp.switch_mlp \
FLASH_MOE_HYBRID_IDS=build/flash-moe-hybrid/captured-layer24-ids.i64 \
build/flash-moe-hybrid/flash-moe-hybrid-oracle \
  build/flash-moe-hybrid/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/moe-hybrid-captured-layer24.json
```

The first command runs uniform spread and concentrated top 10 fixtures. Captured
ID fixtures for layers 0/24/47 were extracted from the first 2,048-row prefill
record in `expert-calibration-sample0.json.expert-ids.jsonl`, validating source
SHA, extent, valid IDs and ten distinct IDs per row. Hidden activations remain
synthetic BF16, declared in the report. `captured-fixtures.json` records original
capture/fixture hashes and CPU geometry predictions:

| Layer | Fixed M64 padded rows | Hybrid padded rows | Jobs in both |
|---|---:|---:|---:|
|0|37,504|28,432|586|
|24|38,464|27,616|601|
|47|37,760|27,024|590|

Those are 24–28% fewer padded tensor rows, not measured performance gains.
The GPU oracle also accepts rows 8,192 and matching raw BF16 activations through
`FLASH_MOE_HYBRID_INPUT` with the corresponding ID file. Pipeline preflight
checks actual static threadgroup memory; Shader Validation may refuse M64,
which must be reported as a refusal rather than a pass. Root owns all GPU runs.
