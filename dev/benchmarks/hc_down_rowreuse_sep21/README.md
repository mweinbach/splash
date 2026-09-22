# Exact packed HC-down row reuse

This is a Root-only GPU component qualifier. It does not edit or select the production graph. The two/four-row shader and bridge are copied from the existing private `flash_hc_down_packed` experiment; the earlier R16/reuse2 screen passed exactness but was slower for all six roles. The previously saved result remains at `build/release/flash/hc-down-packed-reuse2-r16-screen.json`.

The bounded follow-up uses physical rows4/8/16 and original packed Q4/Q5/Q6/Q8 coefficients, reconstructed to F32. Inputs retain BF16. Each row keeps the same lane32 K traversal, separate multiply/add operations, SIMD reduction, BF16 raw-dot boundary, activation and injection boundaries. Mixed injection descriptors stay independent. There is no extra weight cache.

The revised qualifier requires **all raw F32 words, raw BF16 words, BF16 activation words and injection words to match exactly before timing**. A failed case is recorded with `timing_executed:false`; it cannot contribute a speed claim. The literal debug witness must match the production control, and debug/normal candidate outputs must match. Shared output ranges have prefix/suffix guards, sticky diagnostics must remain unchanged, and immutable inputs/source tensors are hashed before/after the complete case.

After all initial proof and guard reads, both routes receive at least150ms actual GPU execution. The following timing pairs use even position balance and do not access CPU tensor, diagnostic or guard buffers until every sample finishes. Reports include first/second-position strata. Kernel timer metadata is read during this interval; tensor payloads are not.

Compile and CPU-check into a fresh directory:

```sh
.venv/bin/python -B dev/benchmarks/hc_down_rowreuse_sep21/prepare.py --build build/hc-down-rowreuse-sep21-v3
```

The prepared v3 build seals247source files and82copied host-object/AIR inputs over the exact pointwise worker. `manifest.json` retains hashes and compiler arguments; `cpu-self-test.json` confirms the200-byte command-timing ABI without creating a Metal device. The failed v2 link closure is preserved; usev3.

Root may run after unloading all other GPU/model work, with a fresh report path:

```sh
SPLASH_FLASH_ALLROWS_FULL512_TARGET=1 SPLASH_FLASH_PLE_SSD_STREAMING=1 \
SPLASH_FLASH_INT8_EXPERT_STORE=/Users/mweinbach/Projects/splash/build/prefill4k-fullcache-artifacts/int8-experts-all512-v1 \
FLASH_HC_DOWN_PACKED_ROWS=4,8 FLASH_HC_DOWN_PACKED_MODES=0,1 FLASH_HC_DOWN_PACKED_PAIRS=10 \
  build/hc-down-rowreuse-sep21-v3/oracle \
  build/hc-down-rowreuse-sep21-v3/splash.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/release/flash/sep21-hc-down-rowreuse-r4-r8-v3.json
```

The private omission loader requires the three `SPLASH` flags above before native original buffers can be created. It checks the certified Full512 metadata, but this oracle performs no expert payload/kernel work. HC packed tensors remain available in the original mapped windows.

The default six original source roles cover different packed formats, mixed injection matrices and the final mixer without injection. `FLASH_HC_DOWN_PACKED_PREFIXES` accepts a comma-separated subset. `FLASH_HC_DOWN_PACKED_INPUT` accepts an exact-sized captured normalized BF16 fixture; otherwise the report explicitly declares deterministic synthetic input. A component gain does not establish a whole-worker decode gain or real-state parity.

Qualified Root result: `build/release/flash/sep21-hc-down-rowreuse-r4-r8-v3.json` passed all24 exact raw F32/BF16 cases, but every candidate was slower after sustained warming and balanced timing. R4/reuse2 speedups ranged0.195–0.661; R4/reuse4 0.072–0.255; R8/reuse2 0.292–0.769; R8/reuse4 0.113–0.297. No whole-worker overlay was made. Do not continue tuning this failed branch.
