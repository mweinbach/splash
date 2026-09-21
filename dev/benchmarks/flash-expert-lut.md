This isolated candidate saves the existing prefill producer's exact BF16 weight
operands in a sidecar. It retains the original packed Q4 codes and replaces
runtime FP32 `q*scale+bias`, BF16 conversion, and coefficient finite checks with
one BF16 table lookup. Input checks, K64 MPP traversal, BF16 SwiGLU boundaries,
stable bucket jobs, down scatter, and canonical combine remain literal.

Each Q4/G64 group saves all sixteen possible coefficients. The converter uses
two separate NumPy FP32 operations followed by BF16 ties-to-even rounding;
rounding the real-number expression directly to BF16 is not equivalent. It
rejects nonfinite source parameters and any nonfinite FP32/BF16 coefficient
before publication. An independent CPU reference covers signed coefficients,
negative scales, signed zero, CPU subnormals, halfway boundaries and malformed
geometry. These are exact operands for the existing BF16 MPP producer. The raw
singleton FP32 QMV route must keep its original coefficient policy.

The saved layout is `[E,Nblock64,Kgroup64,Nlane64,Q16]`, little-endian BF16. An
N64/K64 staging tile addresses contiguous 2 KiB of tables. Each projection table
is 419,430,400 bytes, each layer is 1,258,291,200 bytes (1.171875 GiB), and all
144 target-model tables are 60,397,977,600 bytes (56.25 GiB). Original packed
codes cost another 56.25 GiB across these projections. Retaining both originals
and sidecars therefore adds 56.25 GiB of file mappings; a later standalone
converted model could replace the original metadata, while retaining FP32
coefficient metadata for raw decode requires a separately declared route.

The CPU converter verifies the immutable source manifest and freshly hashes
every touched source shard. It checks all sixteen coefficients of every group,
writes to a temporary directory, records payload/source hashes and geometry,
and publishes atomically. Existing outputs and paths inside the source package
are refused. It never modifies the original checkpoint or local source bundle.

```sh
.venv/bin/python dev/tools/flash_expert_lut.py \
  --package install/local-models/Flash-Next-oQ4e-mtp-v1 \
  --output build/flash-expert-lut-format/layer0-v2 --layer 0

make -f Makefile -f dev/benchmarks/flash_moe_lut_oracle.mk \
  BUILD=build/flash-expert-lut-format flash-moe-lut-oracle

.venv/bin/python -m unittest dev/tests/flash/test_flash_expert_lut_format.py
build/flash-expert-lut-format/flash-moe-lut-oracle --cpu-self-test
build/flash-expert-lut-format/flash-moe-lut-oracle --validate-sidecar \
  build/flash-expert-lut-format/layer0-v2 \
  install/local-models/Flash-Next-oQ4e-mtp-v1
```

The compiled oracle has a private library linked from the unchanged production
AIRs plus the private candidate shader. No production route is changed.
Compilation, the CPU self-test and sidecar validation submit no GPU commands.
Root runs inference serially after stopping the service:

```sh
FLASH_MOE_LUT_ROWS=2048 FLASH_MOE_LUT_TILE=32 FLASH_MOE_LUT_PAIRS=6 \
build/flash-expert-lut-format/flash-moe-lut-oracle \
  build/flash-expert-lut-format/flash-lut.metallib \
  install/local-models/Flash-Next-oQ4e-mtp-v1 \
  build/flash-expert-lut-format/layer0-v2 \
  build/release/flash/moe-lut-layer0-m32-r2048.json
```

Use rows8192/tile64 for the existing M64/SG8 geometry, or set
`FLASH_MOE_LUT_PHASE=gate_up|down|both` to isolate the replaced phase. The default
compares deterministic synthetic BF16 hidden inputs and spread/concentrated
top-10 IDs using actual original weights. `FLASH_MOE_LUT_INPUT` and
`FLASH_MOE_LUT_IDS` accept paired exact raw BF16/I64 captures; both must be set.
`FLASH_MOE_LUT_PREFIX` selects another layer and must match its sidecar. Layers
24 and47 can be converted with the same CPU tool.

Before timing, the oracle exhaustively compares all 629,145,600 saved layer
coefficients with original Metal reconstruction, including every Q4 code rather
than only codes used by a fixture. It then checks every gate/up activation,
canonical down and combined BF16 byte against the qualified Q4x8 producer,
independent CPU stable-bucket expectations, sticky diagnostics and output
canaries. Matched commands alternate order after warmup. Reports include source
and sidecar identities, actual pipeline threadgroup memory, job utilization,
and complete-chain GPU/wall time. This primitive experiment still requires a
full-model HTTP qualification before promotion; CPU identity proves no speedup.
