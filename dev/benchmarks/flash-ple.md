# Flash-Next PLE primitive qualification

The native PLE path consumes the checkpoint's stored I64 hash arrays and all
128 original affine Q4/G32 shards. It gathers selected rows without joining or
dequantizing the complete table. Each dispatch binds eight shards. Reconstruction
uses F32 coefficients and rounds to BF16 before applying the checkpoint's shared
BF16 scale; actual MLX Metal output resolves the possible contraction ambiguity.

The PLE gate retains the actual MLX BF16 sum partition. Widths up to 64 fold
sequentially in BF16. The production width 2560 uses 640 threads, four consecutive
BF16 additions per thread, F32 SIMD reduction followed by BF16, then a second SIMD
reduction with the same boundary. The source's BF16 divisor, signed square root
and sigmoid are retained. This distinction materially changes gating even when
normalized keys, queries and products already match exactly.

The standalone gate sigmoid uses precise exp with BF16 boundaries. Convolution
activation follows compiled `nn.silu` and retains fast exp; the two operator
contracts differ at finite BF16 inputs such as -6.84375.

Build the isolated oracle and its two-kernel library without replacing the default
server's executable or metallib:

```sh
mkdir -p build/release/flash/ple-oracle

xcrun -sdk macosx clang++ -std=c++20 -O3 -Wall -Wextra -Werror \
  -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 \
  -fobjc-arc dev/tests/engine/flash_ple_metal_test.mm \
  runtime/flash/FlashPLE.cpp runtime/flash/FlashHC.cpp \
  runtime/flash/FlashWeights.mm runtime/flash/FlashDescriptor.mm \
  runtime/metal/MetalBackend.mm runtime/metal/DeviceCapabilities.cpp \
  -framework Foundation -framework Metal -framework IOKit \
  -o build/release/flash/ple-oracle/flash-ple-metal-test

xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror \
  -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 \
  -c runtime/metal/kernels/shared/flash_ple.metal \
  -o build/release/flash/ple-oracle/flash_ple.air

xcrun -sdk macosx metal -std=metal4.1 -O3 -Wall -Wextra -Werror \
  -Iruntime -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1 \
  -c runtime/metal/kernels/shared/flash_hc.metal \
  -o build/release/flash/ple-oracle/flash_hc.air

xcrun -sdk macosx metallib \
  build/release/flash/ple-oracle/flash_ple.air \
  build/release/flash/ple-oracle/flash_hc.air \
  -o build/release/flash/ple-oracle/flash_ple.metallib
```

Run GPU work serially. `flash_ple_mlx_oracle.py` requires explicit `--run-gpu` and
a fresh output directory; it preserves its CPU fixtures. The root-run source
goldens at `build/release/flash/ple-mlx-golden-v3` include actual MLX floating
outputs and independent CPU I64 hash expectations. Individual manifest entries
identify their provenance and carry payload hashes.

```sh
MTL_SHADER_VALIDATION=1 \
  build/release/flash/ple-oracle/flash-ple-metal-test \
  build/release/flash/ple-oracle/flash_ple.metallib \
  build/release/flash/ple-mlx-golden-v3 \
  <derived-model-directory>
```

Omit the final model argument to test synthetic source shards and post-projection
fixtures without loading the full converted model. Optional
`SPLASH_PLE_ORACLE_DUMP_DIR` saves native intermediate BF16 arrays before comparison
failures. No prompt text or weight payloads are logged.

The corrected root-run oracle passed Metal shader validation for 1,706,496
elements, including 1,706,494 bit-exact values; the two remaining differences
fit the original numerical allowance, maximum absolute difference 0.00012207.
Native whole-versus-split output/state, injection, per-lane speculative prefix
restoration, and selected actual-checkpoint row gathers are required bit-exact.
Evidence: `build/release/metal41/flash-ple-corrected-mlx-oracle.log`.

Speculative callers retain pre-command token-history and convolution-state
snapshots. `addPLERestorePrefix` reconstructs each lane's explicit retained token
prefix; the runtime maps its `accepted + 1` convention to that count. Post-command
state alone cannot recover a rejected verify suffix.
