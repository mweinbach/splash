# Optional original PLE SSD execution

`SPLASH_FLASH_PLE_SSD_STREAMING=1` selects an optional route; missing/0 keeps the
resident original table. All changes remain local. The loader keeps the original
affine bytes on disk and maps only the remaining model weights into Metal.

Each synchronous executor (singleton forward/verify, joint verify, ordinary
batch decode, and batch prefill) owns a bounded Shared staging arena. The loader
owns one immutable `FlashPLESSDStore`, shared by those executors. Its CPU row
cache is optional and bounded independently of the full table.

Before adding model dispatches, `FlashPLESSD::prepare` computes canonical I64 IDs
using checkpoint multipliers, 16 stored vocabulary sizes and offsets, incoming
token IDs, and private copies of the completed request-owned two-token histories.
It fetches unchanged 100-byte rows: 80 packed-Q4 bytes, 10 BF16 scale bytes, and
10 BF16 bias bytes. A disk failure throws before any target graph is submitted.
Preparation does not mutate request history or convolution state.

The target graph retains `flash_ple_hash` and ordered
`flash_ple_update_history`. `flash_ple_ssd_gather` checks every GPU ID against the
staged CPU ID, then executes the original affine F32 arithmetic, rounds the
affine result to BF16, multiplies by the checkpoint BF16 shared scale, and rounds
again to BF16. An invalid/mismatched ID writes NaN and sets the sticky index bit;
nonfinite reconstructed values retain the sticky numeric diagnostic. Existing
PLE projection, convolution, injection, and prefix restoration are unchanged.

No GPU ID readback or additional submission is needed. Calls are already
synchronous on the GPU owner, so CPU histories are read only after the preceding
command completed. After speculative partial restore, the next call prepares
IDs from the restored canonical history. Staging is single-use per graph.

The two buffers admit `round16K(lanes*rows*16*8)` plus
`round16K(lanes*rows*16*100)` bytes. At the largest four-lane 2048-row geometry,
this is 14,155,776 bytes. These buffers are included in both workspace admission
and the actual allocation ledger. Public `pleSSDStagingBytes()` getters report
the four executor arenas separately and return zero while disabled.

CPU qualification of the production hash passes 17,203,366 checks, 7,070 valid
hash calls, and 289 rejected input cases in both optimized and ASan/UBSan builds.
The test uses independent `__int128` wrap/modulo goldens, exact small checkpoint
metadata, EOS/cold histories, lane permutation, chunking, signed boundary
metadata, R1/4/16/2048 and lanes1/2/4. No Metal backend or GPU is linked.

Changed production C++ objects and the Metal4.1 shader compiled with `-Werror`.
GPU primitive equality and actual service qualification must be run separately
by the root GPU owner; compilation and CPU hash correctness do not qualify the
complete service.
