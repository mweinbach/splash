#pragma once

#include "flash/FlashWeights.hpp"
#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::flash::opt {

// Multi-row affine-quantized GEMV (runtime/metal kernels opt_qmv_*). Every
// weight byte is read once per dispatch and reused for all rows, so small
// verification windows cost about the same traffic as a single row.
// SPLASH_OPT_QMV=0 disables the route (default on).
bool qmvEnabled() noexcept;

inline constexpr uint32_t kQmvMaximumRows = 5;
// Larger windows (batched verification) run as balanced chunks of <= 5 rows.
inline constexpr uint32_t kQmvMaximumSplitRows = 20;
// Short prefill windows (21..128 rows by default) also use the chunked kernels: 16-row
// matrix-unit chunks where the format allows, 5-row SIMD chunks otherwise.
inline constexpr uint32_t kQmvMaximumPrefillRows = 256;
// Runtime cap within kQmvMaximumPrefillRows (SPLASH_OPT_PREFILL_QMV_ROWS).
uint32_t prefillQmvRows() noexcept;

// y[rows, N] = x[rows, K] * W^T for a rank-2 MLX affine projection.
// x and y are contiguous BF16 rows. Returns false, adding nothing, when the
// projection format, row count or alignment is not supported.
bool addQmv(metal::CommandGraph &graph, const metal::MetalBuffer &input,
            const FlashAffineProjection &projection,
            const metal::MetalBuffer &output, uint32_t rows);

// silu(x*gate^T) * (x*up^T) for a shared expert whose gate and up projections
// share a format, with flash_moe_silu_multiply's BF16 boundaries; one dispatch
// per <= 5-row chunk. Returns false, adding nothing, for unsupported formats.
bool addSharedSwiGLU(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                     const FlashAffineProjection &gate, const FlashAffineProjection &up,
                     const metal::MetalBuffer &output, const metal::MetalBuffer &diagnostics,
                     uint32_t rows);

// Tall inputs (rows > kQmvMaximumSplitRows) against small projections
// (N <= 64): one dispatch of four-row blocks, e.g. prefill a/b/gate/injection.
bool addQmvTall(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                const FlashAffineProjection &projection,
                const metal::MetalBuffer &output, uint32_t rows);

// Hyper-connection down projection (10240 -> 320, SiLU(raw/4)) plus the four
// optional injection gates (2*sigmoid(raw/4)), one dispatch.
bool addHCDown(metal::CommandGraph &graph, const metal::MetalBuffer &normalized,
               const FlashAffineProjection &down, const FlashAffineProjection *injection,
               const metal::MetalBuffer &activated, const metal::MetalBuffer &gates,
               uint32_t rows);

// Hyper-connection up projection (320 -> 4x2560) fused with the sigmoid-gated
// stream mix into the 2560-wide block input.
bool addHCUpMix(metal::CommandGraph &graph, const metal::MetalBuffer &normalized,
                const metal::MetalBuffer &activated, const FlashAffineProjection &up,
                const metal::MetalBuffer &mixed, uint32_t rows);

// y[rows, N] = x[rows, K] * W^T for a dense BF16 W[N, K].
bool addDenseBF16Rows(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                      const FlashTensor &weights, const metal::MetalBuffer &output,
                      uint32_t rows);

// Routed experts on the original affine expert weights for small row counts:
// intermediate[rows*sel, 640] = silu(gate) * up, expertDown[rows*sel, 2560].
// Opt-in with SPLASH_OPT_MOE=1: requires GPU-mapped original expert weights.
bool moeEnabled() noexcept;
// SPLASH_OPT_MOE=1: Q4 experts replace the INT8 store on every path and the
// store is not kept resident. SPLASH_OPT_MOE=2: Q4 experts serve <= 5-row
// decode/verify only; prefill and wider batches keep the resident INT8 store.
bool moeReplacesInt8() noexcept;

// SPLASH_OPT_PREPARE_VERIFY=0 disables building the singleton verify graph
// while the MTP head fold runs (default on).
bool prepareVerifyEnabled() noexcept;
// SPLASH_OPT_DRAFT_CHAIN=0 disables running the chained draft steps as one
// command with GPU token feedback (default on).
bool draftChainEnabled() noexcept;
// SPLASH_OPT_QSA_KERNEL=0 restores flash_qsa_mpp_online_partition for the
// few-row selected-block attention partitions (default: vectorized staging).
bool qsaKernelEnabled() noexcept;
// SPLASH_OPT_GDN_KERNEL=0 restores flash_gdn_lazy_verify_sg16 for lazy-rollback
// GDN verification (default: parallel operand phase, barrier-free recurrence).
bool gdnKernelEnabled() noexcept;
// SPLASH_OPT_STATE_POOL=0 frees released request states instead of reusing
// them for the next request (default: pool up to four).
bool statePoolEnabled() noexcept;
// SPLASH_OPT_MPPQ=0 keeps batched (6..20-row) projections on the SIMD
// multi-row kernel instead of the matrix-unit kernel for 4/8-bit weights.
bool mppqEnabled() noexcept;
// SPLASH_OPT_FUSED_SWIGLU=0 keeps separate shared-expert gate/up/SiLU dispatches.
bool fusedSwiGLUEnabled() noexcept;

// Lossless 8-bit copies of 5/6-bit dense projection codes (same scales and
// biases) for the matrix-unit path. FlashForward builds them once at startup
// for eligible projections (SPLASH_OPT_REPACK=0 disables them).
bool repackEnabled() noexcept;
// SPLASH_OPT_PREFILL_QMV=0 keeps 21..64-row prefill projections and HC on the
// dense-cache route.
bool prefillQmvEnabled() noexcept;
bool repackEligible(const FlashAffineProjection &projection) noexcept;
[[nodiscard]] metal::MetalBuffer repackCodes(metal::MetalBackend &backend,
                                             const FlashAffineProjection &projection);
void registerRepackedCodes(const void *codes, const metal::MetalBuffer &repacked);

// SPLASH_MK_QMV=1: 2..8-row windows of wide dense projections run on the
// matrix units over a lossless lane-major tile copy of the codes with
// transposed scales/biases (kernels/mk_dense.metal). FlashForward builds the
// copies at startup for eligible projections.
bool mkDenseEnabled() noexcept;
bool mkTileEligible(const FlashAffineProjection &projection) noexcept;
uint64_t mkTileBytes(const FlashAffineProjection &projection) noexcept;
struct MkTiledProjection {
  metal::MetalBuffer tiles;
  metal::MetalBuffer parameters;  // BF16 scales then biases, [K/G][paddedN] each
  uint32_t paddedN = 0;
};
[[nodiscard]] MkTiledProjection mkTileProjection(metal::MetalBackend &backend,
                                                 const FlashAffineProjection &projection);
void registerMkTiled(const void *codes, const MkTiledProjection &tiled);
// Up to four projections of one input (same K) tiled into shared buffers and
// dispatched together (mk_mpt_multi); e.g. GDN qkv/z/a/b or attention q/k/v/index.
struct MkTiledGroup {
  metal::MetalBuffer tiles, parameters;
  std::vector<std::byte> table;  // MkMultiParams without rows/strides
  uint32_t blocks = 0, K = 0, count = 0;
  uint32_t n[4] = {};
};
bool mkGroupEligible(const std::vector<const FlashAffineProjection *> &projections) noexcept;
uint64_t mkGroupBytes(const std::vector<const FlashAffineProjection *> &projections) noexcept;
[[nodiscard]] MkTiledGroup mkTileGroup(metal::MetalBackend &backend,
                                       const std::vector<const FlashAffineProjection *> &projections);
// outputs[i] receives projection i ([rows, N_i] contiguous). Returns false when
// rows is outside 2..8.
bool addMkGroup(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                const MkTiledGroup &group, const std::vector<metal::MetalBuffer> &outputs,
                uint32_t rows);
// SPLASH_MK_MOE_TILED=1 (with SPLASH_MK_QMV=1 and SPLASH_MK_MOE=1): a lossless
// lane-major tile copy of every layer's Q4 routed experts, with per-expert
// transposed scales/biases, for the few-row MoE kernels (mk_moe_*_t).
bool mkExpertTilesEnabled() noexcept;
struct MkExpertTiles {
  metal::MetalBuffer gate, up, down, gateParameters, upParameters, downParameters;
};
bool mkExpertTilesEligible(const FlashAffineProjection &gate, const FlashAffineProjection &up,
                           const FlashAffineProjection &down) noexcept;
uint64_t mkExpertTileBytes(const FlashAffineProjection &gate, const FlashAffineProjection &up,
                           const FlashAffineProjection &down) noexcept;
[[nodiscard]] MkExpertTiles mkTileExperts(metal::MetalBackend &backend, const FlashAffineProjection &gate,
                                          const FlashAffineProjection &up, const FlashAffineProjection &down);
// Expert tiles registered by FlashForward under the gate and down code
// addresses of their layer, for the blocked-MoE bucket kernels.
void registerMkExpertTiles(const FlashAffineProjection &gate, const FlashAffineProjection &down,
                           const MkExpertTiles &tiles);
const MkExpertTiles *mkExpertTilesFor(const FlashAffineProjection &projection) noexcept;
// Tile copy registered for a dense projection by FlashForward, or null.
const MkTiledProjection *mkTiledProjection(const FlashAffineProjection &projection) noexcept;
// Hyper-connection block (kernels/mk_hc.metal): down (+ inject) as a split-K
// tiled group, up tiled with streams interleaved per 32-column block.
struct MkHC {
  MkTiledGroup down;
  metal::MetalBuffer upTiles, upParameters;
  uint32_t upBits = 0;
  bool injection = false;
};
bool mkHCEligible(const FlashAffineProjection &down, const FlashAffineProjection *inject,
                  const FlashAffineProjection &up) noexcept;
uint64_t mkHCBytes(const FlashAffineProjection &down, const FlashAffineProjection *inject,
                   const FlashAffineProjection &up) noexcept;
[[nodiscard]] MkHC mkTileHC(metal::MetalBackend &backend, const FlashAffineProjection &down,
                            const FlashAffineProjection *inject, const FlashAffineProjection &up);
[[nodiscard]] metal::MetalBuffer mkHCPartials(metal::MetalBackend &backend);
// normalized [rows, 4, 2560] -> mixed [rows, 2560] and (with injection) gates [rows, 4].
bool addMkHC(metal::CommandGraph &graph, const metal::MetalBuffer &normalized, const MkHC &hc,
             const metal::MetalBuffer &partials, const metal::MetalBuffer &mixed,
             const metal::MetalBuffer &gates, uint32_t rows);
bool addMoEExperts(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                   const FlashAffineProjection &gate, const FlashAffineProjection &up,
                   const FlashAffineProjection &down, const metal::MetalBuffer &expertIDs,
                   const metal::MetalBuffer &intermediate, const metal::MetalBuffer &expertDown,
                   uint32_t rows, uint32_t selections);

// Leading vocabulary rows projected for greedy single-row MTP drafts
// (SPLASH_OPT_DRAFT_VOCAB, default 98304; 0 projects the full vocabulary).
uint32_t draftVocabulary() noexcept;

// Top-k MoE routing with one simdgroup per row (512 experts, <= 10 picks).
bool addRoute(metal::CommandGraph &graph, const metal::MetalBuffer &logits,
              const metal::MetalBuffer &expertIDs, const metal::MetalBuffer &routeWeights,
              uint32_t rows, uint32_t experts, uint32_t selections, bool normalizeTopK);

} // namespace splash::flash::opt

namespace splash::flash::opt {
// Diagnostic host timers (SPLASH_OPT_TIMERS=1): accumulate wall time per label
// and print a summary line to stderr every 256 samples of that label.
class ScopedTimer final {
public:
  explicit ScopedTimer(const char *label) noexcept;
  ~ScopedTimer();
  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;
private:
  const char *label_;
  uint64_t start_ = 0;
};
} // namespace splash::flash::opt
