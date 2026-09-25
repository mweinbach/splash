#pragma once

// Megakernel decode phases (kernels/mk_*.metal) for 1..16 row windows: plain
// decode, MTP verification and batched verification of several requests. SPLASH_MK_MOE=1 enables the fused MoE; it needs
// GPU-mapped original Q4 experts (SPLASH_OPT_MOE=1).
#include "flash/FlashWeights.hpp"
#include "flash/OptQmv.hpp"
#include "metal/CommandGraph.hpp"

#include <cstdint>

namespace splash::flash::mk {

inline constexpr uint32_t kMaximumRows = 16;

bool moeEnabled() noexcept;
// SPLASH_MK_CONCURRENT=1: independent projections of one phase share a
// concurrent dispatch group.
bool concurrentEnabled() noexcept;

struct MoEScratch final {
  metal::MetalBuffer logits;        // BF16[16, 512]
  metal::MetalBuffer sharedInter;   // BF16[16, 640]
  metal::MetalBuffer sharedLogit;   // BF16[16]
  metal::MetalBuffer plan;          // MkMoEPlan
  metal::MetalBuffer inter;         // BF16[160][16, 640]
  metal::MetalBuffer expertDown;    // BF16[160, 2560]
  metal::MetalBuffer sharedDown;    // BF16[16, 2560]
};

MoEScratch allocateMoEScratch(metal::MetalBackend &backend);
// Upper bound of allocateMoEScratch's allocations (0 when the fused MoE is off).
uint64_t moeScratchPlannedBytes() noexcept;

// mixed BF16[rows, 2560] -> hyper += inject(branch) and, when nextNorm is set,
// normalized = grouped RMS norm of the updated hyper streams.
// Returns false, adding nothing, when a weight format is unsupported. With
// expert tiles (and a tiled shared down projection) the expert phases run on
// mk_moe_gate_up_t / mk_moe_down_t.
bool addMoE(metal::CommandGraph &graph, const FlashWeights &weights, const std::string &mlpPrefix,
            const metal::MetalBuffer &mixed, const metal::MetalBuffer &gates,
            const metal::MetalBuffer &hyper, const FlashTensor *nextNorm, bool onePlusNorm,
            const metal::MetalBuffer &normalized, const MoEScratch &scratch, uint32_t rows,
            float epsilon, const opt::MkExpertTiles *tiles = nullptr);

} // namespace splash::flash::mk
