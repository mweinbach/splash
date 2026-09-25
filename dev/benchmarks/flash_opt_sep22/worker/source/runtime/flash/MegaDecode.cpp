#include "flash/MegaDecode.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

namespace splash::flash::mk {
namespace {

struct Affine {
  uint64_t wRowStride = 0, wExpertStride = 0, pRowStride = 0, pExpertStride = 0;
  uint32_t bits = 0, group = 0, pad0 = 0, pad1 = 0;
};
static_assert(sizeof(Affine) == 48);

struct RouterParams {
  uint32_t rows = 0, pad0 = 0, pad1 = 0, pad2 = 0;
  Affine sharedGate, sharedUp, sharedLogit;
};
struct MoEParams {
  uint32_t rows = 0, pad0 = 0, pad1 = 0, pad2 = 0;
  Affine gate, up, down, sharedDown;
};
struct TiledParams {
  uint32_t rows = 0, pad0 = 0, pad1 = 0, pad2 = 0;
  uint64_t gateStride = 0, downStride = 0, gateParameterStride = 0, downParameterStride = 0;
  uint32_t sharedBits = 0, sharedGroup = 0, sharedPaddedN = 0, pad3 = 0;
  uint64_t sharedBiasOffset = 0;
};
static_assert(sizeof(TiledParams) == 72);
struct CombineParams {
  uint32_t rows = 0, hasNorm = 0, normConvention = 0;
  float epsilon = 0;
};

constexpr uint64_t kPlanBytes = 4 * (4 + 160 + 160 * 16 + 160 + 160);

bool flag(const char *name) {
  const char *value = std::getenv(name);
  return value && std::strcmp(value, "1") == 0;
}

Affine affine(const FlashAffineProjection &p) {
  return {p.weightRowStrideBytes, p.weightExpertStrideBytes, p.parameterRowStrideBytes / 2,
          p.parameterExpertStrideBytes / 2, p.bits, p.groupSize, 0, 0};
}

bool mapped(const FlashAffineProjection &p) {
  return p.weights && p.scales && p.biases && p.weights->buffer && p.scales->buffer &&
         p.biases->buffer && p.weightRowStrideBytes % 4 == 0 && p.parameterRowStrideBytes % 2 == 0 &&
         p.weightExpertStrideBytes % 4 == 0 && p.parameterExpertStrideBytes % 2 == 0;
}

bool format(const FlashAffineProjection &p, uint32_t experts, uint32_t K, uint32_t N,
            std::initializer_list<uint32_t> bits) {
  if (!mapped(p) || p.experts != experts || p.inputSize != K || p.outputSize != N) return false;
  if (p.groupSize != 64 && p.groupSize != 128) return false;
  for (uint32_t b : bits) if (p.bits == b) return true;
  return false;
}

} // namespace

bool moeEnabled() noexcept {
  static const bool enabled = flag("SPLASH_MK_MOE");
  return enabled;
}

bool concurrentEnabled() noexcept {
  static const bool enabled = flag("SPLASH_MK_CONCURRENT");
  return enabled;
}

constexpr uint64_t kScratchBytes[7] = {
    uint64_t{kMaximumRows} * 512 * 2 + 16384,            // router logits
    uint64_t{kMaximumRows} * 640 * 2 + 16384,            // shared intermediate
    16384,                                               // shared gate logit
    16384,                                               // plan
    uint64_t{kMaximumRows} * 10 * 16 * 640 * 2 + 16384,  // expert intermediate
    uint64_t{kMaximumRows} * 10 * 2560 * 2 + 16384,      // expert down
    uint64_t{kMaximumRows} * 2560 * 2 + 16384};          // shared down
static_assert(kPlanBytes <= 16384);

MoEScratch allocateMoEScratch(metal::MetalBackend &backend) {
  const auto shared = metal::BufferStorage::Shared;
  MoEScratch s;
  s.logits = backend.allocateBuffer(kScratchBytes[0], shared, "mk-moe-router-logits");
  s.sharedInter = backend.allocateBuffer(kScratchBytes[1], shared, "mk-moe-shared-inter");
  s.sharedLogit = backend.allocateBuffer(kScratchBytes[2], shared, "mk-moe-shared-logit");
  s.plan = backend.allocateBuffer(kScratchBytes[3], shared, "mk-moe-plan");
  s.inter = backend.allocateBuffer(kScratchBytes[4], shared, "mk-moe-inter");
  std::memset(s.inter.contents(), 0, s.inter.sizeBytes());
  s.expertDown = backend.allocateBuffer(kScratchBytes[5], shared, "mk-moe-expert-down");
  s.sharedDown = backend.allocateBuffer(kScratchBytes[6], shared, "mk-moe-shared-down");
  return s;
}

uint64_t moeScratchPlannedBytes() noexcept {
  if (!moeEnabled()) return 0;
  uint64_t total = 0;
  for (const uint64_t bytes : kScratchBytes) total += (bytes + 65535) & ~uint64_t(65535);
  return total;
}

bool addMoE(metal::CommandGraph &graph, const FlashWeights &weights, const std::string &mlp,
            const metal::MetalBuffer &mixed, const metal::MetalBuffer &gates,
            const metal::MetalBuffer &hyper, const FlashTensor *nextNorm, bool onePlusNorm,
            const metal::MetalBuffer &normalized, const MoEScratch &s, uint32_t rows,
            float epsilon, const opt::MkExpertTiles *tiles) {
  if (!moeEnabled() || !rows || rows > kMaximumRows) return false;
  const auto &router = weights.tensor(mlp + ".gate.weight");
  if (router.dtype != FlashDType::BF16 || router.shape != std::vector<uint64_t>{512, 2560}) return false;
  const auto &sg = weights.projection(mlp + ".shared_expert.gate_proj");
  const auto &su = weights.projection(mlp + ".shared_expert.up_proj");
  const auto &sd = weights.projection(mlp + ".shared_expert.down_proj");
  const auto &sl = weights.projection(mlp + ".shared_expert_gate");
  const auto &g = weights.projection(mlp + ".switch_mlp.gate_proj");
  const auto &u = weights.projection(mlp + ".switch_mlp.up_proj");
  const auto &d = weights.projection(mlp + ".switch_mlp.down_proj");
  if (!format(sg, 1, 2560, 640, {4, 8}) || !format(su, 1, 2560, 640, {4, 8}) || sg.bits != su.bits ||
      !format(sd, 1, 640, 2560, {4, 8}) || !format(sl, 1, 2560, 1, {4, 8}) ||
      !format(g, 512, 2560, 640, {4}) || !format(u, 512, 2560, 640, {4}) ||
      !format(d, 512, 640, 2560, {4}) || g.groupSize != 64 || u.groupSize != 64 || d.groupSize != 64)
    return false;
  if (nextNorm && nextNorm->dtype != FlashDType::BF16) return false;

  RouterParams rp;
  rp.rows = rows;
  rp.sharedGate = affine(sg);
  rp.sharedUp = affine(su);
  rp.sharedLogit = affine(sl);
  const bool routerMPP = sg.bits == 8 && sg.groupSize == 128 && su.groupSize == 128 &&
                         !flag("SPLASH_MK_ROUTER_SIMD");
  // The SIMD router variants cover up to eight rows.
  if (rows > 8 && !routerMPP) return false;
  graph.add(routerMPP ? std::string("mk_moe_router_mpp") : "mk_moe_router_r" + std::to_string(rows),
            {mixed, router.buffer, sg.weights->buffer, sg.scales->buffer, sg.biases->buffer,
             su.weights->buffer, su.scales->buffer, su.biases->buffer, sl.weights->buffer,
             sl.scales->buffer, sl.biases->buffer, s.logits, s.sharedInter, s.sharedLogit},
            rp, {routerMPP ? 37u : 145u, 1, 1}, {256, 1, 1});

  MoEParams mp;
  mp.rows = rows;
  mp.gate = affine(g);
  mp.up = affine(u);
  mp.down = affine(d);
  mp.sharedDown = affine(sd);
  const uint32_t maxUnique = rows * 10;
  graph.add("mk_moe_plan", {s.logits, s.plan}, mp, {1, 1, 1}, {256, 1, 1});
  const opt::MkTiledProjection *sharedTiles = tiles ? opt::mkTiledProjection(sd) : nullptr;
  if (tiles && sharedTiles) {
    TiledParams tp;
    tp.rows = rows;
    tp.gateStride = g.weightExpertStrideBytes;
    tp.downStride = d.weightExpertStrideBytes;
    tp.gateParameterStride = g.parameterExpertStrideBytes;
    tp.downParameterStride = d.parameterExpertStrideBytes;
    tp.sharedBits = sd.bits;
    tp.sharedGroup = sd.groupSize;
    tp.sharedPaddedN = sharedTiles->paddedN;
    tp.sharedBiasOffset = uint64_t(sd.inputSize / sd.groupSize) * sharedTiles->paddedN;
    const bool wide = rows > 8;
    graph.add(wide ? "mk_moe_gate_up_t16" : "mk_moe_gate_up_t", {mixed, tiles->gate, tiles->up,
              tiles->gateParameters, tiles->upParameters, s.inter, s.plan}, tp, {640 / 32, maxUnique, 1},
              {wide ? 128u : 256u, 1, 1});
    graph.add(wide ? "mk_moe_down_t16" : "mk_moe_down_t", {s.inter, s.sharedInter, tiles->down, tiles->downParameters, sharedTiles->tiles,
              sharedTiles->parameters, s.expertDown, s.sharedDown, s.plan}, tp, {2560 / 32, maxUnique + 1, 1},
              {64, 1, 1});
  } else {
  graph.add("mk_moe_gate_up",
            {mixed, g.weights->buffer, g.scales->buffer, g.biases->buffer,
             u.weights->buffer, u.scales->buffer, u.biases->buffer, s.inter, s.plan},
            mp, {640 / 32, maxUnique, 1}, {128, 1, 1});
  graph.add("mk_moe_down",
            {s.inter, s.sharedInter, d.weights->buffer, d.scales->buffer, d.biases->buffer,
             sd.weights->buffer, sd.scales->buffer, sd.biases->buffer, s.expertDown, s.sharedDown,
             s.plan},
            mp, {2560 / 32, maxUnique + 1, 1}, {160, 1, 1});
  }

  CombineParams cp{rows, nextNorm ? 1u : 0u, onePlusNorm ? 0u : 1u, epsilon};
  graph.add("mk_moe_combine",
            {s.expertDown, s.sharedDown, s.sharedLogit, s.plan, gates, hyper,
             nextNorm ? nextNorm->buffer : hyper, normalized},
            cp, {rows, 4, 1}, {640, 1, 1});
  return true;
}

} // namespace splash::flash::mk
