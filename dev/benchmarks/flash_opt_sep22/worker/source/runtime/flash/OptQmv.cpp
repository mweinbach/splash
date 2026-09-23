#include "flash/OptQmv.hpp"

#include <algorithm>
#include <dispatch/dispatch.h>
#include <unordered_map>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <string>

namespace splash::flash::opt {
namespace {

// Must match OptQmvParams in opt_qmv.metal.
struct QmvParams {
  uint32_t K, N, rows, xStride, yStride, row0;
  uint64_t wRowStride, pRowStride;
};

// Balanced chunks of at most kRowCapacities' largest entry (5 rows).
template <class Fn>
void forEachChunk(uint32_t rows, Fn &&fn) {
  const uint32_t chunks = (rows + 4) / 5;
  const uint32_t per = (rows + chunks - 1) / chunks;
  for (uint32_t row0 = 0; row0 < rows; row0 += per) fn(row0, std::min(per, rows - row0));
}
static_assert(sizeof(QmvParams) == 40);

struct Variant {
  uint32_t rn, sn, sk, vpt;
};

// Kernel row capacities that are compiled; the smallest one >= rows is used.
// Eight-row variants spill registers (4-8x slower), so the route stops at 5.
constexpr uint32_t kRowCapacities[] = {1, 2, 3, 4, 5};

// From the standalone sweep (build/opt-sep22/bench/sweep1.txt) over this
// model's projection shapes on M5 Ultra.
Variant choose(uint32_t K, uint32_t N, uint32_t capacity) {
  if (K <= 1024) return {4, 1, 4, 8};
  if (capacity >= 3) return N <= 1024 ? Variant{2, 1, 16, 8} : Variant{2, 1, 8, 8};
  if (N <= 1024 || K >= 8192) return {2, 1, 16, 8};
  if (K >= 4096) return {4, 1, 4, 16};
  return {4, 1, 8, 16};
}

} // namespace

uint32_t draftVocabulary() noexcept {
  static const uint32_t rows = [] {
    const char *value = std::getenv("SPLASH_OPT_DRAFT_VOCAB");
    return value ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 98304u;
  }();
  return rows;
}

bool qmvEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_QMV");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}

namespace {
std::mutex &repackMutex() { static std::mutex mutex; return mutex; }
std::unordered_map<const void *, metal::MetalBuffer> &repackRegistry() {
  static std::unordered_map<const void *, metal::MetalBuffer> registry;
  return registry;
}
const metal::MetalBuffer *repackedCodes(const void *codes) {
  std::lock_guard lock(repackMutex());
  const auto found = repackRegistry().find(codes);
  return found == repackRegistry().end() ? nullptr : &found->second;
}
} // namespace

uint32_t prefillQmvRows() noexcept {
  static const uint32_t rows = [] {
    const char *value = std::getenv("SPLASH_OPT_PREFILL_QMV_ROWS");
    const unsigned long parsed = value && *value ? std::strtoul(value, nullptr, 10) : 128ul;
    return static_cast<uint32_t>(std::min<unsigned long>(parsed, kQmvMaximumPrefillRows));
  }();
  return rows;
}

bool prefillQmvEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_PREFILL_QMV");
    return qmvEnabled() && (!value || std::strcmp(value, "0") != 0);
  }();
  return enabled;
}

bool repackEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_REPACK");
    return qmvEnabled() && (!value || std::strcmp(value, "0") != 0);
  }();
  return enabled;
}

bool repackEligible(const FlashAffineProjection &p) noexcept {
  return (p.bits == 5 || p.bits == 6) && p.experts == 1 && p.weights && p.scales && p.biases &&
      (p.groupSize == 64 || p.groupSize == 128) && p.outputSize > 1024 &&
      p.outputSize % 32 == 0 && p.inputSize % p.groupSize == 0 &&
      p.inputSize / p.groupSize <= 160 && p.weights->buffer.contents() &&
      p.weightRowStrideBytes >= (uint64_t(p.inputSize) * p.bits + 7) / 8;
}

metal::MetalBuffer repackCodes(metal::MetalBackend &backend, const FlashAffineProjection &p) {
  const uint32_t N = p.outputSize, K = p.inputSize, bits = p.bits;
  auto out = backend.allocateBuffer((uint64_t(N) * K + 16383) & ~uint64_t(16383),
      metal::BufferStorage::Shared, "opt-repacked-u8-codes");
  const auto *source = static_cast<const uint8_t *>(p.weights->buffer.contents());
  auto *destination = static_cast<uint8_t *>(out.contents());
  if (!source || !destination) throw std::logic_error("repacked codes require Shared buffers");
  const uint64_t rowBytes = p.weightRowStrideBytes;
  const uint32_t mask = (1u << bits) - 1u;
  // MLX packs each row as a little-endian bitstream of `bits`-wide codes.
  dispatch_apply(N, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t n) {
    const uint8_t *row = source + n * rowBytes;
    uint8_t *codes = destination + n * K;
    for (uint32_t k = 0; k < K; ++k) {
      const uint64_t bit = uint64_t(k) * bits;
      const uint64_t byte = bit >> 3;
      uint32_t value = row[byte];
      if (byte + 1 < rowBytes) value |= uint32_t(row[byte + 1]) << 8;
      codes[k] = uint8_t((value >> (bit & 7)) & mask);
    }
  });
  return out;
}

void registerRepackedCodes(const void *codes, const metal::MetalBuffer &repacked) {
  std::lock_guard lock(repackMutex());
  repackRegistry()[codes] = repacked;
}

bool addQmv(metal::CommandGraph &graph, const metal::MetalBuffer &input,
            const FlashAffineProjection &p, const metal::MetalBuffer &output,
            uint32_t rows) {
  if (!qmvEnabled() || !rows || rows > kQmvMaximumPrefillRows || p.experts != 1 ||
      !p.weights || !p.scales || !p.biases)
    return false;
  if (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) return false;
  if (p.groupSize != 64 && p.groupSize != 128) return false;
  if (p.bits == 4 && p.groupSize != 64) return false;
  const uint32_t K = p.inputSize, N = p.outputSize;
  if (!K || !N || K % 16 || K % p.groupSize) return false;
  if (p.weightRowStrideBytes % 4 || p.parameterRowStrideBytes % 2) return false;
  const auto *base = static_cast<const unsigned char *>(p.weights->buffer.contents());
  if (!base || reinterpret_cast<uintptr_t>(base) % 4) return false;
  if (input.sizeBytes() < uint64_t(rows) * K * 2 || output.sizeBytes() < uint64_t(rows) * N * 2)
    return false;
  if (input.sameView(output)) return false;
  // Windows wider than five rows (batched verification, short prefill): the
  // matrix units read each 4/8-bit code (or the 8-bit repack of a 5/6-bit
  // projection) once per sixteen-row chunk; a <= 5-row tail uses the SIMD
  // kernel. Row ranges written by the kernels are disjoint.
  uint32_t simdRow0 = 0;
  const metal::MetalBuffer *repacked = (p.bits == 5 || p.bits == 6) && repackEnabled()
      ? repackedCodes(base) : nullptr;
  const bool matrixFormat = (p.bits == 4 && p.groupSize == 64) || p.bits == 8 || repacked;
  // Narrow outputs (N <= 1024) measured faster in-model as SIMD chunks for
  // verification windows; prefill-sized windows (> 20 rows) use the matrix
  // units for them too, with sixteen-column tiles and a 16-way K split.
  const bool narrow = N <= 1024;
  if (rows > kQmvMaximumRows && mppqEnabled() && matrixFormat && N % 32 == 0 &&
      K / p.groupSize <= 160 && (!narrow || rows > kQmvMaximumSplitRows)) {
    const uint32_t nt = narrow ? 16 : 32, sk = narrow ? 16 : 8;
    const uint32_t bits = repacked ? 8 : p.bits;
    const std::string pipeline = "opt_mppq_b" + std::to_string(bits) + "_g" +
        std::to_string(p.groupSize) + "_nt" + std::to_string(nt) + "_sk" + std::to_string(sk);
    const metal::MetalBuffer &codes = repacked ? *repacked : p.weights->buffer;
    const uint64_t codeRowStride = repacked ? uint64_t(K) : p.weightRowStrideBytes;
    while (rows - simdRow0 > kQmvMaximumRows) {
      const uint32_t matrixRows = std::min<uint32_t>(rows - simdRow0, 16);
      const QmvParams params{K, N, matrixRows, K, N, simdRow0, codeRowStride, p.parameterRowStrideBytes};
      graph.add(pipeline, {input, codes, p.scales->buffer, p.biases->buffer, output},
                params, {N / nt, 1, 1}, {32 * sk, 1, 1});
      simdRow0 += matrixRows;
    }
  }
  if (simdRow0 == rows) return true;
  forEachChunk(rows - simdRow0, [&](uint32_t chunk0, uint32_t count) {
    const uint32_t row0 = simdRow0 + chunk0;
    uint32_t capacity = 0;
    for (uint32_t candidate : kRowCapacities)
      if (candidate >= count) { capacity = candidate; break; }
    const Variant v = choose(K, N, capacity);
    const std::string pipeline = "opt_qmv_b" + std::to_string(p.bits) + "_g" +
        std::to_string(p.groupSize) + "_r" + std::to_string(capacity) + "_n" +
        std::to_string(v.rn) + "_sn" + std::to_string(v.sn) + "_sk" +
        std::to_string(v.sk) + "_v" + std::to_string(v.vpt);
    const QmvParams params{K, N, count, K, N, row0, p.weightRowStrideBytes, p.parameterRowStrideBytes};
    const uint32_t perGroup = v.rn * v.sn;
    graph.add(pipeline, {input, p.weights->buffer, p.scales->buffer, p.biases->buffer, output},
              params, {(N + perGroup - 1) / perGroup, 1, 1}, {32 * v.sn * v.sk, 1, 1});
  });
  return true;
}

bool addSharedSwiGLU(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                     const FlashAffineProjection &gate, const FlashAffineProjection &up,
                     const metal::MetalBuffer &output, const metal::MetalBuffer &diagnostics,
                     uint32_t rows) {
  if (!qmvEnabled() || !fusedSwiGLUEnabled() || !rows || rows > kQmvMaximumSplitRows) return false;
  for (const auto *p : {&gate, &up})
    if (p->experts != 1 || !p->weights || !p->scales || !p->biases || p->bits != 8 ||
        (p->groupSize != 64 && p->groupSize != 128) || p->weightRowStrideBytes % 4 ||
        p->parameterRowStrideBytes % 2 || !p->weights->buffer.contents() ||
        reinterpret_cast<uintptr_t>(p->weights->buffer.contents()) % 4)
      return false;
  if (gate.inputSize != up.inputSize || gate.outputSize != up.outputSize ||
      gate.groupSize != up.groupSize || gate.weightRowStrideBytes != up.weightRowStrideBytes ||
      gate.parameterRowStrideBytes != up.parameterRowStrideBytes)
    return false;
  const uint32_t K = gate.inputSize, N = gate.outputSize;
  if (!K || !N || K % 16 || K % gate.groupSize || !diagnostics) return false;
  if (input.sizeBytes() < uint64_t(rows) * K * 2 || output.sizeBytes() < uint64_t(rows) * N * 2)
    return false;
  forEachChunk(rows, [&](uint32_t row0, uint32_t count) {
    uint32_t capacity = 0;
    for (uint32_t candidate : kRowCapacities)
      if (candidate >= count) { capacity = candidate; break; }
    // Same variant as addQmv would choose for each projection, so every gate
    // and up value is summed exactly as by the separate projections.
    const Variant v = choose(K, N, capacity);
    const std::string pipeline = "opt_qmv_swiglu_b8_g" + std::to_string(gate.groupSize) +
        "_r" + std::to_string(capacity) + "_n" + std::to_string(v.rn) + "_sn" +
        std::to_string(v.sn) + "_sk" + std::to_string(v.sk) + "_v" + std::to_string(v.vpt);
    const QmvParams params{K, N, count, K, N, row0, gate.weightRowStrideBytes, gate.parameterRowStrideBytes};
    const uint32_t perGroup = v.rn * v.sn;
    graph.add(pipeline, {input, gate.weights->buffer, gate.scales->buffer, gate.biases->buffer,
              up.weights->buffer, up.scales->buffer, up.biases->buffer, output, diagnostics},
              params, {(N + perGroup - 1) / perGroup, 1, 1}, {32 * v.sn * v.sk, 1, 1});
  });
  return true;
}

bool addQmvTall(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                const FlashAffineProjection &p, const metal::MetalBuffer &output,
                uint32_t rows) {
  if (!qmvEnabled() || rows <= kQmvMaximumSplitRows || p.experts != 1 || p.outputSize > 64 ||
      !p.weights || !p.scales || !p.biases)
    return false;
  if (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) return false;
  if (p.groupSize != 64 && p.groupSize != 128) return false;
  if (p.bits == 4 && p.groupSize != 64) return false;
  const uint32_t K = p.inputSize, N = p.outputSize;
  if (!K || !N || K % 16 || K % p.groupSize) return false;
  if (p.weightRowStrideBytes % 4 || p.parameterRowStrideBytes % 2) return false;
  const auto *base = static_cast<const unsigned char *>(p.weights->buffer.contents());
  if (!base || reinterpret_cast<uintptr_t>(base) % 4) return false;
  if (input.sizeBytes() < uint64_t(rows) * K * 2 || output.sizeBytes() < uint64_t(rows) * N * 2)
    return false;
  // Four-row blocks along grid y. Activation rows come from memory, so the
  // variant maximizes parallel loads: long K splits 16 ways for <= 4 outputs;
  // wide outputs use four 4-output groups with a 2-way K split.
  uint32_t sn = 2, sk = 1;
  if (N <= 4) { sn = 1; sk = 16; }
  else if (N > 8) { sn = 4; sk = 2; }
  const std::string pipeline = "opt_qmv_b" + std::to_string(p.bits) + "_g" +
      std::to_string(p.groupSize) + "_r4_n4_sn" + std::to_string(sn) + "_sk" +
      std::to_string(sk) + "_v8";
  const QmvParams params{K, N, rows, K, N, 0, p.weightRowStrideBytes, p.parameterRowStrideBytes};
  const uint32_t perGroup = 4 * sn;
  graph.add(pipeline, {input, p.weights->buffer, p.scales->buffer, p.biases->buffer, output},
            params, {(N + perGroup - 1) / perGroup, (rows + 3) / 4, 1}, {32 * sn * sk, 1, 1});
  return true;
}

namespace {

struct HCParams {
  uint32_t rows, hasInjection, injectionBits, injectionGroup;
  uint64_t downWRow, downPRow, injWRow, injPRow, upWRow, upPRow;
  uint32_t row0, reserved;
};
static_assert(sizeof(HCParams) == 72);

bool hcFormat(const FlashAffineProjection &p, uint32_t K, uint32_t N) {
  if (p.experts != 1 || !p.weights || !p.scales || !p.biases) return false;
  if (p.inputSize != K || p.outputSize != N || p.groupSize != 64) return false;
  if (p.bits != 4 && p.bits != 5 && p.bits != 6 && p.bits != 8) return false;
  if (p.weightRowStrideBytes % 4 || p.parameterRowStrideBytes % 2) return false;
  const auto *base = static_cast<const unsigned char *>(p.weights->buffer.contents());
  return base && reinterpret_cast<uintptr_t>(base) % 4 == 0;
}

uint32_t rowCapacity(uint32_t rows) {
  for (uint32_t candidate : kRowCapacities)
    if (candidate >= rows) return candidate;
  return 0;
}

} // namespace

bool addHCDown(metal::CommandGraph &graph, const metal::MetalBuffer &normalized,
               const FlashAffineProjection &down, const FlashAffineProjection *injection,
               const metal::MetalBuffer &activated, const metal::MetalBuffer &gates,
               uint32_t rows) {
  if (!qmvEnabled() || !rows || rows > kQmvMaximumPrefillRows || !hcFormat(down, 10240, 320)) return false;
  if (injection) {
    if (injection->experts != 1 || injection->inputSize != 10240 || injection->outputSize != 4 ||
        !injection->weights || !injection->scales || !injection->biases ||
        (injection->bits != 4 && injection->bits != 5 && injection->bits != 6 && injection->bits != 8) ||
        (injection->groupSize != 64 && injection->groupSize != 128) ||
        injection->weightRowStrideBytes % 4 ||
        injection->parameterRowStrideBytes % 2 || !gates)
      return false;
  }
  const FlashAffineProjection &i = injection ? *injection : down;
  forEachChunk(rows, [&](uint32_t row0, uint32_t count) {
    const HCParams params{count, injection ? 1u : 0u, injection ? injection->bits : 0u,
        injection ? injection->groupSize : 0u, down.weightRowStrideBytes, down.parameterRowStrideBytes,
        injection ? injection->weightRowStrideBytes : 0u, injection ? injection->parameterRowStrideBytes : 0u,
        0, 0, row0, 0};
    const std::string pipeline = "opt_hc_down_b" + std::to_string(down.bits) + "_g64_r" +
        std::to_string(rowCapacity(count));
    graph.add(pipeline, {normalized, down.weights->buffer, down.scales->buffer, down.biases->buffer,
        i.weights->buffer, i.scales->buffer, i.biases->buffer, activated,
        injection ? gates : activated}, params, {injection ? 161u : 160u, 1, 1}, {512, 1, 1});
  });
  return true;
}

bool addHCUpMix(metal::CommandGraph &graph, const metal::MetalBuffer &normalized,
                const metal::MetalBuffer &activated, const FlashAffineProjection &up,
                const metal::MetalBuffer &mixed, uint32_t rows) {
  if (!qmvEnabled() || !rows || rows > kQmvMaximumPrefillRows || !hcFormat(up, 320, 10240)) return false;
  forEachChunk(rows, [&](uint32_t row0, uint32_t count) {
    const HCParams params{count, 0, 0, 0, 0, 0, 0, 0, up.weightRowStrideBytes,
        up.parameterRowStrideBytes, row0, 0};
    const std::string pipeline = "opt_hc_up_mix_b" + std::to_string(up.bits) + "_g64_r" +
        std::to_string(rowCapacity(count));
    graph.add(pipeline, {normalized, activated, up.weights->buffer, up.scales->buffer,
        up.biases->buffer, mixed}, params, {640, 1, 1}, {128, 1, 1});
  });
  return true;
}

bool addDenseBF16Rows(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                      const FlashTensor &weights, const metal::MetalBuffer &output,
                      uint32_t rows) {
  if (!qmvEnabled() || !rows || rows > kQmvMaximumSplitRows || weights.dtype != FlashDType::BF16 ||
      weights.shape.size() != 2)
    return false;
  const uint64_t n = weights.shape[0], k = weights.shape[1];
  if (!n || !k || n > UINT32_MAX || k % 8 || k > UINT32_MAX) return false;
  const auto *base = static_cast<const unsigned char *>(weights.buffer.contents());
  if (!base || reinterpret_cast<uintptr_t>(base) % 8) return false;
  const uint32_t N = uint32_t(n), K = uint32_t(k);
  if (input.sizeBytes() < uint64_t(rows) * K * 2 || output.sizeBytes() < uint64_t(rows) * N * 2)
    return false;
  forEachChunk(rows, [&](uint32_t row0, uint32_t count) {
    const QmvParams params{K, N, count, K, N, row0, 0, 0};
    graph.add("opt_dense_bf16_r" + std::to_string(rowCapacity(count)),
              {input, weights.buffer, output}, params, {(N + 3) / 4, 1, 1}, {128, 1, 1});
  });
  return true;
}

namespace {
int moeMode() noexcept {
  static const int mode = [] {
    const char *value = std::getenv("SPLASH_OPT_MOE");
    if (!qmvEnabled() || !value) return 0;
    if (std::strcmp(value, "1") == 0) return 1;
    if (std::strcmp(value, "2") == 0) return 2;
    return 0;
  }();
  return mode;
}
} // namespace

bool moeEnabled() noexcept { return moeMode() != 0; }
bool fusedSwiGLUEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_FUSED_SWIGLU");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool mppqEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_MPPQ");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool statePoolEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_STATE_POOL");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool gdnKernelEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_GDN_KERNEL");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool qsaKernelEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_QSA_KERNEL");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool draftChainEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_DRAFT_CHAIN");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool prepareVerifyEnabled() noexcept {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_PREPARE_VERIFY");
    return !value || std::strcmp(value, "0") != 0;
  }();
  return enabled;
}
bool moeReplacesInt8() noexcept { return moeMode() == 1; }

namespace {
struct MoEParams {
  uint32_t K, N, rows, selections, experts, reserved;
  uint64_t wRow, wExpert, pRow, pExpert;
};
static_assert(sizeof(MoEParams) == 56);

bool expertFormat(const FlashAffineProjection &p, uint32_t K, uint32_t N) {
  if (!p.weights || !p.scales || !p.biases || p.experts < 2) return false;
  if (p.inputSize != K || p.outputSize != N || p.bits != 4 || p.groupSize != 64) return false;
  if (p.weightRowStrideBytes % 4 || p.weightExpertStrideBytes % 4 ||
      p.parameterRowStrideBytes % 2 || p.parameterExpertStrideBytes % 2)
    return false;
  const auto *base = static_cast<const unsigned char *>(p.weights->buffer.contents());
  return base && reinterpret_cast<uintptr_t>(base) % 4 == 0;
}
} // namespace

bool addMoEExperts(metal::CommandGraph &graph, const metal::MetalBuffer &input,
                   const FlashAffineProjection &gate, const FlashAffineProjection &up,
                   const FlashAffineProjection &down, const metal::MetalBuffer &expertIDs,
                   const metal::MetalBuffer &intermediate, const metal::MetalBuffer &expertDown,
                   uint32_t rows, uint32_t selections) {
  if (!moeEnabled() || !rows || rows > kQmvMaximumRows || !selections || selections > 16) return false;
  if (!expertFormat(gate, 2560, 640) || !expertFormat(up, 2560, 640) || !expertFormat(down, 640, 2560))
    return false;
  if (gate.experts != up.experts || gate.experts != down.experts ||
      gate.weightRowStrideBytes != up.weightRowStrideBytes ||
      gate.weightExpertStrideBytes != up.weightExpertStrideBytes ||
      gate.parameterRowStrideBytes != up.parameterRowStrideBytes ||
      gate.parameterExpertStrideBytes != up.parameterExpertStrideBytes)
    return false;
  const uint32_t routes = rows * selections;
  if (expertIDs.sizeBytes() < uint64_t(routes) * 8 || intermediate.sizeBytes() < uint64_t(routes) * 640 * 2 ||
      expertDown.sizeBytes() < uint64_t(routes) * 2560 * 2)
    return false;
  const MoEParams gu{2560, 640, rows, selections, gate.experts, 0, gate.weightRowStrideBytes,
      gate.weightExpertStrideBytes, gate.parameterRowStrideBytes, gate.parameterExpertStrideBytes};
  graph.add("opt_moe_gate_up_b4_g64", {input, gate.weights->buffer, gate.scales->buffer,
      gate.biases->buffer, up.weights->buffer, up.scales->buffer, up.biases->buffer, expertIDs,
      intermediate}, gu, {640 / 8, routes, 1}, {64, 1, 1});
  const MoEParams dn{640, 2560, rows, selections, down.experts, 0, down.weightRowStrideBytes,
      down.weightExpertStrideBytes, down.parameterRowStrideBytes, down.parameterExpertStrideBytes};
  graph.add("opt_moe_down_b4_g64", {intermediate, down.weights->buffer, down.scales->buffer,
      down.biases->buffer, down.weights->buffer, down.scales->buffer, down.biases->buffer, expertIDs,
      expertDown}, dn, {2560 / 8, routes, 1}, {64, 1, 1});
  return true;
}

bool addRoute(metal::CommandGraph &graph, const metal::MetalBuffer &logits,
              const metal::MetalBuffer &expertIDs, const metal::MetalBuffer &routeWeights,
              uint32_t rows, uint32_t experts, uint32_t selections, bool normalizeTopK) {
  if (!qmvEnabled() || !rows || experts != 512 || !selections || selections > 10) return false;
  if (logits.sizeBytes() < uint64_t(rows) * experts * 2 ||
      expertIDs.sizeBytes() < uint64_t(rows) * selections * 8 ||
      routeWeights.sizeBytes() < uint64_t(rows) * selections * 2)
    return false;
  const struct { uint32_t rows, experts, selections, normalize; } params{
      rows, experts, selections, normalizeTopK ? 1u : 0u};
  graph.add("opt_moe_route", {logits, expertIDs, routeWeights}, params, {rows, 1, 1}, {32, 1, 1});
  return true;
}

} // namespace splash::flash::opt

namespace splash::flash::opt {
namespace {
bool timersEnabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SPLASH_OPT_TIMERS");
    return value && std::strcmp(value, "1") == 0;
  }();
  return enabled;
}
struct TimerSlot { const char *label = nullptr; uint64_t count = 0, nanoseconds = 0; };
std::mutex &timerMutex() { static std::mutex mutex; return mutex; }
std::vector<TimerSlot> &timerSlots() { static std::vector<TimerSlot> slots; return slots; }
uint64_t nowNanoseconds() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}
} // namespace

ScopedTimer::ScopedTimer(const char *label) noexcept : label_(label) {
  if (timersEnabled()) start_ = nowNanoseconds();
}
ScopedTimer::~ScopedTimer() {
  if (!start_) return;
  const uint64_t elapsed = nowNanoseconds() - start_;
  std::lock_guard lock(timerMutex());
  auto &slots = timerSlots();
  auto slot = std::find_if(slots.begin(), slots.end(),
      [&](const TimerSlot &candidate) { return std::strcmp(candidate.label, label_) == 0; });
  if (slot == slots.end()) { slots.push_back({label_, 0, 0}); slot = slots.end() - 1; }
  slot->count += 1; slot->nanoseconds += elapsed;
  static const uint64_t every = [] {
    const char *value = std::getenv("SPLASH_OPT_TIMERS_EVERY");
    return value && *value ? std::max<uint64_t>(1, std::strtoull(value, nullptr, 10)) : uint64_t{256};
  }();
  if (slot->count % every == 0)
    std::fprintf(stderr, "opt-timer %s count=%llu mean_us=%.1f\n", slot->label,
        static_cast<unsigned long long>(slot->count),
        double(slot->nanoseconds) / double(slot->count) / 1000.0);
}
} // namespace splash::flash::opt
