#pragma once
#include "metal/abi/FlashAffine.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef RAW_LARGE_R4_POLICY_CPU_ONLY
#include "source_identity.hpp"
#include "flash/FlashAffine.hpp"
#include "flash/FlashFloatDenseCache.hpp"
#include "dev/benchmarks/raw_q4_verify_worker_sep22/policy.hpp"
#endif

namespace splash::flash::raw_large_r4_guard_pair_sep22 {
inline constexpr const char *kFlag = "SPLASH_FLASH_RAW_LARGE_R4_GUARD_PAIR_SEP22";
inline constexpr const char *kModel = "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
inline constexpr const char *kManifest = "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0";
inline constexpr const char *kScope = "singleton main VerifyR4; cache-member selective NULL RAW; constructor-authenticated GDN qkv/z/out, QSA q/o and first PLE key; legacy GDN26 Q4 takes precedence";
inline constexpr uint64_t kAlignment = 16384;
inline constexpr uint32_t kPotentialRoles = 133, kRawRoles = 113, kLegacyRoles = 26, kNewRoles = 87;
inline constexpr std::array<const char *, 9> kDependencies{
    "SPLASH_FLASH_RAW_Q4_ROWPAIR_VERIFY_SEP22",
    "SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22",
    "SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22",
    "SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22",
    "SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22",
    "SPLASH_FLASH_ALLROWS_FULL512_TARGET", "SPLASH_FLASH_QMV_F32",
    "SPLASH_FLASH_FLOAT_DENSE_CACHE", "SPLASH_FLASH_FLOAT_DENSE_SELECTIVE"};
inline void require(bool value, const char *message) {
  if (!value) throw std::invalid_argument(message);
}
enum class FlagState : uint8_t { Missing, Disabled, Enabled };
inline FlagState parse(const char *value) {
  if (!value) return FlagState::Missing;
  if (std::string_view(value) == "0") return FlagState::Disabled;
  if (std::string_view(value) == "1") return FlagState::Enabled;
  throw std::invalid_argument(std::string(kFlag) + " must be exactly 0 or 1");
}
class FrozenFlag final {
  FlagState state_;
public:
  explicit FrozenFlag(const char *value) : state_(parse(value)) {}
  bool check(const char *value) const {
    if (parse(value) != state_) throw std::logic_error("raw large R4 flag changed after freezing");
    return state_ == FlagState::Enabled;
  }
};
inline bool requested() {
  static const FrozenFlag frozen(std::getenv(kFlag));
  return frozen.check(std::getenv(kFlag));
}
template <class Getter> inline void dependencies(bool enabled, Getter get) {
  if (!enabled) return;
  for (const char *name : kDependencies) {
    const char *value = get(name);
    if (!value || std::string_view(value) != "1")
      throw std::invalid_argument(std::string("raw large R4 requires ") + name + "=1");
  }
  const char *phase = get("SPLASH_FLASH_PREFILL_I8_DECODE_Q4_SEP21");
  require(!phase || std::string_view(phase) == "0", "raw large R4 excludes phase-Q4 target");
}
inline void validateDependencies() { dependencies(requested(), [](const char *name) { return std::getenv(name); }); }
inline bool context(uint32_t rows, bool verification, bool singletonMain) noexcept {
  return rows == 4 && verification && singletonMain;
}
enum class Role : uint8_t { None, GDNQKV, GDNZ, GDNOut, QSAQ, QSAOut, PLEKey };
inline Role role(std::string_view prefix, uint32_t firstPLE = 1) noexcept {
  constexpr std::string_view leading = "language_model.model.layers.";
  if (!prefix.starts_with(leading)) return Role::None;
  const auto tail = prefix.substr(leading.size()); const auto dot = tail.find('.');
  if (dot == std::string_view::npos || !dot || dot > 2 || (dot == 2 && tail[0] == '0')) return Role::None;
  uint32_t layer = 0;
  for (char c : tail.substr(0, dot)) {
    if (c < '0' || c > '9') return Role::None;
    layer = layer * 10 + uint32_t(c - '0');
  }
  if (layer >= 48) return Role::None;
  const auto name = tail.substr(dot + 1);
  if (layer == firstPLE && name == "ple.key_proj") return Role::PLEKey;
  if (layer % 4 != 3) {
    if (name == "linear_attn.in_proj_qkv") return Role::GDNQKV;
    if (name == "linear_attn.in_proj_z") return Role::GDNZ;
    if (name == "linear_attn.out_proj") return Role::GDNOut;
  } else {
    if (name == "self_attn.q_proj") return Role::QSAQ;
    if (name == "self_attn.o_proj") return Role::QSAOut;
  }
  return Role::None;
}
inline bool roleDimensions(Role r, uint32_t k, uint32_t n) noexcept {
  switch (r) {
  case Role::GDNQKV: case Role::PLEKey: return k == 2560 && n == 10240;
  case Role::GDNZ: return k == 2560 && n == 6144;
  case Role::GDNOut: case Role::QSAOut: return k == 6144 && n == 2560;
  case Role::QSAQ: return k == 2560 && n == 12288;
  default: return false;
  }
}
struct Signature { uint32_t bits, group, k, n, expected, expectedNew; };
// Root-provided small scalar inventory; no model/profile payload is consumed.
inline constexpr std::array<Signature, 7> kSignatures{{
    {5, 128, 6144, 2560, 36, 36}, {4, 64, 2560, 10240, 27, 1},
    {5, 128, 2560, 6144, 26, 26}, {6, 64, 2560, 6144, 10, 10},
    {4, 64, 2560, 12288, 5, 5}, {5, 64, 2560, 10240, 4, 4},
    {4, 64, 6144, 2560, 5, 5}}};
inline size_t signature(const FlashAffineParams &p) noexcept {
  for (size_t i = 0; i < kSignatures.size(); ++i) {
    const auto &s = kSignatures[i];
    if (p.bits == s.bits && p.group_size == s.group && p.input_size == s.k && p.output_size == s.n) return i;
  }
  return kSignatures.size();
}
inline uint64_t extent(uint32_t n, uint64_t stride, uint64_t rowBytes) {
  require(n > 1 && stride <= (UINT64_MAX - rowBytes) / (n - 1), "raw large R4 row extent overflow");
  return uint64_t(n - 1) * stride + rowBytes;
}
inline bool parametersValid(const FlashAffineParams &p) noexcept {
  if (signature(p) == kSignatures.size() || p.rows != 4 || p.selections != 1 || p.experts != 1 || p.flags) return false;
  const uint64_t code = uint64_t(p.input_size) * p.bits / 8;
  const uint64_t parameter = uint64_t(p.input_size / p.group_size) * 2;
  return p.output_size >= 2048 && p.weight_row_stride_bytes >= code &&
      p.parameter_row_stride_bytes >= parameter && !(p.parameter_row_stride_bytes % 2) &&
      !(p.parameter_expert_stride_bytes % 2) &&
      p.weight_row_stride_bytes <= (UINT64_MAX - code) / (p.output_size - 1) &&
      p.parameter_row_stride_bytes <= (UINT64_MAX - parameter) / (p.output_size - 1);
}
struct Span { uintptr_t address = 0; uint64_t bytes = 0; };
inline bool validSpan(Span s, uint64_t minimum) noexcept {
  return s.address && s.bytes >= minimum && s.bytes <= UINTPTR_MAX - s.address;
}
inline bool overlap(Span a, Span b) noexcept {
  if (!validSpan(a, 1) || !validSpan(b, 1)) return true;
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
enum class DType : uint8_t { U32, BF16, F32, Other };
struct TensorMetadata {
  DType dtype = DType::Other; uint32_t rank = 0;
  std::array<uint64_t, 2> shape{}; uint64_t logicalBytes = 0; Span view;
};
struct Observation {
  std::string prefix; uintptr_t projectionIdentity = 0; FlashAffineParams params{};
  std::array<TensorMetadata, 3> original;
  bool cacheMember = false, selectedF32Tile = false, legacyQ4 = false;
  TensorMetadata cached;
};
inline void tensor(const TensorMetadata &t, DType dtype, uint64_t n, uint64_t k, uint64_t bytes) {
  require(t.dtype == dtype && t.rank == 2 && t.shape == std::array<uint64_t, 2>{n, k} &&
      t.logicalBytes == bytes && t.view.bytes == bytes && validSpan(t.view, bytes) &&
      !(t.view.address % kAlignment), "raw large R4 tensor dtype/shape/extent/effective view offset differs");
}
inline void observation(const Observation &o) {
  const auto &p = o.params; const auto r = role(o.prefix);
  require(o.projectionIdentity && roleDimensions(r, p.input_size, p.output_size) &&
      p.rows == 4 && p.experts == 1 && p.selections == 1 && p.flags == 0 &&
      (p.bits == 4 || p.bits == 5 || p.bits == 6 || p.bits == 8) &&
      (p.group_size == 32 || p.group_size == 64 || p.group_size == 128) &&
      !(p.input_size % p.group_size), "raw large R4 canonical source role/projection geometry differs");
  const uint64_t codeRow = uint64_t(p.input_size) * p.bits / 8, parameterRow = uint64_t(p.input_size / p.group_size) * 2;
  require(p.weight_row_stride_bytes == codeRow && p.parameter_row_stride_bytes == parameterRow &&
      p.weight_expert_stride_bytes == uint64_t(p.output_size) * codeRow &&
      p.parameter_expert_stride_bytes == uint64_t(p.output_size) * parameterRow,
      "raw large R4 original compact source strides differ");
  tensor(o.original[0], DType::U32, p.output_size, uint64_t(p.input_size) * p.bits / 32, uint64_t(p.output_size) * codeRow);
  for (size_t i : {1u, 2u}) tensor(o.original[i], DType::BF16, p.output_size, p.input_size / p.group_size, uint64_t(p.output_size) * parameterRow);
  for (size_t i = 0; i < 3; ++i) for (size_t j = i + 1; j < 3; ++j)
    require(!overlap(o.original[i].view, o.original[j].view), "raw large R4 source planes alias");
  require(o.cacheMember, "raw large R4 original F32 selector membership missing");
  tensor(o.cached, DType::F32, p.output_size, p.input_size, uint64_t(p.output_size) * p.input_size * 4);
  const bool legacy = r == Role::GDNQKV && p.bits == 4 && p.group_size == 64;
  require(o.legacyQ4 == legacy, "raw large R4 legacy precedence classification differs");
  if (!o.selectedF32Tile) require(parametersValid(p), "raw large R4 NULL-policy source is outside seven qualified signatures");
}
struct Entry {
  uintptr_t projectionIdentity; FlashAffineParams params;
  std::array<Span, 3> original; size_t shape;
};
struct Inventory {
  uint32_t potential = 0, raw = 0, legacy = 0, f32 = 0;
  std::array<uint32_t, 7> shapes{}, newShapes{}; std::vector<Entry> entries;
};
inline Inventory inventory(std::string_view source, std::string_view manifest, const std::vector<Observation> &roles) {
  require(source == kModel && manifest == kManifest, "raw large R4 original model/layout identity differs");
  Inventory out; std::set<std::string_view> names; std::set<uintptr_t> identities;
  for (const auto &o : roles) {
    observation(o);
    require(names.insert(o.prefix).second && identities.insert(o.projectionIdentity).second, "raw large R4 inventory duplicate role/identity");
    ++out.potential;
    if (o.selectedF32Tile) { ++out.f32; continue; }
    const size_t shape = signature(o.params); ++out.raw; ++out.shapes[shape];
    if (o.legacyQ4) { ++out.legacy; continue; }
    ++out.newShapes[shape];
    out.entries.push_back({o.projectionIdentity, o.params,
        {o.original[0].view, o.original[1].view, o.original[2].view}, shape});
  }
  require(out.potential == kPotentialRoles && out.raw == kRawRoles && out.legacy == kLegacyRoles &&
      out.f32 == 20 && out.entries.size() == kNewRoles, "raw large R4 inventory requires 133 potential/113 RAW/26 legacy/87 new/20 F32 roles");
  for (size_t i = 0; i < kSignatures.size(); ++i)
    require(out.shapes[i] == kSignatures[i].expected && out.newShapes[i] == kSignatures[i].expectedNew,
        "raw large R4 exact seven-signature inventory differs");
  std::sort(out.entries.begin(), out.entries.end(), [](const Entry &a, const Entry &b) { return a.projectionIdentity < b.projectionIdentity; });
  return out;
}
inline const Entry *select(const Inventory &roles, uintptr_t identity, uint32_t rows, bool verification, bool singletonMain) noexcept {
  if (!context(rows, verification, singletonMain)) return nullptr;
  const auto found = std::lower_bound(roles.entries.begin(), roles.entries.end(), identity,
      [](const Entry &e, uintptr_t key) { return e.projectionIdentity < key; });
  return found != roles.entries.end() && found->projectionIdentity == identity ? &*found : nullptr;
}
struct Descriptor {
  std::string_view pipeline; std::array<uint64_t, 3> groups{}, threads{};
  size_t buffers = 0, payloads = 0; uint32_t paramsIndex = 0; uint64_t paramsBytes = 0;
  bool paramsPresent = false; std::array<uint32_t, 7> indices{}; FlashAffineParams params{};
};
inline std::string originalPipeline(const FlashAffineParams &p) {
  return "flash_affine_mlx_qmv_f32xsum_v1_q" + std::to_string(p.bits) + "_g" + std::to_string(p.group_size);
}
inline std::string candidatePipeline(const FlashAffineParams &p) {
  return "raw_large_R4_guard_pair_sep22_timed_q" + std::to_string(p.bits) + "_g" + std::to_string(p.group_size);
}
inline void descriptor(const Descriptor &d, const Entry &qualified, const std::array<Span, 7> &spans) {
  const auto &p = d.params;
  require(parametersValid(p) && !std::memcmp(&p, &qualified.params, sizeof(p)) &&
      d.pipeline == originalPipeline(p) && d.groups == std::array<uint64_t, 3>{p.output_size / 8, 4, 1} &&
      d.threads == std::array<uint64_t, 3>{64, 1, 1} && d.buffers == 7 && d.payloads == 1 &&
      d.paramsIndex == 7 && d.paramsBytes == sizeof(p) && d.paramsPresent,
      "raw large R4 original descriptor/immutable Params64 differs");
  const std::array<uint64_t, 7> minimum{uint64_t(4) * p.input_size * 2,
      extent(p.output_size, p.weight_row_stride_bytes, uint64_t(p.input_size) * p.bits / 8),
      extent(p.output_size, p.parameter_row_stride_bytes, uint64_t(p.input_size / p.group_size) * 2),
      extent(p.output_size, p.parameter_row_stride_bytes, uint64_t(p.input_size / p.group_size) * 2),
      1, uint64_t(4) * p.output_size * 2, 4};
  for (size_t i = 0; i < spans.size(); ++i) require(d.indices[i] == i && validSpan(spans[i], minimum[i]), "raw large R4 short/misindexed original Shared binding");
  require(spans[0].address == spans[4].address && spans[0].bytes == spans[4].bytes,
      "raw large R4 flags0 dummy must preserve original input view");
  for (size_t i = 0; i < 3; ++i)
    require(spans[i + 1].address == qualified.original[i].address && spans[i + 1].bytes == qualified.original[i].bytes,
        "raw large R4 immutable original source view changed");
  require(!(spans[0].address % 2) && !(spans[2].address % 2) && !(spans[3].address % 2) &&
      !(spans[5].address % 2) && !(spans[6].address % 4), "raw large R4 typed view alignment differs");
  for (size_t w : {5u, 6u}) for (size_t r = 0; r < 5; ++r)
    require(!overlap(spans[w], spans[r]), "raw large R4 writable binding aliases readonly input");
  require(!overlap(spans[5], spans[6]), "raw large R4 output/diagnostics alias");
}
inline std::atomic<uint64_t> graphCalls{0}, graphRows{0}, excludedContextGraphCalls{0};
inline std::atomic<uint32_t> authenticatedRawRoles{0}, authenticatedNewRoles{0};

#ifndef RAW_LARGE_R4_POLICY_CPU_ONLY
inline TensorMetadata metadata(const FlashTensor &t) {
  DType dtype = DType::Other;
  switch (t.dtype) { case FlashDType::U32: dtype = DType::U32; break;
  case FlashDType::BF16: dtype = DType::BF16; break; case FlashDType::F32: dtype = DType::F32; break;
  default: break; }
  TensorMetadata m{dtype, uint32_t(t.shape.size()), {}, t.logicalBytes,
      {reinterpret_cast<uintptr_t>(t.buffer.contents()), t.buffer.sizeBytes()}};
  if (t.shape.size() == 2) m.shape = {t.shape[0], t.shape[1]};
  require(t.buffer && t.buffer.storage() == metal::BufferStorage::Shared, "raw large R4 tensor requires original Shared view");
  return m; // Addresses/lengths only: never dereference tensor contents.
}
class QualifiedRoles final {
  Inventory roles_;
public:
  QualifiedRoles(const FlashWeights &weights, const FlashFloatDenseCache &cache) {
    validateDependencies(); require(requested(), "raw large R4 inventory requires frozen flag1");
    const auto &d = weights.descriptor();
    require(d.layers == 48 && d.layerKinds.size() == 48 && d.pleLayerIndices == std::vector<uint32_t>{1} &&
        weights.normConvention() == NormConvention::OnePlusWeight, "raw large R4 original descriptor/PLE/norm differs");
    std::vector<Observation> observations; observations.reserve(kPotentialRoles);
    const auto append = [&](const std::string &prefix) {
      require(weights.contains(prefix + ".weight"), "raw large R4 expected original main role missing");
      const auto &p = weights.projection(prefix);
      require(p.weights && p.scales && p.biases, "raw large R4 original source tensors missing");
      Observation o; o.prefix = prefix; o.projectionIdentity = reinterpret_cast<uintptr_t>(&p);
      o.params = {4, 1, p.inputSize, p.outputSize, p.experts, p.bits, p.groupSize, 0,
          p.weightRowStrideBytes, p.weightExpertStrideBytes, p.parameterRowStrideBytes, p.parameterExpertStrideBytes};
      o.original = {metadata(*p.weights), metadata(*p.scales), metadata(*p.biases)};
      o.cacheMember = cache.contains(prefix);
      if (o.cacheMember) o.cached = metadata(cache.tensor(prefix));
      o.selectedF32Tile = bool(flashFloatDenseSmallRowsPolicy(prefix, 4, p.outputSize, p.inputSize, p.bits, p.groupSize));
      o.legacyQ4 = raw_q4_verify_sep22::selected(prefix, 4, true, true, p);
      observations.push_back(std::move(o));
    };
    for (uint32_t layer = 0; layer < 48; ++layer) {
      require(d.layerKinds[layer] == (layer % 4 == 3 ? FlashLayerKind::SparseAttention : FlashLayerKind::GatedDeltaNet),
          "raw large R4 original GDN/QSA layer pattern differs");
      const auto prefix = "language_model.model.layers." + std::to_string(layer) + '.';
      if (layer % 4 != 3) for (const char *name : {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"}) append(prefix + name);
      else for (const char *name : {"self_attn.q_proj", "self_attn.o_proj"}) append(prefix + name);
    }
    append("language_model.model.layers.1.ple.key_proj");
    roles_ = inventory(weights.sourceIdentity(), weights.manifestFingerprint(), observations);
    authenticatedRawRoles.store(roles_.raw, std::memory_order_relaxed);
    authenticatedNewRoles.store(uint32_t(roles_.entries.size()), std::memory_order_relaxed);
  }
  bool selected(const FlashAffineProjection &p, uint32_t rows, bool verification, bool singletonMain) const noexcept {
    return select(roles_, reinterpret_cast<uintptr_t>(&p), rows, verification, singletonMain) != nullptr;
  }
  void add(metal::CommandGraph &graph, metal::MetalBuffer input, const FlashAffineProjection &p,
      metal::MetalBuffer output, metal::MetalBuffer diagnostics, uint32_t rows, bool verification, bool singletonMain) const {
    require(requested(), "raw large R4 add requires immutable flag1");
    const auto *entry = select(roles_, reinterpret_cast<uintptr_t>(&p), rows, verification, singletonMain);
    if (!entry) { excludedContextGraphCalls.fetch_add(1, std::memory_order_relaxed); throw std::invalid_argument("raw large R4 unqualified role/context"); }
    // Keep the original callable's tensor/extent/route validation first. Only
    // a checked descriptor is substituted; target graph stays untouched on failure.
    metal::CommandGraph original; addAffine(original, input, p, output, diagnostics, rows);
    require(original.dispatches().size() == 1, "raw large R4 original descriptor count differs");
    const auto &d = original.dispatches()[0]; Descriptor m;
    m.pipeline = d.pipelineName; m.groups = {d.threadgroups.x, d.threadgroups.y, d.threadgroups.z};
    m.threads = {d.threadsPerThreadgroup.x, d.threadsPerThreadgroup.y, d.threadsPerThreadgroup.z};
    m.buffers = d.buffers.size(); m.payloads = d.bytes.size();
    if (d.bytes.size() == 1) { m.paramsIndex = d.bytes[0].index; m.paramsBytes = d.bytes[0].sizeBytes; m.paramsPresent = d.bytes[0].data != nullptr;
      if (m.paramsBytes == sizeof(m.params) && m.paramsPresent) std::memcpy(&m.params, d.bytes[0].data, sizeof(m.params)); }
    std::array<Span, 7> spans{}; std::vector<metal::MetalBuffer> buffers; buffers.reserve(7);
    require(d.buffers.size() == 7, "raw large R4 original buffer count differs");
    for (size_t i = 0; i < 7; ++i) {
      const auto &binding = d.buffers[i]; m.indices[i] = binding.index;
      require(binding.buffer && binding.buffer.storage() == metal::BufferStorage::Shared, "raw large R4 requires original Shared bindings");
      spans[i] = {reinterpret_cast<uintptr_t>(binding.buffer.contents()), binding.buffer.sizeBytes()};
      buffers.push_back(binding.buffer);
    }
    require(d.buffers[4].buffer.sameView(d.buffers[0].buffer) && d.buffers[1].buffer.sameView(p.weights->buffer) &&
        d.buffers[2].buffer.sameView(p.scales->buffer) && d.buffers[3].buffer.sameView(p.biases->buffer),
        "raw large R4 original binding identities/view offsets differ");
    descriptor(m, *entry, spans);
    graph.add(candidatePipeline(m.params), std::move(buffers), m.params, {m.params.output_size / 8, 2, 1}, {64, 1, 1});
    graphCalls.fetch_add(1, std::memory_order_relaxed); graphRows.fetch_add(4, std::memory_order_relaxed);
  }
};
inline std::string marker() {
  return requested() ? std::string(";private-raw-large-R4-guard-pair-87-main-roles-sourceSha256=") + kSourceIdentitySha256 : std::string{};
}
#endif
static_assert(sizeof(FlashAffineParams) == 64);
static_assert(alignof(FlashAffineParams) == 8);
} // namespace splash::flash::raw_large_r4_guard_pair_sep22
