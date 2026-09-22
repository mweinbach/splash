// CPU-only fixed geometry/policy proof. No MetalBackend, FlashWeights, Cache
// or Workspace object is constructed; no model file or coefficient is read.
#include "worker_cache.hpp"
#include "abi.hpp"

#include <iostream>
#include <set>
#include <stdexcept>

namespace w8 = splash::flash::dense_w8a8_sep21;
using splash::flash::FlashDenseTraversal;

namespace {
uint64_t checks = 0;
void require(bool passed, const char *what) {
  ++checks;
  if (!passed) throw std::runtime_error(what);
}
void test() {
  const auto names = w8::selectedPrefixes();
  const std::set<std::string> distinct(names.begin(), names.end());
  require(names.size() == 84 && distinct.size() == 84,
      "Dense W8A8 selection must contain exactly84 unique canonical prefixes");
  uint64_t planned = 0, logical = 0;
  std::array<uint32_t, 3> roleCounts{};
  for (uint32_t layer = 0; layer < 48; ++layer) {
    const auto leading = "language_model.model.layers." + std::to_string(layer) + ".";
    const bool gdn = layer % 4 != 3;
    for (const auto &role : {std::string("linear_attn.in_proj_qkv"),
                           std::string("linear_attn.in_proj_z"),
                           std::string("self_attn.q_proj")}) {
      const auto prefix = leading + role;
      const bool expected = gdn ? role != "self_attn.q_proj" : role == "self_attn.q_proj";
      const auto shape = w8::geometry(prefix);
      require(bool(shape) == expected, "Dense W8A8 layer/role policy differs");
      require(distinct.contains(prefix) == expected, "Dense W8A8 prefix census differs");
      if (!expected) continue;
      const uint32_t output = role == "linear_attn.in_proj_qkv" ? 10240
          : role == "linear_attn.in_proj_z" ? 6144 : 12288;
      ++roleCounts[role == "linear_attn.in_proj_qkv" ? 0 : role == "linear_attn.in_proj_z" ? 1 : 2];
      require(shape.k == 2560 && shape.n == output, "Dense W8A8 canonical geometry differs");
      const std::array<uint64_t, 2> sourceShape{output, 2560};
      const uint64_t sourceBytes = uint64_t{output} * 2560 * 2;
      using splash::flash::FlashDType;
      using splash::metal::BufferStorage;
      require(w8::sourceMetadataMatches(shape, FlashDType::BF16, sourceShape,
          sourceBytes, sourceBytes, BufferStorage::Shared),
          "Dense W8A8 exact cached BF16 source metadata was rejected");
      for (const auto dtype : {FlashDType::U32, FlashDType::I64, FlashDType::F32})
        require(!w8::sourceMetadataMatches(shape, dtype, sourceShape,
            sourceBytes, sourceBytes, BufferStorage::Shared),
            "Dense W8A8 nonBF16 cached source metadata was accepted");
      for (const auto extent : {sourceBytes - 2, sourceBytes + 2, sourceBytes + 16384}) {
        require(!w8::sourceMetadataMatches(shape, FlashDType::BF16, sourceShape,
            extent, sourceBytes, BufferStorage::Shared),
            "Dense W8A8 inexact logical source extent was accepted");
        require(!w8::sourceMetadataMatches(shape, FlashDType::BF16, sourceShape,
            sourceBytes, extent, BufferStorage::Shared),
            "Dense W8A8 inexact backing source extent was accepted");
      }
      const std::array<uint64_t, 2> transposed{2560, output}, wrong{output + 64, 2560};
      const std::array<uint64_t, 3> rank3{1, output, 2560};
      for (const auto metadata : {std::span<const uint64_t>{transposed},
          std::span<const uint64_t>{wrong}, std::span<const uint64_t>{rank3},
          std::span<const uint64_t>{}})
        require(!w8::sourceMetadataMatches(shape, FlashDType::BF16, metadata,
            sourceBytes, sourceBytes, BufferStorage::Shared),
            "Dense W8A8 noncanonical cached source shape was accepted");
      require(!w8::sourceMetadataMatches(shape, FlashDType::BF16, sourceShape,
          sourceBytes, sourceBytes, BufferStorage::Private),
          "Dense W8A8 private cached source metadata was accepted");
      const auto plan = w8::projectionPlan(prefix, 2048, output, 2560, false);
      require(bool(plan) && plan.tileRows == 128 && plan.tileOutputs == 64 && plan.simdGroups == 4,
          "Dense W8A8 eligible SG4 tile plan differs");
      require(plan.traversal == (gdn ? FlashDenseTraversal::Swizzle4 : FlashDenseTraversal::Swizzle8),
          "Dense W8A8 measured role traversal differs");
      require(!w8::projectionPlan(prefix, 2048, output, 2560, true),
          "Dense W8A8 verification route was accepted");
      for (uint32_t rows : {0u, 1u, 16u, 1024u, 2047u, 2049u, 4096u, 8192u, UINT32_MAX})
        require(!w8::projectionPlan(prefix, rows, output, 2560, false),
            "Dense W8A8 non2048-row route was accepted");
      for (uint32_t groups : {0u, 1u, 2u, 8u, UINT32_MAX})
        require(!w8::projectionPlan(prefix, 2048, output, 2560, false, groups),
            "Dense W8A8 unqualified SIMD group choice was accepted");
      require(!w8::projectionPlan(prefix, 2048, output + 64, 2560, false) &&
              !w8::projectionPlan(prefix, 2048, output, 6144, false),
          "Dense W8A8 noncanonical geometry was accepted");
      const uint64_t codeBytes = uint64_t{output} * 2560;
      const uint64_t scaleBytes = uint64_t{output} * 4;
      logical += codeBytes + scaleBytes;
      // Independent integer division accounting, separate from roundedBytes.
      planned += ((codeBytes + 16383) / 16384) * 16384;
      planned += ((scaleBytes + 16383) / 16384) * 16384;
    }
    for (const char *excluded : {"linear_attn.out_proj", "self_attn.o_proj", "self_attn.k_proj",
        "self_attn.v_proj", "self_attn.indexer.index_qk_proj", "mlp.gate", "mlp.shared_expert.up_proj",
        "attn_hyper_connection.input_mix_weight_down", "ple.key_proj"})
      require(!w8::geometry(leading + excluded), "Dense W8A8 excluded role was accepted");
  }
  require(roleCounts == std::array<uint32_t, 3>{36, 36, 12}, "Dense W8A8 role counts differ");
  require(w8::kImmutableBufferCount == 168 && planned == w8::Cache::plannedBytes(),
      "Dense W8A8 separately rounded immutable backing census differs");
  require(planned == 1890975744ULL && logical == 1890385920ULL,
      "Dense W8A8 fixed byte-accounting golden differs");
  for (const char *malformed : {"", "language_model.lm_head", "language_model.mtp.layers.0.linear_attn.in_proj_qkv",
      "language_model.model.layers.00.linear_attn.in_proj_qkv", "language_model.model.layers.01.linear_attn.in_proj_z",
      "language_model.model.layers.-1.linear_attn.in_proj_qkv", "language_model.model.layers.+1.linear_attn.in_proj_qkv",
      "language_model.model.layers.1x.linear_attn.in_proj_qkv", "language_model.model.layers.48.linear_attn.in_proj_qkv",
      "language_model.model.layers.000.linear_attn.in_proj_z", "language_model.model.layers.0.linear_attn.in_proj_z.weight",
      "language_model.model.layers.0.linear_attn.in_proj_z ", "language_model.model.layers.0.linear_attn.in_proj_z\n"})
    require(!w8::geometry(malformed), "Dense W8A8 malformed prefix was accepted");
  for (uint32_t capacity : {0u, 1u, 1024u, 2047u})
    require(!w8::requiresCache(capacity), "Dense W8A8 small-capacity cache plan was accepted");
  for (uint32_t capacity : {2048u, 2049u, 4096u, 8192u, UINT32_MAX})
    require(w8::requiresCache(capacity), "Dense W8A8 derivative cache plan depends on dispatch choice");
  const auto scratch = w8::Workspace::logicalBufferBytes();
  require(scratch == std::array<uint64_t, 3>{12582912ULL, 8192ULL, 4ULL},
      "Dense W8A8 fixed scratch shapes differ");
  uint64_t scratchPlanned = 0;
  for (uint64_t bytes : scratch) scratchPlanned += ((bytes + 16383) / 16384) * 16384;
  require(scratchPlanned == 12615680ULL && scratchPlanned == w8::Workspace::plannedBytes(),
      "Dense W8A8 scratch conservative allocation differs");
  require(sizeof(DenseW8A8QuantizeParams) == 8 && sizeof(FlashDenseCacheParams) == 32,
      "Dense W8A8 converter/matmul ABI differs");
}
} // namespace

int main() {
  try {
    test();
    std::cout << "{\"cpu_checks\":\"passed\",\"checks\":" << checks
        << ",\"gpu_work\":false,\"payload_reads\":false,\"cache_constructed\":false"
        << ",\"projection_count\":84,\"immutable_buffer_count\":168"
        << ",\"cache_planned_bytes\":" << w8::Cache::plannedBytes()
        << ",\"workspace_planned_bytes\":" << w8::Workspace::plannedBytes() << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
