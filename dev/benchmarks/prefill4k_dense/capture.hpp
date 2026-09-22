#pragma once

#include "flash/FlashWeights.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashForward.h"
#include <cstdlib>
#include <exception>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace prefill4k_dense {
struct Capture final {
  std::string projection;
  uint32_t rows = 0, inputs = 0, outputs = 0;
  splash::metal::MetalBuffer input, weights, output;
};
inline std::vector<Capture> captures;
inline std::set<std::string> captured;
inline bool captureRole(const std::string &prefix) {
  for (const char *role : {
      "language_model.model.layers.0.linear_attn.in_proj_qkv",
      "language_model.model.layers.3.self_attn.q_proj",
      "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down",
      "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_up",
      "language_model.model.layers.0.linear_attn.in_proj_z",
      "language_model.model.layers.0.linear_attn.out_proj",
      "language_model.model.layers.3.self_attn.k_proj",
      "language_model.model.layers.3.self_attn.indexer.index_qk_proj",
      "language_model.model.layers.0.mlp.shared_expert.down_proj",
      "language_model.model.layers.1.ple.value_proj"})
    if (prefix == role) return true;
  return false;
}
inline void captureCopy(splash::metal::CommandGraph &graph,
    const splash::metal::MetalBuffer &source, const splash::metal::MetalBuffer &destination,
    uint64_t bytes) {
  if (!bytes || bytes % 4 || source.sizeBytes() < bytes || destination.sizeBytes() < bytes)
    throw std::runtime_error("dense capture extent invalid");
  const uint64_t words = bytes / 4;
  graph.add("flash_forward_copy_words", {source,destination}, FlashForwardCopyParams{words},
      {(words + 255) / 256,1,1});
}

// This helper is inserted only into a private generated FlashForward copy.
// Scope destruction records the model's output after its projection dispatch.
class CaptureScope final {
public:
  CaptureScope(splash::metal::MetalBackend &backend, splash::metal::CommandGraph &graph,
      const std::string &prefix, const splash::metal::MetalBuffer &input,
      const splash::metal::MetalBuffer &output, const splash::flash::FlashTensor *weight,
      uint32_t rows) : graph_(graph), sourceOutput_(output) {
    const char *directory = std::getenv("PREFILL4K_DENSE_CAPTURE");
    if (!directory || !*directory || !weight || rows != 2048 || !captureRole(prefix) || captured.contains(prefix))
      return;
    if (weight->dtype != splash::flash::FlashDType::BF16 || weight->shape.size() != 2)
      throw std::runtime_error("dense capture requires BF16 matrix");
    Capture value;
    value.projection = prefix; value.rows = rows;
    value.inputs = uint32_t(weight->shape[1]); value.outputs = uint32_t(weight->shape[0]);
    if (input.sizeBytes() < uint64_t(rows)*value.inputs*2 ||
        output.sizeBytes() < uint64_t(rows)*value.outputs*2)
      throw std::runtime_error("dense capture source input/output extent invalid");
    value.weights = weight->buffer;
    value.input = backend.allocateBuffer(uint64_t(rows)*value.inputs*2,
        splash::metal::BufferStorage::Shared,"private-dense-capture-input");
    value.output = backend.allocateBuffer(uint64_t(rows)*value.outputs*2,
        splash::metal::BufferStorage::Shared,"private-dense-capture-output");
    captureCopy(graph_,input,value.input,value.input.sizeBytes());
    index_ = captures.size(); captures.push_back(std::move(value)); captured.insert(prefix);
  }
  ~CaptureScope() noexcept(false) {
    if (index_ && std::uncaught_exceptions() == 0) {
      const auto &capture = captures.at(*index_);
      captureCopy(graph_,sourceOutput_,capture.output,capture.output.sizeBytes());
    }
  }
private:
  splash::metal::CommandGraph &graph_;
  splash::metal::MetalBuffer sourceOutput_;
  std::optional<size_t> index_;
};
void writeCaptures();
} // namespace prefill4k_dense
