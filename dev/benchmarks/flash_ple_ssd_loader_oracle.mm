#include "flash/FlashWeights.hpp"
#include "flash/FlashPLESSDLayout.hpp"
#include "flash/FlashPLESSDStore.hpp"
#include "engine/Json.hpp"

#import <Foundation/Foundation.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool valid, const char *message) {
  if (!valid) throw std::runtime_error(message);
}
}

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      require(argc == 4, "usage: flash-ple-ssd-loader-oracle METALLIB PACKAGE FRESH_REPORT_JSON");
      require(!std::filesystem::exists(argv[3]), "report already exists");
      splash::metal::MetalBackend backend(argv[1]);
      const auto initial = backend.memoryStats().allocatedBytes;
      auto weights = splash::flash::FlashWeights::load(backend, argv[2]);
      require(weights.tensorCount() == 3748, "tensor inventory changed");
      require(weights.manifestFingerprint() ==
          "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0", "fingerprint changed");
      const auto statistics = weights.pleSSDStorageStats();
      auto original = weights.immutableWeightBuffers();
      const auto expectedBytes = statistics.enabled ? 74317889536ULL : 106320429056ULL;
      const auto expectedBuffers = statistics.enabled ? 28U : 21U;
      require(original.size() == expectedBuffers, "immutable native window count differs");
      uint64_t bytes = 0;
      for (const auto &buffer : original) {
        require(buffer && buffer.contents() && buffer.sizeBytes() % 16384 == 0,
                "invalid native weight window");
        bytes += buffer.sizeBytes();
      }
      require(bytes == expectedBytes && weights.actualAllocatedBytes() == expectedBytes,
              "weight storage accounting differs");
      uint64_t checkedParts = 0;
      for (uint32_t part = 0; part < 128; ++part) {
        const auto prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards." +
            std::to_string(part);
        if (statistics.enabled) {
          const auto &p = weights.diskProjection(prefix);
          require(p.weights && p.scales && p.biases && p.inputSize == 160 && p.bits == 4 &&
              p.groupSize == 32 && p.weightRowStrideBytes == 80 && p.parameterRowStrideBytes == 10,
              "disk affine metadata differs");
          for (const auto *tensor : {p.weights, p.scales, p.biases})
            require(tensor->offset % 16384 == 0 && tensor->offset <= tensor->fileBytes &&
                tensor->logicalBytes <= tensor->fileBytes - tensor->offset, "disk tensor extent differs");
          bool rejected = false;
          try { static_cast<void>(weights.projection(prefix)); }
          catch (const std::exception &) { rejected = true; }
          require(rejected, "disk-only PLE projection escaped as GPU buffers");
          rejected = false;
          try { static_cast<void>(weights.tensor(prefix + ".weight")); }
          catch (const std::exception &) { rejected = true; }
          require(rejected, "disk-only PLE tensor escaped as GPU buffer");
        } else {
          const auto &p = weights.projection(prefix);
          require(p.weights->buffer && p.scales->buffer && p.biases->buffer,
                  "raw PLE projection has no GPU storage");
        }
        ++checkedParts;
      }
      const auto &anchor = weights.tensor(
          "language_model.model.layers.0.attn_hyper_connection.hc_norm.weight");
      auto anchorView = anchor.buffer;
      std::vector<uint8_t> anchorBytes(static_cast<const uint8_t *>(anchorView.contents()),
          static_cast<const uint8_t *>(anchorView.contents()) + anchor.logicalBytes);
      auto store = weights.pleSSDStore();
      require(statistics.enabled == bool(store), "shared store mode differs");
      if (store) require(store->partCount() == 128 && store->tableRows() == weights.descriptor().pleTableRows,
                         "shared store geometry differs");
      auto moved = std::move(weights);
      require(weights.tensorCount() == 0 && moved.tensorCount() == 3748, "move changed inventory ownership");
      require(backend.memoryStats().allocatedBytes - initial == expectedBytes, "move changed native ledger");
      moved = splash::flash::FlashWeights();
      require(std::memcmp(anchorView.contents(), anchorBytes.data(), anchorBytes.size()) == 0,
              "retained tensor view lost mapped backing after weights release");
      require(backend.memoryStats().allocatedBytes - initial == expectedBytes,
              "immutable bases lost ownership after weights release");
      original.clear();
      require(backend.memoryStats().allocatedBytes > initial,
              "retained tensor view failed to retain its native owner");
      anchorView = {};
      require(backend.memoryStats().allocatedBytes == initial, "weight windows leaked after last view release");
      store.reset();
      std::ofstream report(argv[3]);
      require(bool(report), "cannot open report");
      report << "{\"schema\":\"flash-ple-ssd-loader-lifetime-v12\",\"valid\":true,"
          "\"gpu_commands\":0,\"gpu_buffer_creation\":true,\"streaming_enabled\":"
          << (statistics.enabled ? "true" : "false")
          << ",\"native_weight_windows\":" << expectedBuffers
          << ",\"native_weight_bytes\":" << expectedBytes
          << ",\"disk_only_payload_bytes\":" << statistics.diskOnlyPayloadBytes
          << ",\"disk_only_logical_bytes\":" << statistics.diskOnlyLogicalBytes
          << ",\"checked_parts\":" << checkedParts
          << ",\"tensor_count\":3748,\"unchanged_fingerprint\":true,"
             "\"disk_only_gpu_access_rejected\":true,\"move_and_view_lifetime\":true,"
             "\"no_native_leak\":true}\n";
      require(bool(report), "cannot write report");
      std::cout << "{\"valid\":true,\"report\":" << splash::json::quote(argv[3]) << "}\n";
      return 0;
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
