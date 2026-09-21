// CPU-only native metadata/source/shape/file-size audit; no Metal device or
// payload mapping/hash/conversion. Inference construction verifies payloads.
#include "flash/FlashInt8ExpertStoreMetadata.hpp"
#include "engine/Json.hpp"

#import <Foundation/Foundation.h>

#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2) throw std::invalid_argument("usage: flash-int8-store-metadata-audit STORE_DIRECTORY");
      const auto metadata = splash::flash::loadFlashInt8ExpertStoreMetadata(argv[1],
          "ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e",
          "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",
          splash::flash::NormConvention::OnePlusWeight);
      uint64_t selected = 0;
      for (const auto &layer : metadata.layers) selected += layer.selectedIDs.size();
      std::cout << "{\"valid\":true,\"scope\":\"native CPU metadata, source/layout, shape and file-size checks only\","
        << "\"payload_hashes_verified\":false,\"gpu_commands\":0,\"source_identity_sha256\":"
        << splash::json::quote(metadata.sourceIdentity)
        << ",\"source_manifest_sha256\":" << splash::json::quote(metadata.sourceManifestSha256)
        << ",\"manifest_sha256\":" << splash::json::quote(metadata.identitySha256)
        << ",\"plan_sha256\":" << splash::json::quote(metadata.planSha256)
        << ",\"layers\":" << metadata.layers.size()
        << ",\"selected_experts\":" << selected
        << ",\"selected_experts_per_layer\":" << metadata.layers[0].selectedIDs.size()
        << ",\"mapped_payload_bytes\":" << metadata.totalBytes
        << ",\"planned_allocation_bytes\":" << metadata.plannedBytes << "}\n";
      return 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  }
}
