#pragma once
#include "flash/FlashGDNStaged.hpp"

#include <span>
#include <string>

struct GdnBatchIlpTile { uint32_t values = 32, time = 32, simds = 8; };
struct GdnBatchIlpLane {
  splash::flash::FlashGDNState state;
  uint32_t rows = 0;
};

void validateGdnBatchIlpTile(GdnBatchIlpTile tile);
[[nodiscard]] std::string gdnBatchIlpName(GdnBatchIlpTile tile);
// Pure scalar and missing-live-state preflight; constructs no backend.
[[nodiscard]] uint32_t validateGdnBatchIlpGeometry(std::span<const GdnBatchIlpLane> lanes,
    uint32_t inputRowsStride, GdnBatchIlpTile tile = {}, float epsilon = 1e-6f);

// Private full GDN graph. Each live state remains a separate writable F32
// allocation. Prepare, direct-gamma output normalization and convolution carry
// keep the current per-lane kernels; only recurrence is merged across slots.
// No state copy, temporal reassociation or precision conversion occurs.
void addGdnBatchIlp(splash::metal::MetalBackend &backend,
    splash::metal::CommandGraph &graph, const splash::flash::FlashGDNWeights &weights,
    const splash::flash::FlashGDNBuffers &flatBuffers,
    std::span<const GdnBatchIlpLane> lanes, uint32_t inputRowsStride,
    GdnBatchIlpTile tile = {}, float epsilon = 1e-6f);
