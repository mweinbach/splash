#pragma once
// Generated verbatim guard baseline: source-only; no backend/model.
namespace frozen_baseline {

[[noreturn]] void fail(const char *reason) { throw std::invalid_argument(reason); }
using FlashMoEBlockedTile=::metadata_fake::Tile;

void requireGeometry(uint32_t rows, uint32_t selections) {
  if (!rows || rows > kFlashMoEBucketMaximumRows || !selections ||
      selections > kFlashMoEBucketMaximumSelections)
    throw std::invalid_argument("Flash MoE buckets unsupported row/selection geometry");
}
uint32_t moEBucketJobCapacity(uint32_t rows, uint32_t selections,
                            uint32_t tileRows) {
  requireGeometry(rows, selections);
  if (tileRows != 8 && tileRows != 16 && tileRows != 32 && tileRows != 64)
    throw std::invalid_argument("Flash MoE buckets unsupported matrix tile rows");
  // Sum ceil(count[e]/M) <= ceil(sum(count[e])/M) + E - 1.
  // Counts sum to at most rows*selections, including duplicate legal IDs.
  return (rows * selections + tileRows - 1) / tileRows + kExperts - 1;
}
namespace metal=::metadata_fake;
using FlashMoEBlockedScratch=::metadata_fake::Scratch;
using FlashMoEBlockedTile=::metadata_fake::Tile;

void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes) {
  if (!buffer || buffer.storage() != metal::BufferStorage::Shared ||
      !buffer.contents() || !bytes || buffer.sizeBytes() < bytes)
    fail("INT8 expert store requires sufficient Shared buffer views");
}
void disjoint(const metal::MetalBuffer &a, const metal::MetalBuffer &b) {
  const auto aa = reinterpret_cast<uintptr_t>(a.contents());
  const auto bb = reinterpret_cast<uintptr_t>(b.contents());
  if (!aa || !bb) fail("INT8 expert store requires addressable buffer views");
  if (aa <= bb ? uint64_t(bb - aa) < a.sizeBytes() : uint64_t(aa - bb) < b.sizeBytes())
    fail("INT8 expert writable output overlaps an input or immutable operand");
}
inline bool flashMoEDirectAEnabled(){return ::metadata_fake::configuration->directA;}
inline FlashMoEBlockedTile flashMoEBlockedTile(uint32_t,bool){return ::metadata_fake::configuration->wideM64?FlashMoEBlockedTile::M64N64:FlashMoEBlockedTile::M32N64;}

void allRowsScratch(const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
                    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) {
  const uint32_t m = uint32_t(tile);
  if (!rows || rows > 8192 || !selections || selections > 10 ||
      (m != 16 && m != 32 && m != 64) || rows > s.buckets.rowCapacity ||
      selections > s.buckets.selectionCapacity || uint64_t{rows} * selections > s.buckets.routeCapacity ||
      !flashMoEDirectAEnabled()) fail("all-row Full512 unsupported scratch/Direct-A geometry");
  if (m == 64 && (rows < 1024 || flashMoEBlockedTile(4096, false) != FlashMoEBlockedTile::M64N64))
    fail("all-row Full512 M64 requires the enabled wide policy");
  const uint32_t routes = rows * selections, jobs = moEBucketJobCapacity(rows, selections, m);
  if (jobs > s.buckets.jobCapacity) fail("all-row Full512 insufficient declared matrix jobs");
  requireBytes(diagnostics, 4);
  requireBytes(s.buckets.counts, 512 * 4); requireBytes(s.buckets.offsets, 513 * 4);
  requireBytes(s.buckets.routeMap, uint64_t{routes} * 4);
  requireBytes(s.buckets.canonicalToPacked, uint64_t{routes} * 4);
  requireBytes(s.buckets.jobOffsets, 513 * 4); requireBytes(s.buckets.jobCount, 4);
  requireBytes(s.buckets.tileJobs, uint64_t{jobs} * 8);
  requireBytes(s.buckets.packedInputs, uint64_t{routes + 63} * 2560 * 2);
  requireBytes(s.packedActivated, uint64_t{routes + 63} * 640 * 2);
  requireBytes(s.scatteredDown, uint64_t{routes} * 2560 * 2);
  const std::array scratch{s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics};
  for (size_t i = 0; i < scratch.size(); ++i)
    for (size_t j = i + 1; j < scratch.size(); ++j) disjoint(scratch[i], scratch[j]);
}
struct Impl {using Layer=::metadata_fake::Layer;std::array<Layer,48> layers;::metadata_fake::Metadata metadata;bool compactR4Verify=true;

  const Layer &layer(uint32_t index) const {
    if (index >= 48) fail("saved INT8 expert layer is outside target layers");
    return layers[index];
  }
  void immutableDisjoint(const metal::MetalBuffer &output) const {
    for (const auto &layer : layers) {
      disjoint(output, layer.base); disjoint(output, layer.ranks);
    }
  }
};

class FlashInt8ExpertStore {public: Impl *impl_=nullptr;
bool gatheredMPPEnabled()const{return ::metadata_fake::configuration->gather;}
uint32_t gatheredMPPMaximumRows()const{return ::metadata_fake::configuration->gatherCap;}
bool compactNativeR4VerifyEnabled()const;
void addCompactNativeR4VerifyPack(metal::CommandGraph &,uint32_t,metal::MetalBuffer,metal::MetalBuffer,const FlashMoEBlockedScratch &,metal::MetalBuffer,uint32_t,uint32_t)const;
void addGateUp(metal::CommandGraph &,uint32_t,const FlashMoEBlockedScratch &,metal::MetalBuffer,uint32_t,FlashMoEBlockedTile,uint32_t)const;
void addDownScatter(metal::CommandGraph &,uint32_t,const FlashMoEBlockedScratch &,metal::MetalBuffer,uint32_t,FlashMoEBlockedTile,uint32_t)const;
};
namespace compact_native_r4_verify_sep22 {inline bool requested(){return ::metadata_fake::configuration->requested;}}

bool FlashInt8ExpertStore::compactNativeR4VerifyEnabled() const {
  if (!impl_) fail("compact R4 verifier Store disposed");
  if (compact_native_r4_verify_sep22::requested()!=impl_->compactR4Verify)
    fail("compact R4 verifier flag changed after construction");
  if (impl_->compactR4Verify && (!gatheredMPPEnabled()||gatheredMPPMaximumRows()!=4))
    fail("compact R4 verifier requires original gathered cap exactly4");
  return impl_->compactR4Verify;
}
void FlashInt8ExpertStore::addCompactNativeR4VerifyPack(metal::CommandGraph &graph,uint32_t index,
    metal::MetalBuffer input,metal::MetalBuffer ids,const FlashMoEBlockedScratch &s,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections) const {
  if (!compactNativeR4VerifyEnabled()||rows!=4||selections!=10) fail("compact verifier only canonical R4/S10");
  const auto &layer=impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size()!=512) fail("compact verifier requires Full512 inventory");
  allRowsScratch(s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);
  requireBytes(input,uint64_t{4}*2560*2);requireBytes(ids,40*8);requireBytes(layer.ranks,512*4);
  disjoint(input,ids);
  for (const auto &b:{s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,s.buckets.packedInputs,
      s.buckets.jobOffsets,s.buckets.jobCount,s.buckets.tileJobs,s.packedActivated,s.scatteredDown,diagnostics}) {
    disjoint(input,b);disjoint(ids,b);impl_->immutableDisjoint(b);
  }
  impl_->immutableDisjoint(input);impl_->immutableDisjoint(ids);
  (void)graph; (void)layer;
}

void FlashInt8ExpertStore::addGateUp(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  const auto &layer = impl_->layer(index);
  allRowsScratch(s, diagnostics, rows, tile, selections);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("all-row target dispatch requires Full512");
  for (const auto &buffer : {s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  (void)graph; (void)layer;
}

void FlashInt8ExpertStore::addDownScatter(metal::CommandGraph &graph, uint32_t index,
    const FlashMoEBlockedScratch &s, metal::MetalBuffer diagnostics,
    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  const auto &layer = impl_->layer(index);
  allRowsScratch(s, diagnostics, rows, tile, selections);
  if (impl_->metadata.layers[index].selectedIDs.size() != 512)
    fail("all-row target dispatch requires Full512");
  for (const auto &buffer : {s.buckets.counts, s.buckets.offsets, s.buckets.routeMap,
      s.buckets.canonicalToPacked, s.buckets.packedInputs, s.buckets.jobOffsets,
      s.buckets.jobCount, s.buckets.tileJobs, s.packedActivated, s.scatteredDown, diagnostics})
    impl_->immutableDisjoint(buffer);
  (void)graph; (void)layer;
}

inline void validateBundled(FlashInt8ExpertStore &store,metal::CommandGraph &graph,uint32_t index,metal::MetalBuffer input,metal::MetalBuffer ids,
const FlashMoEBlockedScratch &s,metal::MetalBuffer diagnostics,uint32_t rows,uint32_t selections){
auto *impl_=store.impl_;
const auto compactNativeR4VerifyEnabled=[&]{return store.compactNativeR4VerifyEnabled();};


  if (!compactNativeR4VerifyEnabled()||rows!=4||selections!=10) fail("compact verifier only canonical R4/S10");
  const auto &layer=impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size()!=512) fail("compact verifier requires Full512 inventory");

::splash::flash::compact_r4_preflight_sep22::validateComplete(s,input,ids,layer.ranks,diagnostics,rows,selections,
[](const auto &scratch,auto diag,uint32_t r,uint32_t k){allRowsScratch(scratch,diag,r,FlashMoEBlockedTile::M16N64,k);},
[](const auto &buffer,uint64_t bytes){requireBytes(buffer,bytes);},
[](const auto &a,const auto &b){disjoint(a,b);},
[&](const auto &buffer){impl_->immutableDisjoint(buffer);});
(void)graph;
}
}
