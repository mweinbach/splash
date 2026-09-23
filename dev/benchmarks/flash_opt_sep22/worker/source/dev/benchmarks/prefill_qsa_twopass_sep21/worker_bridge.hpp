#pragma once
#include "twopass.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <atomic>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string_view>
namespace splash::flash::prefill_qsa_twopass_sep21 {
inline constexpr const char *kFlag="SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21";
inline constexpr const char *kPolicy="private-singleton-main-fresh-r2048-global-f32P-wholeK256-QK-wholeK2048-packedV-PV-source-bf16-gate-v1";
// Filled from the authenticated strict Root-qualified component source snapshot.
inline constexpr const char *kShaderSHA="6bb23ded512839feda31e04a3812802d6aab5eef6d4b8e33474b794a8247a131";
inline constexpr const char *kHostSHA="1f9f56f5ac77bedaa054fb51d0d3ed8b5d6e12d5eb1d5e8688a5b0fd9fde735e";
inline bool parse(const char *value) {
  if (!value || std::string_view(value)=="0") return false;
  if (std::string_view(value)=="1") return true;
  throw std::invalid_argument(std::string(kFlag)+" must be 0 or 1");
}
inline bool requested() {
  static const bool selected=[] {
    if (!parse(std::getenv(kFlag))) return false;
    for (const char *dependency:{"SPLASH_FLASH_QSA_F32","SPLASH_FLASH_QSA_MPP","SPLASH_FLASH_QSA_ROW_TILES",
        "SPLASH_FLASH_QSA_BULK_PREFILL","SPLASH_FLASH_QSA_BULK_PREFILL_SG8"}) {
      const char *value=std::getenv(dependency);
      if (!value || std::string_view(value)!="1") throw std::invalid_argument(std::string(kFlag)+"=1 requires "+dependency+"=1");
    }
    return true;
  }();
  return selected;
}
constexpr bool mainEligible(uint32_t begin,uint32_t rows,bool verification,bool singleton,bool selected) noexcept {
  return selected && singleton && !verification && begin==0 && rows==2048;
}
constexpr uint64_t plannedExtraBytes(uint32_t maximumRows,bool selected) noexcept {
  return selected && maximumRows>=2048?prefill4k::twoPassExtraBytes():0;
}
inline const char *selectionMarker(bool selected) noexcept {
  return selected?";private-prefill-qsa-twopass-fresh-r2048-global-f32P-packedV-main-nonverify-v1":"";
}
inline std::string numericalIdentity(std::string_view base,bool selected) {
  if (!selected) return std::string(base);
  const std::string input=std::string(base)+"\n"+kPolicy+"\n"+kShaderSHA+"\n"+kHostSHA;
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  if (!CC_SHA256(input.data(),CC_LONG(input.size()),digest)) throw std::runtime_error("private QSA numerical identity SHA256 failed");
  std::ostringstream result;for (unsigned char byte:digest) result<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(byte);return result.str();
}
struct EncodedCounters {std::atomic<uint64_t> forwards{0},attentionLayers{0},constructedArenas{0},constructedArenaBytes{0};};
inline EncodedCounters &encodedCounters() {static EncodedCounters counters;return counters;}
inline void recordForward() {encodedCounters().forwards.fetch_add(1,std::memory_order_relaxed);}
inline void recordLayer() {encodedCounters().attentionLayers.fetch_add(1,std::memory_order_relaxed);}
struct Workspace final {
  metal::MetalBuffer arena;
  prefill4k::TwoPassWorkspace qualified;
  uint64_t allocatedBytes=0;
  Workspace(metal::MetalBackend &backend,const prefill4k::DenseCoalescedWorkspace &prepared) {
    if (prepared.maximumRows!=2048) throw std::logic_error("private QSA must reuse existing fresh2K prepared planes");
    const uint64_t before=backend.memoryStats().allocatedBytes;
    arena=backend.allocateBuffer(prefill4k::twoPassExtraBytes(),metal::BufferStorage::Shared,"private QSA singleton global F32P arena");
    qualified={prepared,backend.view(arena,0,25165824),backend.view(arena,25165824,402653184),backend.view(arena,427819008,50331648)};
    const uint64_t after=backend.memoryStats().allocatedBytes;
    if (after<before || after-before!=prefill4k::twoPassExtraBytes() || arena.sizeBytes()!=prefill4k::twoPassExtraBytes())
      throw std::logic_error("private QSA single arena allocation/admission mismatch");
    allocatedBytes=after-before;encodedCounters().constructedArenas.fetch_add(1,std::memory_order_relaxed);
    encodedCounters().constructedArenaBytes.fetch_add(allocatedBytes,std::memory_order_relaxed);
  }
};
}
