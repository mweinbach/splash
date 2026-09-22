// Pure CPU checks: no allocator, Metal backend or model is constructed.
#include "dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy.hpp"

#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>

namespace fixed=splash::flash::fixed_sg2_prefill_sep21;
using splash::flash::FlashMoEBlockedTile;

namespace {
void require(bool pass,const char *reason) {
  if (!pass) throw std::runtime_error(reason);
}
void rejects(const char *value) {
  bool rejected=false;
  try {(void)fixed::parseSelection(value);} catch (const std::invalid_argument &) {rejected=true;}
  require(rejected,"invalid fixed SG2 variant was accepted");
}
void selfTest() {
  require(fixed::parseSelection(nullptr)==0 && fixed::parseSelection("0")==0 &&
      fixed::parseSelection("7")==7,"fixed SG2 strict parser golden differs");
  for (const char *invalid:{"","1","2","3","4","5","6","8","9","10","11",
      "12","-7","+7","07","007","7 "," 7","7\n","true","false","0x7","00"})
    rejects(invalid);
  require(std::string_view(fixed::markerFor(0)).empty() &&
      std::string_view(fixed::producerNameFor(true,0)).empty() &&
      std::string_view(fixed::producerNameFor(false,0)).empty() &&
      !fixed::producerThreadsFor(0),"fixed SG2 flag0 changes baseline identity/producer policy");
  constexpr std::string_view marker=
      ";private-prefill-moe-sg2-k128-sep21-main-nonverification-r2048-m32-original-i8-f32lateScale-bf16-boundaries-v1";
  require(fixed::markerFor(7)==marker,"fixed SG2 active marker differs");
  require(std::string_view(fixed::producerNameFor(true,7))==
      "prefill_moe_sep21_memory_fixed_gate_up_m32_n64_k128_sg2" &&
      std::string_view(fixed::producerNameFor(false,7))==
      "prefill_moe_sep21_memory_fixed_down_scatter_m32_n64_k128_sg2" &&
      fixed::producerThreadsFor(7)==64,"fixed SG2 variant7 exact pipeline inventory differs");
  require(fixed::eligibleFor(2048,FlashMoEBlockedTile::M32N64,false,7),
      "fixed SG2 active canonical main R2048 rejected");
  require(!fixed::eligibleFor(2048,FlashMoEBlockedTile::M32N64,true,7),
      "fixed SG2 verification route eligible");
  for (uint32_t rows:{0u,1u,2u,16u,256u,512u,1024u,2047u,2049u,4096u,8192u,8193u,UINT32_MAX})
    require(!fixed::eligibleFor(rows,FlashMoEBlockedTile::M32N64,false,7),
        "fixed SG2 nonR2048 route eligible");
  for (auto tile:{FlashMoEBlockedTile::M8N64,FlashMoEBlockedTile::M16N64,
      FlashMoEBlockedTile::M64N64,static_cast<FlashMoEBlockedTile>(0),static_cast<FlashMoEBlockedTile>(128)})
    require(!fixed::eligibleFor(2048,tile,false,7),"fixed SG2 nonM32 route eligible");
  for (uint32_t selected:{0u,1u,2u,3u,4u,5u,6u,8u,9u,10u,11u,UINT32_MAX})
    require(!fixed::eligibleFor(2048,FlashMoEBlockedTile::M32N64,false,selected),
        "fixed SG2 non7 variant eligible");
  static_assert(std::is_trivially_copyable_v<fixed::Counters>);
  const fixed::Counters empty;
  require(!empty.enabled && !empty.gateCalls && !empty.gateRows && !empty.downCalls && !empty.downRows,
      "fixed SG2 default graph counters are not empty");

  // The process's live environment exercises early parsing and immutable mode.
  const char *raw=std::getenv("SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT");
  const bool wasSet=raw!=nullptr;
  const std::string previous=raw ? raw : "";
  const uint32_t selected=fixed::parseSelection(raw),frozen=fixed::selection();
  require(frozen==selected && fixed::requested()==(selected==7) &&
      std::string_view(fixed::marker())==fixed::markerFor(selected) &&
      std::string_view(fixed::producerName(true))==fixed::producerNameFor(true,selected) &&
      std::string_view(fixed::producerName(false))==fixed::producerNameFor(false,selected) &&
      fixed::producerThreads()==fixed::producerThreadsFor(selected) &&
      fixed::eligible(2048,FlashMoEBlockedTile::M32N64,false)==(selected==7),
      "fixed SG2 frozen live policy differs");
  require(setenv("SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT",selected==7 ? "0" : "7",1)==0,
      "fixed SG2 CPU fixture cannot change environment");
  require(fixed::selection()==frozen && fixed::requested()==(frozen==7),
      "fixed SG2 selection is not frozen on first use");
  require((wasSet ? setenv("SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT",previous.c_str(),1) :
      unsetenv("SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT"))==0,
      "fixed SG2 CPU fixture cannot restore environment");
}
} // namespace

int main() {
  try {
    selfTest();
    std::cout<<"{\"cpu_checks\":\"passed\",\"gpu_work\":false,\"payload_reads\":false,"
        "\"frozen_selection\":"<<fixed::selection()<<",\"requested_threads\":"
        <<fixed::producerThreads()<<",\"allocator_used\":false,\"whole_model_equality\":\"pending_root_qualification\"}\n";
    return 0;
  } catch (const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
