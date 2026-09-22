// CPU-only policy/byte-accounting checks: no MetalBackend or model is created.
#include "dev/benchmarks/prefill_moe_sep21/w8a8/worker.hpp"

#include <iostream>
#include <string>
#include <string_view>

namespace w8=splash::flash::prefill_w8a8_sep21;
using splash::flash::FlashMoEBlockedTile;

namespace {
void require(bool passed,const char *message) {
  if (!passed) throw std::runtime_error(message);
}
template<class F> void rejects(F &&operation) {
  bool rejected=false;
  try {operation();} catch (const std::invalid_argument &) {rejected=true;}
  require(rejected,"invalid W8A8 policy/geometry was accepted");
}
void test() {
  require(w8::parseMode(nullptr)==0 && w8::parseMode("0")==0 &&
      w8::parseMode("1")==1 && w8::parseMode("2")==2,"strict mode parser golden differs");
  for (const char *invalid:{"","3","-1","+1","01","1 "," 1","true","false","1\n","0x1"})
    rejects([&]{(void)w8::parseMode(invalid);});
  require(std::string_view(w8::policyFor(0)).empty() &&
      std::string_view(w8::markerFor(0)).empty() &&
      std::string_view(w8::producerNameFor(true,0)).empty() &&
      !w8::producerThreadsFor(0),"flag0 changes baseline numerical policy");
  for (uint32_t selected:{1u,2u}) {
    require(!std::string_view(w8::policyFor(selected)).empty() &&
        std::string_view(w8::markerFor(selected)).substr(1)==w8::policyFor(selected),
        "active W8A8 marker does not encode full policy");
    require(w8::eligibleFor(2048,FlashMoEBlockedTile::M32N64,false,selected),
        "canonical main 2K W8A8 prefill rejected");
    require(!w8::eligibleFor(2048,FlashMoEBlockedTile::M32N64,true,selected),
        "W8A8 verification route accepted");
    for (uint32_t rows:{0u,1u,16u,1024u,2047u,2049u,4096u,8192u,8193u})
      require(!w8::eligibleFor(rows,FlashMoEBlockedTile::M32N64,false,selected),
          "W8A8 non2K route accepted");
    for (auto tile:{FlashMoEBlockedTile::M8N64,FlashMoEBlockedTile::M16N64,FlashMoEBlockedTile::M64N64})
      require(!w8::eligibleFor(2048,tile,false,selected),"W8A8 nonM32 route accepted");
    const auto gate=std::string_view(w8::producerNameFor(true,selected));
    const auto down=std::string_view(w8::producerNameFor(false,selected));
    const auto suffix=selected==1 ? "m32_n64_sg4" : "m32_n64_sg2";
    require(gate.starts_with("private_w8a8_worker_gate_up_") && gate.ends_with(suffix) &&
        down.starts_with("private_w8a8_worker_down_scatter_") && down.ends_with(suffix) &&
        w8::producerThreadsFor(selected)==(selected==1 ? 128u : 64u),"W8A8 pipeline inventory differs");
  }
  for (uint32_t selected:{0u,3u,UINT32_MAX})
    require(!w8::eligibleFor(2048,FlashMoEBlockedTile::M32N64,false,selected),
        "inactive/invalid W8A8 mode eligible");
  require(w8::Workspace::logicalBytes(2048)==65901944ULL &&
      w8::Workspace::plannedBytes(2048)==65945600ULL,
      "2K W8A8 four-buffer byte accounting differs");
  require(w8::Workspace::logicalBytes(8192)==263001464ULL &&
      w8::Workspace::plannedBytes(8192)==263045120ULL,
      "8K W8A8 four-buffer byte accounting differs");
  require(w8::Workspace::logicalBytes(1)==234184ULL &&
      w8::Workspace::plannedBytes(1)==278528ULL,"minimum W8A8 extent differs");
  for (uint32_t invalid:{0u,8193u,UINT32_MAX}) {
    rejects([&]{(void)w8::Workspace::logicalBytes(invalid);});
    rejects([&]{(void)w8::Workspace::plannedBytes(invalid);});
  }
  for (uint32_t rows:{1u,2048u,8192u}) {
    uint64_t independentlyPlanned=0,independentlyLogical=0;
    const uint64_t padded=uint64_t{rows}*10+63;
    const std::array<uint64_t,4> independentlySized{padded*2560,padded*640,padded*4,padded*4};
    require(w8::Workspace::sizes(rows)==independentlySized,"W8A8 global quant buffer shapes differ");
    for (uint64_t bytes:independentlySized) {
      independentlyLogical+=bytes;
      independentlyPlanned+=((bytes+64+16383)/16384)*16384;
    }
    require(w8::Workspace::logicalBytes(rows)==independentlyLogical &&
        w8::Workspace::plannedBytes(rows)==independentlyPlanned,
        "W8A8 per-buffer 64-byte guards/16KB rounding differ");
  }
  require(sizeof(w8::QuantParams)==16,"W8A8 quantizer uint4 ABI differs");
  // Validate the live flag strictly, then demonstrate immutable mode on reuse.
  const char *original=std::getenv("SPLASH_FLASH_PREFILL_MOE_W8A8");
  const bool wasSet=original!=nullptr;
  const std::string previous=original ? original : "";
  const uint32_t expected=w8::parseMode(original),frozen=w8::mode();
  require(frozen==expected && std::string_view(w8::policy())==w8::policyFor(expected) &&
      std::string_view(w8::marker())==w8::markerFor(expected),"live W8A8 policy differs");
  require(setenv("SPLASH_FLASH_PREFILL_MOE_W8A8",expected==2 ? "1" : "2",1)==0,
      "CPU mode-freeze fixture cannot set environment");
  require(w8::mode()==frozen,"W8A8 mode is not frozen on first use");
  require((wasSet ? setenv("SPLASH_FLASH_PREFILL_MOE_W8A8",previous.c_str(),1) :
      unsetenv("SPLASH_FLASH_PREFILL_MOE_W8A8"))==0,"CPU mode-freeze fixture cannot restore environment");
}
} // namespace

int main() {
  try {
    test();
    std::cout<<"{\"cpu_checks\":\"passed\",\"gpu_work\":false,\"payload_reads\":false,"
        "\"frozen_mode\":"<<w8::mode()<<",\"workspace_buffer_count\":4,"
        "\"r2048_logical_bytes\":65901944,\"r2048_planned_bytes\":65945600,"
        "\"r8192_logical_bytes\":263001464,\"r8192_planned_bytes\":263045120}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr<<error.what()<<'\n';return 1;
  }
}
