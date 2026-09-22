#pragma once
#include "flash/FlashMoEBlocked.hpp"
#include "metal/abi/FlashInt8ExpertStore.h"
#include <array>
#include <cstring>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <string_view>

namespace splash::flash::prefill_moe_sep21 {
struct Variant {
  const char *kind;
  uint32_t k, sg;
  bool registerOperand, staticExtent, differentReduction;
  bool rowParts =false;
};
inline constexpr std::array<Variant,11> variants{{
  {"whole",0,1,false,false,false}, {"whole",0,2,false,false,false},
  {"static",0,1,false,true,false}, {"static",0,2,false,true,false},
  {"fixed",64,1,false,true,true}, {"fixed",128,1,false,true,true},
  {"fixed",128,2,false,true,true}, {"register",64,1,true,true,true},
  {"register",128,1,true,true,true}, {"register",64,1,true,true,true,true},
  {"register",128,1,true,true,true,true}
}};
inline std::string pipelineName(const Variant &v,bool gate) {
  return std::string("prefill_moe_sep21_") +(v.registerOperand ? "register_" :"memory_" +std::string(v.kind) +"_") +
      (gate ? "gate_up" :"down_scatter") +"_m32_n64_k" +std::to_string(v.k) +"_sg" +std::to_string(v.sg) +(v.rowParts ? "_m16parts" :"");
}
inline uint32_t parseVariantIndex(const char *raw) {
  if (!raw ||std::string_view(raw) =="0") return 0;
  for (uint32_t index=1;index <=variants.size();++index)
    if (std::string_view(raw) ==std::to_string(index)) return index;
  throw std::invalid_argument("SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT must be0..11");
}
inline uint32_t requestedVariantIndex() {
  static const uint32_t selected=parseVariantIndex(std::getenv("SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT"));
  return selected;
}
inline std::string policyIdentity() {
  const uint32_t selected=requestedVariantIndex();
  if (!selected) return {};
  const auto &v=variants[selected -1];
  return std::string("private-prefill-moe-sep21-variant=") +std::to_string(selected) +
      ";gate=" +pipelineName(v,true) +";down=" +pipelineName(v,false) +
      ";native-M32-jobs-original-i8-post-f32-row-scale-bf16-boundaries-v1" +
      (v.registerOperand ? ";register-part-M" +std::to_string(v.rowParts ? 16 :32) +"N32K32-exact-i8-to-bf16" :"");
}
// No new allocation or parameter bytes. Source graph remains the owner of all
// inline parameter payloads. Native bucket jobs and Q4 misses stay unchanged.
class HitCommands {
 public:
  std::vector<metal::ComputeDispatch> commands;
  HitCommands() =default;
  HitCommands(HitCommands &&) =default;
  HitCommands &operator=(HitCommands &&) =default;
  HitCommands(const HitCommands &) =delete;
  HitCommands &operator=(const HitCommands &) =delete;
  HitCommands(std::span<const metal::ComputeDispatch> source,uint32_t rows,const Variant &variant) {
    if (!rows ||rows >8192 ||(variant.sg !=1 &&variant.sg !=2) ||
        (variant.k !=0 &&variant.k !=64 &&variant.k !=128))
      throw std::invalid_argument("low SIMD expert kernel geometry unsupported");
    bool gateSeen=false,downSeen=false;
    metal::MetalBuffer offsets,jobs,count;
    for (const auto &original :source) {
      auto d=original;
      const bool gate=d.pipelineName.starts_with("flash_int8_expert_store_gate_up_m") &&
          !d.pipelineName.starts_with("flash_int8_expert_store_gate_up_miss_");
      const bool down=d.pipelineName.starts_with("flash_int8_expert_store_down_scatter_");
      if (!gate &&!down) { commands.push_back(d);continue; }
      if (d.bytes.size() !=1 ||d.bytes[0].sizeBytes !=sizeof(FlashInt8ExpertStoreParams) ||!d.bytes[0].data)
        throw std::invalid_argument("low SIMD expert source parameter ABI differs");
      FlashInt8ExpertStoreParams p;std::memcpy(&p,d.bytes[0].data,sizeof(p));
      if (p.rows !=rows ||p.selections !=10 ||p.route_capacity !=rows *10 ||
          p.tile_rows !=32 ||p.job_capacity !=(rows *10 +31) /32 +511 ||
          !p.stored_experts ||p.stored_experts >512 ||p.scale_group_size ||p.reserved ||
          d.threadgroups.y !=p.job_capacity ||d.threadgroups.z !=1)
        throw std::invalid_argument("low SIMD expert requires original M32 jobs and extents");
      if (gate) {
        if (gateSeen ||downSeen ||d.buffers.size() !=11 ||d.bytes[0].index !=11 ||d.threadgroups.x !=10)
          throw std::invalid_argument("low SIMD expert gate ABI/order differs");
        gateSeen=true;offsets=d.buffers[6].buffer;jobs=d.buffers[7].buffer;count=d.buffers[8].buffer;
      } else {
        if (!gateSeen ||downSeen ||d.buffers.size() !=10 ||d.bytes[0].index !=10 ||d.threadgroups.x !=40 ||
            !d.buffers[4].buffer.sameView(offsets) ||!d.buffers[5].buffer.sameView(jobs) ||!d.buffers[6].buffer.sameView(count))
          throw std::invalid_argument("low SIMD expert down ABI/jobs/order differs");
        downSeen=true;
      }
      d.pipelineName=pipelineName(variant,gate);
      d.threadsPerThreadgroup={uint64_t(variant.sg) *32,1,1};commands.push_back(d);
    }
    if (!gateSeen ||!downSeen ||commands.size() !=source.size())
      throw std::invalid_argument("low SIMD expert source producer missing or dispatch count differs");
  }
};
inline void cpuSelfTest() {
  if (parseVariantIndex(nullptr) ||parseVariantIndex("0")) throw std::logic_error("private expert control parser invalid");
  for (uint32_t index=1;index <=variants.size();++index)
    if (parseVariantIndex(std::to_string(index).c_str()) !=index) throw std::logic_error("private expert variant parser invalid");
  for (const char *raw :{"", "-1", "+1", "01", "1 ", " 1", "12", "1x"}) {
    bool rejected=false;try { (void)parseVariantIndex(raw); } catch (const std::invalid_argument &) { rejected=true; }
    if (!rejected) throw std::logic_error("private expert invalid variant accepted");
  }
  for (const auto &v :variants) {
    if (pipelineName(v,true) ==pipelineName(v,false) ||v.sg >2 ||
        (v.registerOperand &&v.sg !=1) ||(v.k &&(2560 %v.k ||640 %v.k)))
      throw std::logic_error("private expert kernel inventory invalid");
  }
}
} // namespace splash::flash::prefill_moe_sep21
