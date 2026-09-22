#include "telemetry.hpp"
#include <algorithm>
#include <iostream>
#include <limits>

using namespace splash::flash::gdn_wy_hc_sep21::telemetry;
namespace {
void require(bool ok,const char *why) { if (!ok) throw why; }
void add(Snapshot &s,uint32_t slot,uint64_t amount) {
  auto &value=s.words[slot];
  const auto limit=std::numeric_limits<uint64_t>::max();
  if (amount>limit-value) { value=limit; s.words[Saturated]=1; }
  else value+=amount;
}
void observe(Snapshot &s,const std::array<uint32_t,48> &flags,bool replay) {
  uint64_t selected=0,zero=0;uint32_t mask=0;
  for (auto reason:flags) {
    selected+=reason!=0;zero+=reason==0;mask|=reason;
    if (replay) {
      add(s,RangeHeads,(reason&1)!=0);add(s,CancellationHeads,(reason&2)!=0);
      add(s,NonfiniteHeads,(reason&4)!=0);add(s,NormRangeHeads,(reason&8)!=0);
      add(s,UnknownReasonHeads,(reason&~15u)!=0);
      if (!(reason&~15u)) add(s,HistogramBegin+(reason&15u),1);
    }
  }
  if (replay) {add(s,ReplayCalls,1);add(s,ReplayedHeads,selected);add(s,UnflaggedHeads,zero);
    s.words[LastReplayedHeads]=selected;s.words[LastReasonMask]=mask;
  } else {add(s,PrepareCalls,1);add(s,ScheduledHeads,48);add(s,EligibleHeads,zero);
    add(s,AppliedHeads,zero);s.words[LastEligibleHeads]=zero;
  }
}
} // namespace
int main() {
  try {
    constexpr uint64_t coeff=164757504,snapshot=3145728,flags=192,arena=167919616;
    constexpr uint64_t offset=coeff+snapshot+flags;
    static_assert(offset%8==0 && offset+kBytes<=arena);
    Snapshot s;std::array<uint32_t,48> f{};
    observe(s,f,false);for (uint32_t i=0;i<48;++i) f[i]=i%16;observe(s,f,true);
    require(s[ScheduledHeads]==48 && s[EligibleHeads]==48 && s[AppliedHeads]==48,"first eligible phase");
    require(s[ReplayedHeads]==45 && s[UnflaggedHeads]==3,"first replay phase");
    for (auto value:s.histogram()) require(value==3,"exact mask histogram");
    require(s[RangeHeads]==24 && s[CancellationHeads]==24 && s[NonfiniteHeads]==24 && s[NormRangeHeads]==24,"overlapping reason counts");
    f.fill(0);observe(s,f,false);observe(s,f,true);
    require(s[ScheduledHeads]==96 && s[EligibleHeads]==96 && s[AppliedHeads]==96,"accumulation across reused flag arenas");
    require(s[ReplayedHeads]==45 && s[UnflaggedHeads]==51 && s[PrepareCalls]==2 && s[ReplayCalls]==2,"second phase accumulation");
    f.fill(1);observe(s,f,false);observe(s,f,true);
    require(s[ScheduledHeads]==144 && s[EligibleHeads]==96 && s[AppliedHeads]==96 && s[ReplayedHeads]==93,"range-ineligible heads stay sticky");
    f.fill(16);observe(s,f,true);require(s[UnknownReasonHeads]==48,"unknown reason preserved");
    const auto copied=readCompleted(s.words.data());require(copied.words==s.words,"native uint64 safe-point ABI");
    s.words[ScheduledHeads]=std::numeric_limits<uint64_t>::max()-1;add(s,ScheduledHeads,48);
    require(s[ScheduledHeads]==std::numeric_limits<uint64_t>::max() && s[Saturated]==1,"counter saturation never wraps");
    std::cout << "{\"pass\":true,\"gpu_work\":false,\"telemetry_bytes\":256,\"telemetry_offset\":" << offset
      << ",\"arena_bytes\":" << arena << ",\"accumulation_across_layers\":true,\"reason_masks_and_overlap\":true,\"safe_point_abi\":true}\n";
  } catch (const char *reason) { std::cerr<<reason<<'\n';return 1; }
}
