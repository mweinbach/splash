// Pure CPU policy checks. No backend, model, allocator or device header.
#include "bridge.hpp"
#include <array>
#include <iostream>
#include <limits>

namespace {
namespace policy=splash::flash::compact_native_r4_verify_sep22;
uint64_t checks=0;
void require(bool value,const char *message){++checks;if(!value)throw std::runtime_error(message);}
template<class Function>void rejects(Function f,const char *message){++checks;try{f();}
  catch(const std::invalid_argument &){return;}catch(const std::logic_error &){return;}throw std::runtime_error(message);}
void setFlag(const char *s){require((s ? ::setenv(policy::kFlag,s,1) : ::unsetenv(policy::kFlag))==0,"Cannot set CPU fixture environment");}
void pureChecks() {
  require(!policy::parseSwitch(nullptr)&&!policy::parseSwitch("0")&&policy::parseSwitch("1"),"Strict optional flag literal contract differs");
  require(policy::parseState(nullptr)==policy::SwitchState::Missing&&policy::parseState("0")==policy::SwitchState::Disabled,
      "Missing and explicit zero must freeze separately");
  for(const char *bad:{""," "," 0","0 "," 1","1 ","00","01","10","2","-1","+1","true","false","TRUE","yes","no","on","off","\t1","1\n","0\r","1.0","0x1"})
    rejects([&]{(void)policy::parseSwitch(bad);},"Malformed truthiness or whitespace was accepted");
  for(bool enabled:{false,true})for(bool verification:{false,true}) {
    for(uint32_t rows=0;rows<=8192;++rows)require(policy::eligibleFor(rows,verification,enabled)==(enabled&&rows==4&&verification),
        "Eligibility changed prefill/AR/non-R4 or disabled verification");
    for(uint32_t rows:{8193u,std::numeric_limits<uint32_t>::max()})require(!policy::eligibleFor(rows,verification,enabled),"Unsupported geometry became eligible");
  }
  require(policy::sourceIdentityValid(policy::kSourceIdentitySha256),"Generated identity format differs");
  for(const std::string &s:std::array<std::string,5>{"",std::string(63,'0'),std::string(65,'0'),std::string(64,'G'),std::string(64,'A')})
    require(!policy::sourceIdentityValid(s),"Malformed source identity was accepted");
  require(policy::markerFor(false).empty(),"Disabled marker changed inherited identity");
  const auto marker=policy::markerFor(true);
  for(std::string_view piece:{"physicalR4-singleton-main-verification-only","INTEGERONLY-compact-native-M16-exact-original-float-kernels-six-stages",
      "planner=expert_r4_compact_native_sep22_plan-parallel-SIMDcount-TG2432B","native-pack-GU-excludedPoison-prepareDown-down-unchanged","no-model-quality-claim"})
    require(marker.find(piece)!=std::string::npos,"Execution marker lacks scope, planner or unchanged native stages");
  require(marker.ends_with(policy::kSourceIdentitySha256),"Execution marker does not bind generated source identity");
  require(marker.find("softwaredot")==std::string::npos,"Rejected software-dot route entered identity");
  const policy::Counters counters;require(!counters.enabled&&!counters.planCalls&&!counters.planRows&&!counters.gateCalls&&!counters.gateRows&&!counters.downCalls&&!counters.downRows,
      "Graph counters must initialize disabled and zero");
  for(bool enabled:{false,true})for(uint32_t mask=0;mask<32;++mask)for(uint32_t cap:{0u,1u,3u,4u,5u,16u,8192u,std::numeric_limits<uint32_t>::max()}) {
    const bool full512=(mask&1)!=0,gather=(mask&2)!=0,blocked=(mask&4)!=0,directA=(mask&8)!=0,q4x8=(mask&16)!=0;
    const bool valid=mask==31&&cap==4;require(policy::dependenciesValid(full512,gather,cap,blocked,directA,q4x8)==valid,"Pure qualified dependency contract differs");
    if(enabled&&!valid)rejects([&]{policy::validateDependenciesFor(enabled,full512,gather,cap,blocked,directA,q4x8);},"Enabled unqualified config was accepted");
    else {policy::validateDependenciesFor(enabled,full512,gather,cap,blocked,directA,q4x8);require(true,"Qualified/off dependency configuration failed");}
  }
}
void runtimeChecks(bool enabled) {
  require(policy::requested()==enabled,"Frozen runtime flag differs");require(policy::implementationMarker()==policy::markerFor(enabled),"Runtime identity differs");
  for(uint32_t rows=0;rows<=8192;++rows)for(bool verification:{false,true})
    require(policy::eligible(rows,verification)==(enabled&&rows==4&&verification),"Runtime eligibility differs");
  policy::validateDependencies(true,true,4,true,true,true);require(true,"Qualified runtime dependencies failed");
  if(enabled)rejects([]{policy::validateDependencies(true,true,5,true,true,true);},"Runtime accepted cap different from4");
  else {policy::validateDependencies(false,false,0,false,false,false);require(true,"Disabled route altered inherited dependencies");}
}
void modeChecks(std::string_view mode) {
  bool enabled=false;const char *initial=nullptr;
  if(mode=="--freeze0")initial="0";
  else if(mode=="--freeze1"){initial="1";enabled=true;}
  else if(mode=="--missing")initial=nullptr;
  else if(mode=="--retry0"||mode=="--retry1") {
    setFlag("bad");rejects([]{(void)policy::requested();},"Invalid first initialization did not throw");
    enabled=mode=="--retry1";initial=enabled?"1":"0";
  }else throw std::invalid_argument("Unknown strict CPU process mode");
  setFlag(initial);runtimeChecks(enabled);
  for(const char *mutation:std::array<const char *,6>{nullptr,"0","1","bad"," ","1 "}) {
    if((!initial&&!mutation)||(initial&&mutation&&std::string_view(initial)==mutation))continue;
    setFlag(mutation);rejects([]{(void)policy::requested();},"Frozen environment changed");
    rejects([]{(void)policy::eligible(4,true);},"Verification dispatch ignored frozen mutation");
    rejects([]{(void)policy::eligible(2048,false);},"Inherited prefill ignored frozen mutation");
    rejects([]{policy::validateDependencies(true,true,4,true,true,true);},"Dependencies ignored frozen mutation");
    rejects([]{(void)policy::implementationMarker();},"Identity ignored frozen mutation");
    setFlag(initial);require(policy::requested()==enabled,"Exact frozen environment restoration failed");
  }
  std::cout<<"{\"kind\":\"compact_native_r4_verify_policy_cpu\",\"mode\":\""<<mode<<"\",\"enabled\":"<<(enabled?"true":"false")
      <<",\"checks\":"<<checks<<",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";
}
} // namespace
int main(int argc,char **argv) {
  try {
    pureChecks();if(argc==2){modeChecks(argv[1]);return 0;}
    if(argc!=1)throw std::invalid_argument("Use no arguments or --freeze0, --freeze1, --missing, --retry0, --retry1");
    const bool enabled=policy::parseSwitch(std::getenv(policy::kFlag));runtimeChecks(enabled);
    std::cout<<"{\"kind\":\"compact_native_r4_verify_policy_cpu\",\"rows_checked\":\"0..8192\",\"enabled\":"<<(enabled?"true":"false")
        <<",\"checks\":"<<checks<<",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";return 0;
  }catch(const std::exception &error){std::cerr<<"Compact verify CPU policy failed: "<<error.what()<<'\n';return 1;}
}
