// CPU-only strict-policy qualification. No backend, model or device headers.
#include "bridge.hpp"
#include <array>
#include <iostream>
#include <limits>

namespace {
namespace policy=splash::flash::gemv_decode_r1_sep21;
uint64_t checks=0;
void require(bool value,const char *message) {
  ++checks;
  if(!value) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function,const char *message) {
  ++checks;
  try {function();} catch(const std::invalid_argument &) {return;}
    catch(const std::logic_error &) {return;}
  throw std::runtime_error(message);
}
void setFlag(const char *value) {
  const int result=value ? ::setenv(policy::kFlag,value,1) : ::unsetenv(policy::kFlag);
  require(result==0,"Cannot set strict CPU fixture environment");
}
void parserChecks() {
  require(!policy::parseSwitch(nullptr),"Missing flag must be disabled");
  require(!policy::parseSwitch("0"),"Literal zero must be disabled");
  require(policy::parseSwitch("1"),"Literal one must be enabled");
  require(policy::parseState(nullptr)==policy::SwitchState::Missing,"Optional missing state was collapsed");
  require(policy::parseState("0")==policy::SwitchState::Disabled,"Explicit zero state differs");
  require(policy::parseState("1")==policy::SwitchState::Enabled,"Explicit one state differs");
  for(const char *bad:{""," "," 0","0 "," 1","1 ","00","01","10","2","-1","+1",
      "true","false","TRUE","False","yes","no","on","off","\t1","1\n","0\r","1.0","0x1"})
    rejects([&]{(void)policy::parseSwitch(bad);},"Malformed truthy/whitespace flag was accepted");
}
void identityChecks() {
  require(policy::sourceIdentityValid(policy::kSourceIdentitySha256),"Generated identity must be 64 lowercase hexadecimal characters");
  for(const std::string &bad:std::array<std::string,5>{"",std::string(63,'0'),std::string(65,'0'),std::string(64,'G'),std::string(64,'A')})
    require(!policy::sourceIdentityValid(bad),"Malformed source identity was accepted");
  require(policy::markerFor(false).empty(),"Flag zero changed base identity");
  const auto enabled=policy::markerFor(true);
  require(enabled.starts_with(";private-R1-nonverification-decode-only-numerical-alternative"),"Enabled marker lacks the qualified R1 scope");
  for(std::string_view part:{"direct-BF16xI8-vector4-L32O4-fourpartials-descendingXOR",
      "kernel=gemv_decode_sep21_v4_l32_o4","cert=RNorRTZ-u23-FTZ4lambda-lateF32scale-BF16SwiGLU-v1b"})
    require(enabled.find(part)!=std::string::npos,"Enabled marker lost kernel or numerical certificate semantics");
  require(enabled.ends_with(policy::kSourceIdentitySha256),"Enabled marker does not bind generated source SHA256");
  require(enabled.find("l16_o8")==std::string::npos,"Unqualified R4/L16 producer entered R1 identity");
  const policy::Counters counters;
  require(!counters.enabled&&!counters.gateCalls&&!counters.gateRows&&!counters.downCalls&&!counters.downRows,
      "Graph construction counters must initialize disabled and empty");
}
void exhaustiveEligibilityChecks() {
  for(bool enabled:{false,true}) for(bool verification:{false,true}) {
    for(uint32_t rows=1;rows<=8192;++rows)
      require(policy::eligibleFor(rows,verification,enabled)==(enabled&&rows==1&&!verification),
          "Eligibility changed verification or R>=2 base dispatch");
    for(uint32_t rows:{0u,8193u,std::numeric_limits<uint32_t>::max()})
      require(!policy::eligibleFor(rows,verification,enabled),"Unsupported row geometry became eligible");
  }
}
void runtimeChecks(bool enabled) {
  require(policy::requested()==enabled,"Frozen runtime state differs");
  require(policy::implementationMarker()==policy::markerFor(enabled),"Runtime identity differs from strict policy");
  for(uint32_t rows=1;rows<=8192;++rows) for(bool verification:{false,true})
    require(policy::eligible(rows,verification)==(enabled&&rows==1&&!verification),"Runtime eligibility differs");
}
void freezeChecks(std::string_view mode) {
  const char *initial=nullptr;
  bool enabled=false;
  if(mode=="--freeze0") initial="0";
  else if(mode=="--freeze1") {initial="1";enabled=true;}
  else if(mode=="--missing") initial=nullptr;
  else if(mode=="--retry0"||mode=="--retry1") {
    setFlag("bad");
    rejects([]{(void)policy::requested();},"Invalid first initialization did not throw");
    enabled=mode=="--retry1"; initial=enabled?"1":"0";
  } else throw std::invalid_argument("Unknown CPU policy process mode");
  setFlag(initial); runtimeChecks(enabled);
  for(const char *mutation:std::array<const char *,6>{nullptr,"0","1","bad"," ","1 "}) {
    if((!initial&&!mutation)||(initial&&mutation&&std::string_view(initial)==mutation)) continue;
    setFlag(mutation);
    rejects([]{(void)policy::requested();},"Post-freeze environment mutation was accepted");
    rejects([]{(void)policy::eligible(1,false);},"Eligible decode ignored changed frozen environment");
    rejects([]{(void)policy::eligible(2,true);},"Base verification path ignored changed frozen environment");
    rejects([]{(void)policy::implementationMarker();},"Runtime identity ignored changed frozen environment");
    setFlag(initial);
    require(policy::requested()==enabled,"Restoring exact frozen environment failed");
  }
  std::cout<<"{\"kind\":\"gemv_decode_r1_policy_cpu\",\"mode\":\""<<mode
      <<"\",\"enabled\":"<<(enabled?"true":"false")<<",\"checks\":"<<checks
      <<",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";
}
} // namespace
int main(int argc,char **argv) {
  try {
    parserChecks(); identityChecks(); exhaustiveEligibilityChecks();
    if(argc==2) {freezeChecks(argv[1]);return 0;}
    if(argc!=1) throw std::invalid_argument("Use no arguments or --freeze0, --freeze1, --missing, --retry0, --retry1");
    const bool enabled=policy::parseSwitch(std::getenv(policy::kFlag)); runtimeChecks(enabled);
    std::cout<<"{\"kind\":\"gemv_decode_r1_policy_cpu\",\"rows_checked\":\"1..8192\",\"enabled\":"
      <<(enabled?"true":"false")<<",\"checks\":"<<checks
      <<",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";
    return 0;
  } catch(const std::exception &error) {
    std::cerr<<"R1 GEMV policy CPU check failed: "<<error.what()<<'\n';return 1;
  }
}
