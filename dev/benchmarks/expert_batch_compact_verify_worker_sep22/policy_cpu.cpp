// Pure CPU selector/dependency/frozen flag checks; no device/model/allocator.
#include "bridge.hpp"
#include <array>
#include <iostream>
#include <limits>
namespace {
namespace p=splash::flash::compact_native_batch_verify_sep22;
uint64_t checks=0;
void require(bool v,const char*m){++checks;if(!v)throw std::runtime_error(m);}
template<class F>void rejects(F f,const char*m){++checks;try{f();}catch(const std::invalid_argument&){return;}catch(const std::logic_error&){return;}throw std::runtime_error(m);}
void setFlag(const char*s){require((s ? ::setenv(p::kFlag,s,1) : ::unsetenv(p::kFlag))==0,"CPU env set failed");}
void pure(){
 require(!p::parseSwitch(nullptr)&&!p::parseSwitch("0")&&p::parseSwitch("1"),"strict switch");
 require(p::parseState(nullptr)!=p::parseState("0"),"missing differs from explicit zero");
 for(const char*s:{""," "," 0","0 "," 1","1 ","00","01","10","2","-1","+1","true","false","TRUE","yes","no","on","off","\t1","1\n","0\r","1.0","0x1"})rejects([&]{(void)p::parseSwitch(s);},"malformed flag accepted");
 for(bool enabled:{false,true})for(bool verify:{false,true})for(uint32_t lanes=0;lanes<=8;++lanes)for(uint32_t rows=0;rows<=8192;++rows)
  require(p::eligibleFor(lanes,rows,verify,enabled)==(enabled&&verify&&rows==4&&(lanes==2||lanes==4)),"ineligible caller selected");
 require(!p::eligibleFor(UINT32_MAX,4,true,true)&&!p::eligibleFor(2,UINT32_MAX,true,true),"overflow geometry selected");
 require(p::sourceIdentityValid(p::kSourceIdentitySha256),"generated identity");
 for(const auto&s:std::array<std::string,5>{"",std::string(63,'0'),std::string(65,'0'),std::string(64,'G'),std::string(64,'A')})require(!p::sourceIdentityValid(s),"malformed identity accepted");
 require(p::markerFor(false).empty()&&p::markerFor(true).ends_with(p::kSourceIdentitySha256),"default off marker or provenance differs");
 for(bool enabled:{false,true})for(uint32_t mask=0;mask<64;++mask)for(uint32_t cap:{0u,1u,3u,4u,5u,16u,8192u,UINT32_MAX}){
  bool a=mask&1,b=mask&2,c=mask&4,d=mask&8,e=mask&16,f=mask&32;bool valid=mask==63&&cap==4;
  require(p::dependenciesValid(a,b,cap,c,d,e,f)==valid,"qualified dependency tuple differs");
  if(enabled&&!valid)rejects([&]{p::validateDependenciesFor(enabled,a,b,cap,c,d,e,f);},"unqualified config accepted");
  else{p::validateDependenciesFor(enabled,a,b,cap,c,d,e,f);require(true,"qualified/off config");}
 }
 const p::Counters s;require(!s.enabled&&!s.r8.planCalls&&!s.r8.planRows&&!s.r8.gateCalls&&!s.r8.downCalls&&!s.r16.planCalls&&!s.r16.planRows,"counters nonzero");
}
void runtime(bool enabled){
 require(p::requested()==enabled,"frozen flag differs");
 for(uint32_t lanes=0;lanes<=4;++lanes)for(uint32_t rows=0;rows<=4;++rows)require(p::eligible(lanes,rows,true)==(enabled&&rows==4&&(lanes==2||lanes==4)),"actual active span policy differs");
 p::validateDependencies(true,true,4,true,true,true,true);require(true,"qualified runtime tuple");
 require(p::implementationMarker()==p::markerFor(enabled),"runtime marker differs");
}
}
int main(int argc,char**argv){try{
 pure();std::string_view mode=argc==2?argv[1]:"--missing";const char*initial=nullptr;bool enabled=false;
 if(mode=="--freeze0")initial="0";else if(mode=="--freeze1"){initial="1";enabled=true;}else if(mode=="--missing")initial=nullptr;
 else if(mode=="--retry0"||mode=="--retry1"){setFlag("bad");rejects([]{(void)p::requested();},"invalid initialization accepted");enabled=mode=="--retry1";initial=enabled?"1":"0";}else throw std::invalid_argument("unknown CPU mode");
 setFlag(initial);runtime(enabled);
 for(const char*mutation:std::array<const char*,6>{nullptr,"0","1","bad"," ","1 "}){
  if((!initial&&!mutation)||(initial&&mutation&&std::string_view(initial)==mutation))continue;
  setFlag(mutation);rejects([]{(void)p::requested();},"frozen policy mutation accepted");rejects([]{(void)p::eligible(2,4,true);},"dispatch ignored mutation");rejects([]{p::validateDependencies(true,true,4,true,true,true,true);},"dependency ignored mutation");rejects([]{(void)p::implementationMarker();},"marker ignored mutation");setFlag(initial);require(p::requested()==enabled,"restoration failed");
 }
 std::cout<<"{\"kind\":\"compact_native_batch_verify_policy_cpu\",\"mode\":\""<<mode<<"\",\"enabled\":"<<(enabled?"true":"false")<<",\"checks\":"<<checks<<",\"gpu_work\":false,\"payload_reads\":false,\"pass\":true}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
