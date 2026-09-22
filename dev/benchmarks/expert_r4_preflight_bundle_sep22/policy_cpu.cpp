#include "policy.hpp"
#include <iostream>
#include <array>
namespace p=splash::flash::compact_r4_preflight_sep22;
namespace {
uint64_t checks=0;
void require(bool v){++checks;if(!v)throw std::runtime_error("CPU policy assertion failed");}
template<class F>void reject(F fn){++checks;try{fn();}catch(const std::logic_error&){return;}throw std::runtime_error("CPU policy accepted invalid metadata");}
void set(const char *value){require((value?setenv(p::kFlag,value,1):unsetenv(p::kFlag))==0);}
}
int main(int argc,char **argv) {
 try {
  require(p::parse(nullptr)==p::State::Missing);require(p::parse("0")==p::State::Disabled);require(p::parse("1")==p::State::Enabled);
  for(const char *bad:{""," ","00","01","true","false","+1","-1","2","1 "," 1","1\n"})reject([&]{(void)p::parse(bad);});
  if(argc>2)throw std::invalid_argument("Use no arguments or --missing, --freeze0, --freeze1, --retry0 or --retry1");
  const std::string_view mode=argc==1?"--default":argv[1];const char *value=nullptr;bool enabled=false;
  const char *raw=std::getenv(p::kFlag);const std::string initial=raw?raw:"";
  if(mode=="--default"){value=raw?initial.c_str():nullptr;enabled=p::parse(value)==p::State::Enabled;}
  else if(mode=="--freeze0")value="0";
  else if(mode=="--freeze1"){value="1";enabled=true;}
  else if(mode=="--missing")value=nullptr;
  else if(mode=="--retry0"||mode=="--retry1"){set("bad");reject([]{(void)p::requested();});enabled=mode=="--retry1";value=enabled?"1":"0";}
  else throw std::invalid_argument("Unknown CPU policy mode");
  set(value);require(p::requested()==enabled);p::validateDependencies(true);
  if(enabled)reject([]{p::validateDependencies(false);});else p::validateDependencies(false);
  require(p::marker().empty()==!enabled);
  if(enabled)require(p::marker().ends_with(p::kPreflightSourceIdentitySha256));
  for(const char *mutation:std::array<const char *,6>{nullptr,"0","1","bad"," ","1 "}) {
   if((!value&&!mutation)||(value&&mutation&&std::string_view(value)==mutation))continue;
   set(mutation);reject([]{(void)p::requested();});reject([]{p::validateDependencies(true);});reject([]{(void)p::marker();});set(value);require(p::requested()==enabled);
  }
  std::cout<<"{\"mode\":\""<<mode<<"\",\"checks\":"<<checks<<",\"pass\":true,\"GPU_work\":false,\"payload_reads\":false}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
