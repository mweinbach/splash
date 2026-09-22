// Device-free actual current BQSA4 policy checks; no model/device creation.
#include "dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp"
#include <iostream>
namespace p=splash::flash::batch_prefill_twopass_sep22;
uint64_t checks=0;
void req(bool v){++checks;if(!v)throw std::runtime_error("BQSA CPU contract mismatch");}
template<class F>void rejects(F f){++checks;try{f();}catch(const std::invalid_argument&){return;}catch(const std::logic_error&){return;}throw std::runtime_error("BQSA missing refusal");}
void setFlag(const char*s){if(s)::setenv(p::flag,s,1);else::unsetenv(p::flag);}
int main(int argc,char**argv){try{
 for(bool enabled:{false,true})for(bool fresh:{false,true})for(uint32_t lanes=0;lanes<=8;++lanes)for(uint32_t rows=0;rows<=8192;++rows)req(p::eligible(lanes,rows,fresh,enabled)==(enabled&&fresh&&lanes==4&&rows==2048));
 for(const char*s:{"","2","true"," 1","1 ","01"})rejects([&]{(void)p::state(s);});
 req(p::state(nullptr)!=p::state("0"));std::string_view mode=argc==2?argv[1]:"--missing";const char*initial=nullptr;bool enabled=false;
 if(mode=="--freeze1"||mode=="--retry1"){initial="1";enabled=true;}else if(mode=="--freeze0")initial="0";else if(mode!="--missing")throw std::invalid_argument("unknown CPU mode");
 if(mode=="--retry1"){setFlag("bad");rejects([]{(void)p::requested();});}
 for(const char *dep:{"SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_BATCH_QSA_BULK_PREFILL","SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21","SPLASH_FLASH_QSA_F32","SPLASH_FLASH_QSA_MPP","SPLASH_FLASH_QSA_ROW_TILES","SPLASH_FLASH_QSA_BULK_PREFILL","SPLASH_FLASH_QSA_BULK_PREFILL_SG8","SPLASH_FLASH_MTP","SPLASH_FLASH_BATCH_MTP","SPLASH_FLASH_BATCH_MTP_PREFILL","SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY","SPLASH_FLASH_BATCH_MTP_TEACHER_CACHE_ONLY"})::setenv(dep,"1",1);
 ::setenv("SPLASH_FLASH_MTP_DRAFT_DEPTH","3",1);setFlag(initial);req(p::requested()==enabled);
 for(const char*m:{"bad"," ","1 "}){setFlag(m);rejects([]{(void)p::requested();});setFlag(initial);req(p::requested()==enabled);}
 setFlag(enabled?"0":"1");rejects([]{(void)p::requested();});setFlag(initial);req(p::requested()==enabled);
 std::cout<<"{\"kind\":\"B4only-BQSA-policy-CPU\",\"mode\":\""<<mode<<"\",\"checks\":"<<checks<<",\"pass\":true,\"GPU_work\":false,\"payload_reads\":false}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
