#include "index.hpp"
#include "metadata.hpp"
#include "original_guard.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace guard_metadata;
namespace candidate=splash::flash::immutable_interval_index_sep22;
using Buffer=metal::MetalBuffer;
struct Result{bool accepted;int category,first;std::string message;};
uint64_t checks=0;
void require(bool yes,const char*text){if(!yes)throw std::runtime_error(text);++checks;}
template<class Call> Result outcome(Call call){
  counters={};countCalls=true;
  try{call();return {true,0,-1,{}};}
  catch(const std::invalid_argument&e){return {false,1,counters.currentImmutable,e.what()};}
  catch(const std::exception&e){return {false,2,counters.currentImmutable,e.what()};}
}
void equal(const Result&a,const Result&b){
  if(!(a.accepted==b.accepted&&a.category==b.category&&a.first==b.first&&a.message==b.message)){
    std::cerr<<"parity context checks="<<checks<<" old="<<a.accepted<<"/"<<a.category<<"/"<<a.first<<"/"<<a.message<<" new="<<b.accepted<<"/"<<b.category<<"/"<<b.first<<"/"<<b.message<<"\n";
    throw std::runtime_error("literal old/new admission category/text/first-failure mismatch");}
  ++checks;
}
template<class Address> std::array<candidate::Span<Address>,96> spans(const Original&source){
  std::array<candidate::Span<Address>,96> result{};
  for(size_t i=0;i<48;++i){result[2*i]={Address(reinterpret_cast<uintptr_t>(source.layers[i].base.contents())),source.layers[i].base.length};result[2*i+1]={Address(reinterpret_cast<uintptr_t>(source.layers[i].ranks.contents())),source.layers[i].ranks.length};}
  return result;
}
Original source(uint64_t offset=0x10000000,uint64_t spacing=0x100000){
  Original result;
  for(size_t i=0;i<48;++i){result.layers[i].base={uintptr_t(offset+2*i*spacing),65536,true,true,int(2*i)};
    result.layers[i].ranks={uintptr_t(offset+(2*i+1)*spacing),16384,true,true,int(2*i+1)};}
  return result;
}
template<class Address> void decision(const Original&source,const candidate::Index96<Address>&index,Buffer query){
  const auto old=outcome([&]{source.immutableDisjoint(query);});
  const auto now=outcome([&]{index.validate({Address(reinterpret_cast<uintptr_t>(query.contents())),query.length},[&]{source.immutableDisjoint(query);});});
  equal(old,now);
}
template<class Address> void campaign(){
  auto s=source();candidate::Index96<Address> index(spans<Address>(s));require(index.indexable(),"valid fixed96 index unavailable");
  for(const auto&layer:s.layers)for(const auto&b:{layer.base,layer.ranks}){
    for(uintptr_t address:std::array<uintptr_t,6>{b.address-1,b.address,b.address+1,uintptr_t(b.address+b.length-1),uintptr_t(b.address+b.length),uintptr_t(b.address+b.length+1)})
      for(uint64_t length:std::array<uint64_t,6>{0,1,2,16384,65536,UINT64_MAX})decision(s,index,{address,length});
    decision(s,index,{b.address-16,16});decision(s,index,{b.address-16,17});decision(s,index,{b.address-16,b.length+32});
  }
  for(uint64_t length:std::array<uint64_t,3>{0,1,UINT64_MAX})decision(s,index,{0,length});
  const auto maximum=std::numeric_limits<Address>::max();
  for(uint64_t length:std::array<uint64_t,5>{0,1,2,UINT64_MAX,uint64_t(maximum)})
    for(uintptr_t begin:{uintptr_t(maximum),uintptr_t(maximum-1),uintptr_t(1)})decision(s,index,{begin,length});
  // Sorting, duplicate starts, containment and overlapping sources.
  std::mt19937_64 random(0x964813);
  for(uint32_t layout=0;layout<24;++layout){
    auto varied=s;
    for(auto&layer:varied.layers)for(auto*b:{&layer.base,&layer.ranks}){
      b->address=uintptr_t(0x100000+random()%0x40000);b->length=1+random()%0x20000;
    }
    if(layout%3==0)varied.layers[2].base=varied.layers[1].base;
    if(layout%3==1){varied.layers[2].base.address=varied.layers[1].base.address;varied.layers[2].base.length*=2;}
    candidate::Index96<Address> x(spans<Address>(varied));require(x.indexable(),"valid overlap source index unavailable");
    for(uint32_t q=0;q<4096;++q)decision(varied,x,{uintptr_t(0x100000+random()%0x80000),random()%0x40000});
  }
  // Any bad source disables the WHOLE index, preserving earlier-failure order.
  for(uint32_t ordinal=0;ordinal<96;++ordinal)for(uint32_t bad=0;bad<4;++bad){
    auto broken=s;auto&b=ordinal%2?broken.layers[ordinal/2].ranks:broken.layers[ordinal/2].base;
    if(bad==0)b.address=0;if(bad==1)b.length=0;
    if(bad==2){b.address=uintptr_t(maximum);b.length=2;}
    if(bad==3){b.address=1;b.length=UINT64_MAX;}
    candidate::Index96<Address> x(spans<Address>(broken));require(!x.indexable(),"partially indexed invalid source table");
    for(Buffer query:std::array<Buffer,5>{{{1,1},{0,0},{s.layers[0].base.address,1},{uintptr_t(maximum),UINT64_MAX},{0x8000000,4096}}})decision(broken,x,query);
  }
  if constexpr(sizeof(Address)==4){
    auto tooLong=s;tooLong.layers[4].base.length=uint64_t(UINT32_MAX)+1;
    candidate::Index96<Address> x(spans<Address>(tooLong));require(!x.indexable(),"uint64 source length narrowed before32bit limit");
    decision(tooLong,x,{1,uint64_t(UINT32_MAX)+1});
  }
}
struct Roles{
  FlashMoEBlockedScratch scratch;Buffer input,ids,diag;
};
Roles roles(){
  Roles r;uintptr_t at=0x1000000;const auto b=[&](uint64_t size){Buffer out{at,size};at+=0x100000;return out;};
  r.scratch.buckets.counts=b(2048);r.scratch.buckets.offsets=b(2052);
  r.scratch.buckets.routeMap=b(160);r.scratch.buckets.canonicalToPacked=b(160);
  r.scratch.buckets.packedInputs=b(527360);r.scratch.buckets.jobOffsets=b(2052);
  r.scratch.buckets.jobCount=b(4);r.scratch.buckets.tileJobs=b(4112);
  r.scratch.packedActivated=b(131840);r.scratch.scatteredDown=b(204800);
  r.input=b(20480);r.ids=b(320);r.diag=b(4);return r;
}
std::array<Buffer*,13> pointers(Roles&r){
  auto&s=r.scratch;return {&s.buckets.counts,&s.buckets.offsets,&s.buckets.routeMap,&s.buckets.canonicalToPacked,
    &s.buckets.packedInputs,&s.buckets.jobOffsets,&s.buckets.jobCount,&s.buckets.tileJobs,
    &s.packedActivated,&s.scatteredDown,&r.diag,&r.input,&r.ids};
}
template<class Immutable> void complete(const Original&s,const Roles&r,uint32_t rows,uint32_t selections,Immutable imm){
  splash::flash::compact_r4_preflight_sep22::validateComplete(r.scratch,r.input,r.ids,s.layers[0].ranks,r.diag,rows,selections,
    [](const auto&s,auto d,uint32_t r,uint32_t n){allRowsScratch(s,d,r,FlashMoEBlockedTile::M16N64,n);},
    [](const auto&b,uint64_t n){requireBytes(b,n);},
    [](const auto&a,const auto&b){disjoint(a,b);},imm);
}
void fullGuards(){
  const auto s=source();const candidate::Index96<> index(spans<uintptr_t>(s));
  const auto test=[&](const Roles&r,uint32_t rows=4,uint32_t selections=10){
    equal(outcome([&]{complete(s,r,rows,selections,[&](const auto&b){s.immutableDisjoint(b);});}),
          outcome([&]{complete(s,r,rows,selections,[&](const auto&b){index.validate({reinterpret_cast<uintptr_t>(b.contents()),b.sizeBytes()},[&]{s.immutableDisjoint(b);});});}));
  };
  test(roles());
  for(uint32_t role=0;role<13;++role)for(uint32_t ordinal=0;ordinal<96;++ordinal)for(uint32_t offset:{0U,1U}){
    auto r=roles();auto&b=*pointers(r)[role];const auto&immutable=ordinal%2?s.layers[ordinal/2].ranks:s.layers[ordinal/2].base;
    b.address=immutable.address+offset;test(r);
  }
  for(uint32_t role=0;role<13;++role)for(uint32_t bad=0;bad<5;++bad){
    auto r=roles();auto&b=*pointers(r)[role];if(bad==0)b.present=false;if(bad==1)b.shared=false;if(bad==2)b.address=0;if(bad==3)b.length=0;if(bad==4)b.length=1;test(r);
  }
  for(uint32_t a=0;a<13;++a)for(uint32_t b=0;b<13;++b)if(a!=b){
    auto r=roles();auto p=pointers(r);p[a]->address=p[b]->address;test(r);
  }
  for(uint32_t field=0;field<4;++field){auto r=roles();switch(field){case 0:r.scratch.buckets.rowCapacity=3;break;
      case 1:r.scratch.buckets.selectionCapacity=9;break;case 2:r.scratch.buckets.routeCapacity=39;break;case 3:r.scratch.buckets.jobCapacity=513;break;}test(r);}
  for(uint32_t row:{0U,1U,4U,8193U})for(uint32_t selections:{0U,1U,10U,11U})test(roles(),row,selections);
}
volatile uint64_t witness=0;
template<class Call> double timed(Call call,uint32_t iterations){
  const auto begin=std::chrono::steady_clock::now();uint64_t checksum=0;
  for(uint32_t n=0;n<iterations;++n)for(uint32_t layer=0;layer<48;++layer)checksum+=call()+layer+n;
  witness=checksum;return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
void benchmark(){
  const auto s=source();const candidate::Index96<> index(spans<uintptr_t>(s));const auto r=roles();
  countCalls=true;counters={};complete(s,r,4,10,[&](const auto&b){s.immutableDisjoint(b);});const auto oldCalls=counters;
  counters={};complete(s,r,4,10,[&](const auto&b){index.validate({reinterpret_cast<uintptr_t>(b.contents()),b.sizeBytes()},[&]{s.immutableDisjoint(b);});});const auto newCalls=counters;
  require(oldCalls.contents==2666&&oldCalls.sizes==1340&&newCalls.contents==183&&newCalls.sizes==105,"actual original/new full13guard metadata call counts");
  countCalls=false;
  auto old=[&]{complete(s,r,4,10,[&](const auto&b){s.immutableDisjoint(b);});return uint64_t(13);};
  auto now=[&]{complete(s,r,4,10,[&](const auto&b){index.validate({reinterpret_cast<uintptr_t>(b.contents()),b.sizeBytes()},[&]{s.immutableDisjoint(b);});});return uint64_t(13);};
  (void)timed(old,1);(void)timed(now,1);double warmOld=0,warmNew=0;
  while(warmOld<50||warmNew<50){warmOld+=timed(old,8);warmNew+=timed(now,8);}
  std::vector<double>a,b;constexpr uint32_t iterations=32;
  for(uint32_t pair=0;pair<20;++pair){const bool first=pair%4==0||pair%4==3;
    if(first){a.push_back(timed(old,iterations));b.push_back(timed(now,iterations));}
    else{b.push_back(timed(now,iterations));a.push_back(timed(old,iterations));}}
  auto median=[](std::vector<double>x){std::sort(x.begin(),x.end());return (x[9]+x[10])/2;};
  std::cout<<",\"metadata_benchmark\":{\"warm_old_ms\":"<<warmOld<<",\"warm_new_ms\":"<<warmNew
    <<",\"balanced_pairs\":20,\"full13role_validation_layers_per_sample\":"<<48*iterations
    <<",\"old_median_ms\":"<<median(a)<<",\"new_median_ms\":"<<median(b)
    <<",\"adapter_speedup\":"<<median(a)/median(b)<<",\"checksum_witness\":"<<witness
    <<",\"actual_old_contents_calls_per_layer\":"<<oldCalls.contents<<",\"actual_new_contents_calls_per_layer\":"<<newCalls.contents
    <<",\"actual_old_size_calls_per_layer\":"<<oldCalls.sizes<<",\"actual_new_size_calls_per_layer\":"<<newCalls.sizes
    <<",\"native_ObjC_or_GPU_or_whole_decode_attribution\":false}";
  const auto scenario=[&](const char*name,auto legacy,auto indexed){
    double wa=0,wb=0;while(wa<50||wb<50){wa+=timed(legacy,8);wb+=timed(indexed,8);}
    std::vector<double>x,y;for(uint32_t pair=0;pair<20;++pair){
      const bool first=pair%4==0||pair%4==3;
      if(first){x.push_back(timed(legacy,16));y.push_back(timed(indexed,16));}
      else{y.push_back(timed(indexed,16));x.push_back(timed(legacy,16));}}
    std::cout<<",\""<<name<<"\":{\"warm_old_ms\":"<<wa<<",\"warm_new_ms\":"<<wb
      <<",\"balanced_pairs\":20,\"old_median_ms\":"<<median(x)<<",\"new_median_ms\":"<<median(y)
      <<",\"adapter_speedup\":"<<median(x)/median(y)<<",\"checksum_witness\":"<<witness
      <<",\"native_attribution\":false}";
  };
  auto rejected=r;rejected.input.address=s.layers[47].ranks.address+1;
  auto rejection=[&](bool fast){try{complete(s,rejected,4,10,[&](const auto&b){
      if(fast)index.validate({reinterpret_cast<uintptr_t>(b.contents()),b.sizeBytes()},[&]{s.immutableDisjoint(b);});
      else s.immutableDisjoint(b);});return uint64_t(0);}
    catch(const std::invalid_argument&){return uint64_t(1);}};
  scenario("rejected_overlap_full13guard",[&]{return rejection(false);},[&]{return rejection(true);});
  const Buffer point{s.layers[0].base.address,0};
  auto zero=[&](bool fast){for(uint32_t role=0;role<13;++role){
      if(fast)index.validate({reinterpret_cast<uintptr_t>(point.contents()),point.sizeBytes()},[&]{s.immutableDisjoint(point);});
      else s.immutableDisjoint(point);}return uint64_t(13);};
  scenario("zero_length_original_fallback_immutable13queries",[&]{return zero(false);},[&]{return zero(true);});
}
int main(int argc,char**argv){
  try{
    campaign<uint64_t>();campaign<uint32_t>();fullGuards();
    std::cout<<std::setprecision(16)<<"{\"pass\":true,\"decision_checks\":"<<checks
      <<",\"actual_shipping_header_used\":true,\"literal_old_separate_TU\":true,\"model_or_device_or_data_payload_access\":false";
    if(argc==2&&std::string(argv[1])=="--benchmark")benchmark();
    else require(argc==1,"oracle [--benchmark]");
    std::cout<<",\"GPU_executed\":false,\"Store_or_worker_integration\":false}\n";return 0;
  }catch(const std::exception&e){std::cerr<<"{\"pass\":false,\"reason\":\""<<e.what()<<"\"}\n";return 1;}
}
