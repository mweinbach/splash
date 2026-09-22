#import <Foundation/Foundation.h>
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashMoE.h"
#include "metal/abi/FlashMoEBlocked.h"
#include "engine/Json.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

namespace {
using namespace splash::metal;
void require(bool v,const std::string &why) {if(!v) throw std::runtime_error(why);}
constexpr uint32_t sticky=0x80000080u;
constexpr uint64_t guardBytes=64;
struct Guarded {
  MetalBuffer base,view;uint64_t bytes;
  std::vector<std::byte> frozen;
  Guarded(MetalBackend &b,uint64_t n):bytes(n) {
    require(n>0,"zero buffer extent");base=b.allocateBuffer(n+2*guardBytes,BufferStorage::Shared);
    view=b.view(base,guardBytes,n);std::memset(base.contents(),0xa5,n+2*guardBytes);
  }
  void guards() const {
    const auto *p=static_cast<const unsigned char *>(base.contents());
    for(uint64_t i=0;i<guardBytes;++i)
      require(p[i]==0xa5 && p[guardBytes+bytes+i]==0xa5,"pointwise canary changed");
  }
  void freeze() {const auto *p=static_cast<const std::byte *>(view.contents());frozen={p,p+bytes};}
  void immutable() const {guards();require(frozen.size()==bytes && std::memcmp(view.contents(),frozen.data(),bytes)==0,"input changed");}
  std::vector<std::byte> snapshot() const {const auto *p=static_cast<const std::byte *>(view.contents());return {p,p+bytes};}
};
uint16_t bf16(float v) {
  uint32_t bits=std::bit_cast<uint32_t>(v);
  if((bits&0x7f800000u)==0x7f800000u && (bits&0x007fffffu))return 0x7fc0;
  return uint16_t((bits+0x7fffu+((bits>>16)&1))>>16);
}
float fp32(uint16_t v) {return std::bit_cast<float>(uint32_t(v)<<16);}
uint16_t plus(uint16_t a,uint16_t b) {return bf16(fp32(a)+fp32(b));}
uint16_t times(uint16_t a,uint16_t b) {return bf16(fp32(a)*fp32(b));}
uint16_t reduce(const uint16_t *down,const uint16_t *weights,uint32_t slots) {
  uint16_t p[8]{};
  for(uint32_t y=0;y<8;++y) for(uint32_t s=y;s<slots;s+=8)p[y]=plus(p[y],times(down[s],weights[s]));
  uint16_t v=p[0];for(uint32_t y=1;y<8;++y)v=plus(v,p[y]);return v;
}
enum class Fault {None,NegativeID,HugeID,Duplicate08,Duplicate19,WeightNegative,WeightAboveOne,
  WeightNaN,WeightInf,DownNaN,ZeroTimesInf,SharedNaN,GateNaN,GateInf,Overflow,MixedFaults,Cancellation};
const char *faultName(Fault f) {
  constexpr const char *names[]={"finite","negative_id","huge_id","duplicate_0_8","duplicate_1_9",
      "negative_weight","weight_above_one","weight_nan","weight_inf","down_nan","zero_times_inf",
      "shared_nan","gate_nan","gate_inf","sum_overflow","mixed_id_down_faults","finite_cancellation"};
  return names[uint32_t(f)];
}
struct CombineFixture {
  FlashMoEPointwiseParams p;uint32_t seed;
  Guarded down,ids,weights,shared,gate,out,diag;
  CombineFixture(MetalBackend &b,uint32_t rows,uint32_t width,uint32_t slots,Fault fault,uint32_t diagSeed=sticky)
      :p{rows,width,slots,512},seed(diagSeed),down(b,uint64_t(rows)*width*slots*2),ids(b,uint64_t(rows)*slots*8),
       weights(b,uint64_t(rows)*slots*2),shared(b,uint64_t(rows)*width*2),gate(b,uint64_t(rows)*2),
       out(b,uint64_t(rows)*width*2),diag(b,4) {
    auto *d=static_cast<uint16_t *>(down.view.contents()),*w=static_cast<uint16_t *>(weights.view.contents());
    auto *sh=static_cast<uint16_t *>(shared.view.contents()),*g=static_cast<uint16_t *>(gate.view.contents());
    auto *id=static_cast<int64_t *>(ids.view.contents());
    constexpr uint16_t gates[]={0x0000,0x8000,0x3f00,0xbf00,0x3f80,0xbf80,0x4280,0xc280,0x42c8,0xc2c8};
    for(uint32_t row=0;row<rows;++row) {
      g[row]=gates[row%10];
      for(uint32_t slot=0;slot<slots;++slot) {
        const uint64_t route=uint64_t(row)*slots+slot;
        id[route]=(row*11+slot)%512;w[route]=slot%4==0?0x8000:bf16(float((slot%3)+1)/8);
        for(uint32_t c=0;c<width;++c)d[route*width+c]=bf16(float(int((row*17+slot*13+c*7)%129)-64)/16);
      }
      for(uint32_t c=0;c<width;++c)sh[uint64_t(row)*width+c]=bf16(float(int((row*7+c*3)%65)-32)/16);
    }
    // Faults affect one row/column, exposing both local poisoning and shared row metadata.
    const uint32_t row=rows-1;const uint64_t r=uint64_t(row)*slots,index=uint64_t(row)*width+width-1;
    switch(fault) {
    case Fault::None:break;
    case Fault::NegativeID:id[r]=-1;break;
    case Fault::HugeID:id[r]=INT64_MAX;break;
    case Fault::Duplicate08:require(slots>8,"duplicate08 fixture K");id[r+8]=id[r];break;
    case Fault::Duplicate19:require(slots>9,"duplicate19 fixture K");id[r+9]=id[r+1];break;
    case Fault::WeightNegative:w[r]=0x8001;break;
    case Fault::WeightAboveOne:w[r]=0x3f81;break;
    case Fault::WeightNaN:w[r]=0x7fa1;break;
    case Fault::WeightInf:w[r]=0x7f80;break;
    case Fault::DownNaN:d[r*width+width-1]=0xffa1;break;
    case Fault::ZeroTimesInf:w[r]=0;d[r*width+width-1]=0x7f80;break;
    case Fault::SharedNaN:sh[index]=0x7fa1;break;
    case Fault::GateNaN:g[row]=0xffa1;break;
    case Fault::GateInf:g[row]=0x7f80;break;
    case Fault::Overflow:
      for(uint32_t s=0;s<slots;++s) {w[r+s]=0x3f80;d[(r+s)*width+width-1]=0x7f7f;}break;
    case Fault::MixedFaults:id[r]=INT64_MIN;w[r]=0x8001;d[r*width+width-1]=0x7f80;break;
    case Fault::Cancellation: {
      require(slots==10,"cancellation fixture K");
      const uint16_t cancellation[10]={0x4300,0xc300,0x3f80,0,0,0,0,0,0x3f00,0x3f00};
      for(uint32_t s=0;s<10;++s){w[r+s]=0x3f80;d[(r+s)*width+width-1]=cancellation[s];}
      sh[index]=0;break;
    }
    }
    for(auto *x:{&down,&ids,&weights,&shared,&gate})x->freeze();reset();
  }
  void reset() {std::memset(out.view.contents(),0x59,out.bytes);*static_cast<uint32_t *>(diag.view.contents())=seed;}
  void guards() const {for(const auto *x:{&down,&ids,&weights,&shared,&gate,&out,&diag})x->guards();}
  void immutable() const {for(const auto *x:{&down,&ids,&weights,&shared,&gate})x->immutable();}
  CommandGraph graph(const char *name,bool extra=false,uint32_t threads=256) {
    const uint64_t columns=std::strcmp(name,"private_moe_combine_cta_row")==0?1:(p.width+255)/256;
    CommandGraph g;g.add(name,{down.view,ids.view,weights.view,shared.view,gate.view,out.view,diag.view},p,
        {columns+(extra?1:0),p.rows+(extra?1:0),1},{threads,1,1});return g;
  }
};
struct PoisonFixture {
  FlashMoEBlockedDownParams p;uint32_t seed;Guarded inverse,out,diag;std::vector<std::byte> initial;
  PoisonFixture(MetalBackend &b,uint32_t rows,uint32_t mode,uint32_t diagSeed=sticky)
      :p{{rows,10,640,2560,512,0,0,0,320,uint64_t(2560)*320,20,uint64_t(2560)*20},rows*10,rows*10,16,0},
       seed(diagSeed),inverse(b,uint64_t(rows)*10*4),out(b,uint64_t(rows)*10*2560*2),diag(b,4) {
    auto *inv=static_cast<uint32_t *>(inverse.view.contents());auto *o=static_cast<uint16_t *>(out.view.contents());
    for(uint32_t r=0;r<p.route_capacity;++r) {
      inv[r]=r;
      if(mode==1 || (mode==2 && r%5==0))inv[r]=r%3==0?UINT32_MAX:p.route_capacity+(r%2);
      for(uint32_t c=0;c<2560;++c)o[uint64_t(r)*2560+c]=uint16_t((r*43+c*13)%0x10000);
    }
    inverse.freeze();initial=out.snapshot();reset();
  }
  void reset() {std::memcpy(out.view.contents(),initial.data(),initial.size());*static_cast<uint32_t *>(diag.view.contents())=seed;}
  void verifyExpected() const {
    const auto *inv=static_cast<const uint32_t *>(inverse.view.contents());
    const auto *o=static_cast<const uint16_t *>(out.view.contents());
    const auto *before=reinterpret_cast<const uint16_t *>(initial.data());bool invalid=false;
    for(uint32_t r=0;r<p.route_capacity;++r) {
      const bool excluded=inv[r]>=p.route_capacity;invalid|=excluded;
      for(uint32_t c=0;c<2560;++c)require(o[uint64_t(r)*2560+c]==(excluded?0x7fc0:before[uint64_t(r)*2560+c]),"poison canonical bits or untouched bytes differ");
    }
    require(*static_cast<const uint32_t *>(diag.view.contents())==(seed|(invalid?5u:0u)),"poison diagnostics differ");
  }
  void guards() const {inverse.guards();out.guards();diag.guards();}
  CommandGraph graph(const char *name,uint32_t threads,uint32_t columns,bool extra=false) {
    CommandGraph g;g.add(name,{inverse.view,out.view,diag.view},p,
        {columns+(extra?1:0),p.route_capacity+(extra?1:0),1},{threads,1,1});return g;
  }
};
void valid(CommandTiming t) {require(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>0&&t.gpuSeconds<60&&std::isfinite(t.wallSeconds)&&t.wallSeconds>0,"invalid timing ABI");}
double median(std::vector<double> v) {
  require(!v.empty(),"empty timing sample");std::sort(v.begin(),v.end());const auto i=v.size()/2;
  return v.size()%2?v[i]:(v[i-1]+v[i])/2;
}
void compare(const Guarded &out,const Guarded &diag,const std::vector<std::byte> &expected,uint32_t d,const std::string &name) {
  if(std::memcmp(out.view.contents(),expected.data(),out.bytes)) {
    const auto *a=static_cast<const uint16_t *>(out.view.contents());const auto *b=reinterpret_cast<const uint16_t *>(expected.data());
    for(uint64_t i=0;i<out.bytes/2;++i)if(a[i]!=b[i])
      throw std::runtime_error(name+" BF16 mismatch index="+std::to_string(i)+" expected="+std::to_string(b[i])+" observed="+std::to_string(a[i]));
  }
  require(*static_cast<const uint32_t *>(diag.view.contents())==d,name+" diagnostics differ");
}
template<class Fixture,size_t N> void rotatedTimings(MetalBackend &b,Fixture &f,
    const std::array<CommandGraph,N> &graphs,const std::array<const char *,N> &names,
    std::ostream &report) {
  f.reset();valid(b.submitCommand(graphs[0].dispatches()));f.guards();
  const auto expected=f.out.snapshot();const auto d=*static_cast<const uint32_t *>(f.diag.view.contents());
  std::array<std::vector<double>,N> gpu,wall;
  report<<"\"orders\":[";
  for(uint32_t cycle=0;cycle<11;++cycle) {
    if(cycle>=3) {if(cycle>3)report<<',';report<<'[';}
    for(uint32_t k=0;k<N;++k) {
      const uint32_t kind=(cycle%2==0?(k+cycle)%N:(cycle+N-k)%N);
      f.reset();const auto t=b.submitCommand(graphs[kind].dispatches());valid(t);
      compare(f.out,f.diag,expected,d,names[kind]);f.guards();
      if(cycle>=3) {if(k)report<<',';report<<kind;gpu[kind].push_back(t.gpuSeconds*1000);wall[kind].push_back(t.wallSeconds*1000);}
    }
    if(cycle>=3)report<<']';
  }
  report<<"],\"variants\":[";
  for(uint32_t k=0;k<N;++k) {
    if(k)report<<',';std::vector<double> ratios;uint32_t positive=0;
    for(uint32_t i=0;i<gpu[k].size();++i) {const double r=gpu[0][i]/gpu[k][i];ratios.push_back(r);positive+=r>1;}
    report<<"{\"name\":"<<splash::json::quote(names[k])<<",\"median_gpu_ms\":"<<median(gpu[k])
        <<",\"median_wall_ms\":"<<median(wall[k])<<",\"median_paired_speedup\":"<<median(ratios)
        <<",\"positive_pairs\":"<<positive<<",\"gpu_ms\":[";
    for(uint32_t i=0;i<gpu[k].size();++i) {if(i)report<<',';report<<gpu[k][i];}report<<"]}";
    std::cerr<<names[k]<<" medianGPUms="<<median(gpu[k])<<" pairedSpeedup="<<median(ratios)<<'\n';
  }
  report<<']';
}
constexpr std::array<const char *,5> combineNames{{"flash_moe_combine","private_moe_combine_simd_lane0","private_moe_combine_simd_slots","private_moe_combine_cta","private_moe_combine_cta_row"}};
constexpr std::array<const char *,3> poisonNames{{"flash_moe_blocked_poison_excluded_routes","private_moe_poison_route256","private_moe_poison_route32"}};
void cpuTest() {
  uint64_t checks=0;
  for(uint32_t width:{1,31,32,33,255,256,257,2559,2560}) {
    std::vector<uint32_t> hits(width);
    for(uint32_t group=0;group<(width+255)/256;++group)for(uint32_t tid=0;tid<256;++tid)
      if(group*256+tid<width)++hits[group*256+tid];
    require(std::all_of(hits.begin(),hits.end(),[](uint32_t n){return n==1;}),"combine column ownership");++checks;
  }
  for(uint32_t threads:{32,256}) {
    std::array<uint32_t,2560> hits{};
    for(uint32_t tid=0;tid<threads;++tid)for(uint32_t c=tid;c<2560;c+=threads)++hits[c];
    require(std::all_of(hits.begin(),hits.end(),[](uint32_t n){return n==1;}),"poison column ownership");++checks;
  }
  for(uint32_t slots=1;slots<=10;++slots) {
    uint16_t d[10],w[10];for(uint32_t s=0;s<10;++s){d[s]=bf16(float(s)-4);w[s]=0x3e80;}
    const uint16_t ref=reduce(d,w,slots);require(std::isfinite(fp32(ref)),"CPU BF16 reducer finite");++checks;
  }
  // This cancellation witnesses that sequential K10 is not the source reduction.
  uint16_t d[10]={0x4300,0xc300,0x3f80,0,0,0,0,0,0x3f00,0x3f00},w[10];std::fill_n(w,10,0x3f80);
  uint16_t seq=0;for(uint32_t s=0;s<10;++s)seq=plus(seq,times(d[s],w[s]));
  require(reduce(d,w,10)!=seq,"reduction fixture must distinguish slot grouping");++checks;
  std::cout<<"{\"cpu_self_test\":\"passed\",\"checks\":"<<checks<<",\"gpu_work\":false,\"numerical_qualification\":\"pending_gpu\"}\n";
}
}

int main(int argc,char **argv) {
  @autoreleasepool {try {
    if(argc==2&&std::string(argv[1])=="--cpu-self-test"){cpuTest();return 0;}
    require(argc==4&&std::string(argv[1])!="--cpu-self-test","usage: oracle METALLIB REPORT_JSON --gpu");
    require(std::string(argv[3])=="--gpu","GPU execution requires explicit --gpu");
    MetalBackend b(argv[1]);std::ostringstream report;uint64_t combineCases=0,poisonCases=0;
    report<<std::setprecision(12)<<"{\"schema\":\"splash-moe-pointwise-synthetic-v1\",\"pass\":true,\"actual_model_inputs\":false,"
        <<"\"error_limit\":\"zero bytes BF16 outputs, diagnostic words, inputs and canaries\","
        <<"\"timing_size_bytes\":"<<sizeof(CommandTiming)<<",\"sticky_seed\":"<<sticky<<",\"timings\":[";
    bool first=true;
    auto qualifyCombine=[&](uint32_t rows,uint32_t width,uint32_t slots,Fault fault,bool extra,bool time,uint32_t seed=sticky) {
      CombineFixture f(b,rows,width,slots,fault,seed);std::array<CommandGraph,5> graphs;
      for(uint32_t k=0;k<5;++k)graphs[k]=f.graph(combineNames[k],extra);
      f.reset();valid(b.submitCommand(graphs[0].dispatches()));f.guards();
      const auto expected=f.out.snapshot();const uint32_t d=*static_cast<const uint32_t *>(f.diag.view.contents());
      for(uint32_t k=1;k<5;++k) {f.reset();valid(b.submitCommand(graphs[k].dispatches()));
        compare(f.out,f.diag,expected,d,combineNames[k]);f.guards();}
      if(time) {
        for(uint32_t k=0;k<5;++k)graphs[k]=f.graph(combineNames[k]);
        if(!first)report<<',';first=false;report<<"{\"kind\":\"combine\",\"rows\":"<<rows<<",\"width\":"<<width<<",\"selections\":"<<slots
            <<",\"fault\":"<<splash::json::quote(faultName(fault))<<',';
        rotatedTimings(b,f,graphs,combineNames,report);report<<'}';
      }
      f.immutable();++combineCases;
    };
    for(uint32_t rows:{1,4,16,2048})for(uint32_t width:{1,31,32,33,255,256,257,2559,2560})
      qualifyCombine(rows,width,10,Fault::None,true,width==2560);
    for(uint32_t slots=1;slots<=10;++slots)qualifyCombine(4,33,slots,Fault::None,true,false);
    qualifyCombine(8192,33,10,Fault::None,true,false);
    for(uint32_t i=1;i<=uint32_t(Fault::Cancellation);++i)for(uint32_t width:{1,33,2560})
      qualifyCombine(4,width,10,Fault(i),true,false);
    for(Fault fault:{Fault::None,Fault::NegativeID,Fault::MixedFaults,Fault::Cancellation})
      qualifyCombine(16,257,10,fault,true,false,sticky|1u);
    // Zero/oversized geometry and wrong launch dimensions must touch no outputs.
    for(uint32_t invalid=0;invalid<8;++invalid) {
      CombineFixture f(b,4,33,10,Fault::None);
      switch(invalid){case 0:f.p.rows=0;break;case 1:f.p.rows=8193;break;case 2:f.p.width=0;break;case 3:f.p.width=2561;break;
        case 4:f.p.selections=0;break;case 5:f.p.selections=11;break;case 6:f.p.experts=0;break;case 7:f.p.experts=9;break;}
      std::vector<std::byte> expected;uint32_t d=0;
      for(uint32_t k=0;k<5;++k){f.reset();CommandGraph g;g.add(combineNames[k],{f.down.view,f.ids.view,f.weights.view,f.shared.view,f.gate.view,f.out.view,f.diag.view},f.p,{1,1,1},{256,1,1});
        valid(b.submitCommand(g.dispatches()));if(!k){expected=f.out.snapshot();d=*static_cast<const uint32_t *>(f.diag.view.contents());}
        compare(f.out,f.diag,expected,d,combineNames[k]);f.guards();}
      require(d==(sticky|2u),"combine malformed geometry diagnostics");f.immutable();++combineCases;
    }
    for(uint32_t xThreads:{32,128,512}) {
      CombineFixture f(b,4,33,10,Fault::None);
      const auto expected=f.out.snapshot();
      for(const char *name:combineNames) {
        f.reset();auto g=f.graph(name,false,xThreads);valid(b.submitCommand(g.dispatches()));
        compare(f.out,f.diag,expected,sticky|2u,name);f.guards();
      }
      f.immutable();++combineCases;
    }
    for(uint32_t rows:{1,4,16,2048})for(uint32_t mode=0;mode<3;++mode) {
      PoisonFixture f(b,rows,mode);std::array<CommandGraph,3> graphs{{f.graph(poisonNames[0],256,10,true),f.graph(poisonNames[1],256,1,true),f.graph(poisonNames[2],32,1,true)}};
      for(uint32_t k=0;k<3;++k){f.reset();valid(b.submitCommand(graphs[k].dispatches()));f.guards();f.verifyExpected();}
      graphs={f.graph(poisonNames[0],256,10),f.graph(poisonNames[1],256,1),f.graph(poisonNames[2],32,1)};
      if(!first)report<<',';first=false;report<<"{\"kind\":\"poison\",\"rows\":"<<rows<<",\"mode\":"<<mode<<',';
      rotatedTimings(b,f,graphs,poisonNames,report);report<<'}';f.inverse.immutable();++poisonCases;
    }
    for(uint32_t invalid=0;invalid<8;++invalid) {
      PoisonFixture f(b,4,2);
      switch(invalid){case 0:f.p.affine.rows=0;break;case 1:f.p.affine.rows=8193;break;
        case 2:f.p.affine.selections=0;break;case 3:f.p.affine.selections=11;break;
        case 4:f.p.affine.input_size=641;break;case 5:f.p.affine.output_size=2559;break;
        case 6:f.p.affine.experts=511;break;case 7:++f.p.route_capacity;break;}
      std::array<CommandGraph,3> graphs{{f.graph(poisonNames[0],256,10),f.graph(poisonNames[1],256,1),f.graph(poisonNames[2],32,1)}};
      for(uint32_t k=0;k<3;++k){f.reset();valid(b.submitCommand(graphs[k].dispatches()));
        compare(f.out,f.diag,f.initial,sticky|2u,poisonNames[k]);f.guards();}
      f.inverse.immutable();++poisonCases;
    }
    // Poison intentionally does not validate these unrelated affine/job fields.
    for(uint32_t mode=0;mode<3;++mode) {
      PoisonFixture f(b,4,mode,sticky|1u);f.p.affine.reserved0=1;f.p.affine.reserved1=2;f.p.affine.reserved2=3;
      f.p.reserved=4;f.p.job_capacity=0;f.p.tile_rows=17;f.p.affine.weight_row_stride_bytes=0;
      f.p.affine.weight_expert_stride_bytes=1;f.p.affine.parameter_row_stride_bytes=UINT64_MAX;
      f.p.affine.parameter_expert_stride_bytes=0;
      for(uint32_t replay=0;replay<2;++replay) {
        std::array<CommandGraph,3> graphs{{f.graph(poisonNames[0],256,10,true),f.graph(poisonNames[1],256,1,true),f.graph(poisonNames[2],32,1,true)}};
        for(uint32_t k=0;k<3;++k){f.reset();valid(b.submitCommand(graphs[k].dispatches()));f.verifyExpected();f.guards();}
        f.inverse.immutable();++poisonCases;
        auto *inv=static_cast<uint32_t *>(f.inverse.view.contents());
        for(uint32_t r=0;r<f.p.route_capacity;++r)inv[r]=inv[r]>=f.p.route_capacity?r:UINT32_MAX;
        f.inverse.freeze();
      }
    }
    report<<"],\"combine_qualified_cases\":"<<combineCases<<",\"poison_qualified_cases\":"<<poisonCases<<",\"warmup_cycles\":3,\"matched_cycles\":8}\n";
    std::ofstream output(argv[2]);require(bool(output),"cannot write pointwise report");output<<report.str();return 0;
  }catch(const std::exception &e){std::cerr<<"pointwise oracle failed: "<<e.what()<<'\n';return 1;}}
}
