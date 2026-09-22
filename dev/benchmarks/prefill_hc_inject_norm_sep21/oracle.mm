#include "dev/benchmarks/prefill_hc_inject_norm_sep21/bridge.hpp"
#include "metal/abi/FlashHC.h"
#include "engine/Json.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
using namespace splash::metal;
using namespace splash::flash;
namespace candidate = splash::flash::prefill_hc_inject_norm_sep21;
constexpr uint64_t guard = 64;
constexpr uint32_t sticky = 0x80000080u;
void require(bool ok, const std::string &why) { if (!ok) throw std::runtime_error(why); }
uint16_t bf16(float v) {
  uint32_t x = std::bit_cast<uint32_t>(v);
  if ((x & 0x7f800000u) == 0x7f800000u && (x & 0x007fffffu)) return 0x7fc0;
  return uint16_t((x + 0x7fffu + ((x >> 16) & 1)) >> 16);
}
float fp32(uint16_t v) { return std::bit_cast<float>(uint32_t(v) << 16); }
uint16_t injected(uint16_t h, uint16_t b, uint16_t gate) {
  return bf16(fp32(h) + fp32(bf16(fp32(b) * fp32(gate))));
}
bool finite(uint16_t v) { return (v & 0x7f80) != 0x7f80; }
bool normalOrZero(uint16_t v) {
  return finite(v) && ((v & 0x7f80) || !(v & 0x007f));
}
struct Guarded {
  MetalBuffer base, view; uint64_t bytes;
  std::vector<std::byte> frozen;
  Guarded(MetalBackend &b, uint64_t n) : bytes(n) {
    base = b.allocateBuffer(n + 2 * guard, BufferStorage::Shared);
    view = b.view(base, guard, n); std::memset(base.contents(), 0xa5, n + 2 * guard);
  }
  void checkGuard() const {
    const auto *p = static_cast<const unsigned char *>(base.contents());
    for (uint64_t i = 0; i < guard; ++i)
      require(p[i] == 0xa5 && p[guard + bytes + i] == 0xa5, "guard changed");
  }
  std::vector<std::byte> snapshot() const {
    const auto *p = static_cast<const std::byte *>(view.contents()); return {p, p + bytes};
  }
  void freeze() { frozen = snapshot(); }
  void unchanged() const {
    checkGuard(); require(frozen.size() == bytes && !std::memcmp(view.contents(), frozen.data(), bytes), "read-only input changed");
  }
};
enum class Fault : uint32_t {
  Finite, Rounding, Cancellation, Zero, Subnormal, LargeFinite, HyperNaN,
  BranchInf, GateNaN, ZeroTimesInf, WeightNaN, WeightInf, ResidualOverflow
};
const char *name(Fault f) {
  constexpr const char *v[] = {"finite", "rounding_boundaries", "cancellation", "signed_zero", "subnormal",
      "finite_square_overflow", "hyper_nan", "branch_inf", "gate_nan", "zero_times_inf", "weight_nan", "weight_inf", "residual_overflow"};
  return v[uint32_t(f)];
}
struct Fixture {
  FlashHCGeometry geometry; NormConvention convention; FlashTensor normWeight;
  Guarded hyper, branch, gates, weight, updated, norm, diag;
  uint32_t seed;
  Fixture(MetalBackend &b, uint32_t rows, bool f32, NormConvention c, Fault fault, uint32_t seed_ = sticky)
      : geometry{rows,2560,4,1e-6f}, convention(c),
        hyper(b,uint64_t(rows)*10240*2), branch(b,uint64_t(rows)*2560*2),
        gates(b,uint64_t(rows)*4*2), weight(b,10240*(f32?4:2)),
        updated(b,uint64_t(rows)*10240*2), norm(b,uint64_t(rows)*10240*2), diag(b,4), seed(seed_) {
    normWeight = {weight.view,f32?FlashDType::F32:FlashDType::BF16,{10240},weight.bytes};
    auto *h=static_cast<uint16_t *>(hyper.view.contents()); auto *br=static_cast<uint16_t *>(branch.view.contents());
    auto *g=static_cast<uint16_t *>(gates.view.contents());
    constexpr uint16_t gateValues[] = {0,0x8000,0x3f00,0x3f80,0x3f81,0x3fff,0x4000,0xbf80};
    for (uint32_t row=0; row<rows; ++row) {
      for(uint32_t s=0;s<4;++s) g[uint64_t(row)*4+s]=gateValues[(row*3+s)%8];
      for(uint32_t col=0;col<2560;++col) {
        br[uint64_t(row)*2560+col]=bf16(float(int((row*13+col*7)%1025)-512)/128);
        for(uint32_t s=0;s<4;++s)
          h[(uint64_t(row)*4+s)*2560+col]=bf16(float(int((row*17+s*3+col*11)%513)-256)/128);
      }
    }
    for(uint32_t i=0;i<10240;++i) {
      // These values exercise an F32 1+w scale that must not round to BF16.
      const float v = c==NormConvention::DirectGamma ? 1.0f+float(int(i%65)-32)/4096
          : (i%17==0 ? -1.000244140625f : float(int(i%65)-32)/4096);
      if(f32)static_cast<float *>(weight.view.contents())[i]=v;
      else static_cast<uint16_t *>(weight.view.contents())[i]=bf16(v);
    }
    const uint64_t r=uint64_t(rows-1), idx=(r*4+3)*2560+2559, bidx=r*2560+2559, gidx=r*4+3;
    switch(fault) {
      case Fault::Finite:break;
      case Fault::Rounding:h[idx]=0xbf82;br[bidx]=0x3f81;g[gidx]=0x3f81;break;
      case Fault::Cancellation:h[idx]=0xc300;br[bidx]=0x4300;g[gidx]=0x3f80;break;
      case Fault::Zero:
        std::fill_n(h,hyper.bytes/2,0x8000);std::fill_n(br,branch.bytes/2,0);std::fill_n(g,gates.bytes/2,0x8000);break;
      case Fault::Subnormal:h[idx]=1;br[bidx]=0x8001;g[gidx]=0x3f00;break;
      case Fault::LargeFinite:std::fill_n(h+idx-2559,2560,0x7f7f);br[bidx]=0;break;
      case Fault::HyperNaN:h[idx]=0xffa1;break;
      case Fault::BranchInf:br[bidx]=0x7f80;g[gidx]=0x3f80;break;
      case Fault::GateNaN:g[gidx]=0xffa1;break;
      case Fault::ZeroTimesInf:br[bidx]=0x7f80;g[gidx]=0;break;
      case Fault::WeightNaN:
        if(f32)static_cast<float *>(weight.view.contents())[10239]=std::bit_cast<float>(0x7fa12345u);
        else static_cast<uint16_t *>(weight.view.contents())[10239]=0xffa1;break;
      case Fault::WeightInf:
        if(f32)static_cast<float *>(weight.view.contents())[10239]=std::numeric_limits<float>::infinity();
        else static_cast<uint16_t *>(weight.view.contents())[10239]=0x7f80;break;
      case Fault::ResidualOverflow:h[idx]=0x7f7f;br[bidx]=0x7f7f;g[gidx]=0x4000;break;
    }
    for(auto *x:{&hyper,&branch,&gates,&weight}) x->freeze(); reset();
  }
  void reset(bool restoreHyper=true) {
    if(restoreHyper)std::memcpy(hyper.view.contents(),hyper.frozen.data(),hyper.bytes);
    std::memset(updated.view.contents(),0x59,updated.bytes);std::memset(norm.view.contents(),0x59,norm.bytes);
    *static_cast<uint32_t *>(diag.view.contents())=seed;
  }
  void check(bool mutatedHyper=false) const {
    for(const auto *x:{&hyper,&branch,&gates,&weight,&updated,&norm,&diag}) x->checkGuard();
    for(const auto *x:{&branch,&gates,&weight}) x->unchanged();
    if(!mutatedHyper)hyper.unchanged();
  }
  CommandGraph graph(bool fusion, bool inPlace=false) {
    CommandGraph graph;
    const auto out = inPlace ? hyper.view : updated.view;
    if(fusion)candidate::addPrivatePrefillHCInjectNormSep21(graph,hyper.view,branch.view,gates.view,normWeight,
        out,norm.view,diag.view,geometry,convention);
    else {
      addHCInject(graph,hyper.view,branch.view,gates.view,out,geometry);
      addHCGroupedNorm(graph,out,normWeight,norm.view,geometry,convention);
    }
    return graph;
  }
};
void valid(CommandTiming t) {
  require(sizeof(CommandTiming)==200,"native CommandTiming ABI drift");
  require(std::isfinite(t.gpuSeconds)&&t.gpuSeconds>0&&t.gpuSeconds<60&&
      std::isfinite(t.wallSeconds)&&t.wallSeconds>0,"invalid command timing");
}
uint32_t diagnostic(const std::vector<std::byte> &updated,const std::vector<std::byte> &norm,uint32_t seed) {
  for(const auto *v:{&updated,&norm}) {
    const auto *p=reinterpret_cast<const uint16_t *>(v->data());
    for(uint64_t i=0;i<v->size()/2;++i)if(!finite(p[i]))return seed|4u;
  }
  return seed;
}
void exact(const MetalBuffer &b,const std::vector<std::byte> &expected,const std::string &what) {
  if(std::memcmp(b.contents(),expected.data(),expected.size())) {
    const auto *a=static_cast<const uint16_t *>(b.contents());const auto *e=reinterpret_cast<const uint16_t *>(expected.data());
    for(uint64_t i=0;i<expected.size()/2;++i)if(a[i]!=e[i])
      throw std::runtime_error(what+" mismatch at "+std::to_string(i)+" expected="+std::to_string(e[i])+" observed="+std::to_string(a[i]));
  }
}
void qualify(MetalBackend &backend,Fixture &f,Fault fault) {
  f.reset();auto native=f.graph(false);valid(backend.submitCommand(native.dispatches()));f.check();
  const auto expectedUpdated=f.updated.snapshot(), expectedNorm=f.norm.snapshot();
  const uint32_t expectedDiag=diagnostic(expectedUpdated,expectedNorm,f.seed);
  // Finite original operands independently witness both BF16 injection stages.
  const auto *h=reinterpret_cast<const uint16_t *>(f.hyper.frozen.data());
  const auto *b=reinterpret_cast<const uint16_t *>(f.branch.frozen.data());
  const auto *g=reinterpret_cast<const uint16_t *>(f.gates.frozen.data());
  const auto *u=reinterpret_cast<const uint16_t *>(expectedUpdated.data());
  for(uint32_t r=0;r<f.geometry.rows;++r)for(uint32_t s=0;s<4;++s)for(uint32_t c=0;c<2560;++c) {
    const uint64_t i=(uint64_t(r)*4+s)*2560+c,bi=uint64_t(r)*2560+c,gi=uint64_t(r)*4+s;
    const float rawProduct=fp32(b[bi])*fp32(g[gi]);
    const uint16_t product=bf16(rawProduct);
    const float rawSum=fp32(h[i])+fp32(product);
    const uint16_t cpu=injected(h[i],b[bi],g[gi]);
    // Subnormal behavior is qualified against the native Metal control only;
    // the independent CPU witness must not prescribe a device denormal mode.
    if((std::fpclassify(rawProduct)==FP_NORMAL||std::fpclassify(rawProduct)==FP_ZERO)&&
        (std::fpclassify(rawSum)==FP_NORMAL||std::fpclassify(rawSum)==FP_ZERO)&&
        normalOrZero(h[i])&&normalOrZero(b[bi])&&normalOrZero(g[gi])&&
        normalOrZero(product)&&normalOrZero(cpu)&&normalOrZero(u[i]))
      require(u[i]==cpu,"native CPU BF16 injection boundary mismatch");
  }
  for(bool inPlace:{false,true}) {
    f.reset();auto graph=f.graph(true,inPlace);valid(backend.submitCommand(graph.dispatches()));f.check(inPlace);
    exact(inPlace?f.hyper.view:f.updated.view,expectedUpdated,std::string(name(fault))+" updated");
    exact(f.norm.view,expectedNorm,std::string(name(fault))+" normalized");
    require(*static_cast<const uint32_t *>(f.diag.view.contents())==expectedDiag,"sticky nonfinite diagnostic mismatch");
    f.reset();auto control=f.graph(false,inPlace);valid(backend.submitCommand(control.dispatches()));f.check(inPlace);
    exact(inPlace?f.hyper.view:f.updated.view,expectedUpdated,"native in-place updated");
    exact(f.norm.view,expectedNorm,"native in-place normalized");
  }
  f.reset();
}
double median(std::vector<double> v) {std::sort(v.begin(),v.end());const auto i=v.size()/2;return v.size()%2?v[i]:(v[i-1]+v[i])/2;}
void timings(MetalBackend &backend,Fixture &f,uint32_t pairs,std::ostream &out) {
  f.reset();auto native=f.graph(false), fused=f.graph(true);
  std::array<CommandGraph *,2> graphs{&native,&fused};
  // No CPU tensor accesses from first warm command through the final pair.
  double warmGpu=0;uint32_t warmCommands=0;
  while(warmGpu<.150 || warmCommands<8) {
    for(uint32_t k=0;k<2;++k){const auto t=backend.submitCommand(graphs[(warmCommands+k)%2]->dispatches());valid(t);warmGpu+=t.gpuSeconds;}
    warmCommands+=2;
  }
  std::array<std::vector<double>,2> gpu,wall;
  out<<"\"warm_gpu_ms\":"<<warmGpu*1000<<",\"warm_commands\":"<<warmCommands<<",\"orders\":[";
  for(uint32_t p=0;p<pairs;++p) {
    if(p)out<<',';out<<'[';
    for(uint32_t position=0;position<2;++position) {
      const uint32_t route=(p+position)%2;
      const auto t=backend.submitCommand(graphs[route]->dispatches());valid(t);
      gpu[route].push_back(t.gpuSeconds*1000);wall[route].push_back(t.wallSeconds*1000);
      if(position)out<<',';out<<route;
    }
    out<<']';
  }
  // An even AB/BA schedule ends in native. One untimed fused command ensures
  // the post-warm exact comparison witnesses the candidate's warmed output.
  valid(backend.submitCommand(fused.dispatches()));
  f.check();const auto u=f.updated.snapshot(),n=f.norm.snapshot();
  require(*static_cast<const uint32_t *>(f.diag.view.contents())==diagnostic(u,n,f.seed),"timing diagnostic mismatch");
  // Post-timing final graph output must still agree with a fresh native control.
  f.reset();valid(backend.submitCommand(native.dispatches()));exact(f.updated.view,u,"timed updated");exact(f.norm.view,n,"timed normalized");f.check();
  std::vector<double> ratios;uint32_t wins=0;
  for(uint32_t i=0;i<pairs;++i){const double r=gpu[0][i]/gpu[1][i];ratios.push_back(r);wins+=r>1;}
  out<<"],\"median_paired_speedup\":"<<median(ratios)<<",\"positive_pairs\":"<<wins<<",\"routes\":[";
  for(uint32_t route=0;route<2;++route) {
    if(route)out<<',';
    out<<"{\"name\":"<<splash::json::quote(route?"private_prefill_hc_inject_norm_sep21":"native_inject_then_norm")
        <<",\"median_gpu_ms\":"<<median(gpu[route])<<",\"median_wall_ms\":"<<median(wall[route])<<",\"gpu_ms\":[";
    for(uint32_t i=0;i<pairs;++i){if(i)out<<',';out<<gpu[route][i];}out<<"],\"wall_ms\":[";
    for(uint32_t i=0;i<pairs;++i){if(i)out<<',';out<<wall[route][i];}out<<"]}";
  }
  out<<']';
  std::cerr<<"R"<<f.geometry.rows<<" dtype="<<(f.normWeight.dtype==FlashDType::F32?"F32":"BF16")
      <<" convention="<<uint32_t(f.convention)<<" native="<<median(gpu[0])<<" fused="<<median(gpu[1])<<" ms speedup="<<median(ratios)<<'\n';
}
template<class F> void rejects(F &&fn,const char *what) {
  bool rejected=false;try{fn();}catch(const std::invalid_argument &){rejected=true;}
  require(rejected,std::string("guard did not reject ")+what);
}
void cpuTest() {
  uint64_t checks=0;
  for(uint32_t rows:{512,513,1024,2048})for(bool f32:{false,true})for(auto c:{NormConvention::OnePlusWeight,NormConvention::DirectGamma}) {
    FlashTensor weight{{},f32?FlashDType::F32:FlashDType::BF16,{10240},uint64_t(10240)*(f32?4:2)};
    const auto p=candidate::checkedParams({rows,2560,4,1e-6f},weight,c);
    require(p.rows==rows&&p.norm_is_float==f32&&p.norm_convention==uint32_t(c),"private params");++checks;
  }
  FlashTensor weight{{},FlashDType::BF16,{10240},20480};
  for(uint32_t i=0;i<9;++i) {
    FlashHCGeometry g{2048,2560,4,1e-6f};auto w=weight;auto c=NormConvention::OnePlusWeight;
    switch(i){case 0:g.rows=511;break;case 1:g.rows=2049;break;case 2:g.width=2559;break;case 3:g.streams=3;break;
      case 4:g.epsilon=0;break;case 5:g.epsilon=std::numeric_limits<float>::infinity();break;
      case 6:w.logicalBytes--;break;case 7:w.dtype=FlashDType::U32;break;case 8:c=NormConvention(99);break;}
    rejects([&]{(void)candidate::checkedParams(g,w,c);},"CPU geometry/dtype/convention");++checks;
  }
  rejects([&]{auto w=weight;w.shape={4,2560};(void)candidate::checkedParams({2048,2560,4,1e-6f},w,NormConvention::OnePlusWeight);},"weight rank");++checks;
  std::array<uint32_t,2560> hits{};for(uint32_t tid=0;tid<640;++tid)for(uint32_t e=0;e<4;++e)++hits[tid*4+e];
  require(std::all_of(hits.begin(),hits.end(),[](uint32_t n){return n==1;}),"exact column ownership");++checks;
  require(injected(0xbf82,0x3f81,0x3f81)!=bf16(fp32(0xbf82)+fp32(0x3f81)*fp32(0x3f81)),"rounding fixture must distinguish product boundary");++checks;
  const float raw=1.0f/256, norm=fp32(0x3f81);
  require(bf16(norm*(1+raw))!=bf16(norm*fp32(bf16(1+raw))),"scale fixture must distinguish BF16-rounding");++checks;
  std::cout<<"{\"cpu_self_test\":\"passed\",\"checks\":"<<checks<<",\"gpu_executed\":false,\"numerical_qualification\":\"pending_gpu\"}\n";
}
void hostGuards(MetalBackend &backend) {
  Fixture f(backend,512,false,NormConvention::OnePlusWeight,Fault::Finite);
  auto call=[&](MetalBuffer output,MetalBuffer normalized,MetalBuffer diagnostics,const FlashTensor &w) {
    CommandGraph graph;candidate::addPrivatePrefillHCInjectNormSep21(graph,f.hyper.view,f.branch.view,f.gates.view,w,
        output,normalized,diagnostics,f.geometry,f.convention);
  };
  rejects([&]{call(backend.view(f.hyper.base,guard+2,f.hyper.bytes),f.norm.view,f.diag.view,f.normWeight);},"partial hyper output alias");
  rejects([&]{call(f.updated.view,f.hyper.view,f.diag.view,f.normWeight);},"norm/hyper alias");
  rejects([&]{call(f.updated.view,f.norm.view,backend.view(f.updated.base,guard,4),f.normWeight);},"diagnostic/output alias");
  rejects([&]{call(backend.view(f.updated.base,guard,f.updated.bytes-2),f.norm.view,f.diag.view,f.normWeight);},"short updated buffer");
  rejects([&]{auto w=f.normWeight;w.buffer=backend.view(f.weight.base,guard,f.weight.bytes-2);call(f.updated.view,f.norm.view,f.diag.view,w);},"short norm weight");
  rejects([&]{auto w=f.normWeight;w.buffer=backend.allocateBuffer(f.weight.bytes,BufferStorage::Private);call(f.updated.view,f.norm.view,f.diag.view,w);},"non-addressable private weight");
  call(f.hyper.view,f.norm.view,f.diag.view,f.normWeight); // exact in-place accepted, no submit
  f.check();
}
void shaderGuards(MetalBackend &backend) {
  Fixture f(backend,512,false,NormConvention::OnePlusWeight,Fault::Finite);
  const auto untouchedUpdated=f.updated.snapshot(),untouchedNorm=f.norm.snapshot();
  for(uint32_t bad=0;bad<18;++bad) {
    f.reset();auto p=candidate::checkedParams(f.geometry,f.normWeight,f.convention);DispatchSize threads{640,1,1};
    switch(bad){case 0:p.rows=0;break;case 1:p.rows=511;break;case 2:p.rows=2049;break;case 3:p.width=2559;break;
      case 4:p.streams=3;break;case 5:p.lowrank=319;break;case 6:p.norm_is_float=2;break;case 7:p.norm_convention=2;break;
      case 8:p.norm_epsilon=0;break;case 9:p.norm_epsilon=std::numeric_limits<float>::quiet_NaN();break;
      case 10:p.arithmetic_mode=2;break;case 11:p.simdgroups=3;break;case 12:threads.x=32;break;case 13:threads.x=512;break;
      case 14:threads={320,2,1};break;case 15:threads={320,1,2};break;case 16:p.has_injection=2;break;case 17:p.write_raw_up=2;break;}
    CommandGraph graph;graph.add("private_prefill_hc_inject_norm_sep21",{f.hyper.view,f.branch.view,f.gates.view,f.weight.view,f.updated.view,f.norm.view,f.diag.view},
        p,{1,1,1},threads);valid(backend.submitCommand(graph.dispatches()));
    exact(f.updated.view,untouchedUpdated,"invalid-launch updated");exact(f.norm.view,untouchedNorm,"invalid-launch norm");
    require(*static_cast<const uint32_t *>(f.diag.view.contents())==(f.seed|2u),"invalid-launch diagnostic");f.check();
  }
  f.reset();auto p=candidate::checkedParams(f.geometry,f.normWeight,f.convention);
  CommandGraph graph;graph.add("private_prefill_hc_inject_norm_sep21",{f.hyper.view,f.branch.view,f.gates.view,f.weight.view,f.updated.view,f.norm.view,f.diag.view},
      p,{f.geometry.rows+1,f.geometry.streams+1,2},{640,1,1});valid(backend.submitCommand(graph.dispatches()));f.check();
  require(*static_cast<const uint32_t *>(f.diag.view.contents())==f.seed,"extra-grid sticky diagnostics changed");
  const auto u=f.updated.snapshot(),n=f.norm.snapshot();f.reset();auto native=f.graph(false);valid(backend.submitCommand(native.dispatches()));
  exact(f.updated.view,u,"extra-grid updated");exact(f.norm.view,n,"extra-grid normalized");f.check();
}
} // namespace
int main(int argc,char **argv) {
  @autoreleasepool {try {
    if(argc==2&&std::string(argv[1])=="--cpu-self-test"){cpuTest();return 0;}
    require(argc==5&&std::string(argv[1])=="--gpu","usage: oracle --gpu METALLIB REPORT_JSON PAIRS_EVEN");
    const uint32_t pairs=uint32_t(std::stoul(argv[4]));require(pairs>=4&&pairs<=64&&pairs%2==0,"matched pairs must be even,4..64");
    MetalBackend backend(argv[2]);hostGuards(backend);shaderGuards(backend);
    uint64_t cases=0;std::ostringstream report;report<<std::setprecision(12)
        <<"{\"schema\":\"splash-prefill-hc-inject-norm-sep21-v1\",\"pass\":true,\"actual_model_inputs\":false,"
        <<"\"error_limit\":\"zero bytes updated BF16, normalized BF16, inputs, canaries; exact sticky diagnostics\","
        <<"\"timing_size_bytes\":"<<sizeof(CommandTiming)<<",\"sticky_seed\":"<<sticky<<",\"host_guard_cases\":7,\"shader_guard_cases\":19,"
        <<"\"timing_protocol\":\"150ms minimum GPU warm, balanced AB/BA, no CPU buffer access inside warm/timing interval\",\"matched_pairs\":"<<pairs<<",\"timings\":[";
    bool first=true;
    for(uint32_t rows:{512,1024,2048})for(bool f32:{false,true})for(auto c:{NormConvention::OnePlusWeight,NormConvention::DirectGamma}) {
      Fixture f(backend,rows,f32,c,Fault::Finite);qualify(backend,f,Fault::Finite);++cases;
      if(!first)report<<',';first=false;report<<"{\"rows\":"<<rows<<",\"weight_dtype\":"<<splash::json::quote(f32?"F32":"BF16")
          <<",\"norm_convention\":"<<splash::json::quote(c==NormConvention::OnePlusWeight?"one_plus_weight":"direct_gamma")<<',';
      timings(backend,f,pairs,report);report<<'}';
    }
    for(uint32_t fault=1;fault<=uint32_t(Fault::ResidualOverflow);++fault)for(bool f32:{false,true})for(auto c:{NormConvention::OnePlusWeight,NormConvention::DirectGamma}) {
      Fixture f(backend,512,f32,c,Fault(fault),fault%2?sticky|1u:sticky);qualify(backend,f,Fault(fault));++cases;
    }
    report<<"],\"qualified_fixtures\":"<<cases<<",\"in_place_and_out_of_place\":true,\"gpu_executed\":true}\n";
    std::ofstream output(argv[3]);require(bool(output),"cannot write report");output<<report.str();return 0;
  }catch(const std::exception &e){
    std::cerr<<"prefill HC inject/norm oracle failed: "<<e.what()<<'\n';
    if(argc==5&&std::string(argv[1])=="--gpu") {
      std::ofstream failure(std::string(argv[3])+".failure.json");
      failure<<"{\"schema\":\"splash-prefill-hc-inject-norm-failure-v1\",\"pass\":false,\"error\":"<<splash::json::quote(e.what())<<"}\n";
    }
    return 1;
  }}
}
