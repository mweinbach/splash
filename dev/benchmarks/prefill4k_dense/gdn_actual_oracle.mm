#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashGDN.h"
#include "engine/Json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace splash::metal;
constexpr uint32_t kCanary=0x7fc1a597,kSticky=0x40000000;
constexpr uint64_t kGuard=64;
static_assert(sizeof(CommandTiming)==200,"Reject stale timing ABI");
void require(bool value,const std::string &message) {if (!value) throw std::runtime_error(message);}
std::string sha(const void *data,uint64_t bytes) {
  require(bytes<=UINT32_MAX,"SHA input too large");unsigned char result[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data,CC_LONG(bytes),result);std::ostringstream out;
  for (auto byte : result) out<<std::hex<<std::setfill('0')<<std::setw(2)<<unsigned(byte);
  return out.str();
}
struct File {std::string path,digest;uint64_t bytes=0;};
struct Source {uint32_t rows=0;float epsilon=0;std::map<std::string,File> files;};
Source manifest(const std::string &path) {
  NSData *data=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
  require(data!=nil,"cannot read actual GDN manifest");NSError *error=nil;
  NSDictionary *document=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
  require(error==nil && [document isKindOfClass:[NSDictionary class]],"invalid GDN JSON");
  require([document[@"schema"] isEqualToString:@"splash-actual-gdn-layer-v1"],"invalid GDN schema");
  Source result;result.rows=[document[@"rows"] unsignedIntValue];result.epsilon=[document[@"norm_epsilon"] floatValue];
  require(result.rows==2048 && [document[@"lanes"] unsignedIntValue]==1 && std::isfinite(result.epsilon) && result.epsilon>0,
          "actual GDN fixture must be one2048-row lane");
  NSDictionary *files=document[@"files"];require([files isKindOfClass:[NSDictionary class]],"missing GDN files");
  for (NSString *name in files) {
    NSDictionary *entry=files[name];File file;
    file.path=[entry[@"path"] UTF8String];file.digest=[entry[@"sha256"] UTF8String];file.bytes=[entry[@"bytes"] unsignedLongLongValue];
    result.files[std::string(name.UTF8String)]=std::move(file);
  }
  return result;
}
std::vector<std::byte> read(const Source &source,const char *name,uint64_t expected) {
  const auto &file=source.files.at(name);require(file.bytes==expected && std::filesystem::file_size(file.path)==expected,
      std::string("GDN fixture extent mismatch: ")+name);
  std::vector<std::byte> bytes(expected);std::ifstream input(file.path,std::ios::binary);
  input.read(reinterpret_cast<char *>(bytes.data()),std::streamsize(expected));require(bool(input),"short GDN read");
  require(sha(bytes.data(),bytes.size())==file.digest,"GDN fixture SHA mismatch");return bytes;
}
struct Guarded {
  MetalBuffer base,view;uint64_t bytes;
  Guarded(MetalBackend &backend,uint64_t count) :bytes(count) {
    require(count && count%4==0,"guard extent notwordaligned");
    base=backend.allocateBuffer(count+2*kGuard*4,BufferStorage::Shared);
    view=backend.view(base,kGuard*4,count);clear();
  }
  void clear() {std::fill_n(static_cast<uint32_t *>(base.contents()),base.sizeBytes()/4,kCanary);}
  void check() const {
    const auto *words=static_cast<const uint32_t *>(base.contents());
    for (uint64_t i=0;i<kGuard;++i)
      require(words[i]==kCanary && words[kGuard+bytes/4+i]==kCanary,"GDN canary changed");
  }
  void load(const std::vector<std::byte> &data) {require(data.size()==bytes,"GDN load extent");std::memcpy(view.contents(),data.data(),bytes);}
};
struct Fixture {
  Guarded mixed,decay,beta,z,norm,state,recurrence,output;MetalBuffer diagnostics;
  std::array<std::string,5> immutableSHA;
  Fixture(MetalBackend &backend,const Source &source,const std::map<std::string,std::vector<std::byte>> &files)
      :mixed(backend,uint64_t(source.rows)*10240*2),decay(backend,uint64_t(source.rows)*48*4),
       beta(backend,uint64_t(source.rows)*48*2),z(backend,uint64_t(source.rows)*6144*2),norm(backend,128*2),
       state(backend,48*128*128*4),recurrence(backend,uint64_t(source.rows)*6144*2),output(backend,uint64_t(source.rows)*6144*2) {
    diagnostics=backend.allocateBuffer(64,BufferStorage::Shared);
    mixed.load(files.at("mixed"));decay.load(files.at("decay"));beta.load(files.at("beta"));z.load(files.at("z"));norm.load(files.at("norm"));
    uint32_t i=0;for (const auto *buffer : {&mixed,&decay,&beta,&z,&norm})
      immutableSHA[i++]=sha(buffer->view.contents(),buffer->bytes);
  }
  void reset(const std::vector<std::byte> &initial) {
    state.clear();state.load(initial);recurrence.clear();output.clear();
    std::memset(diagnostics.contents(),0,diagnostics.sizeBytes());*static_cast<uint32_t *>(diagnostics.contents())=kSticky;
  }
  void verify() const {
    for (const auto *buffer : {&mixed,&decay,&beta,&z,&norm,&state,&recurrence,&output}) buffer->check();
    require(*static_cast<const uint32_t *>(diagnostics.contents())==kSticky,"GDN diagnostics changed");
  }
  void immutable() const {
    uint32_t i=0;for (const auto *buffer : {&mixed,&decay,&beta,&z,&norm})
      require(sha(buffer->view.contents(),buffer->bytes)==immutableSHA[i++],"GDN preparedinput orweight changed");
  }
};
struct Variant {const char *name;uint32_t threads;};
constexpr std::array<Variant,4> variants{{{"private_gdn_delayq_v16_t16",512},{"private_gdn_delayq_v16_t32",512},
    {"private_gdn_ilp_v16_t16_s8",256},{"private_gdn_ilp_v16_t32_s8",256}}};
CommandGraph graph(Fixture &f,const Source &source,const char *name,uint32_t threads,bool output=true) {
  CommandGraph result;
  const FlashGDNParams p{source.rows,1,16,48,128,128,4,source.epsilon,3*10240*2,48*128*128*4};
  result.add(name,{f.mixed.view,f.decay.view,f.beta.view,f.state.view,f.recurrence.view,f.diagnostics},p,
      {48,8,1},{threads,1,1});
  if (output) result.add("flash_gdn_output",{f.recurrence.view,f.z.view,f.norm.view,f.output.view,f.diagnostics},p,
      {48,source.rows,1},{32,1,1});
  return result;
}
void same(const Fixture &a,const Fixture &b,bool complete) {
  require(std::memcmp(a.state.view.contents(),b.state.view.contents(),a.state.bytes)==0,"GDN F32 state differs");
  require(std::memcmp(a.recurrence.view.contents(),b.recurrence.view.contents(),a.recurrence.bytes)==0,"GDN BF16 recurrence differs");
  if (complete) require(std::memcmp(a.output.view.contents(),b.output.view.contents(),a.output.bytes)==0,"GDN final BF16 output differs");
  a.verify();b.verify();
}
void valid(CommandTiming t) {
  require(std::isfinite(t.gpuSeconds) && t.gpuSeconds>1e-9 && t.gpuSeconds<600 &&
      std::isfinite(t.wallSeconds) && t.wallSeconds>1e-9 && t.wallSeconds<600,"Reject invalid/stale ABI timings");
}
double median(std::vector<double> values) {std::sort(values.begin(),values.end());return values[values.size()/2];}
void cpuTest() {
  uint64_t checks=0;
  for (uint32_t head=0;head<48;++head) {
    std::vector<uint32_t> old(128*128),ilp(128*128);
    for (uint32_t block=0;block<8;++block) {
      for (uint32_t sg=0;sg<16;++sg) for (uint32_t lane=0;lane<32;++lane) for (uint32_t i=0;i<4;++i)
        ++old[(block*16+sg)*128+lane*4+i];
      for (uint32_t sg=0;sg<8;++sg) for (uint32_t r=0;r<2;++r)
        for (uint32_t lane=0;lane<32;++lane) for (uint32_t i=0;i<4;++i)
          ++ilp[(block*16+r*8+sg)*128+lane*4+i];
    }
    require(old==ilp && std::all_of(old.begin(),old.end(),[](uint32_t n){return n==1;}),"GDN state owner map differs");++checks;
  }
  std::cout<<"{\"gdn_pair_cpu\":\"passed\",\"checks\":"<<checks<<",\"gpu_work\":false}\n";
}
}
int main(int argc,char **argv) {
  @autoreleasepool {
    try {
      if (argc==2 && std::string(argv[1])=="--cpu-self-test") {cpuTest();return 0;}
      require(argc==4,"usage: oracle METALLIB ACTUAL_MANIFEST REPORT_JSON");
      const auto source=manifest(argv[2]);const uint64_t rows=source.rows;
      std::map<std::string,std::vector<std::byte>> files;
      for (const auto &[name,bytes] : std::array<std::pair<const char *,uint64_t>,9>{{
          {"mixed",rows*10240*2},{"decay",rows*48*4},{"beta",rows*48*2},{"z",rows*6144*2},{"norm",128*2},
          {"initial_state",48*128*128*4},{"expected_state",48*128*128*4},
          {"expected_recurrence",rows*6144*2},{"expected_output",rows*6144*2}}}) files[name]=read(source,name,bytes);
      require(std::any_of(files["expected_state"].begin(),files["expected_state"].end(),[](std::byte x){return x!=std::byte{};}),
          "actual carried state unexpectedlyzero");
      MetalBackend backend(argv[1]);Fixture control(backend,source,files),candidate(backend,source,files);
      std::ostringstream report;report<<std::setprecision(12)<<"{\"schema\":\"splash-actual-gdn-pair-v1\",\"pass\":true,"
          <<"\"error_limit\":\"zero bytes across all BF16 outputs/F32 state/canaries\",\"timing_size_bytes\":"<<sizeof(CommandTiming)
          <<",\"rows\":"<<rows<<",\"layer\":0,\"cases\":[";
      bool first=true;
      for (bool carried : {false,true}) for (const auto variant : variants) {
        const auto &initial=files.at(carried?"expected_state":"initial_state");
        auto cg=graph(control,source,"flash_gdn_staged_v16_t16",512);
        auto ng=graph(candidate,source,variant.name,variant.threads);
        control.reset(initial);candidate.reset(initial);
        valid(backend.submitCommand(cg.dispatches()));valid(backend.submitCommand(ng.dispatches()));same(candidate,control,true);
        if (!carried) {
          require(std::memcmp(control.state.view.contents(),files.at("expected_state").data(),control.state.bytes)==0 &&
              std::memcmp(control.recurrence.view.contents(),files.at("expected_recurrence").data(),control.recurrence.bytes)==0 &&
              std::memcmp(control.output.view.contents(),files.at("expected_output").data(),control.output.bytes)==0,
              "actual captured model does not match production GDN baseline");
        }
        std::vector<double> cGPU,nGPU,cWall,nWall;
        for (uint32_t pair=0;pair<11;++pair) {
          control.reset(initial);candidate.reset(initial);CommandTiming c,n;
          if (pair%2) {n=backend.submitCommand(ng.dispatches());c=backend.submitCommand(cg.dispatches());}
          else {c=backend.submitCommand(cg.dispatches());n=backend.submitCommand(ng.dispatches());}
          valid(c);valid(n);same(candidate,control,true);
          if (pair>=2) {cGPU.push_back(c.gpuSeconds);nGPU.push_back(n.gpuSeconds);cWall.push_back(c.wallSeconds);nWall.push_back(n.wallSeconds);}
        }
        control.immutable();candidate.immutable();
        if (!first) report<<',';first=false;
        report<<"{\"variant\":"<<splash::json::quote(variant.name)<<",\"threads\":"<<variant.threads
            <<",\"carried_after_actual_2k\":"<<(carried?"true":"false")
            <<",\"all_output_state_canaries_byte_exact\":true,\"pairs\":9,\"control_gpu_ms\":"<<median(cGPU)*1000
            <<",\"candidate_gpu_ms\":"<<median(nGPU)*1000<<",\"gpu_speedup\":"<<median(cGPU)/median(nGPU)
            <<",\"control_wall_ms\":"<<median(cWall)*1000<<",\"candidate_wall_ms\":"<<median(nWall)*1000<<'}';
        std::cerr<<variant.name<<" carried="<<carried<<" exact speedup="<<median(cGPU)/median(nGPU)<<'\n';
      }
      report<<"]}\n";std::ofstream out(argv[3]);require(bool(out),"cannot write GDN paired report");out<<report.str();return 0;
    } catch (const std::exception &e) {std::cerr<<"actual GDN paired oracle failed: "<<e.what()<<'\n';return 1;}
  }
}
