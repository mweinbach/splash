// Root-only current-2K state/capture/numerical qualifier. CPU mode creates no device.
#include "inspect.hpp"
#include "capture.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashMoE.hpp"
#include "flash/FlashGreedyGPU.hpp"
#include "flash/FlashGatheredMPP.hpp"
#include "dev/benchmarks/gemv_decode_r1_worker_sep21/abi.hpp"
#include "dev/benchmarks/gemv_decode_sep21_v1b/quality.hpp"
#include "dev/benchmarks/dense_w8a8_sep21/worker_cache.hpp"
#include "dev/benchmarks/prefill4k_allrows_qmv_probe.h"
#include "engine/MemoryGovernor.hpp"
#include "engine/Json.hpp"
#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::flash;
using metal::MetalBuffer;using metal::MetalBackend;using metal::CommandGraph;
using Access=FlashDeepPrefixOracle;
namespace fs=std::filesystem;
namespace qual=gemv_quality;
constexpr uint32_t kSticky=0x80000000u;
constexpr uint64_t kLimit=4ULL<<30;
void require(bool p,const std::string &m){if(!p)throw std::runtime_error(m);}
std::string hash(const void *p,uint64_t n){CC_SHA256_CTX c{};CC_SHA256_Init(&c);const auto*b=static_cast<const uint8_t*>(p);
 while(n){auto s=CC_LONG(std::min<uint64_t>(n,1ULL<<30));CC_SHA256_Update(&c,b,s);b+=s;n-=s;}
 std::array<uint8_t,32>out{};CC_SHA256_Final(out.data(),&c);std::ostringstream s;for(auto v:out)s<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(v);return s.str();}
std::string digest(MetalBuffer b){return hash(b.contents(),b.sizeBytes());}
std::vector<uint8_t> bytes(MetalBuffer b){auto*p=static_cast<const uint8_t*>(b.contents());return {p,p+b.sizeBytes()};}
void exact(MetalBuffer b,const std::vector<uint8_t>&a,const char*m){require(a.size()==b.sizeBytes()&&!std::memcmp(a.data(),b.contents(),a.size()),m);}
uint32_t diag(MetalBuffer b){return *static_cast<const uint32_t*>(b.contents());}
void clear(MetalBuffer b){*static_cast<uint32_t*>(b.contents())=kSticky;}
struct Guard{MetalBuffer base,view;uint64_t size;void check()const{auto*p=static_cast<const uint8_t*>(base.contents());
 for(uint32_t i=0;i<64;++i)require(p[i]==0x5a,"capture prefix canary changed");
 for(uint64_t i=size+64;i<base.sizeBytes();++i)require(p[i]==0x5a,"capture suffix canary changed");}};
MetalBuffer guarded(MetalBackend&b,uint64_t n,std::vector<Guard>&g){auto x=b.allocateBuffer((n+128+16383)&~uint64_t(16383),metal::BufferStorage::Shared,"R1 governed diagnostic guard");
 std::memset(x.contents(),0x5a,x.sizeBytes());auto v=b.view(x,64,n);std::memset(v.contents(),0xa5,n);g.push_back({x,v,n});return v;}
std::span<const uint16_t> bf(MetalBuffer b){return {static_cast<const uint16_t*>(b.contents()),size_t(b.sizeBytes()/2)};}
struct Tap{MetalBuffer raw,scaled,value,diag;};
Tap tap(MetalBackend&b,uint32_t n,std::vector<Guard>&g){return {guarded(b,uint64_t(10)*n*4,g),guarded(b,uint64_t(10)*n*4,g),guarded(b,uint64_t(10)*n*2,g),guarded(b,4,g)};}
using Coeff=std::array<MetalBuffer,7>;
void chain(CommandGraph&g,const Coeff&c,MetalBuffer x,MetalBuffer ids,MetalBuffer a,MetalBuffer d,MetalBuffer flags,bool vector){
 const FlashGEMVDecodeR1Params p{1,10,512,0};
 g.add(vector?"gemv_decode_sep21_v4_l32_o4_gate_up":"flash_gathered_mpp_gate_up_m16_n64_sg4",{x,c[0],c[1],c[2],c[3],c[6],ids,a,flags},p,{vector?160u:10u,1,10},{128,1,1});
 g.add(vector?"gemv_decode_sep21_v4_l32_o4_down":"flash_gathered_mpp_down_m16_n64_sg4",{a,c[4],c[5],c[6],ids,d,flags},p,{vector?640u:40u,1,10},{128,1,1});}
void projection(CommandGraph&g,const Coeff&c,uint32_t plane,MetalBuffer x,MetalBuffer ids,const Tap&t,bool vector){
 const uint32_t k=plane==2?640:2560,n=plane==2?2560:640;
 g.add(vector?"r1diag_vector_projection":"r1diag_gathered_projection",{x,c[plane*2],c[plane*2+1],c[6],ids,t.raw,t.scaled,t.value,t.diag},
   FlashQMVProbeParams{1,10,k,n,uint32_t(plane==2),vector?4u:0u,512,0},{n/(vector?4u:64u),1,10},{128,1,1});}
struct Numeric{uint64_t samples=0,certFailures=0,strictSensitive=0,strictFailures=0;std::array<uint64_t,10>routeSamples{},routeCertFailures{},routeStrictFailures{};std::vector<std::string>firstFailures;bool pass()const{return !certFailures;}};
Numeric f64(const Coeff&c,uint32_t plane,MetalBuffer x,MetalBuffer ids,const Tap&t){
 uint32_t k=plane==2?640:2560,n=plane==2?2560:640;Numeric out;
 auto*id=static_cast<const int64_t*>(ids.contents());auto*ranks=static_cast<const uint32_t*>(c[6].contents());
 auto*codes=static_cast<const int8_t*>(c[plane*2].contents());auto*scales=static_cast<const float*>(c[plane*2+1].contents());
 auto*a=static_cast<const uint16_t*>(x.contents());auto*dot=static_cast<const float*>(t.raw.contents());auto*scaled=static_cast<const float*>(t.scaled.contents());
 auto*v=static_cast<const uint16_t*>(t.value.contents());
 for(uint32_t r=0;r<10;++r)for(uint32_t j=0;j<64;++j){uint32_t col=j*(n/64)+(r*17+plane*7)%(n/64);
  require(id[r]>=0&&id[r]<512&&ranks[id[r]]<512,"current captured ID/rank invalid");uint64_t row=uint64_t(ranks[id[r]])*n+col,at=uint64_t(r)*n+col;
  auto ref=qual::vectorReference(std::span<const uint16_t>(a+(plane==2?r:0)*k,k),std::span<const int8_t>(codes+row*k,k),scales[row],32);
  auto report=qual::assess(ref,dot[at],scaled[at],v[at],diag(t.diag));++out.samples;++out.routeSamples[r];out.certFailures+=!report.certifiedBoundSignFinitePass;out.routeCertFailures[r]+=!report.certifiedBoundSignFinitePass;out.routeStrictFailures[r]+=report.strictSensitiveFailure;
  out.strictSensitive+=report.strict.strictSensitive;out.strictFailures+=report.strictSensitiveFailure;
  if((!report.certifiedBoundSignFinitePass||report.strictSensitiveFailure)&&out.firstFailures.size()<16){std::ostringstream q;
   q<<"{\"route\":"<<r<<",\"column\":"<<col<<",\"dot_reference\":";qual::detail::jsonNumber(q,ref.dot);q<<",\"dot_observed\":";qual::detail::jsonNumber(q,dot[at]);
   q<<",\"dot_absolute_bound\":";qual::detail::jsonNumber(q,ref.dotAbsoluteBound);q<<",\"scaled_reference\":";qual::detail::jsonNumber(q,ref.scaled);q<<",\"scaled_observed\":";qual::detail::jsonNumber(q,scaled[at]);
   q<<",\"scaled_absolute_bound\":";qual::detail::jsonNumber(q,ref.scaledAbsoluteBound);q<<",\"reference_bf16_bits\":"<<ref.bf16<<",\"observed_bf16_bits\":"<<v[at]
    <<",\"exceptional\":"<<(ref.exceptional?"true":"false")<<",\"strict_sensitive\":"<<(report.strict.strictSensitive?"true":"false")<<",\"strict_sensitive_failure\":"<<(report.strictSensitiveFailure?"true":"false")
    <<",\"certified_bound_sign_finite_pass\":"<<(report.certifiedBoundSignFinitePass?"true":"false")<<'}';out.firstFailures.push_back(q.str());}}
 return out;}
void numericJSON(std::ostream&out,const Numeric&n){out<<"{\"samples\":"<<n.samples<<",\"certified_failures\":"<<n.certFailures<<",\"strict_sensitive\":"<<n.strictSensitive<<",\"strict_sensitive_failures\":"<<n.strictFailures<<",\"first_failures\":[";
 for(size_t i=0;i<n.firstFailures.size();++i){if(i)out<<',';out<<n.firstFailures[i];}out<<"],\"per_route_sampled_results\":[";
 for(uint32_t r=0;r<10;++r){if(r)out<<',';out<<"{\"route\":"<<r<<",\"samples\":"<<n.routeSamples[r]<<",\"certified_failures\":"<<n.routeCertFailures[r]<<",\"strict_sensitive_failures\":"<<n.routeStrictFailures[r]<<'}';}out<<"]}";}
void normJSON(std::ostream&out,MetalBuffer a,MetalBuffer b,uint32_t width){auto av=bf(a),bv=bf(b);out<<'[';
 for(uint32_t route=0;route<10;++route){if(route)out<<',';double as=0,bs=0;uint64_t nf=0;
  for(uint32_t i=0;i<width;++i){double x=qual::frozen::bf16Number(av[route*width+i]),y=qual::frozen::bf16Number(bv[route*width+i]);if(!std::isfinite(x)||!std::isfinite(y)){++nf;continue;}as+=x*x;bs+=y*y;}
  out<<"{\"route\":"<<route<<",\"reference_squared_norm\":";qual::detail::jsonNumber(out,as);out<<",\"candidate_squared_norm\":";qual::detail::jsonNumber(out,bs);
  out<<",\"reference_exact_zero_norm\":"<<(as==0&&!nf?"true":"false")<<",\"candidate_exact_zero_norm\":"<<(bs==0&&!nf?"true":"false")<<",\"nonfinite_pairs\":"<<nf<<'}';}out<<']';}
void f32ComparisonJSON(std::ostream&out,MetalBuffer a,MetalBuffer b,uint32_t width){auto*av=static_cast<const float*>(a.contents());auto*bv=static_cast<const float*>(b.contents());
 auto*ab=static_cast<const uint32_t*>(a.contents());auto*bb=static_cast<const uint32_t*>(b.contents());uint64_t mismatches=0,nf=0,maxAt=0;double sum=0,ref=0,max=0;
 out<<"{\"routes\":[";for(uint32_t route=0;route<10;++route){if(route)out<<',';uint64_t rm=0,rnf=0,rat=0;double rs=0,rr=0,rmax=0;
  for(uint32_t col=0;col<width;++col){uint64_t at=uint64_t(route)*width+col;rm+=ab[at]!=bb[at];if(!std::isfinite(av[at])||!std::isfinite(bv[at])){++rnf;continue;}double d=double(av[at])-bv[at];rs+=d*d;rr+=double(av[at])*av[at];if(std::abs(d)>rmax){rmax=std::abs(d);rat=at;}}
  mismatches+=rm;nf+=rnf;sum+=rs;ref+=rr;if(rmax>max){max=rmax;maxAt=rat;}
  out<<"{\"route\":"<<route<<",\"bit_mismatches\":"<<rm<<",\"nonfinite_pairs\":"<<rnf<<",\"relative_l2\":";qual::detail::jsonNumber(out,rr?std::sqrt(rs/rr):rs?INFINITY:0);
  out<<",\"max_abs\":";qual::detail::jsonNumber(out,rmax);out<<",\"max_mismatch_column\":"<<rat%width<<'}';}
 out<<"],\"global_bit_mismatches\":"<<mismatches<<",\"global_nonfinite_pairs\":"<<nf<<",\"global_relative_l2\":";qual::detail::jsonNumber(out,ref?std::sqrt(sum/ref):sum?INFINITY:0);
 out<<",\"max_abs\":";qual::detail::jsonNumber(out,max);out<<",\"max_mismatch_route\":"<<maxAt/width<<",\"max_mismatch_column\":"<<maxAt%width
  <<",\"reference_bits_at_max\":"<<ab[maxAt]<<",\"candidate_bits_at_max\":"<<bb[maxAt]<<'}';}
std::string diagnoseFailedChain(MetalBackend&backend,const Coeff&c,r1_capture::Layer&l,uint32_t layer,MetalBuffer mppA,MetalBuffer vecA,MetalBuffer mppD,MetalBuffer vecD,
 std::array<std::array<Tap,2>,3>&taps,const Tap&own,const qual::MetricReport&am,const qual::MetricReport&dm,const fs::path&report,MetalBuffer recon,MetalBuffer flags,bool&additionalGateFailed){
 std::ostringstream out;out<<std::setprecision(17)<<"{\"layer\":"<<layer<<",\"first_failed_stage\":"<<json::quote(!am.pass()?"activated":!dm.pass()?"own_activation_down":"none")
  <<",\"activated_failed\":"<<(!am.pass()?"true":"false")<<",\"down_failed\":"<<(!dm.pass()?"true":"false")<<",\"original_thresholds_unchanged\":true,\"chain_activated\":";am.json(out);
 out<<",\"chain_down_own_activation\":";dm.json(out);out<<",\"activated_norms_and_zero_finite\":";normJSON(out,mppA,vecA,640);out<<",\"down_norms_and_zero_finite\":";normJSON(out,mppD,vecD,2560);
 out<<",\"original_certificate_inputs\":{\"source_identity_sha256\":\"bd384554f00dafdc285bf6df2dd34bebfe2d963fa1c955e0834c8a08e656f29b\",\"kernel\":\"gemv_decode_sep21_v4_l32_o4\",\"producer_threads\":128,\"lanes\":32,\"outputs_per_CTA\":4,\"gate_K\":2560,\"down_K\":640,\"gate_depth\":28,\"down_depth\":13,\"sampled_columns_per_route\":64,\"u32\":\"2^-23 RN/RTZ\",\"FTZ\":\"4*N*2^-126*(1+gamma(D,u32))\",\"perroute_max_relative_l2\":0.0001,\"perroute_min_cosine\":0.999999},\"actual_hidden_sha256\":"<<json::quote(digest(l.hidden))<<",\"actual_IDs_sha256\":"<<json::quote(digest(l.ids))<<",\"rank_sha256\":"<<json::quote(digest(c[6]))<<",\"actual_original_IDs\":[";
 auto*ids=static_cast<const int64_t*>(l.ids.contents());auto*ranks=static_cast<const uint32_t*>(c[6].contents());for(uint32_t r=0;r<10;++r){if(r)out<<',';out<<"{\"route\":"<<r<<",\"ID\":"<<ids[r]<<",\"rank\":"<<(ids[r]>=0&&ids[r]<512?ranks[ids[r]]:UINT32_MAX)<<'}';}out<<"],\"projection_diagnostics\":[";
 for(uint32_t p=0;p<3;++p){if(p)out<<',';auto input=p==2?mppA:l.hidden;for(uint32_t v=0;v<2;++v){clear(taps[p][v].diag);CommandGraph g;projection(g,c,p,input,l.ids,taps[p][v],bool(v));(void)backend.submitCommand(g.dispatches());}
  auto metrics=qual::metrics(bf(taps[p][0].value),bf(taps[p][1].value),p==2?2560:640);auto n=f64(c,p,input,l.ids,taps[p][1]);
  additionalGateFailed|=!metrics.pass()||!n.pass()||diag(taps[p][0].diag)!=kSticky||diag(taps[p][1].diag)!=kSticky;
  out<<"{\"plane\":"<<p<<",\"stage\":"<<json::quote(p==0?"gate":p==1?"up":"isolated_down_on_same_MPP_activation")<<",\"gathered_diagnostic\":"<<diag(taps[p][0].diag)<<",\"vector_diagnostic\":"<<diag(taps[p][1].diag)<<",\"bf16_perroute_global\":";metrics.json(out);
  out<<",\"raw_f32_comparison\":";f32ComparisonJSON(out,taps[p][0].raw,taps[p][1].raw,p==2?2560:640);out<<",\"scaled_f32_comparison\":";f32ComparisonJSON(out,taps[p][0].scaled,taps[p][1].scaled,p==2?2560:640);
  out<<",\"original_vector_F64_certificate\":";numericJSON(out,n);out<<'}';}
 clear(own.diag);CommandGraph g;projection(g,c,2,vecA,l.ids,own,true);(void)backend.submitCommand(g.dispatches());auto n=f64(c,2,vecA,l.ids,own);out<<"],\"own_vector_down_F64_certificate\":";numericJSON(out,n);
 additionalGateFailed|=!n.pass()||digest(own.value)!=digest(vecD)||diag(own.diag)!=kSticky;
 out<<",\"own_vector_down_tap_matches_shipping_bits\":"<<(digest(own.value)==digest(vecD)?"true":"false")<<",\"compiled_swiglu_tap_self_bit_matches\":[";
 for(uint32_t v=0;v<2;++v){if(v)out<<',';clear(flags);CommandGraph q;addSiLUMultiply(q,taps[0][v].value,taps[1][v].value,recon,flags,1,640,10);(void)backend.submitCommand(q.dispatches());bool match=digest(recon)==digest(v?vecA:mppA)&&diag(flags)==kSticky;additionalGateFailed|=!match;out<<(match?"true":"false");}out<<']';
#if !SPLASH_R1_DIAGNOSTIC_ALL_LAYERS
 // Failure-only bounded exports retain actual register inputs and both own
 // chains. Original full512 payloads are never exported or duplicated.
 fs::path dir=report.string()+".failed-layer";fs::create_directory(dir);std::vector<std::pair<std::string,MetalBuffer>>files{{"actual-hidden.bf16",l.hidden},{"actual-IDs.i64",l.ids},{"actual-activated.bf16",l.activated},{"actual-down.bf16",l.down},{"gathered-activated.bf16",mppA},{"vector-activated.bf16",vecA},{"gathered-down.bf16",mppD},{"vector-down.bf16",vecD},{"actual-ranks.u32",c[6]}};
 for(uint32_t p=0;p<3;++p)for(uint32_t v=0;v<2;++v){std::string name="plane"+std::to_string(p)+(v?"-vector":"-gathered");files.push_back({name+"-raw.f32",taps[p][v].raw});files.push_back({name+"-scaled.f32",taps[p][v].scaled});files.push_back({name+"-projection.bf16",taps[p][v].value});}
 out<<",\"failure_capture_exports\":[";bool first=true;uint64_t total=0;for(auto&[name,buffer]:files){auto path=dir/name;std::ofstream file(path,std::ios::binary);file.write(static_cast<const char*>(buffer.contents()),buffer.sizeBytes());require(bool(file),"failed-layer diagnostic export write failed");total+=buffer.sizeBytes();if(!first)out<<',';first=false;out<<"{\"path\":"<<json::quote(path.string())<<",\"bytes\":"<<buffer.sizeBytes()<<",\"sha256\":"<<json::quote(digest(buffer))<<'}';}
 out<<"],\"failure_capture_export_bytes\":"<<total<<",\"export_scope\":\"actual selected layer inputs/ranks/outputs/taps only; readonly coefficient planes identified by constructor certificate, not copied\"}";
#else
 (void)report;out<<",\"capture_exported\":false,\"diagnostic_only\":true,\"qualification_not_granted\":true}";
#endif
 return out.str();}
std::string selectedWeights(const Coeff&c,MetalBuffer ids){std::string s;auto*id=static_cast<const int64_t*>(ids.contents());auto*ranks=static_cast<const uint32_t*>(c[6].contents());
 for(uint32_t p=0;p<3;++p){uint32_t n=p==2?2560:640,k=p==2?640:2560;for(uint32_t r=0;r<10;++r){require(id[r]>=0&&id[r]<512&&ranks[id[r]]<512,"invalid actual coefficient selection");
   uint64_t rank=ranks[id[r]];s+=hash(static_cast<const int8_t*>(c[p*2].contents())+rank*n*k,uint64_t(n)*k);
   s+=hash(static_cast<const float*>(c[p*2+1].contents())+rank*n,uint64_t(n)*4);}}
 s+=digest(c[6]);return hash(s.data(),s.size());}
std::vector<uint32_t> prompt(const char*path){NSData*d=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];require(d!=nil,"canonical token JSON missing");
 require(hash(d.bytes,d.length)=="4985e55294b83c72cb9e51e00c40f918460b6c4f560cb5b32d4be3662e540b57","canonical2K file differs");
 NSError*e=nil;id j=[NSJSONSerialization JSONObjectWithData:d options:0 error:&e];require(!e&&[j isKindOfClass:[NSArray class]],"token JSON invalid");
 std::vector<uint32_t>v;for(NSNumber*x:static_cast<NSArray*>(j)){auto n=x.unsignedIntValue;require(n<248320,"token outside vocabulary");v.push_back(n);}require(v.size()==2048,"exact2K tokens required");return v;}
std::vector<uint8_t> stateBlob(const FlashForward&t,const FlashRequestState&s,MetalBuffer logits={},MetalBuffer hidden={},MetalBuffer greedy={}){
 std::vector<uint8_t>out;auto add=[&](const void*p,uint64_t n){require(out.size()+n<kLimit,"state spill exceeds4GiB");auto*b=static_cast<const uint8_t*>(p);out.insert(out.end(),b,b+n);};
 auto meta=Access::metadata(t,s);uint64_t m=meta.size();add(&m,8);add(meta.data(),meta.size());
 for(auto&p:Access::planes(s)){uint64_t n=p.buffer.sizeBytes();add(&n,8);add(p.buffer.contents(),n);}
 for(auto b:{logits,hidden,greedy}){uint64_t n=b?b.sizeBytes():0;add(&n,8);if(n)add(b.contents(),n);}return out;}
void stateFile(const fs::path&p,const std::vector<uint8_t>&b,bool compare){if(compare){require(fs::file_size(p)==b.size(),"prefill state extent differs");std::ifstream in(p,std::ios::binary);std::array<uint8_t,65536>s{};size_t at=0;
 while(at<b.size()){size_t n=std::min(s.size(),b.size()-at);in.read(reinterpret_cast<char*>(s.data()),n);require(in.gcount()==std::streamsize(n)&&!std::memcmp(s.data(),b.data()+at,n),"prefill134-plane/outputs mismatch");at+=n;}}
 else{require(!fs::exists(p),"fresh state spill required");std::ofstream o(p,std::ios::binary);o.write(reinterpret_cast<const char*>(b.data()),b.size());require(bool(o),"state spill write failed");}}
uint32_t winner(MetalBuffer logits){auto*p=static_cast<const uint16_t*>(logits.contents());uint32_t best=0;float v=-INFINITY;for(uint32_t i=0;i<248320;++i){float x=std::bit_cast<float>(uint32_t(p[i])<<16);require(std::isfinite(x),"nonfinite actual logits");if(x>v){v=x;best=i;}}return best;}
template<class F>void rejected(F&&f,FlashForward&t,FlashRequestState&s){auto before=stateBlob(t,s);auto bound=Access::binding(s);bool fail=false;try{f();}catch(const std::exception&){fail=true;}
 require(fail,"invalid API accepted");require(stateBlob(t,s)==before&&Access::binding(s)==bound,"rejected API mutated state/ownership");}
void safety(MetalBackend&backend,const Coeff&original,r1_capture::Layer&input,std::vector<Guard>&guards){
 auto invalid=guarded(backend,80,guards),duplicates=guarded(backend,80,guards),badX=guarded(backend,5120,guards),zeroX=guarded(backend,5120,guards);
 auto copiedRanks=guarded(backend,original[6].sizeBytes(),guards),badIntermediate=guarded(backend,12800,guards),a=guarded(backend,12800,guards),d=guarded(backend,51200,guards),flags=guarded(backend,4,guards);
 std::memset(invalid.contents(),0xff,80);std::memcpy(duplicates.contents(),input.ids.contents(),80);
 static_cast<int64_t*>(duplicates.contents())[1]=static_cast<const int64_t*>(input.ids.contents())[0];
 std::memcpy(badX.contents(),input.hidden.contents(),5120);std::memcpy(zeroX.contents(),input.hidden.contents(),5120);
 auto*bad=static_cast<uint16_t*>(badX.contents());auto*zero=static_cast<uint16_t*>(zeroX.contents());bad[0]=0x7fc0;bad[1]=0x7f80;bad[2]=0xff80;zero[0]=zero[1]=zero[2]=0;
 std::memcpy(copiedRanks.contents(),original[6].contents(),copiedRanks.sizeBytes());auto id0=static_cast<const int64_t*>(input.ids.contents())[0];static_cast<uint32_t*>(copiedRanks.contents())[id0]=512;
 std::fill_n(static_cast<uint16_t*>(badIntermediate.contents()),6400,uint16_t(0x7fc0));
 auto gate=[&](const Coeff&c,MetalBuffer x,MetalBuffer ids){clear(flags);CommandGraph g;
  g.add("gemv_decode_sep21_v4_l32_o4_gate_up",{x,c[0],c[1],c[2],c[3],c[6],ids,a,flags},FlashGEMVDecodeR1Params{1,10,512,0},{160,1,10},{128,1,1});(void)backend.submitCommand(g.dispatches());};
 auto down=[&](const Coeff&c,MetalBuffer x,MetalBuffer ids){clear(flags);CommandGraph g;
  g.add("gemv_decode_sep21_v4_l32_o4_down",{x,c[4],c[5],c[6],ids,d,flags},FlashGEMVDecodeR1Params{1,10,512,0},{640,1,10},{128,1,1});(void)backend.submitCommand(g.dispatches());};
 auto poison=[](MetalBuffer b,uint64_t count){auto*p=static_cast<const uint16_t*>(b.contents());for(uint64_t i=0;i<count;++i)require(p[i]==0x7fc0,"invalid route was not canonicalNaN poisoned");};
 const auto badHash=digest(badX),zeroHash=digest(zeroX),rankHash=digest(copiedRanks),idsHash=digest(input.ids);
 gate(original,input.hidden,invalid);require(diag(flags)==(kSticky|1),"invalidgate diagnostic differs");poison(a,6400);
 gate(original,badX,invalid);require(diag(flags)==(kSticky|5),"invalidgate hidden scan did not preservebit4");poison(a,6400);
 down(original,badIntermediate,invalid);require(diag(flags)==(kSticky|5),"invaliddown skip/poison diagnostic differs");poison(d,25600);
 gate(original,badX,input.ids);require(diag(flags)==(kSticky|4),"nonfinite sanitization diagnostic differs");auto sanitized=bytes(a);
 gate(original,zeroX,input.ids);require(diag(flags)==kSticky,"positivezero reference diagnostic changed");exact(a,sanitized,"nonfinite operands not positivezero equivalent");
 gate(original,input.hidden,duplicates);require(diag(flags)==(kSticky|1),"duplicate route diagnostic differs");require(!std::memcmp(a.contents(),static_cast<const uint8_t*>(a.contents())+1280,1280),"duplicate routes didnotcompute independently/equally");
 auto corrupt=original;corrupt[6]=copiedRanks;gate(corrupt,input.hidden,input.ids);require(diag(flags)==(kSticky|1),"corrupt rank gate diagnostic differs");poison(a,640);
 down(corrupt,badIntermediate,input.ids);require((diag(flags)&5)==5,"corrupt rank down poison diagnostic differs");poison(d,2560);
 require(digest(badX)==badHash&&digest(zeroX)==zeroHash&&digest(copiedRanks)==rankHash&&digest(input.ids)==idsHash,"exceptional fixtures/original IDs mutated");
 for(auto&g:guards)g.check();
}

int main(int argc,char**argv){@autoreleasepool{
 fs::path report;std::string failureDiagnostic;std::vector<std::string>allLayerDiagnostics;bool anyChainFailed=false;try{
 if(argc==2&&std::string_view(argv[1])=="--cpu-only"){require(qual::cpuSelfTest(),"frozen numerical certificate CPU failed");
  require(sizeof(FlashGEMVDecodeR1Params)==16&&sizeof(FlashQMVProbeParams)==32,"ABI mismatch");std::cout<<"{\"pass\":true,\"gpu_work\":false,\"model_payload_reads\":false}\n";return 0;}
 require(argc==8&&std::string_view(argv[1])=="--root-gpu","usage: qualifier --root-gpu METALLIB MODEL TOKENS_JSON FRESH_REPORT PREFILL_SPILL export|compare");
 report=argv[5];require(!fs::exists(report),"fresh qualifier report required");bool compare=std::string_view(argv[7])=="compare";
 require(compare||std::string_view(argv[7])=="export","role invalid");require(compare==bool(SPLASH_R1_CANDIDATE),"compiled role mismatch");
 const auto tokens=prompt(argv[4]);uint64_t peak=0,planned=0,mappedBaseline=0,targetBytes=0,stateBytes=0,diagnosticBytes=0,currentDelta=0,afterTargetRelease=0,afterWeightsRelease=0,afterStop=0;
 bool backendCreated=false,backendDestroyed=false,governorGrowth=false,hostValid=false;
 uint32_t guardsPassed=0,numericLayers=0;Numeric totals;uint64_t observedBF16Differences=0;
 bool replayExact=false,prefillExact=false,ownershipStable=false,canaries=true;std::vector<std::string> causalHashes;
 {struct Lifetime{bool &destroyed;~Lifetime(){destroyed=true;}}lifetime{backendDestroyed};MetalBackend backend(argv[2]);backendCreated=true;
 {const auto weights=FlashWeights::load(backend,argv[3]);mappedBaseline=backend.refreshMemoryStats().allocatedBytes;
 uint64_t physical=NSProcessInfo.processInfo.physicalMemory,reserve=std::max<uint64_t>(16ULL<<30,physical/10);
  engine::MemoryGovernor governor(backend,physical-reserve,reserve);
  uint64_t targetPlan=FlashForward::workspacePlannedBytes(4096,2048,4)+FlashForward::expertCachePlannedBytes(weights)+FlashForward::floatDenseCachePlannedBytes(weights)+FlashForward::int8HeadPlannedBytes(weights)
    +FlashDenseCache::plannedBytes(weights,FlashDenseCache::defaultPrefixes(weights,true))+flashMoEBlockedWorkspacePlannedBytes(2048,10)+dense_w8a8_sep21::Cache::plannedBytes();
  planned=targetPlan+2*FlashForward::requestStateBytes(4096)+2*Access::guardedStateAllowance()+(96ULL<<20);
  auto admission=governor.tryReserve(planned);require(bool(admission),"normalGovernor denied diagnostic/state/category plan");
  {FlashForward target(backend,weights,4096,2048,4);
   const auto afterTarget=backend.refreshMemoryStats().allocatedBytes;require(afterTarget>=mappedBaseline,"target allocation ledger regressed");targetBytes=afterTarget-mappedBaseline;
   require(targetBytes==target.workspaceBytes()&&targetBytes<=targetPlan,"actual target/category delta exceeds reserved plan");
   auto state=target.createState(),replay=target.createState();auto stateGuards=Access::guardState(backend,state),replayGuards=Access::guardState(backend,replay);
   const auto afterStates=backend.refreshMemoryStats().allocatedBytes;require(afterStates>=afterTarget,"state allocation ledger regressed");stateBytes=afterStates-afterTarget;
   require(stateBytes<=2*FlashForward::requestStateBytes(4096)+2*Access::guardedStateAllowance(),"actual two guarded states exceed reserved state allowance");
   std::vector<Guard>guards;r1_capture::Context capture;
   for(auto&l:capture.layers)l={guarded(backend,5120,guards),guarded(backend,80,guards),guarded(backend,12800,guards),guarded(backend,51200,guards)};
   auto mppA=guarded(backend,12800,guards),vecA=guarded(backend,12800,guards),mppD=guarded(backend,51200,guards),vecD=guarded(backend,51200,guards),flags=guarded(backend,4,guards),recon=guarded(backend,12800,guards);
   std::array<std::array<Tap,2>,3>taps;for(uint32_t p=0;p<3;++p)for(auto&t:taps[p])t=tap(backend,p==2?2560:640,guards);
   auto own=tap(backend,2560,guards);const auto afterDiagnostic=backend.refreshMemoryStats().allocatedBytes;
   require(afterDiagnostic>=afterStates&&afterDiagnostic-afterStates<=(96ULL<<20),"initial actual captures/taps exceed96MiB allowance");
   require(afterDiagnostic>=mappedBaseline&&afterDiagnostic-mappedBaseline<=planned,"initial live allocation delta exceeds reserved plan");
   // All large owners now have actual charges. Commit the admitted reservation
   // before normal execution, so growth checks do not double-count its bytes.
   admission->commit();
   const auto pref=target.forward(state,tokens,false,true);require(state.logicalLength()==2048,"actualpref length differs");
#if !SPLASH_R1_DIAGNOSTIC_ALL_LAYERS
   auto prefBlob=stateBlob(target,state,pref.logitsBF16,pref.hiddenBF16,pref.greedyResultsU32);stateFile(argv[6],prefBlob,compare);prefillExact=true;
#endif
   auto token=winner(pref.logitsBF16);Access::clone(state,replay);
   const auto binding=Access::binding(state);FlashForwardResult first;
   {r1_capture::Scoped scope(capture);
#if SPLASH_R1_CANDIDATE
    first=target.forwardStandardDecode(state,std::span(&token,1));
#else
    first=target.forward(state,std::span(&token,1));
#endif
   }
   require(std::all_of(capture.seen.begin(),capture.seen.end(),[](bool x){return x;}),"actualR1 causal capture must cover48layers");
   require(state.logicalLength()==2049&&target.ownsState(state)&&binding==Access::binding(state),"R1 owner/length changed unexpectedly");ownershipStable=true;
   auto firstBlob=stateBlob(target,state,first.logitsBF16,Access::hidden(target,1),first.greedyResultsU32);
   const auto*store=target.batchInt8ExpertStore();require(store&&store->identitySha256()=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","actualFull512 coeffstore differs");
   for(uint32_t layer=0;layer<48;++layer){auto&l=capture.layers[layer];auto c=Access::coefficients(*store,layer);const auto immutable=selectedWeights(c,l.ids),xh=digest(l.hidden),ih=digest(l.ids);
    causalHashes.push_back(hash((xh+ih+immutable).data(),(xh+ih+immutable).size()));
    for(bool vector:{false,true}){clear(flags);CommandGraph g;chain(g,c,l.hidden,l.ids,vector?vecA:mppA,vector?vecD:mppD,flags,vector);(void)backend.submitCommand(g.dispatches());require(diag(flags)==kSticky,"shipping replay diagnostic changed");}
#if SPLASH_R1_CANDIDATE
    require(digest(vecA)==digest(l.activated)&&digest(vecD)==digest(l.down),"actualvectorregister/canonical capture differs from same-plane shipping replay");
#else
    require(digest(mppA)==digest(l.activated)&&digest(mppD)==digest(l.down),"actualgatheredregister/canonical capture differs from same-plane shipping replay");
#endif
    auto am=qual::metrics(bf(mppA),bf(vecA),640),dm=qual::metrics(bf(mppD),bf(vecD),2560);
    anyChainFailed|=!am.pass()||!dm.pass();
    bool additionalGateFailed=false;
#if SPLASH_R1_DIAGNOSTIC_ALL_LAYERS
    allLayerDiagnostics.push_back(diagnoseFailedChain(backend,c,l,layer,mppA,vecA,mppD,vecD,taps,own,am,dm,report,recon,flags,additionalGateFailed));anyChainFailed|=additionalGateFailed;
    for(auto&guard:guards)guard.check();require(selectedWeights(c,l.ids)==immutable&&digest(l.hidden)==xh&&digest(l.ids)==ih,"diagnostic original inputs/coefficients changed");++numericLayers;continue;
#else
    if(!am.pass()||!dm.pass())failureDiagnostic=diagnoseFailedChain(backend,c,l,layer,mppA,vecA,mppD,vecD,taps,own,am,dm,report,recon,flags,additionalGateFailed);
#endif
    require(am.pass()&&dm.pass(),"frozen per-global/per-route shipping-chain1e-4/.999999 gate failed");observedBF16Differences+=am.aggregate.mismatches+dm.aggregate.mismatches;
    for(uint32_t p=0;p<3;++p){auto input=p==2?mppA:l.hidden;for(uint32_t v=0;v<2;++v){auto&t=taps[p][v];clear(t.diag);CommandGraph g;projection(g,c,p,input,l.ids,t,bool(v));(void)backend.submitCommand(g.dispatches());require(diag(t.diag)==kSticky,"projection tap diagnostic changed");}
     require(qual::metrics(bf(taps[p][0].value),bf(taps[p][1].value),p==2?2560:640).pass(),"frozen isolated projection per-route gate failed");
     auto n=f64(c,p,input,l.ids,taps[p][1]);totals.samples+=n.samples;totals.certFailures+=n.certFailures;totals.strictSensitive+=n.strictSensitive;totals.strictFailures+=n.strictFailures;require(n.pass(),"originalRN/RTZ/FTZ/F64 sampled vector envelope failed");}
    for(uint32_t v=0;v<2;++v){clear(flags);CommandGraph g;addSiLUMultiply(g,taps[0][v].value,taps[1][v].value,recon,flags,1,640,10);(void)backend.submitCommand(g.dispatches());require(digest(recon)==digest(v?vecA:mppA)&&diag(flags)==kSticky,"projection taps do not reproduce own shipping compiledSwiGLU bits");}
    clear(own.diag);CommandGraph ownGraph;projection(ownGraph,c,2,vecA,l.ids,own,true);(void)backend.submitCommand(ownGraph.dispatches());require(digest(own.value)==digest(vecD)&&diag(own.diag)==kSticky,"own vector down tap does not reproduce shipping bits");
    auto n=f64(c,2,vecA,l.ids,own);require(n.pass(),"own activation F64 down envelope failed");totals.samples+=n.samples;totals.certFailures+=n.certFailures;totals.strictSensitive+=n.strictSensitive;totals.strictFailures+=n.strictFailures;
    const auto savedA=bytes(vecA),savedD=bytes(vecD);std::memset(vecA.contents(),0xa5,12800);std::memset(vecD.contents(),0xa5,51200);clear(flags);CommandGraph replayGraph;chain(replayGraph,c,l.hidden,l.ids,vecA,vecD,flags,true);(void)backend.submitCommand(replayGraph.dispatches());exact(vecA,savedA,"poisoned vector activated replay differs");exact(vecD,savedD,"poisoned vector down replay differs");
    require(diag(flags)==kSticky&&selectedWeights(c,l.ids)==immutable&&digest(l.hidden)==xh&&digest(l.ids)==ih,"actual immutableinput/rank/scale/coefficient snapshot changed");++numericLayers;
   }
#if !SPLASH_R1_DIAGNOSTIC_ALL_LAYERS
   safety(backend,Access::coefficients(*store,0),capture.layers[0],guards);
   FlashForwardResult again;
#if SPLASH_R1_CANDIDATE
   again=target.forwardStandardDecode(replay,std::span(&token,1));
#else
   again=target.forward(replay,std::span(&token,1));
#endif
   require(stateBlob(target,replay,again.logitsBF16,Access::hidden(target,1),again.greedyResultsU32)==firstBlob,"same-owner/current2K stdR1 full134state+output replay differs");replayExact=true;
   // Repeat a SECOND tagged stdR1 from the real progressed state, independently
   // of the first replay. No exactness to the MPP state is claimed.
   auto futureToken=winner(again.logitsBF16);FlashForwardResult s0,s1;
#if SPLASH_R1_CANDIDATE
   s0=target.forwardStandardDecode(state,std::span(&futureToken,1));auto stdFuture=stateBlob(target,state,s0.logitsBF16,Access::hidden(target,1),s0.greedyResultsU32);
   s1=target.forwardStandardDecode(replay,std::span(&futureToken,1));
#else
   s0=target.forward(state,std::span(&futureToken,1));auto stdFuture=stateBlob(target,state,s0.logitsBF16,Access::hidden(target,1),s0.greedyResultsU32);
   s1=target.forward(replay,std::span(&futureToken,1));
#endif
   require(stateBlob(target,replay,s1.logitsBF16,Access::hidden(target,1),s1.greedyResultsU32)==stdFuture,"second actual stdR1future state/output replay differs");
   // Original callers remain byte repeatable after the new tagged path.
   auto next=winner(again.logitsBF16);auto f0=target.forward(state,std::span(&next,1));auto expected=stateBlob(target,state,f0.logitsBF16,Access::hidden(target,1),f0.greedyResultsU32);
   auto f1=target.forward(replay,std::span(&next,1));require(stateBlob(target,replay,f1.logitsBF16,Access::hidden(target,1),f1.greedyResultsU32)==expected,"excludedforward R1future replay differs");
#if SPLASH_R1_CANDIDATE
   rejected([&]{(void)target.forwardStandardDecode(state,{});},target,state);++guardsPassed;
   std::array<uint32_t,2>two{token,token};rejected([&]{(void)target.forwardStandardDecode(state,two);},target,state);++guardsPassed;
   Access::poison(state,true);rejected([&]{(void)target.forwardStandardDecode(state,std::span(&token,1));},target,state);Access::poison(state,false);++guardsPassed;
   Access::pending(state,true);rejected([&]{(void)target.forwardStandardDecode(state,std::span(&token,1));},target,state);Access::pending(state,false);++guardsPassed;
   auto oldOwner=Access::changeOwner(state,std::make_shared<const uint8_t>(0));rejected([&]{(void)target.forwardStandardDecode(state,std::span(&token,1));},target,state);(void)Access::changeOwner(state,oldOwner);++guardsPassed;
   auto oldLength=Access::length(replay,0);rejected([&]{(void)target.forwardStandardDecode(replay,std::span(&token,1));},target,replay);(void)Access::length(replay,oldLength);++guardsPassed;
   // Actual Store host extent/alias validation, before any graph/counter mutation.
   const auto counters=store->gemvDecodeR1Counters();auto &l=capture.layers[0];
   for(uint32_t which=0;which<3;++which){CommandGraph g;bool refused=false;try{store->addGEMVDecodeR1GateUp(g,0,which==0?backend.view(l.hidden,0,5118):l.hidden,l.ids,which==1?l.hidden:which==2?backend.view(l.activated,0,12798):l.activated,flags,1);}catch(const std::exception&){refused=true;}
    require(refused&&g.dispatches().empty(),"short/alias Store view did not refuse before graph mutation");++guardsPassed;}
   const auto after=store->gemvDecodeR1Counters();require(counters.gateCalls==after.gateCalls&&counters.downCalls==after.downCalls,"refused host views changed graph counters");
#endif
   // Original untagged target verification truncates a provisional four-token
   // window to its retained three-token prefix. The replay uses the same real
   // same-owner arena sequentially; this is not a trained-head/MTP proof.
   std::array<uint32_t,4>window{token,next,token,next};
   (void)target.verify(state,window);require(Access::pending(state),"verify didnotpublish pending state");
#if SPLASH_R1_CANDIDATE
   rejected([&]{(void)target.forwardStandardDecode(state,std::span(&token,1));},target,state);++guardsPassed;
#endif
   (void)target.commitVerify(state,3);require(!Access::pending(state),"commit didnotresolve pending state");auto retained=stateBlob(target,state);
   (void)target.verify(replay,window);(void)target.commitVerify(replay,3);require(stateBlob(target,replay)==retained,"original verification retained-prefix/truncate replay differs");
   auto recovered=target.forward(state,std::span(&token,1));auto recovery=stateBlob(target,state,recovered.logitsBF16,Access::hidden(target,1),recovered.greedyResultsU32);
   auto restored=target.forward(replay,std::span(&token,1));require(stateBlob(target,replay,restored.logitsBF16,Access::hidden(target,1),restored.greedyResultsU32)==recovery,"healthy future recovery after guards/commit differs");
#endif
   for(auto&g:guards)g.check();for(auto&g:stateGuards)g.check();for(auto&g:replayGuards)g.check();require(!Access::pending(state)&&!Access::pending(replay),"qualifier left pending state");
   const auto actual=backend.refreshMemoryStats();peak=actual.peakAllocatedBytes;
   require(actual.allocatedBytes>=mappedBaseline&&peak>=actual.allocatedBytes,"actual current/peak ledger regressed");currentDelta=actual.allocatedBytes-mappedBaseline;
   require(currentDelta<=planned&&peak-mappedBaseline<=planned,"actual live/peak delta exceeds reserved target/state/diagnostic plan");
   require(actual.allocatedBytes>=afterStates&&peak>=afterStates,"diagnostic allocation ledger regressed");diagnosticBytes=peak-afterStates;
   require(diagnosticBytes<=(96ULL<<20),"actual peak diagnostic/transient allowance exceeds96MiB");
   const auto gov=governor.snapshot();governorGrowth=gov.growthAllowed;hostValid=gov.hostMeasurementValid;
   require(gov.deniedReservations==0&&governorGrowth&&hostValid,"normalGovernor denied growth/host measurement or reservation");
  }
  afterTargetRelease=backend.refreshMemoryStats().allocatedBytes;
  require(afterTargetRelease==mappedBaseline,"target/states/guard/capture/tap owners didnotreturn to mapped-model baseline");
 }
 afterWeightsRelease=backend.refreshMemoryStats().allocatedBytes;
 require(afterWeightsRelease==0,"weights owners didnotrelease tozero");
 backend.stop();afterStop=backend.refreshMemoryStats().allocatedBytes;require(afterStop==0,"stoppedbackend retainedcharged owners");}
 require(backendCreated&&backendDestroyed&&afterTargetRelease==mappedBaseline&&afterWeightsRelease==0&&afterStop==0,"measured teardown/backend destruction incomplete");
 std::ofstream out(report);
#if SPLASH_R1_DIAGNOSTIC_ALL_LAYERS
 (void)guardsPassed;(void)totals;(void)observedBF16Differences;(void)replayExact;(void)prefillExact;(void)ownershipStable;(void)canaries;(void)currentDelta;(void)targetBytes;(void)stateBytes;
 out<<"{\"schema\":\"TeacherV5-current2K-standard-R1-all-layer-diagnostic-v1\",\"diagnostic_completed\":true,\"numeric_qualification\":false,\"pass\":"<<(!anyChainFailed?"true":"false")<<",\"qualified_for_standard_benchmark\":false,\"diagnostic_only\":true,\"timing_attempted\":false,\"original_thresholds_unchanged\":true,\"any_original_shipping_projection_F64_or_selftap_gate_failed\":"<<(anyChainFailed?"true":"false")
  <<",\"captured_actual_layers\":"<<numericLayers<<",\"current_input_provenance\":\"actual original control R1 after canonical current2K prefill\",\"all_layer_diagnostics\":[";
 for(size_t i=0;i<allLayerDiagnostics.size();++i){if(i)out<<',';out<<allLayerDiagnostics[i];}
 out<<"],\"clean_teardown\":true,\"backend_destroyed_before_publication\":true,\"mapped_model_baseline_bytes\":"<<mappedBaseline<<",\"after_target_owner_release_bytes\":"<<afterTargetRelease<<",\"after_weights_owner_release_bytes\":"<<afterWeightsRelease
  <<",\"after_backend_stop_bytes\":"<<afterStop<<",\"actual_peak_delta_bytes\":"<<peak-mappedBaseline<<",\"planned_Governor_bytes\":"<<planned<<",\"actual_peak_diagnostic_delta_bytes\":"<<diagnosticBytes<<",\"normal_Governor_growth_allowed\":true,\"normal_Governor_host_measurement_valid\":true,\"denied_reservations\":0}\n";
 require(bool(out),"diagnostic report write failed");std::cout<<"{\"diagnostic_completed\":true,\"Root_GPU_work\":true,\"qualified_for_standard_benchmark\":false}\n";return anyChainFailed?2:0;
#else
 (void)anyChainFailed;out<<"{\"schema\":\"TeacherV5-current2K-standard-R1-numeric-state-qualifier-v1\",\"pass\":true,\"qualified_for_standard_benchmark\":"<<(compare?"true":"false")
  <<",\"role\":\""<<(compare?"candidate":"control")<<"\",\"one_forward_per_process\":true,\"Root_GPU_executed\":true,\"current_input_provenance\":\"actual causal R1 producers after current canonical2K prefill\",\"captured_layers\":"<<numericLayers
  <<",\"prefill134state_outputs_exact\":"<<(prefillExact?"true":"false")<<",\"vector_same_plane_shipping_replay_and_tap_bits_exact\":true,\"MPP_bit_parity_claim\":false,\"perroute_global_frozen_stage_gates_pass\":true,\"F64_RN_RTZ_FTZ_certified_samples\":"<<totals.samples
  <<",\"F64_certified_failures\":"<<totals.certFailures<<",\"strict_sensitive_samples\":"<<totals.strictSensitive<<",\"strict_sensitive_failures_separately_reported\":"<<totals.strictFailures
  <<",\"observed_shipping_BF16_differences\":"<<observedBF16Differences<<",\"future_and_same_owner_replay_exact\":"<<(replayExact?"true":"false")<<",\"exceptional_ID_rank_duplicate_nonfinite_safety_pass\":true,\"ownership_stable\":"<<(ownershipStable?"true":"false")<<",\"canaries_clean\":"<<(canaries?"true":"false")
  <<",\"actual_guard_rejections\":"<<guardsPassed<<",\"clean_teardown\":true,\"backend_created\":"<<(backendCreated?"true":"false")<<",\"backend_destroyed_before_publication\":"<<(backendDestroyed?"true":"false")
  <<",\"planned_Governor_bytes\":"<<planned<<",\"mapped_model_baseline_bytes\":"<<mappedBaseline<<",\"actual_target_delta_bytes\":"<<targetBytes<<",\"actual_two_guarded_state_delta_bytes\":"<<stateBytes
  <<",\"actual_peak_diagnostic_delta_bytes\":"<<diagnosticBytes<<",\"actual_current_delta_bytes\":"<<currentDelta<<",\"peak_allocated_bytes\":"<<peak<<",\"actual_peak_delta_bytes\":"<<peak-mappedBaseline
  <<",\"after_target_owner_release_bytes\":"<<afterTargetRelease<<",\"after_weights_owner_release_bytes\":"<<afterWeightsRelease<<",\"after_backend_stop_bytes\":"<<afterStop
  <<",\"normal_Governor_growth_allowed\":"<<(governorGrowth?"true":"false")<<",\"normal_Governor_host_measurement_valid\":"<<(hostValid?"true":"false")<<",\"denied_reservations\":0,\"causal_selected_input_coeff_hashes\":[";
 for(size_t i=0;i<causalHashes.size();++i){if(i)out<<',';out<<json::quote(causalHashes[i]);}out<<"]}\n";require(bool(out),"qualifier report failed");std::cout<<"{\"pass\":true,\"Root_GPU_work\":true}\n";return 0;
#endif
 }catch(const std::exception&e){if(!report.empty()){std::ofstream o(report);o<<"{\"schema\":\"TeacherV5-current2K-standard-R1-numeric-state-qualifier-v1\",\"pass\":false,\"qualified_for_standard_benchmark\":false,\"error\":"<<json::quote(e.what());if(!failureDiagnostic.empty())o<<",\"failed_stage_diagnostic\":"<<failureDiagnostic;o<<"}\n";}std::cerr<<e.what()<<'\n';return 1;}
}}
