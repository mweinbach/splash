#pragma once
// Root-only GPU metadata/packing checks; including this header executes none.
#include "metadata.hpp"
#include "prefill4k_allrows_qmv_one_layer.hpp"
#include "flash/FlashMoEBlocked.hpp"
#include "metal/CommandGraph.hpp"
#include <cstring>
#include <sstream>
#include <vector>
namespace splash::flash::compact_native_r4_safety {
using metal::MetalBackend;using metal::MetalBuffer;using metal::CommandGraph;
struct Case {std::string name;bool pass=false;uint32_t diagnostic=0;};
struct Report {std::vector<Case>cases;
 bool pass()const{return !cases.empty()&&std::all_of(cases.begin(),cases.end(),[](const Case&c){return c.pass;});}
 std::string json()const{std::ostringstream o;o<<"{\"pass\":"<<(pass()?"true":"false")<<",\"native_bad_rank_differs_from_gather\":true,\"cases\":[";
 for(size_t i=0;i<cases.size();++i){if(i)o<<',';o<<"{\"name\":\""<<cases[i].name<<"\",\"pass\":"<<(cases[i].pass?"true":"false")<<",\"diagnostic\":"<<cases[i].diagnostic<<'}';}o<<"]}";return o.str();}
};
inline bool overlaps(MetalBuffer a,MetalBuffer b){const auto aa=reinterpret_cast<uintptr_t>(a.contents()),bb=reinterpret_cast<uintptr_t>(b.contents());return aa>=bb?aa-bb<b.sizeBytes():bb-aa<a.sizeBytes();}
inline void require(MetalBuffer b,uint64_t size,uint32_t align){if(!b||b.storage()!=metal::BufferStorage::Shared||!b.contents()||b.sizeBytes()<size||reinterpret_cast<uintptr_t>(b.contents())%align)throw std::invalid_argument("compact native view extent/storage/alignment differs");}
inline void validate(const qmv_one_layer::OneLayerPayload&l,const FlashMoEBlockedScratch&s,MetalBuffer h,MetalBuffer ids,MetalBuffer diag){
 require(h,4*2560*2,8);require(ids,40*8,8);require(diag,4,4);require(s.buckets.counts,512*4,4);require(s.buckets.offsets,513*4,4);require(s.buckets.routeMap,40*4,4);require(s.buckets.canonicalToPacked,40*4,4);require(s.buckets.jobOffsets,513*4,4);require(s.buckets.tileJobs,514*8,8);require(s.buckets.jobCount,4,4);require(s.buckets.packedInputs,103*2560*2,8);require(s.packedActivated,103*640*2,8);require(s.scatteredDown,40*2560*2,8);
 const std::array<MetalBuffer,13>v{h,ids,diag,s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,s.buckets.jobOffsets,s.buckets.tileJobs,s.buckets.jobCount,s.buckets.packedInputs,s.packedActivated,s.scatteredDown};
 for(uint32_t i=0;i<v.size();++i){if(overlaps(v[i],l.base)||overlaps(v[i],l.ranks))throw std::invalid_argument("compact native aliases immutable payload/ranks");for(uint32_t j=i+1;j<v.size();++j)if(overlaps(v[i],v[j]))throw std::invalid_argument("compact native operand/metadata aliases");}
}
inline void oldInteger(CommandGraph&g,const FlashMoEBlockedScratch&s,MetalBuffer ids,MetalBuffer diag){
 const FlashMoEBucketParams base{4,10,2560,512,40,0,0,0},jobs{4,10,2560,512,40,16,514,0};
 g.add("flash_moe_bucket_histogram",{ids,s.buckets.counts,s.buckets.canonicalToPacked,diag},base,{512,1,1},{256,1,1});
 g.add("flash_moe_bucket_prefix",{s.buckets.counts,s.buckets.offsets,diag},base,{1,1,1},{256,1,1});
 g.add("flash_moe_bucket_stable_map",{ids,s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,diag},base,{512,1,1},{256,1,1});
 g.add("flash_moe_bucket_job_prefix",{s.buckets.counts,s.buckets.offsets,s.buckets.jobOffsets,s.buckets.jobCount,diag},jobs,{1,1,1},{256,1,1});
 g.add("flash_moe_bucket_jobs",{s.buckets.offsets,s.buckets.jobOffsets,s.buckets.jobCount,s.buckets.tileJobs,diag},jobs,{3,1,1},{256,1,1});
}
inline void compactInteger(CommandGraph&g,const FlashMoEBlockedScratch&s,MetalBuffer ids,MetalBuffer diag,FlashMoEBucketParams p={4,10,2560,512,40,16,514,0},bool wrongGrid=false){
 g.add("expert_r4_compact_native_sep22_plan",{ids,s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,s.buckets.jobOffsets,s.buckets.tileJobs,s.buckets.jobCount,diag},p,{wrongGrid?2u:1u,1,1},{256,1,1});
}
inline void originalPack(CommandGraph&g,const FlashMoEBlockedScratch&s,MetalBuffer h,MetalBuffer diag){g.add("flash_moe_direct_a_pack",{h,s.buckets.offsets,s.buckets.routeMap,s.buckets.packedInputs,diag},FlashMoEBucketParams{4,10,2560,512,40,0,0,0},{103,1,1},{256,1,1});}
inline bool expected(const FlashMoEBlockedScratch&s,const compact_native_r4_metadata::Expected&e){return !std::memcmp(e.counts.data(),s.buckets.counts.contents(),512*4)&&!std::memcmp(e.offsets.data(),s.buckets.offsets.contents(),513*4)&&!std::memcmp(e.jobOffsets.data(),s.buckets.jobOffsets.contents(),513*4)&&!std::memcmp(e.routeMap.data(),s.buckets.routeMap.contents(),40*4)&&!std::memcmp(e.inverse.data(),s.buckets.canonicalToPacked.contents(),40*4)&&!std::memcmp(e.jobs.data(),s.buckets.tileJobs.contents(),514*8)&&*static_cast<const uint32_t*>(s.buckets.jobCount.contents())==e.jobCount;}
template<class Allocate,class T>MetalBuffer upload(Allocate&a,const std::vector<T>&v){auto b=a(v.size()*sizeof(T));std::memcpy(b.contents(),v.data(),b.sizeBytes());return b;}
template<class Allocate>
Report run(MetalBackend&backend,const qmv_one_layer::OneLayerPayload&l,const std::vector<uint16_t>&hidden,const std::vector<int64_t>&ids,Allocate a){
 if(hidden.size()!=4*2560||ids.size()!=40)throw std::invalid_argument("compact native safety requires exact R4 fixtures");
 FlashMoEBlockedScratch s{};s.buckets.counts=a(512*4);s.buckets.offsets=a(513*4);s.buckets.routeMap=a(40*4);s.buckets.canonicalToPacked=a(40*4);s.buckets.jobOffsets=a(513*4);s.buckets.jobCount=a(4);s.buckets.tileJobs=a(516*8);s.buckets.packedInputs=a(103*2560*2);s.packedActivated=a(103*640*2);s.scatteredDown=a(40*2560*2);
 auto h=upload(a,hidden),id=upload(a,ids),diag=a(4);validate(l,s,h,id,diag);
 auto clear=[&]{*static_cast<uint32_t*>(diag.contents())=0x80000000;for(auto b:{s.buckets.counts,s.buckets.offsets,s.buckets.routeMap,s.buckets.canonicalToPacked,s.buckets.jobOffsets,s.buckets.jobCount,s.buckets.tileJobs,s.buckets.packedInputs})std::memset(b.contents(),0xa5,b.sizeBytes());};
 auto diagnostic=[&]{return *static_cast<const uint32_t*>(diag.contents());};
 Report r;
 std::array<std::vector<int64_t>,4>patterns{ids,std::vector<int64_t>(40,-1),std::vector<int64_t>(40,7),ids};patterns[3][0]=-1;patterns[3][1]=512;
 for(uint32_t test=0;test<patterns.size();++test){auto t=upload(a,patterns[test]);const auto ref=compact_native_r4_metadata::expected(patterns[test]);
  clear();CommandGraph old;oldInteger(old,s,t,diag);originalPack(old,s,h,diag);(void)backend.submitCommand(old.dispatches());
  const uint32_t oldDiag=diagnostic();const bool oldMeta=expected(s,ref);std::vector<uint8_t>packed(s.buckets.packedInputs.sizeBytes());std::memcpy(packed.data(),s.buckets.packedInputs.contents(),packed.size());
  clear();CommandGraph now;compactInteger(now,s,t,diag);originalPack(now,s,h,diag);(void)backend.submitCommand(now.dispatches());
  const auto*tail=static_cast<const uint8_t*>(s.buckets.tileJobs.contents())+514*8;const bool tailSafe=std::all_of(tail,tail+2*8,[](uint8_t x){return x==0xa5;});
  r.cases.push_back({test==0?"metadata_all_fields_and_padding_exact":test==1?"all_invalid_metadata_and_padding_exact":test==2?"forty_duplicates_m16_jobs_step16_exact":"invalid_bounds_old_native_ownership_exact",oldMeta&&expected(s,ref)&&oldDiag==diagnostic()&&!std::memcmp(packed.data(),s.buckets.packedInputs.contents(),packed.size())&&tailSafe,diagnostic()});
 }
 auto bad=hidden;bad[3*2560]=0x7fc0;bad[3*2560+1]=0x7f80;bad[3*2560+2]=0xff80;auto badH=upload(a,bad);std::vector<int64_t>allInvalid(40,-1);auto invalid=upload(a,allInvalid);
 clear();CommandGraph badOld;oldInteger(badOld,s,invalid,diag);originalPack(badOld,s,badH,diag);(void)backend.submitCommand(badOld.dispatches());const uint32_t oldBad=diagnostic();
 clear();CommandGraph badNew;compactInteger(badNew,s,invalid,diag);originalPack(badNew,s,badH,diag);(void)backend.submitCommand(badNew.dispatches());r.cases.push_back({"original_pack_all_hidden_nonfinite_evidence",oldBad==0x80000005u&&diagnostic()==oldBad,diagnostic()});
 for(uint32_t test=0;test<2;++test){clear();auto badp=FlashMoEBucketParams{4,10,2560,512,40,16,514,0};if(!test)badp.tile_rows=4;CommandGraph malformed;compactInteger(malformed,s,id,diag,badp,test==1);originalPack(malformed,s,h,diag);(void)backend.submitCommand(malformed.dispatches());
  const auto empty=compact_native_r4_metadata::expected(allInvalid);r.cases.push_back({test?"malformed_multi_cta_grid_clears_stale_ranges":"malformed_native_tile4_rejected",expected(s,empty)&&diagnostic()==0x80000002u,diagnostic()});}
 bool alias=false;try{auto bads=s;bads.buckets.counts=h;validate(l,bads,h,id,diag);}catch(const std::invalid_argument&){alias=true;}r.cases.push_back({"host_metadata_operand_alias_rejected",alias,0});
 r.cases.push_back({"input_fixture_immutability",!std::memcmp(hidden.data(),h.contents(),h.sizeBytes())&&!std::memcmp(ids.data(),id.contents(),id.sizeBytes())&&!std::memcmp(bad.data(),badH.contents(),badH.sizeBytes())&&!std::memcmp(allInvalid.data(),invalid.contents(),invalid.sizeBytes()),0});
 return r;
}
} // namespace splash::flash::compact_native_r4_safety
