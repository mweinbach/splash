// Actual frozen guard extraction versus actual shipping callback preflight.
// Fake addresses expose metadata only; no byte at any address is dereferenced.
#include "guard.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace metadata_fake {
enum class BufferStorage:uint8_t {Shared,Private,Managed};
enum class Tile:uint32_t {M8N64=8,M16N64=16,M32N64=32,M64N64=64};
struct MetalBuffer {
  uintptr_t address=0;uint64_t bytes=0;bool valid=true;BufferStorage kind=BufferStorage::Shared;uint64_t allocation=0,offset=0;
  explicit operator bool()const noexcept{return valid;}
  BufferStorage storage()const noexcept{return kind;}
  void *contents()const noexcept{return reinterpret_cast<void *>(address);}
  uint64_t sizeBytes()const noexcept{return bytes;}
  bool sameView(const MetalBuffer &other)const noexcept{return allocation==other.allocation&&offset==other.offset&&bytes==other.bytes;}
};
struct Buckets {
  MetalBuffer counts,offsets,routeMap,canonicalToPacked,packedInputs,jobOffsets,jobCount,tileJobs;
  uint32_t rowCapacity=4,selectionCapacity=10,routeCapacity=40,jobCapacity=516;
};
struct Scratch {Buckets buckets;MetalBuffer packedActivated,scatteredDown;};
struct Layer {MetalBuffer base,ranks;};
struct Inventory {size_t count=512;size_t size()const{return count;}};
struct Entry {Inventory selectedIDs;};
struct Metadata {std::array<Entry,48> layers;};
struct Configuration {bool directA=true,wideM64=false,requested=true,gather=true;uint32_t gatherCap=4;};
inline const Configuration *configuration=nullptr;
struct CommandGraph {};
} // namespace metadata_fake

enum:uint32_t {kFlashMoEBucketMaximumRows=8192,kFlashMoEBucketMaximumSelections=10};
constexpr uint32_t kExperts=512;
#include "baseline_generated.hpp"

namespace {
using metadata_fake::MetalBuffer;
constexpr std::array<const char *,13> roleNames{"H","IDs","counts","offsets","routeMap","inverse","packedH","jobOffsets","jobCount","jobs","packedAct","down","diag"};
constexpr std::array<uint64_t,13> required{20480,320,2048,2052,160,160,527360,2052,4,4112,131840,204800,4};
struct Fixture {
  metadata_fake::Configuration config;
  frozen_baseline::Impl impl;
  std::array<MetalBuffer,13> roles;
  metadata_fake::Scratch scratch;
  uint32_t layer=0,rows=4,selections=10;
  bool disposed=false;
  Fixture() {
    for(uint32_t role=0;role<roles.size();++role)roles[role]={0x10000000ULL+uint64_t{role}*0x100000ULL,required[role],true,metadata_fake::BufferStorage::Shared,100+role,0};
    for(uint32_t i=0;i<48;++i) {
      const uintptr_t address=0x100000000ULL+uint64_t{i}*0x200000000ULL;
      impl.layers[i].base={address,2524446720ULL,true,metadata_fake::BufferStorage::Shared,1000+i*2,0};
      impl.layers[i].ranks={address+0x100000000ULL,16384,true,metadata_fake::BufferStorage::Shared,1001+i*2,0};
    }
  }
  void bind() {
    scratch.buckets.counts=roles[2];scratch.buckets.offsets=roles[3];scratch.buckets.routeMap=roles[4];
    scratch.buckets.canonicalToPacked=roles[5];scratch.buckets.packedInputs=roles[6];scratch.buckets.jobOffsets=roles[7];
    scratch.buckets.jobCount=roles[8];scratch.buckets.tileJobs=roles[9];scratch.packedActivated=roles[10];scratch.scatteredDown=roles[11];
  }
};
struct Counts {uint64_t cases=0,accepted=0,rejected=0,geometry=0,view=0,alias=0,immutable=0,context=0;};
Counts counts;
bool decision(Fixture fixture,bool bundled) {
  fixture.bind();metadata_fake::configuration=&fixture.config;
  frozen_baseline::FlashInt8ExpertStore store;store.impl_=fixture.disposed?nullptr:&fixture.impl;
  metadata_fake::CommandGraph graph;
  try {
    if(bundled)frozen_baseline::validateBundled(store,graph,fixture.layer,fixture.roles[0],fixture.roles[1],fixture.scratch,fixture.roles[12],fixture.rows,fixture.selections);
    else {
      store.addCompactNativeR4VerifyPack(graph,fixture.layer,fixture.roles[0],fixture.roles[1],fixture.scratch,fixture.roles[12],fixture.rows,fixture.selections);
      store.addGateUp(graph,fixture.layer,fixture.scratch,fixture.roles[12],fixture.rows,metadata_fake::Tile::M16N64,fixture.selections);
      store.addDownScatter(graph,fixture.layer,fixture.scratch,fixture.roles[12],fixture.rows,metadata_fake::Tile::M16N64,fixture.selections);
    }
    return true;
  }catch(const std::invalid_argument &){return false;}
}
void compare(Fixture fixture,const std::string &name,int expected=-1) {
  const bool original=decision(fixture,false),bundled=decision(fixture,true);++counts.cases;
  if(original!=bundled)throw std::runtime_error("Verbatim-old/bundled decision differs: "+name);
  if(expected>=0&&original!=bool(expected))throw std::runtime_error("Source-contract case expectation differs: "+name);
  if(original)++counts.accepted;else++counts.rejected;
}
void geometryCases() {
  for(uint32_t rows=0;rows<=8192;++rows){Fixture f;f.rows=rows;compare(f,"physical rows "+std::to_string(rows),rows==4);++counts.geometry;}
  for(uint32_t selections=0;selections<=12;++selections){Fixture f;f.selections=selections;compare(f,"selections "+std::to_string(selections),selections==10);++counts.geometry;}
  for(uint32_t rows:{8193u,UINT32_MAX}){Fixture f;f.rows=rows;compare(f,"unsupported rows",0);++counts.geometry;}
  for(uint32_t value:{0u,1u,3u,4u,5u,8192u,UINT32_MAX}){Fixture f;f.scratch.buckets.rowCapacity=value;compare(f,"row capacity",value>=4);++counts.geometry;}
  for(uint32_t value:{0u,1u,9u,10u,11u,UINT32_MAX}){Fixture f;f.scratch.buckets.selectionCapacity=value;compare(f,"selection capacity",value>=10);++counts.geometry;}
  for(uint32_t value:{0u,1u,39u,40u,41u,UINT32_MAX}){Fixture f;f.scratch.buckets.routeCapacity=value;compare(f,"route capacity",value>=40);++counts.geometry;}
  for(uint32_t value:{0u,1u,513u,514u,515u,516u,UINT32_MAX}){Fixture f;f.scratch.buckets.jobCapacity=value;compare(f,"job capacity",value>=514);++counts.geometry;}
}
void viewCases() {
  for(uint32_t role=0;role<13;++role) {
    for(uint64_t size:{0ULL,required[role]-1,required[role],required[role]+1,UINT64_MAX}) {
      Fixture f;f.roles[role].bytes=size;compare(f,std::string(roleNames[role])+" extent");++counts.view;
    }
    for(auto storage:{metadata_fake::BufferStorage::Shared,metadata_fake::BufferStorage::Private,metadata_fake::BufferStorage::Managed}) {
      Fixture f;f.roles[role].kind=storage;compare(f,std::string(roleNames[role])+" storage",storage==metadata_fake::BufferStorage::Shared);++counts.view;
    }
    Fixture invalid;invalid.roles[role].valid=false;compare(invalid,std::string(roleNames[role])+" null handle",0);++counts.view;
    Fixture null;null.roles[role].address=0;compare(null,std::string(roleNames[role])+" null address",0);++counts.view;
    Fixture unaligned;unaligned.roles[role].address+=1;compare(unaligned,std::string(roleNames[role])+" source accepts unaligned metadata",1);++counts.view;
    Fixture high;high.roles[role].address=UINTPTR_MAX-required[role];compare(high,std::string(roleNames[role])+" high address",1);++counts.view;
  }
}
void aliasCases() {
  for(uint32_t a=0;a<13;++a)for(uint32_t b=a+1;b<13;++b) {
    Fixture same;same.roles[b].address=same.roles[a].address;compare(same,"exact alias roles",0);++counts.alias;
    Fixture forward;forward.roles[b].address=forward.roles[a].address+forward.roles[a].bytes-1;compare(forward,"last-byte overlap",0);++counts.alias;
    Fixture reverse;reverse.roles[b].address=reverse.roles[a].address-reverse.roles[b].bytes+1;compare(reverse,"reverse last-byte overlap",0);++counts.alias;
    Fixture adjacent;adjacent.roles[b].address=adjacent.roles[a].address+adjacent.roles[a].bytes;compare(adjacent,"adjacent independent views",1);++counts.alias;
    Fixture slices;slices.roles[a].allocation=slices.roles[b].allocation=99999;
    slices.roles[b].address=slices.roles[a].address+slices.roles[a].bytes;slices.roles[b].offset=slices.roles[a].bytes;
    compare(slices,"disjoint shared-allocation slices",1);++counts.alias;
  }
}
void immutableCases() {
  for(uint32_t layer=0;layer<48;++layer)for(uint32_t plane=0;plane<2;++plane)for(uint32_t role=0;role<13;++role) {
    Fixture f;const auto &immutable=plane?f.impl.layers[layer].ranks:f.impl.layers[layer].base;
    f.roles[role].address=immutable.address;compare(f,"each role versus each immutable layer",0);++counts.immutable;
    Fixture partial;const auto &p=plane?partial.impl.layers[layer].ranks:partial.impl.layers[layer].base;
    partial.roles[role].address=p.address+p.bytes-1;compare(partial,"immutable last-byte overlap",0);++counts.immutable;
  }
  for(uint32_t layer=0;layer<48;++layer)for(uint32_t plane=0;plane<2;++plane) {
    Fixture f;(plane?f.impl.layers[layer].ranks:f.impl.layers[layer].base).address=0;compare(f,"addressless immutable layer",0);++counts.immutable;
  }
  for(uint64_t bytes:{0ULL,2047ULL,2048ULL,16384ULL}){Fixture f;f.impl.layers[0].ranks.bytes=bytes;compare(f,"selected rank extent",bytes>=2048);++counts.immutable;}
}
void contextCases() {
  for(uint32_t index=0;index<=48;++index){Fixture f;f.layer=index;compare(f,"selected layer bounds",index<48);++counts.context;}
  for(uint32_t index:{49u,UINT32_MAX}){Fixture f;f.layer=index;compare(f,"large layer bound",0);++counts.context;}
  for(uint32_t layer=0;layer<48;++layer)for(size_t inventory:{size_t(0),size_t(511),size_t(512),size_t(513)}) {
    Fixture f;f.layer=layer;f.impl.metadata.layers[layer].selectedIDs.count=inventory;compare(f,"selected layer inventory",inventory==512);++counts.context;
  }
  Fixture disposed;disposed.disposed=true;compare(disposed,"disposed Store",0);++counts.context;
  Fixture disabled;disabled.impl.compactR4Verify=false;disabled.config.requested=false;compare(disabled,"compact disabled",0);++counts.context;
  Fixture mutation;mutation.config.requested=false;compare(mutation,"frozen flag mutation",0);++counts.context;
  Fixture gather;gather.config.gather=false;compare(gather,"gather dependency",0);++counts.context;
  Fixture direct;direct.config.directA=false;compare(direct,"DirectA dependency",0);++counts.context;
  for(uint32_t cap:{0u,3u,4u,5u,UINT32_MAX}){Fixture f;f.config.gatherCap=cap;compare(f,"gather cap",cap==4);++counts.context;}
}
} // namespace
int main() {
  try {
    compare(Fixture{},"supported default",1);geometryCases();viewCases();aliasCases();immutableCases();contextCases();
    std::cout<<"{\"schema\":\"verbatim-native-vs-bundled-preflight-decision-cpu-v1\",\"pass\":true,\"cases\":"<<counts.cases
      <<",\"accepted\":"<<counts.accepted<<",\"rejected\":"<<counts.rejected<<",\"geometry_cases\":"<<counts.geometry
      <<",\"view_cases\":"<<counts.view<<",\"alias_cases\":"<<counts.alias<<",\"immutable_cases\":"<<counts.immutable<<",\"context_cases\":"<<counts.context
      <<",\"bundle_roles\":13,\"immutable_layers\":48,\"baseline_verbatim_frozen_source\":true,\"actual_shipping_header_used\":true,"
      "\"test_not_new_implementation_mirror\":true,\"backend_created\":false,\"gpu_work\":false,\"payload_content_access_or_hash\":false}\n";
    return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
