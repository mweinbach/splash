#pragma once
// Private intended-arithmetic golden: actual old batch projections/HC/GDN/MoE;
// only the QSA emitter uses the literal established singleton packed-V API.
#include "dev/benchmarks/prefill4k_attention/bulk.hpp"
#include "dev/benchmarks/prefill_qsa_twopass_sep21/twopass.hpp"
#include <array>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace splash::flash::batch_qsa_intended_reference {
inline constexpr uint64_t arenaBytes=509607936;
inline constexpr std::string_view modelSource="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
inline constexpr std::string_view scope="TEST_ONLY old batch projection executor; literal existing singleton SAME-projection packedV helper; no original SG8 bit-parity or EVERYROW-pass claim";
struct Session {
 metal::MetalBackend *backend;
 prefill4k::TwoPassWorkspace qualified;
 uint64_t actualCharge=0,encodedLaneLayerCalls=0,completedForwards=0,outerViews=0;
 uint32_t lanes=0,rows=0,capacity=0;
 bool cohortValidated=false;
 std::array<uint32_t,48> visits{};
 std::string actualSource;
 Session(metal::MetalBackend &b,uint64_t admittedBytes):backend(&b) {
  if(admittedBytes<arenaBytes)throw std::invalid_argument("intended reference requires complete509607936 Gov reservation before construction");
  for(const char *flag:{"SPLASH_FLASH_BATCH_PREFILL","SPLASH_FLASH_BATCH_QSA_BULK_PREFILL","SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21","SPLASH_FLASH_QSA_F32","SPLASH_FLASH_QSA_MPP","SPLASH_FLASH_QSA_ROW_TILES","SPLASH_FLASH_QSA_BULK_PREFILL","SPLASH_FLASH_QSA_BULK_PREFILL_SG8"}) {
   const char *raw=std::getenv(flag);if(!raw||std::string_view(raw)!="1")throw std::invalid_argument(std::string("intended reference requires ")+flag+"=1");
  }
  const uint64_t before=b.memoryStats().allocatedBytes;
  qualified=prefill4k::allocateTwoPassWorkspace(b,admittedBytes);
  const uint64_t after=b.memoryStats().allocatedBytes;
  if(after<before||after-before!=arenaBytes)throw std::logic_error("intended reference actual allocatedSize differs from509607936 admitted arena");actualCharge=after-before;
 }
 std::array<metal::MetalBuffer,6> planes() const {return {qualified.prepared.queries,qualified.prepared.indexQueries,qualified.prepared.selectedBlocks,qualified.packedQueries,qualified.scoresAndProbabilities,qualified.rawAttention};}
};
inline thread_local Session *active=nullptr;
struct Install {
 Session *prior;
 explicit Install(Session &value):prior(active){if(prior)throw std::logic_error("nested intended-reference scopes forbidden");active=&value;}
 ~Install(){active=prior;}
};
inline void disjoint(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
 if(!a||!b)return;
 if(a.sameView(b))throw std::invalid_argument("intended reference arena aliases other source storage");
 if(b.storage()!=metal::BufferStorage::Shared)return;
 const auto x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());
 if(!x||!y||a.storage()!=metal::BufferStorage::Shared||a.sizeBytes()>std::numeric_limits<uintptr_t>::max()-x||b.sizeBytes()>std::numeric_limits<uintptr_t>::max()-y||(x<y+b.sizeBytes()&&y<x+a.sizeBytes()))throw std::invalid_argument("intended reference arena overlaps batch/request/immutable storage");
}
inline void beforeWhole(metal::MetalBackend &backend,uint32_t lanes,uint32_t rows,uint32_t capacity,
    std::span<const uint64_t> lengths,const std::vector<metal::MetalBuffer>& others,std::string_view source) {
 auto *s=active;if(!s)return;
 if(s->backend!=&backend||s->cohortValidated||(lanes!=2&&lanes!=4)||rows!=2048||(capacity!=4096&&capacity!=16384)||lengths.size()!=lanes||source!=modelSource)throw std::invalid_argument("intended reference permits one exact fresh2048 realB2/B4 current-source cohort only");
 for(auto length:lengths)if(length)throw std::invalid_argument("intended reference whole cohort must begin0");
 const auto planes=s->planes();constexpr std::array<uint64_t,6> extents{25165824,2097152,4194304,25165824,402653184,50331648};
 for(size_t i=0;i<planes.size();++i){if(!planes[i]||planes[i].storage()!=metal::BufferStorage::Shared||!planes[i].contents()||planes[i].sizeBytes()!=extents[i])throw std::invalid_argument("intended reference exact owned Shared six-plane arena required");(void)backend.view(planes[i],0,extents[i]);for(size_t j=0;j<i;++j)disjoint(planes[i],planes[j]);for(const auto &other:others)disjoint(planes[i],other);}
 s->lanes=lanes;s->rows=rows;s->capacity=capacity;s->outerViews=others.size();s->actualSource=source;s->cohortValidated=true;
}
inline void emit(metal::MetalBackend &backend,metal::CommandGraph &graph,const FlashQSAFastInputs &input,
 FlashQSAState &state,FlashQSAWorkspace &ordinary,FlashQSAFastWorkspace &fast,
 prefill4k::BulkExactWorkspace &legacy,uint32_t layer,uint32_t lane,uint32_t lanes,uint32_t rows,uint64_t begin) {
 auto *s=active;if(!s){prefill4k::addBulkExactQSA(backend,graph,input,state,ordinary,fast,legacy,0,rows,true);return;}
 if(!s->cohortValidated||s->backend!=&backend||lanes!=s->lanes||rows!=s->rows||state.capacity!=s->capacity||begin||layer>=48||lane>=lanes||input.epsilon!=1e-6||input.theta!=1e7)throw std::invalid_argument("intended reference actual per-lane QSA integration differs from admitted fresh cohort");
 // This API and every floating shader are byte-identical to qualified B1.
 prefill4k::addTwoPassQSA(backend,graph,input,state,ordinary,fast,s->qualified,0,rows,false,true);
 ++s->encodedLaneLayerCalls;++s->visits[layer];
}
inline void completed(bool healthy) {
 auto *s=active;if(!s)return;
 if(!healthy||!s->cohortValidated||s->completedForwards||s->encodedLaneLayerCalls!=uint64_t(12)*s->lanes)throw std::logic_error("intended reference complete healthy twelveQSA-layer cohort required");
 uint32_t layers=0;for(auto visits:s->visits)if(visits){if(visits!=s->lanes)throw std::logic_error("intended reference QSA layer missing real lane");++layers;}if(layers!=12)throw std::logic_error("intended reference exact12QSA layers required");++s->completedForwards;
}
} // namespace splash::flash::batch_qsa_intended_reference
