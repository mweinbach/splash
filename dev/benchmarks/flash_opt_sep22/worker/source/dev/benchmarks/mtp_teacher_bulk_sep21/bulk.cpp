#include "dev/benchmarks/mtp_teacher_bulk_sep21/bulk.hpp"
#include "flash/FlashAffine.hpp"
#include "flash/FlashDenseCache.hpp"
#include "flash/FlashHC.hpp"
#include "flash/FlashMTPStateInternal.hpp"
#include "flash/FlashQSAFast.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashForward.h"
#include "metal/abi/FlashMTP.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace splash::flash {
namespace {
constexpr uint32_t width = 2560, hyper = 10240;
enum Slot : uint32_t {
  TokenIDs, Embedding, EmbeddingNormalized, EmbeddingProjected,
  HiddenNormalized, HiddenProjected, Hyper, HCNormalized, HCDown, HCUp,
  HCRawInjection, HCInjectionWeights, Mixed, QProjection, KProjection,
  VProjection, IndexProjection, Diagnostics, Count
};
constexpr std::array<uint32_t, Count - 1> perRowBytes{
  8,5120,5120,5120,20480,20480,20480,20480,640,20480,8,8,
  5120,24576,1024,1024,1280};
constexpr uint64_t round16K(uint64_t value) { return (value+16383)/16384*16384; }
constexpr uint64_t allocationPlan() {
  uint64_t value=16384;
  for (auto bytes:perRowBytes) value+=round16K(uint64_t{bytes}*2048);
  return value;
}
static_assert(allocationPlan()==FlashMTPTeacherBulkForward::plannedBytes);
bool overlaps(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  if (!a || !b) return false;
  if (a.sameView(b)) return true;
  const auto x=reinterpret_cast<uintptr_t>(a.contents());
  const auto y=reinterpret_cast<uintptr_t>(b.contents());
  if (!x || !y) return false;
  if (a.sizeBytes()>UINTPTR_MAX-x || b.sizeBytes()>UINTPTR_MAX-y)
    throw std::invalid_argument("teacher bulk alias extent overflows");
  return x<y+b.sizeBytes() && y<x+a.sizeBytes();
}
}
struct FlashMTPTeacherBulkForward::Impl final {
  FlashMTPForward &head;
  metal::MetalBackend &backend;
  const FlashWeights &weights;
  std::array<metal::MetalBuffer,Count> scratch;
  uint64_t allocated=0;
  Impl(FlashMTPForward &owner,metal::MetalBackend &device,const FlashWeights &model)
      :head(owner),backend(device),weights(model) {
    const auto before=backend.memoryStats().allocatedBytes;
    for (uint32_t i=0;i<Count-1;++i)
      scratch[i]=backend.allocateBuffer(round16K(uint64_t{perRowBytes[i]}*2048),
          metal::BufferStorage::Shared,"private singleton teacher bulk scratch");
    scratch[Diagnostics]=backend.allocateBuffer(16384,metal::BufferStorage::Shared,
        "private singleton teacher bulk diagnostics");
    const auto after=backend.memoryStats().allocatedBytes;
    if(after<before || after-before>allocationPlan())
      throw std::runtime_error("teacher bulk allocation exceeds reserved plan");
    allocated=after-before;
  }
  metal::MetalBuffer bf(Slot slot,uint32_t rows,uint32_t columns) {
    return backend.view(scratch[slot],0,uint64_t{rows}*columns*2);
  }
};
FlashMTPTeacherBulkForward::FlashMTPTeacherBulkForward(FlashMTPForward &head) {
  std::lock_guard lock(head.batchMutex());
  if (!head.batchDenseCache() || head.teacherBulkMaximumRows()!=128 ||
      !head.batchQSAF32() || !head.batchQSAMPP())
    throw std::invalid_argument("teacher bulk requires original cached128-row F32/MPP trained head");
  impl_=std::make_unique<Impl>(head,head.batchBackend(),head.batchWeights());
}
FlashMTPTeacherBulkForward::~FlashMTPTeacherBulkForward()=default;
uint64_t FlashMTPTeacherBulkForward::workspaceBytes() const noexcept {
  return impl_?impl_->allocated:0;
}
std::array<metal::MetalBuffer,18> FlashMTPTeacherBulkForward::oracleWorkspaceBuffers() const {
  if(!impl_)throw std::logic_error("teacher bulk is not initialized");
  return impl_->scratch;
}
metal::CommandTiming FlashMTPTeacherBulkForward::primeTeacherCache(
    FlashMTPState &request,metal::MetalBuffer previousHidden,
    std::span<const uint32_t> nextTokens) {
  if(!impl_) throw std::logic_error("teacher bulk is not initialized");
  auto &p=*impl_;
  std::lock_guard lock(p.head.batchMutex());
  if(!p.head.ownsState(request))
    throw std::invalid_argument("teacher bulk requires its healthy authoritative head state");
  auto &state=*request.impl_;
  const auto count=nextTokens.size();
  if(!count || count>maximumRows || count%rowQuantum ||
      state.length>state.qsa.capacity || count>state.qsa.capacity-state.length)
    throw std::invalid_argument("teacher bulk requires complete128-row windows within2048/context capacity");
  const uint32_t rows=static_cast<uint32_t>(count);
  if(!previousHidden || previousHidden.storage()!=metal::BufferStorage::Shared ||
      !previousHidden.contents() || previousHidden.sizeBytes()<uint64_t{rows}*hyper*2)
    throw std::invalid_argument("teacher bulk requires actual BF16[rows,10240] target features");
  const auto actualInput=p.backend.view(previousHidden,0,uint64_t{rows}*hyper*2);
  // Host guards complete before touching any owned scratch or dispatch graph.
  for(auto token:nextTokens) if(token>=p.weights.descriptor().vocabularySize)
    throw std::invalid_argument("teacher bulk token is outside vocabulary");
  for(const auto &buffer:p.scratch) if(overlaps(actualInput,buffer))
    throw std::invalid_argument("teacher bulk features alias mutable scratch");
  for(const auto &buffer:{state.qsa.keys,state.qsa.values,state.qsa.rawIndexKeys,
      state.qsa.pooledKeys,state.qsa.indexPositions}) if(overlaps(actualInput,buffer))
    throw std::invalid_argument("teacher bulk features alias persistent head cache");
  for(const auto &buffer:p.weights.immutableWeightBuffers()) if(overlaps(actualInput,buffer))
    throw std::invalid_argument("teacher bulk features alias original coefficients");
  for(const auto &buffer:p.head.batchDenseCache()->immutableWeightBuffers()) if(overlaps(actualInput,buffer))
    throw std::invalid_argument("teacher bulk features alias cached coefficients");
  const auto bf=[&](Slot slot,uint32_t columns){return p.bf(slot,rows,columns);};
  const auto diag=p.backend.view(p.scratch[Diagnostics],0,4);
  const auto tokens=p.backend.view(p.scratch[TokenIDs],0,uint64_t{rows}*8);
  metal::CommandGraph graph;
  const auto project=[&](const std::string &name,metal::MetalBuffer input,
      metal::MetalBuffer output,uint32_t projectedRows,uint32_t logicalRows) {
    p.head.batchProject(graph,name,input,output,diag,projectedRows,logicalRows,false);
  };
  const auto &descriptor=p.weights.descriptor();
  const FlashHCGeometry hc{rows,width,4,static_cast<float>(descriptor.normEpsilon)};
  addAffineEmbedding(graph,p.weights.projection("language_model.model.embed_tokens"),
      tokens,bf(Embedding,width),diag,rows);
  addHCGroupedNorm(graph,bf(Embedding,width),p.weights.tensor("mtp.pre_fc_norm_embedding.weight"),
      bf(EmbeddingNormalized,width),{rows,width,1,hc.epsilon},
      p.weights.normConvention("mtp.pre_fc_norm_embedding.weight"));
  addHCGroupedNorm(graph,actualInput,p.weights.tensor("mtp.pre_fc_norm_hidden.weight"),
      bf(HiddenNormalized,hyper),{rows,hyper,1,hc.epsilon},
      p.weights.normConvention("mtp.pre_fc_norm_hidden.weight"));
  project("mtp.fc_embedding",bf(EmbeddingNormalized,width),bf(EmbeddingProjected,width),rows,rows);
  project("mtp.fc_hidden",bf(HiddenNormalized,hyper),bf(HiddenProjected,hyper),rows*4,rows);
  graph.add("flash_mtp_fuse_inputs",{bf(EmbeddingProjected,width),bf(HiddenProjected,hyper),
      bf(Hyper,hyper),diag},FlashMTPFuseParams{rows,width,4},
      {(uint64_t{rows}*hyper+255)/256,1,1});
  const std::string hcPrefix="mtp.layers.0.attn_hyper_connection";
  const auto norm=hcPrefix+".hc_norm.weight";
  addHCGroupedNorm(graph,bf(Hyper,hyper),p.weights.tensor(norm),bf(HCNormalized,hyper),
      hc,p.weights.normConvention(norm));
  project(hcPrefix+".input_mix_weight_down",bf(HCNormalized,hyper),bf(HCDown,320),rows,rows);
  graph.add("flash_forward_hc_silu",{bf(HCDown,320),bf(HCDown,320),diag},
      FlashForwardActivationParams{rows,320,4},{(uint64_t{rows}*320+255)/256,1,1});
  project(hcPrefix+".input_mix_weight_up",bf(HCDown,320),bf(HCUp,hyper),rows,rows);
  project(hcPrefix+".block_inject_weight",bf(HCNormalized,hyper),bf(HCRawInjection,4),rows,rows);
  addHCMixWithInjection(graph,bf(HCNormalized,hyper),bf(HCUp,hyper),bf(HCRawInjection,4),
      bf(Mixed,width),bf(HCInjectionWeights,4),hc);
  const std::string attention="mtp.layers.0.self_attn";
  project(attention+".q_proj",bf(Mixed,width),bf(QProjection,12288),rows,rows);
  project(attention+".k_proj",bf(Mixed,width),bf(KProjection,512),rows,rows);
  project(attention+".v_proj",bf(Mixed,width),bf(VProjection,512),rows,rows);
  project(attention+".indexer.index_qk_proj",bf(Mixed,width),bf(IndexProjection,640),rows,rows);
  // Validate every original128-row chronological cache graph before appending
  // any cache mutation to the final graph. Scratch is shared only sequentially.
  std::vector<metal::CommandGraph> prefixes(rows/rowQuantum);
  for(uint32_t slice=0;slice<rows/rowQuantum;++slice) {
    const auto view=[&](Slot slot,uint32_t columns) {
      return p.backend.view(p.scratch[slot],uint64_t{slice}*rowQuantum*columns*2,
          uint64_t{rowQuantum}*columns*2);
    };
    FlashQSAFastInputs input;
    input.qProjection=view(QProjection,12288);input.kProjection=view(KProjection,512);
    input.vProjection=view(VProjection,512);input.indexProjection=view(IndexProjection,640);
    input.qNorm=&p.weights.tensor(attention+".q_norm.weight");
    input.kNorm=&p.weights.tensor(attention+".k_norm.weight");
    input.indexQNorm=&p.weights.tensor(attention+".indexer.q_layernorm.weight");
    input.indexKNorm=&p.weights.tensor(attention+".indexer.k_layernorm.weight");
    input.qConvention=p.weights.normConvention(attention+".q_norm.weight");
    input.kConvention=p.weights.normConvention(attention+".k_norm.weight");
    input.indexQConvention=p.weights.normConvention(attention+".indexer.q_layernorm.weight");
    input.indexKConvention=p.weights.normConvention(attention+".indexer.k_layernorm.weight");
    input.epsilon=descriptor.normEpsilon;input.theta=descriptor.rotaryTheta;input.diagnostics=diag;
    p.head.teacherBulkAddPrefix(prefixes[slice],input,state.qsa,
        static_cast<uint32_t>(state.length)+slice*rowQuantum,rowQuantum);
  }
  for(const auto &prefix:prefixes) p.head.teacherBulkAppendPrefix(graph,prefix);
  auto *hostTokens=static_cast<int64_t *>(tokens.contents());
  for(uint32_t row=0;row<rows;++row) hostTokens[row]=nextTokens[row];
  std::memset(diag.contents(),0,4);
  metal::CommandTiming timing;
  try {
    timing=p.backend.submitCommand(graph.dispatches());
    uint32_t status=0;std::memcpy(&status,diag.contents(),4);
    if(status) throw std::runtime_error("teacher bulk sticky diagnostics failed: "+std::to_string(status));
  } catch(...) {state.poisoned=true;throw;}
  state.length+=rows;
  return timing;
}
} // namespace splash::flash
