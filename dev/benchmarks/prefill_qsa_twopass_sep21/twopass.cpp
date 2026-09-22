#include "twopass.hpp"
#include "metal/abi/FlashQSAFast.h"
#include <cstring>
#include <stdexcept>
#include <vector>
namespace splash::flash::prefill4k {
namespace {
void require(bool v,const char *message) {if (!v) throw std::invalid_argument(message);}
void disjoint(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  const auto x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());
  require(x && y && (x<=y?y-x>=a.sizeBytes():x-y>=b.sizeBytes()),"Two-pass QSA planes overlap");
}
}
bool twoPassGeometry(uint32_t begin,uint32_t rows,uint32_t capacity,bool verification) noexcept {
  return !verification && !begin && rows==2048 && capacity>=2048 && capacity<=262144;
}
TwoPassWorkspace allocateTwoPassWorkspace(metal::MetalBackend &backend,uint64_t reservedBytes) {
  require(reservedBytes>=twoPassPlannedBytes(),"Two-pass QSA requires complete reservation before allocation");
  return {allocateDenseCoalescedWorkspace(backend,2048),
    backend.allocateBuffer(25165824,metal::BufferStorage::Shared,"two-pass Q pack"),
    backend.allocateBuffer(402653184,metal::BufferStorage::Shared,"two-pass F32 scores/P union"),
    backend.allocateBuffer(50331648,metal::BufferStorage::Shared,"two-pass raw F32 attention")};
}
void addTwoPassQSA(metal::MetalBackend &backend,metal::CommandGraph &graph,
    const FlashQSAFastInputs &input,FlashQSAState &state,FlashQSAWorkspace &ordinary,
    FlashQSAFastWorkspace &fast,TwoPassWorkspace &w,uint32_t begin,uint32_t rows,bool verification,bool packedV) {
  require(twoPassGeometry(begin,rows,state.capacity,verification),"Two-pass only admits fresh non-verification dense2K");
  require(fast.maximumRows==128 && fast.maximumPartitions==32 && ordinary.maximumRows==128,
      "Two-pass keeps original ordinary32-partition service scratch");
  for (const auto &[plane,size]:std::vector<std::pair<metal::MetalBuffer,uint64_t>>{
      {w.packedQueries,25165824},{w.scoresAndProbabilities,402653184},{w.rawAttention,50331648}})
    require(plane && plane.contents() && plane.sizeBytes()>=size,"Two-pass arena is short");
  const std::vector<metal::MetalBuffer> newPlanes{w.packedQueries,w.scoresAndProbabilities,w.rawAttention};
  for (uint32_t i=0;i<newPlanes.size();++i) {
    for (uint32_t j=0;j<i;++j) disjoint(newPlanes[i],newPlanes[j]);
    for (const auto &old:{w.prepared.queries,w.prepared.indexQueries,w.prepared.selectedBlocks,
        state.keys,state.values,state.rawIndexKeys,state.pooledKeys,state.indexPositions,
        input.qProjection,input.kProjection,input.vProjection,input.indexProjection,input.output,input.diagnostics})
      disjoint(newPlanes[i],old);
    for (const auto *norm:{input.qNorm,input.kNorm,input.indexQNorm,input.indexKNorm}) {
      require(norm,"Two-pass norm tensor is missing");disjoint(newPlanes[i],norm->buffer);
    }
    if (input.positions) disjoint(newPlanes[i],input.positions);
    for (const auto &scratch:{ordinary.queries,ordinary.indexQueries,ordinary.blockScores,
        ordinary.selectedBlocks,ordinary.attentionScores,ordinary.probabilities,
        fast.partitionStatistics,fast.partitionValues}) disjoint(newPlanes[i],scratch);
  }
  metal::CommandGraph qualified,result;
  addDenseCoalescedQSA(backend,qualified,input,state,ordinary,fast,w.prepared,begin,rows);
  const auto prefix=qualified.dispatches();
  require(prefix.size()==35 && prefix[0].pipelineName=="flash_qsa_fast_prepare" &&
      prefix[1].pipelineName.starts_with("flash_qsa_pool_rope_") &&
      prefix[2].pipelineName=="flash_qsa_select_blocks","Two-pass original prefix drift");
  for (uint32_t i=0;i<3;++i) {
    const auto &d=prefix[i];std::vector<metal::MetalBuffer> buffers;
    for (const auto &b:d.buffers) {require(b.index==buffers.size(),"Two-pass prefix binding drift");buffers.push_back(b.buffer);}
    require(d.bytes.size()==1,"Two-pass original prefix bytes drift");
    if (d.bytes[0].sizeBytes==sizeof(FlashQSAParams)) {FlashQSAParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));
      result.add(d.pipelineName,buffers,p,d.threadgroups,d.threadsPerThreadgroup);}
    else {require(d.bytes[0].sizeBytes==sizeof(FlashQSAFastParams),"Two-pass prefix ABI drift");
      FlashQSAFastParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));result.add(d.pipelineName,buffers,p,d.threadgroups,d.threadsPerThreadgroup);}
  }
  const FlashQSAFastParams p{{2048,0,state.capacity,(state.capacity+3)/4,0,512,0,0,1e-6f,1e7f,0,0},0,0,4,32};
  result.add("sep21_qsa_twopass_pack_q",{w.prepared.queries,w.packedQueries,w.prepared.selectedBlocks,input.diagnostics},p,{49152,1,1},{256,1,1});
  result.add("sep21_qsa_twopass_qk_m128_n64",{w.packedQueries,state.keys,w.scoresAndProbabilities,input.diagnostics},p,{32,192,2},{256,1,1});
  result.add("sep21_qsa_twopass_softmax_f32",{w.scoresAndProbabilities,input.diagnostics},p,{24576,2,1},{256,1,1});
  // QK is finished with packed Q; its first2MiB now hold transposed V.
  if (packedV) result.add("sep21_qsa_twopass_pack_v",{state.values,w.packedQueries,input.diagnostics},p,{4096,1,1},{256,1,1});
  result.add(packedV?"sep21_qsa_twopass_pv_packed_v_m128_n64":"sep21_qsa_twopass_pv_m128_n64",
      {w.scoresAndProbabilities,packedV?w.packedQueries:state.values,w.rawAttention,input.diagnostics},p,{4,192,2},{256,1,1});
  result.add("sep21_qsa_twopass_unpack_gate",{w.rawAttention,input.qProjection,input.output,input.diagnostics},p,{49152,1,1},{256,1,1});
  // No mutation of the caller graph until every host geometry/alias/prefix check passes.
  for (const auto &d:result.dispatches()) {
    std::vector<metal::MetalBuffer> buffers;for (const auto &b:d.buffers) buffers.push_back(b.buffer);
    if (d.bytes[0].sizeBytes==sizeof(FlashQSAParams)) {FlashQSAParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));graph.add(d.pipelineName,buffers,p,d.threadgroups,d.threadsPerThreadgroup);}
    else {FlashQSAFastParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));graph.add(d.pipelineName,buffers,p,d.threadgroups,d.threadsPerThreadgroup);}
  }
}
}
