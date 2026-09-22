#pragma once
#include "flash/FlashQSAFast.hpp"
#include "metal/abi/FlashForward.h"
#include <array>
#include <stdexcept>
#include <string>
#include <limits>
#include <cstring>
#include <span>
#include <vector>

// Private development capture. No public executor/header/layout or normal
// Worker entrypoint changes. The serialized Root oracle owns this session.
namespace splash::flash::batch_qsa_capture {
constexpr uint64_t inputsBytes = 57147392;
constexpr uint64_t outputBytes = 25165824;
constexpr uint64_t reservedBytes = 84ULL * 1024 * 1024;
struct Session {
  uint32_t layer=3,lane=0,actualLanes=0,rows=0,capacity=0;
  bool cohortValidated=false,inputEncoded=false,outputEncoded=false,completed=false;
  uint64_t insertionDispatch=0,outputInsertionDispatch=0,actualCharge=0;
  uint64_t disjointOtherViews=0;
  std::array<uint64_t,6> legacyProducerOrdinals{};
  std::array<std::string,6> legacyProducerPipelines{};
  std::array<uint64_t,6> legacyProducerBufferCounts{};
  std::array<uint64_t,6> legacyProducerParameterBytes{};
  std::string sourceIdentity;
  std::array<metal::MetalBuffer,4> projected;
  std::array<metal::MetalBuffer,9> guardedOwners;
  std::array<FlashTensor,4> norms;
  std::array<NormConvention,4> conventions{};
  metal::MetalBuffer output;
  Session(metal::MetalBackend &backend,const std::array<const FlashTensor*,4>& original,
      uint64_t admittedBytes) {
    if(admittedBytes<reservedBytes)throw std::invalid_argument("complete capture reservation before allocation required");
    const uint64_t before=backend.memoryStats().allocatedBytes;
    constexpr std::array<uint64_t,4> bytes{50331648,2097152,2097152,2621440};
    const auto guarded=[&](size_t slot,uint64_t bytes) {
      guardedOwners[slot]=backend.allocateBuffer(bytes+512,metal::BufferStorage::Shared,"Root retained guarded capture owner");
      std::memset(guardedOwners[slot].contents(),0xa5,bytes+512);
      return backend.view(guardedOwners[slot],256,bytes);
    };
    for(size_t i=0;i<4;++i)projected[i]=guarded(i,bytes[i]);
    output=guarded(4,outputBytes);
    for(size_t i=0;i<4;++i) {
      if(!original[i] || !original[i]->logicalBytes || original[i]->logicalBytes%4 ||
          (original[i]->dtype!=FlashDType::BF16 && original[i]->dtype!=FlashDType::F32))
        throw std::invalid_argument("actual norm tensor dtype/shape/logical extent required");
      uint64_t elements=1;
      if(original[i]->shape.empty())throw std::invalid_argument("actual norm requires real nonempty shape");
      for(auto extent:original[i]->shape) {
        if(!extent || extent>std::numeric_limits<uint64_t>::max()/elements)
          throw std::invalid_argument("actual norm shape product invalid");
        elements*=extent;
      }
      const uint64_t word=original[i]->dtype==FlashDType::F32?4:2;
      if(elements>std::numeric_limits<uint64_t>::max()/word || elements*word!=original[i]->logicalBytes)
        throw std::invalid_argument("actual norm logical bytes differ from its dtype and shape");
      norms[i]=*original[i];norms[i].buffer=guarded(5+i,norms[i].logicalBytes);
    }
    const uint64_t after=backend.memoryStats().allocatedBytes;
    if(after<before || after-before>admittedBytes)throw std::logic_error("capture actual allocatedSize charge exceeds admitted plan");
    actualCharge=after-before;
  }
};
inline thread_local Session *activeSession=nullptr;
struct Install {
  Session *previous;
  explicit Install(Session &value):previous(activeSession) {
    if(previous)throw std::logic_error("nested capture sessions forbidden");activeSession=&value;
  }
  ~Install(){activeSession=previous;}
};
inline void requireDisjoint(const metal::MetalBuffer &a,const metal::MetalBuffer &b) {
  if(!a || !b) return;
  if(a.sameView(b))throw std::invalid_argument("capture owner aliases existing allocation view");
  // A Shared capture allocation cannot share an owner with Private storage.
  if(b.storage()!=metal::BufferStorage::Shared)return;
  const auto x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());
  if(!x || !y || a.storage()!=metal::BufferStorage::Shared ||
      a.sizeBytes()>std::numeric_limits<uintptr_t>::max()-x ||
      b.sizeBytes()>std::numeric_limits<uintptr_t>::max()-y ||
      (x<y+b.sizeBytes() && y<x+a.sizeBytes()))
    throw std::invalid_argument("capture owner overlaps batch, request, immutable or capture storage");
}
inline void validateWholeCohort(metal::MetalBackend &backend,uint32_t lanes,uint32_t rows,
    uint32_t capacity,std::span<const uint64_t> lengths,const std::vector<metal::MetalBuffer> &others) {
  auto *s=activeSession;if(!s)return;
  if(s->cohortValidated || (lanes!=2 && lanes!=4) || rows!=2048 || capacity!=16384 ||
      lengths.size()!=lanes || s->lane>=lanes)
    throw std::invalid_argument("capture requires one complete real fresh2048 B2/B4 cohort");
  for(auto length:lengths)if(length)throw std::invalid_argument("capture whole cohort must have fresh context0");
  for(size_t i=0;i<s->guardedOwners.size();++i) {
    const auto &owner=s->guardedOwners[i];
    if(!owner || owner.storage()!=metal::BufferStorage::Shared || !owner.contents())
      throw std::invalid_argument("capture requires owned Shared guarded buffers");
    // Verify allocation ownership on the actual backend before any normal writes.
    (void)backend.view(owner,0,owner.sizeBytes());
    for(size_t j=0;j<i;++j)requireDisjoint(owner,s->guardedOwners[j]);
    for(const auto &other:others)requireDisjoint(owner,other);
  }
  constexpr std::array<uint64_t,4> projectionBytes{50331648,2097152,2097152,2621440};
  for(size_t i=0;i<9;++i) {
    const uint64_t bytes=i<4?projectionBytes[i]:i==4?outputBytes:s->norms[i-5].logicalBytes;
    const auto &view=i<4?s->projected[i]:i==4?s->output:s->norms[i-5].buffer;
    if(!view.sameView(backend.view(s->guardedOwners[i],256,bytes)))
      throw std::invalid_argument("capture destination differs from its exact guarded owner view");
  }
  s->actualLanes=lanes;s->rows=rows;s->capacity=capacity;
  s->disjointOtherViews=others.size();s->cohortValidated=true;
}
inline void copy(metal::CommandGraph& graph,const metal::MetalBuffer &from,
    const metal::MetalBuffer &to,uint64_t bytes) {
  if(!from || !to || from.sizeBytes()<bytes || to.sizeBytes()!=bytes || bytes%4 || from.sameView(to) ||
      to.storage()!=metal::BufferStorage::Shared || !to.contents())
    throw std::invalid_argument("capture GPU copy requires exact owned nonalias views");
  const auto address=reinterpret_cast<uintptr_t>(to.contents());
  if(address%4 || bytes>std::numeric_limits<uintptr_t>::max()-address)
    throw std::invalid_argument("capture destination word alignment or address extent invalid");
  if(from.storage()==metal::BufferStorage::Shared) {
    const auto source=reinterpret_cast<uintptr_t>(from.contents());
    if(!source || source%4 || bytes>std::numeric_limits<uintptr_t>::max()-source ||
        (source<address+bytes && address<source+bytes))
      throw std::invalid_argument("capture source/destination overlapping range or word extent invalid");
  }
  const uint64_t words=bytes/4;
  graph.add("flash_forward_copy_words",{from,to},FlashForwardCopyParams{words},{(words+255)/256,1,1},{256,1,1});
}
inline void beforeQSA(metal::CommandGraph& graph,const FlashQSAFastInputs &input,
    const FlashQSAState &state,uint32_t layer,uint32_t lane,uint32_t lanes,uint32_t rows,
    uint64_t begin,std::string_view source) {
  auto *s=activeSession;if(!s || s->layer!=layer || s->lane!=lane)return;
  if(!s->cohortValidated || s->inputEncoded || lanes!=s->actualLanes || rows!=s->rows || begin || state.capacity!=s->capacity ||
      input.epsilon!=1e-6 || input.theta!=1e7)throw std::invalid_argument("actual capture must be unique real fresh2K B2/B4 lane0/layer3");
  s->insertionDispatch=graph.dispatches().size();s->actualLanes=lanes;s->rows=rows;s->capacity=state.capacity;s->sourceIdentity=source;
  const std::array<metal::MetalBuffer,4> from{input.qProjection,input.kProjection,input.vProjection,input.indexProjection};
  const std::array<const FlashTensor*,4> ns{input.qNorm,input.kNorm,input.indexQNorm,input.indexKNorm};
  s->conventions={input.qConvention,input.kConvention,input.indexQConvention,input.indexKConvention};
  for(size_t i=0;i<4;++i) {
    if(from[i].sizeBytes()!=s->projected[i].sizeBytes())throw std::invalid_argument("actual captured projection must have exact2048 real extent");
    if(!ns[i] || ns[i]->dtype!=s->norms[i].dtype || ns[i]->shape!=s->norms[i].shape || ns[i]->logicalBytes!=s->norms[i].logicalBytes)
      throw std::invalid_argument("capture norm differs from actual selected layer tensor");
    copy(graph,from[i],s->projected[i],s->projected[i].sizeBytes());copy(graph,ns[i]->buffer,s->norms[i].buffer,s->norms[i].logicalBytes);
  }
  s->inputEncoded=true;
}
inline void afterQSA(metal::CommandGraph& graph,const metal::MetalBuffer &output,uint32_t layer,uint32_t lane) {
  auto *s=activeSession;if(!s || s->layer!=layer || s->lane!=lane)return;
  if(!s->inputEncoded || s->outputEncoded)throw std::logic_error("capture output producer position invalid");
  if(output.sizeBytes()!=outputBytes)throw std::invalid_argument("capture output must have exact real2048 BF16 extent");
  s->outputInsertionDispatch=graph.dispatches().size();
  if(s->outputInsertionDispatch!=s->insertionDispatch+8+6)
    throw std::logic_error("capture actual legacy six-dispatch graph extent changed");
  const auto dispatches=graph.dispatches();
  for(size_t i=0;i<6;++i) {
    s->legacyProducerOrdinals[i]=s->insertionDispatch+8+i;
    s->legacyProducerPipelines[i]=dispatches[s->legacyProducerOrdinals[i]].pipelineName;
    const auto &dispatch=dispatches[s->legacyProducerOrdinals[i]];
    s->legacyProducerBufferCounts[i]=dispatch.buffers.size();
    if(dispatch.bytes.size()!=1)throw std::logic_error("capture legacy producer parameter binding changed");
    s->legacyProducerParameterBytes[i]=dispatch.bytes[0].sizeBytes;
  }
  if(s->legacyProducerPipelines[0]!="flash_qsa_fast_prepare" ||
      !s->legacyProducerPipelines[1].starts_with("flash_qsa_pool_rope_") ||
      s->legacyProducerPipelines[2]!="flash_qsa_select_blocks" ||
      s->legacyProducerPipelines[3]!="flash_qsa_mpp_prefill_bulk_early_2048" ||
      s->legacyProducerPipelines[4]!="flash_qsa_mpp_prefill_bulk_temporal_sg8_2048" ||
      s->legacyProducerPipelines[5]!="flash_qsa_fast_prefill_bulk_reduce_2048")
    throw std::logic_error("capture actual legacy producer pipeline contract changed");
  copy(graph,output,s->output,outputBytes);s->outputEncoded=true;
}
inline void completed(bool healthy) {
  auto *s=activeSession;if(!s)return;
  if(!healthy || !s->cohortValidated || !s->inputEncoded || !s->outputEncoded)throw std::logic_error("capture actual native submission/projection/output copies incomplete");
  s->completed=true;
}
inline void guardCheck(const Session &s) {
  std::array<uint64_t,9> bytes{s.projected[0].sizeBytes(),s.projected[1].sizeBytes(),s.projected[2].sizeBytes(),
      s.projected[3].sizeBytes(),s.output.sizeBytes(),s.norms[0].logicalBytes,s.norms[1].logicalBytes,s.norms[2].logicalBytes,s.norms[3].logicalBytes};
  for(size_t owner=0;owner<9;++owner) {
    const auto *p=static_cast<const uint8_t*>(s.guardedOwners[owner].contents());
    for(uint64_t byte=0;byte<256;++byte)
      if(p[byte]!=0xa5 || p[256+bytes[owner]+byte]!=0xa5)throw std::runtime_error("owned capture guard changed");
  }
}
} // namespace splash::flash::batch_qsa_capture
