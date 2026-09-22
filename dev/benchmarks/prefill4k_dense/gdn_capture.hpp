#pragma once
#include "flash/FlashGDNStaged.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashForward.h"
#include "metal/abi/FlashGDN.h"
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace private_gdn_capture {
struct Snapshot {
  uint32_t rows = 0;
  FlashGDNParams params{};
  splash::metal::MetalBuffer mixed,decay,beta,z,norm,initialState,expectedState,
      expectedRecurrence,expectedOutput;
};
inline std::optional<Snapshot> snapshot;
inline void copy(splash::metal::CommandGraph &graph,const splash::metal::MetalBuffer &source,
                 const splash::metal::MetalBuffer &destination,uint64_t bytes) {
  if (!bytes || bytes%4 || source.sizeBytes() < bytes || destination.sizeBytes() < bytes)
    throw std::runtime_error("GDN capture copy extent invalid");
  graph.add("flash_forward_copy_words",{source,destination},FlashForwardCopyParams{bytes/4},
      {(bytes/4+255)/256,1,1});
}
inline void append(splash::metal::CommandGraph &graph,const splash::metal::ComputeDispatch &d) {
  if (d.bytes.size()!=1 || d.bytes[0].sizeBytes!=sizeof(FlashGDNParams))
    throw std::runtime_error("GDN capture dispatch ABI drift");
  FlashGDNParams p{};std::memcpy(&p,d.bytes[0].data,sizeof(p));
  std::vector<splash::metal::MetalBuffer> buffers;
  for (const auto &b : d.buffers) {
    if (b.index!=buffers.size()) throw std::runtime_error("GDN capture binding order drift");
    buffers.push_back(b.buffer);
  }
  graph.add(d.pipelineName,std::move(buffers),p,d.threadgroups,d.threadsPerThreadgroup);
}
inline void add(splash::metal::MetalBackend &backend,splash::metal::CommandGraph &graph,
    const splash::flash::FlashGDNWeights &w,const splash::flash::FlashGDNBuffers &b,
    const splash::flash::FlashGDNState &state,uint32_t rows,uint32_t layer,float epsilon) {
  using namespace splash::flash;using namespace splash::metal;
  const char *directory=std::getenv("PREFILL4K_GDN_CAPTURE");
  if (!directory || !*directory || snapshot || layer!=0 || rows!=2048) {
    addGDNStagedPrefill(graph,w,b,state,rows,1,FlashGDNStageTile::Values16Time16,epsilon);return;
  }
  CommandGraph validated;
  addGDNStagedPrefill(validated,w,b,state,rows,1,FlashGDNStageTile::Values16Time16,epsilon);
  if (validated.dispatches().size()!=4) throw std::runtime_error("GDN capture graph ABI drift");
  Snapshot s;s.rows=rows;
  std::memcpy(&s.params,validated.dispatches()[1].bytes[0].data,sizeof(s.params));
  const auto allocation=[&](uint64_t bytes) {return backend.allocateBuffer(bytes,BufferStorage::Shared,"private-actual-gdn-capture");};
  s.mixed=allocation(uint64_t(rows)*10240*2);s.decay=allocation(uint64_t(rows)*48*4);
  s.beta=allocation(uint64_t(rows)*48*2);s.z=allocation(uint64_t(rows)*6144*2);
  s.initialState=allocation(flashGDNRecurrentLaneBytes());s.expectedState=allocation(flashGDNRecurrentLaneBytes());
  s.expectedRecurrence=allocation(uint64_t(rows)*6144*2);s.expectedOutput=allocation(uint64_t(rows)*6144*2);
  s.norm=backend.view(w.norm->buffer,0,128*2);
  copy(graph,state.recurrent,s.initialState,s.initialState.sizeBytes());
  append(graph,validated.dispatches()[0]);
  copy(graph,b.mixed,s.mixed,s.mixed.sizeBytes());copy(graph,b.decay,s.decay,s.decay.sizeBytes());
  copy(graph,b.beta,s.beta,s.beta.sizeBytes());copy(graph,b.z,s.z,s.z.sizeBytes());
  append(graph,validated.dispatches()[1]);
  copy(graph,state.recurrent,s.expectedState,s.expectedState.sizeBytes());
  copy(graph,b.recurrentRows,s.expectedRecurrence,s.expectedRecurrence.sizeBytes());
  append(graph,validated.dispatches()[2]);
  copy(graph,b.output,s.expectedOutput,s.expectedOutput.sizeBytes());
  append(graph,validated.dispatches()[3]);snapshot=std::move(s);
}
void write();
} // namespace private_gdn_capture
