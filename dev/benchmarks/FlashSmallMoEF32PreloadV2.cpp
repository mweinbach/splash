#include "FlashSmallMoEF32.hpp"
#include <cstring>
#include <array>
#include <stdexcept>
#include <string>

namespace splash::flash::candidate {
namespace {
void require(bool value,const char *message){if(!value)throw std::invalid_argument(message);}
void geometry(uint32_t rows,uint32_t gr){require(rows && rows<=16 && (gr==1 || gr==2 || gr==4),"small F32 MoE row/group geometry invalid");}
void size(const metal::MetalBuffer &b,uint64_t bytes){require(b && b.sizeBytes()>=bytes,"small F32 MoE buffer extent too short");}
bool overlap(const metal::MetalBuffer &a,const metal::MetalBuffer &b){
  if(a.sameView(b))return true;
  const auto x=reinterpret_cast<uintptr_t>(a.contents()),y=reinterpret_cast<uintptr_t>(b.contents());
  return x && y && (x<=y?y-x<a.sizeBytes():x-y<b.sizeBytes());
}
void scratch(const SmallMoEF32Scratch &s,uint32_t rows){require(s.routeCapacity>=rows*10,"small F32 MoE job capacity insufficient");size(s.jobs,uint64_t(rows)*10*32);size(s.count,4);require(!overlap(s.jobs,s.count),"small F32 MoE job/count alias");}
FlashAffineParams validated(metal::MetalBuffer input,const FlashAffineProjection &projection,
    metal::MetalBuffer ids,metal::MetalBuffer output,metal::MetalBuffer diagnostics,uint32_t rows,bool perSelection){
  metal::CommandGraph source;addGatheredAffine(source,input,projection,ids,output,diagnostics,rows,10,perSelection);
  require(source.dispatches().size()==1,"small F32 MoE affine validator differs");
  const auto &d=source.dispatches()[0];
  require(d.pipelineName==(perSelection?"flash_expert_qmv_contig_k8_sg2_c4":"flash_expert_qmv_contig_k16_sg4_c2") &&
      d.bytes.size()==1 && d.bytes[0].sizeBytes==64,"small F32 MoE requires current contiguous QMV producer");
  FlashAffineParams p;std::memcpy(&p,d.bytes[0].data,sizeof(p));return p;
}
void disjoint(std::span<const metal::MetalBuffer> outputs,std::span<const metal::MetalBuffer> inputs){
  for(size_t i=0;i<outputs.size();++i){for(size_t j=i+1;j<outputs.size();++j)require(!overlap(outputs[i],outputs[j]),"small F32 MoE outputs alias");
    for(const auto &input:inputs)require(!overlap(outputs[i],input),"small F32 MoE output aliases immutable/source buffer");}
}
}
SmallMoEF32Scratch allocateSmallMoEF32Scratch(metal::MetalBackend &backend,uint32_t rows){
  geometry(rows,1);return {backend.allocateBuffer(uint64_t(rows)*10*32,metal::BufferStorage::Shared,"private small F32 MoE jobs"),
    backend.allocateBuffer(4,metal::BufferStorage::Shared,"private small F32 MoE count"),rows*10};
}
void addSmallMoEF32Jobs(metal::CommandGraph &graph,metal::MetalBuffer ids,const SmallMoEF32Scratch &s,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t gr){
  geometry(rows,gr);scratch(s,rows);size(ids,uint64_t(rows)*10*8);size(diagnostics,4);
  const std::array<metal::MetalBuffer,2> output{s.jobs,s.count},input{ids,diagnostics};disjoint(output,input);
  if(gr>1)graph.add("flash_small_moe_f32_preload_v2_jobs",{ids,s.jobs,s.count,diagnostics},
    FlashSmallMoEF32JobParams{rows,10,rows*10,gr,0,0,0,0},{1,1,1},{256,1,1});
}
void addSmallMoEF32Gate(metal::CommandGraph &graph,metal::MetalBuffer input,const FlashAffineProjection &gate,
    const FlashAffineProjection &up,metal::MetalBuffer ids,const SmallMoEF32Scratch &s,metal::MetalBuffer gout,
    metal::MetalBuffer uout,metal::MetalBuffer activation,metal::MetalBuffer gtap,metal::MetalBuffer utap,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t gr,bool taps){
  geometry(rows,gr);scratch(s,rows);const auto g=validated(input,gate,ids,gout,diagnostics,rows,false);
  const auto u=validated(input,up,ids,uout,diagnostics,rows,false);
  require(g.input_size==2560 && g.output_size==640 && u.input_size==2560 && u.output_size==640,"small F32 MoE gate/up matrix geometry differs");
  size(activation,uint64_t(rows)*10*640*2);size(gtap,uint64_t(rows)*10*640*4);size(utap,uint64_t(rows)*10*640*4);
  const std::array<metal::MetalBuffer,5> output{gout,uout,activation,gtap,utap};
  const std::array<metal::MetalBuffer,11> inputs{input,ids,s.jobs,s.count,diagnostics,gate.weights->buffer,gate.scales->buffer,
      gate.biases->buffer,up.weights->buffer,up.scales->buffer,up.biases->buffer};disjoint(output,inputs);
  graph.add("flash_small_moe_f32_preload_v2_gate_up_gr"+std::to_string(gr),
    {input,gate.weights->buffer,gate.scales->buffer,gate.biases->buffer,up.weights->buffer,up.scales->buffer,
     up.biases->buffer,ids,s.jobs,s.count,gout,uout,activation,gtap,utap,diagnostics},
    FlashSmallMoEF32GateParams{g,u,gr,rows*10,uint32_t(taps),0},{80,rows*10,1},{128,1,1});
}
void addSmallMoEF32Down(metal::CommandGraph &graph,metal::MetalBuffer input,const FlashAffineProjection &down,
    metal::MetalBuffer ids,const SmallMoEF32Scratch &s,metal::MetalBuffer output,metal::MetalBuffer tap,
    metal::MetalBuffer diagnostics,uint32_t rows,uint32_t gr,bool taps){
  geometry(rows,gr);scratch(s,rows);const auto p=validated(input,down,ids,output,diagnostics,rows,true);
  size(tap,uint64_t(rows)*10*2560*4);
  const std::array<metal::MetalBuffer,2> outputs{output,tap};
  const std::array<metal::MetalBuffer,8> inputs{input,ids,s.jobs,s.count,diagnostics,down.weights->buffer,down.scales->buffer,down.biases->buffer};disjoint(outputs,inputs);
  graph.add("flash_small_moe_f32_preload_v2_down_gr"+std::to_string(gr),
    {input,down.weights->buffer,down.scales->buffer,down.biases->buffer,ids,s.jobs,s.count,output,tap,diagnostics},
    FlashSmallMoEF32DownParams{p,gr,rows*10,uint32_t(taps),0},{320,rows*10,1},{64,1,1});
}
void addSmallMoEF32Reference(metal::CommandGraph &graph,metal::MetalBuffer input,const FlashAffineProjection &p,
    metal::MetalBuffer ids,metal::MetalBuffer output,metal::MetalBuffer tap,metal::MetalBuffer diagnostics,
    uint32_t rows,bool perSelection){
  const auto parameters=validated(input,p,ids,output,diagnostics,rows,perSelection);size(tap,uint64_t(rows)*10*p.outputSize*4);
  const std::array<metal::MetalBuffer,2> outputs{output,tap};
  const std::array<metal::MetalBuffer,6> inputs{input,ids,diagnostics,p.weights->buffer,p.scales->buffer,p.biases->buffer};disjoint(outputs,inputs);
  graph.add(perSelection?"flash_small_moe_reference_contig_k8_sg2_c4":"flash_small_moe_reference_contig_k16_sg4_c2",
    {input,p.weights->buffer,p.scales->buffer,p.biases->buffer,ids,output,diagnostics,tap},parameters,
    {p.outputSize/8,rows,10},{perSelection?64u:128u,1,1});
}
} // namespace splash::flash::candidate
