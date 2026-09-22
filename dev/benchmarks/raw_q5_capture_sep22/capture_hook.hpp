#pragma once
// Oracle-clone-only observer. No shipping header, shader or public API changes.
#include "metal/CommandGraph.hpp"
#include "metal/abi/FlashForward.h"
#include "metal/abi/FlashAffine.h"
#include "dev/benchmarks/raw_q5_rowpair_sep22/storage.hpp"
#include <cstring>
#include <sstream>

namespace raw_q5_capture_sep22 {
using namespace splash::metal;
inline constexpr const char *kPrefix="language_model.model.layers.0.linear_attn.out_proj";
inline constexpr uint64_t kBytes=49152,kAllocation=65536;
inline bool eligible(std::string_view prefix,uint32_t rows,bool verify,uint32_t begin) noexcept {
  return prefix==kPrefix&&rows==4&&verify&&begin==2048;
}
class Observer final {
public:
  explicit Observer(MetalBackend &backend) {
    const auto before=backend.memoryStats().allocatedBytes;
    output_=raw_q5_rowpair_sep22::Guarded::allocate(backend,kBytes,"owned current VerifyR4 layer0 GDN output input copy");
    const auto after=backend.memoryStats().allocatedBytes;
    if(after<before||after-before!=kAllocation)throw std::logic_error("capture owner actual allocation differs");
    allocated_=after-before;
  }
  Observer(const Observer&)=delete;Observer&operator=(const Observer&)=delete;
  Observer(Observer&&)=delete;Observer&operator=(Observer&&)=delete;
  void before(CommandGraph &graph,std::string_view prefix,const MetalBuffer &input,
      uint32_t rows,bool verify,uint32_t begin) {
    if(!eligible(prefix,rows,verify,begin))return;
    if(armed_||completed_)throw std::logic_error("duplicate current layer0 GDN output capture");
    if(!input||input.storage()!=BufferStorage::Shared||!input.contents()||input.sizeBytes()!=kBytes||
       reinterpret_cast<uintptr_t>(input.contents())%4||!output_.view.contents()||
       reinterpret_cast<uintptr_t>(output_.view.contents())%4)
      throw std::invalid_argument("current GDN output capture input extent/storage/alignment differs");
    const uintptr_t a=reinterpret_cast<uintptr_t>(input.contents()),b=reinterpret_cast<uintptr_t>(output_.view.contents());
    if(input.sameView(output_.view)||a>UINTPTR_MAX-kBytes||b>UINTPTR_MAX-kBytes||
       (a<b+kBytes&&b<a+kBytes))throw std::invalid_argument("current GDN output capture aliases producer input");
    source_=input;copyIndex_=graph.dispatches().size();
    graph.add("flash_forward_copy_words",{input,output_.view},FlashForwardCopyParams{kBytes/4},{48,1,1},{256,1,1});
    projectionIndex_=graph.dispatches().size();armed_=true;
  }
  void after(const CommandGraph &graph,std::string_view prefix,uint32_t rows,bool verify,uint32_t begin) {
    if(!eligible(prefix,rows,verify,begin))return;
    if(!armed_||completed_||graph.dispatches().size()!=projectionIndex_+1)
      throw std::logic_error("current GDN output capture projection descriptor count differs");
    const auto&d=graph.dispatches()[projectionIndex_];
    if(d.pipelineName!="flash_affine_mlx_qmv_f32xsum_v1_q5_g128"||d.threadgroups.x!=320||d.threadgroups.y!=4||d.threadgroups.z!=1||
       d.threadsPerThreadgroup.x!=64||d.threadsPerThreadgroup.y!=1||d.threadsPerThreadgroup.z!=1||d.buffers.size()!=7||d.bytes.size()!=1||
       d.bytes[0].index!=7||d.bytes[0].sizeBytes!=64||!d.bytes[0].data)
      throw std::logic_error("actual current projection is not original raw Q5/G128 descriptor");
    std::memcpy(&params_,d.bytes[0].data,64);
    if(params_.rows!=4||params_.selections!=1||params_.input_size!=6144||params_.output_size!=2560||params_.experts!=1||params_.bits!=5||
       params_.group_size!=128||params_.flags||params_.weight_row_stride_bytes!=3840||params_.parameter_row_stride_bytes!=96)
      throw std::logic_error("actual current raw projection parameters differ");
    constexpr std::array<uint64_t,7>minimum{49152,9830400,245760,245760,1,20480,4};
    for(uint32_t i=0;i<7;++i){
      if(d.buffers[i].index!=i||!d.buffers[i].buffer||d.buffers[i].buffer.sizeBytes()<minimum[i])
        throw std::logic_error("actual current projection binding extent/index differs");
      bindingBytes_[i]=d.buffers[i].buffer.sizeBytes();
    }
    if(!d.buffers[0].buffer.sameView(source_))throw std::logic_error("captured source differs from actual GDN output binding0");
    const auto&c=graph.dispatches()[copyIndex_];
    if(c.pipelineName!="flash_forward_copy_words"||c.buffers.size()!=2||!c.buffers[0].buffer.sameView(source_)||!c.buffers[1].buffer.sameView(output_.view)||
       c.bytes.size()!=1||c.bytes[0].index!=2||c.bytes[0].sizeBytes!=sizeof(FlashForwardCopyParams)||!c.bytes[0].data||c.threadgroups.x!=48||c.threadgroups.y!=1||c.threadgroups.z!=1||
       c.threadsPerThreadgroup.x!=256||c.threadsPerThreadgroup.y!=1||c.threadsPerThreadgroup.z!=1)
      throw std::logic_error("actual owned copy descriptor differs");
    FlashForwardCopyParams p{};std::memcpy(&p,c.bytes[0].data,sizeof(p));if(p.words!=12288)throw std::logic_error("copy word count differs");
    completed_=1;armed_=false;
  }
  const MetalBuffer &output()const {if(completed_!=1||armed_)throw std::logic_error("actual capture was not completed exactly once");return output_.view;}
  const MetalBuffer &diagnosticOwnerView()const noexcept{return output_.view;}
  void check()const {if(!output_.clean())throw std::logic_error("current input capture redzones changed");}
  uint64_t allocatedBytes()const noexcept{return allocated_;}
  uint64_t completed()const noexcept{return completed_;}
  std::string metadata()const {
    std::ostringstream o;o<<"{\"copy_pipeline\":\"flash_forward_copy_words\",\"copy_dispatch_index\":"<<copyIndex_<<",\"projection_dispatch_index\":"<<projectionIndex_
      <<",\"copy_words\":12288,\"copy_grid\":[48,1,1],\"copy_threads\":[256,1,1],\"actual_projection\":\"flash_affine_mlx_qmv_f32xsum_v1_q5_g128\",\"actual_projection_grid\":[320,4,1],\"actual_projection_threads\":[64,1,1],\"input_binding0_exact_copied_view\":true,\"weight_row_stride_bytes\":"<<params_.weight_row_stride_bytes<<",\"parameter_row_stride_bytes\":"<<params_.parameter_row_stride_bytes<<",\"actual_binding_bytes\":[";
    for(uint32_t i=0;i<7;++i){if(i)o<<',';o<<bindingBytes_[i];}o<<"],\"encoded_captures\":"<<completed_<<",\"owned_logical_bytes\":49152,\"actual_capture_allocation_bytes\":"<<allocated_<<",\"capture_is_borrowed_scratch\":false}";return o.str();
  }
private:
  raw_q5_rowpair_sep22::Guarded output_;MetalBuffer source_;
  bool armed_=false;uint64_t completed_=0,copyIndex_=0,projectionIndex_=0,allocated_=0;
  FlashAffineParams params_{};std::array<uint64_t,7>bindingBytes_{};
};
inline thread_local Observer *active=nullptr;
class Scope final {
public:explicit Scope(Observer&observer){if(active)throw std::logic_error("nested current input capture observer");active=&observer;}
  Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
  ~Scope(){active=nullptr;}
};
inline void before(CommandGraph&g,std::string_view p,const MetalBuffer&input,uint32_t r,bool v,uint32_t b){if(active)active->before(g,p,input,r,v,b);}
inline void after(const CommandGraph&g,std::string_view p,uint32_t r,bool v,uint32_t b){if(active)active->after(g,p,r,v,b);}
} // namespace raw_q5_capture_sep22
