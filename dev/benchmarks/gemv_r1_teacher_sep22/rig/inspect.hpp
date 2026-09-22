#pragma once
#include "flash/FlashRequestStateInternal.hpp"
#include "flash/FlashGDNLazyRollback.hpp"
#include "flash/FlashInt8ExpertStore.hpp"
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>
namespace splash::flash {
class FlashDeepPrefixOracle final {
public:
  enum class Type : uint32_t { BF16=1, F32=2, I64=3, U32=4 };
  struct Plane { std::string label; Type type; metal::MetalBuffer buffer; uint64_t live; };
  static std::vector<Plane> planes(const FlashRequestState &request) {
    if (!request.impl_) throw std::logic_error("uninitialized snapshot state");
    const auto &s=*request.impl_; std::vector<Plane> out;
    const auto add=[&](std::string label,Type type,const metal::MetalBuffer &buffer,uint64_t live) {
      if (!buffer || !buffer.contents() || live>buffer.sizeBytes())
        throw std::logic_error("invalid state plane extent: "+label);
      out.push_back({std::move(label),type,buffer,live});
    };
    for (uint32_t layer=0;layer<48;++layer) {
      const auto p="layer."+std::to_string(layer)+".";
      if (s.gdn[layer].recurrent) {
        add(p+"gdn.convolution",Type::BF16,s.gdn[layer].convolution,flashGDNConvolutionLaneBytes());
        add(p+"gdn.recurrent",Type::F32,s.gdn[layer].recurrent,flashGDNRecurrentLaneBytes());
      } else {
        const auto &q=s.qsa[layer];
        add(p+"qsa.keys",Type::BF16,q.keys,s.length*1024);
        add(p+"qsa.values",Type::BF16,q.values,s.length*1024);
        add(p+"qsa.raw_index_keys",Type::BF16,q.rawIndexKeys,s.length*256);
        add(p+"qsa.pooled_keys",Type::BF16,q.pooledKeys,(s.length/4)*256);
        add(p+"qsa.index_positions",Type::I64,q.indexPositions,s.length*8);
      }
    }
    add("ple.history",Type::I64,s.pleHistory,16);
    add("ple.convolution",Type::BF16,s.pleConvolution,9ULL*10240*2);
    if (out.size()!=134) throw std::logic_error("state inventory must contain134 planes");
    return out;
  }
  static std::array<uintptr_t,3> binding(const FlashRequestState &request) {
    if (!request.impl_) throw std::logic_error("uninitialized owner binding");
    return {reinterpret_cast<uintptr_t>(request.impl_.get()),
      reinterpret_cast<uintptr_t>(request.impl_->owner.get()),
      reinterpret_cast<uintptr_t>(request.impl_->identity.get())};
  }
  static std::string metadata(const FlashForward &target,const FlashRequestState &request) {
    if (!request.impl_) throw std::logic_error("uninitialized state metadata");
    const auto &s=*request.impl_; std::ostringstream out;
    out<<"{\"length\":"<<s.length<<",\"capacity\":"<<s.capacity
      <<",\"poisoned\":"<<(s.poisoned?"true":"false")
      <<",\"pending\":"<<(s.pendingVerification?"true":"false")
      <<",\"owned\":"<<(target.ownsState(request)?"true":"false")<<",\"geometry\":[";
    for (uint32_t i=0;i<48;++i) {
      if (i) out<<',';
      if (s.gdn[i].recurrent) out<<'['<<i<<",\"gdn\","<<s.gdn[i].convolutionLaneStrideBytes
        <<','<<s.gdn[i].recurrentLaneStrideBytes<<']';
      else out<<'['<<i<<",\"qsa\","<<s.qsa[i].capacity<<']';
    }
    out<<"]}"; return out.str();
  }
  static bool pending(const FlashRequestState &request) {
    return request.impl_ && request.impl_->pendingVerification;
  }
  struct Guard { metal::MetalBuffer base,view; uint64_t bytes; void check()const {
    const auto *p=static_cast<const uint8_t *>(base.contents());
    for(uint32_t i=0;i<64;++i)if(p[i]!=0x5a)throw std::runtime_error("request prefix redzone changed");
    for(uint64_t i=64+bytes;i<base.sizeBytes();++i)if(p[i]!=0x5a)throw std::runtime_error("request suffix redzone changed");
  }};
  static constexpr uint64_t guardedStateAllowance(){return 134ULL*16384;}
  static std::vector<Guard> guardState(metal::MetalBackend &backend,FlashRequestState &request){
    if(!request.impl_)throw std::logic_error("uninitialized guarded state");
    std::vector<Guard> guards; guards.reserve(134);
    const auto wrap=[&](metal::MetalBuffer &buffer){
      const uint64_t bytes=buffer.sizeBytes();
      if(!buffer||!buffer.contents()||!bytes)throw std::logic_error("state guard source unavailable");
      auto base=backend.allocateBuffer((bytes+128+16383)&~uint64_t{16383},metal::BufferStorage::Shared,"oracle request redzones");
      std::memset(base.contents(),0x5a,base.sizeBytes());
      auto view=backend.view(base,64,bytes);std::memcpy(view.contents(),buffer.contents(),bytes);
      buffer=view;guards.push_back({base,view,bytes});
    };
    auto &s=*request.impl_;
    for(uint32_t i=0;i<48;++i)if(s.gdn[i].recurrent){wrap(s.gdn[i].convolution);wrap(s.gdn[i].recurrent);}
      else{wrap(s.qsa[i].keys);wrap(s.qsa[i].values);wrap(s.qsa[i].rawIndexKeys);wrap(s.qsa[i].pooledKeys);wrap(s.qsa[i].indexPositions);}
    wrap(s.pleHistory);wrap(s.pleConvolution);
    if(guards.size()!=134)throw std::logic_error("request guard census differs");return guards;
  }
  struct Undefined { std::string label; metal::MetalBuffer buffer; uint64_t begin; };
  static metal::MetalBuffer hidden(const FlashForward &target,uint32_t rows);
  static std::array<metal::MetalBuffer,7> coefficients(const FlashInt8ExpertStore &store,uint32_t layer);
  static void clone(const FlashRequestState &source,FlashRequestState &destination) {
    if(!source.impl_||!destination.impl_||source.impl_->owner!=destination.impl_->owner)
      throw std::logic_error("same-owner state clone required");
    auto a=planes(source),b=planes(destination);
    for(size_t i=0;i<a.size();++i) {
      if(a[i].label!=b[i].label||a[i].buffer.sizeBytes()!=b[i].buffer.sizeBytes())throw std::logic_error("clone schema mismatch");
      std::memcpy(b[i].buffer.contents(),a[i].buffer.contents(),a[i].buffer.sizeBytes());
    }
    destination.impl_->length=source.impl_->length;destination.impl_->poisoned=source.impl_->poisoned;
    destination.impl_->pendingVerification=source.impl_->pendingVerification;
  }
  static void poison(FlashRequestState &request,bool value){request.impl_->poisoned=value;}
  static uint64_t length(FlashRequestState &request,uint64_t value){auto old=request.impl_->length;request.impl_->length=value;return old;}
  static void pending(FlashRequestState &request,bool value){request.impl_->pendingVerification=value;}
  static std::shared_ptr<const uint8_t> changeOwner(FlashRequestState &request,std::shared_ptr<const uint8_t> owner){
    auto old=request.impl_->owner;request.impl_->owner=std::move(owner);return old;
  }
  static std::vector<Plane> tapePlanes(const FlashForward &target);
  static std::vector<Undefined> tapeUndefined(const FlashForward &target);
  static std::string tapeMetadata(const FlashForward &target);
  static bool tapeCanaries(const FlashForward &target);
  static uint32_t retainedValue(const FlashForward &target);
  static bool rawOwns(const FlashForward &target,const FlashRequestState &request);
  static std::array<uint64_t,3> tapeStatus(const FlashForward &target);

};

}
