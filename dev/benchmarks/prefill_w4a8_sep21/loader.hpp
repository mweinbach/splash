#pragma once
// The metadata entry point reads only manifest JSON/checksum text. load() is
// called only by Root's --gpu oracle, maps nine exact one-layer tensor ranges,
// and creates three temporary centered-code planes. No full-model store loads.
#include "flash/FlashWeights.hpp"
#include "prefill4k_allrows_qmv_one_layer.hpp"
#import <Foundation/Foundation.h>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

namespace splash::bench::w4a8 {
using metal::MetalBuffer;
using metal::MetalBackend;
inline constexpr uint64_t kAlignment=16384;
inline constexpr const char *kSourceIdentity="ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e";
inline constexpr const char *kQualifiedManifestSHA="0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402";
inline void require(bool value,const std::string &reason) {if(!value)throw std::invalid_argument("one-layer W4A8: "+reason);}
inline uint64_t rounded(uint64_t value) {require(value&&value<=UINT64_MAX-kAlignment,"bad allocation extent");return(value+kAlignment-1)&~(kAlignment-1);}
inline std::string hash(const void *p,uint64_t bytes) {return flash::qmv_one_layer::detail::hash(p,bytes);}
inline std::string hash(const MetalBuffer &b) {return hash(b.contents(),b.sizeBytes());}
struct TensorRange {
  std::string name,path,sourcePath,shardSHA;
  uint64_t fileBytes=0,offset=0,bytes=0,sourceOffset=0;
  std::vector<uint64_t> shape;
  flash::FlashDType dtype=flash::FlashDType::BF16;
};
struct LayerMetadata {
  std::filesystem::path root;
  std::string manifestSHA,sourceIdentity;
  uint32_t layer=0;
  std::array<TensorRange,9> tensors;
  uint64_t originalBytes=0,centeredBytes=0;
};
inline LayerMetadata inspect(const std::filesystem::path &directory,uint32_t layer) {
  require(layer<48,"layer outside0..47");LayerMetadata result;result.root=std::filesystem::canonical(directory);result.layer=layer;
  const auto manifestPath=result.root/"manifest.json";
  const auto manifestBytes=std::filesystem::file_size(manifestPath);
  require(manifestBytes>0&&manifestBytes<(16ULL<<20),"manifest is missing/oversized before read");
  NSData *bytes=[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:manifestPath.c_str()]];
  require(bytes&&bytes.length<(16ULL<<20),"manifest is missing/oversized");
  result.manifestSHA=hash(bytes.bytes,bytes.length);
  require(result.manifestSHA==kQualifiedManifestSHA,"qualified original metadata seal differs");
  std::ifstream check(result.root/"manifest.sha256");std::string expected;check>>expected;
  require(expected==result.manifestSHA,"locked native manifest checksum differs");
  NSError *error=nil;id parsed=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:&error];
  require(!error&&[parsed isKindOfClass:[NSDictionary class]],"manifest JSON invalid");NSDictionary *manifest=parsed;
  const auto string=[](id value){require([value isKindOfClass:[NSString class]],"manifest string invalid");return std::string([value UTF8String]);};
  const auto integer=[](id value){require([value isKindOfClass:[NSNumber class]],"manifest integer invalid");return uint64_t([value unsignedLongLongValue]);};
  require(string(manifest[@"schema"])=="splash-local-qwen4-affine-v1"&&integer(manifest[@"alignment"])==kAlignment,"native manifest schema differs");
  result.sourceIdentity=string(manifest[@"source_identity_sha256"]);require(result.sourceIdentity==kSourceIdentity,"original checkpoint source identity differs");
  NSDictionary *tensorMap=manifest[@"tensors"],*quantization=manifest[@"quantization"];
  require([tensorMap isKindOfClass:[NSDictionary class]]&&[quantization isKindOfClass:[NSDictionary class]],"tensor/quantization metadata invalid");
  std::map<std::string,NSDictionary *> shards;
  for(NSDictionary *record in manifest[@"shards"])shards.emplace(string(record[@"path"]),record);
  const std::string prefix="language_model.model.layers."+std::to_string(layer)+".mlp.switch_mlp.";
  const std::array<std::string,3> roles{"gate_proj","up_proj","down_proj"},suffixes{"weight","scales","biases"};
  for(uint32_t plane=0;plane<3;++plane) {
    const uint32_t n=plane==2?2560:640,k=plane==2?640:2560;
    const std::string projection=prefix+roles[plane];NSDictionary *q=quantization[[NSString stringWithUTF8String:projection.c_str()]];
    const uint64_t bits=q?integer(q[@"bits"]):integer(quantization[@"bits"]),group=q?integer(q[@"group_size"]):integer(quantization[@"group_size"]);
    require(bits==4&&group==64,"one-layer source must remain original unsignedQ4/G64");
    for(uint32_t field=0;field<3;++field) {
      auto &entry=result.tensors[plane*3+field];entry.name=projection+"."+suffixes[field];
      NSDictionary *record=tensorMap[[NSString stringWithUTF8String:entry.name.c_str()]];
      require([record isKindOfClass:[NSDictionary class]],"source tensor missing");entry.path=string(record[@"shard"]);
      const std::filesystem::path relative(entry.path);require(!relative.is_absolute(),"absolute source tensor shard");
      for(const auto &part:relative)require(part!="..","source tensor shard escapes package");
      require(shards.contains(entry.path),"unknown source shard");NSDictionary *shard=shards.at(entry.path);
      entry.fileBytes=integer(shard[@"bytes"]);entry.shardSHA=string(shard[@"sha256"]);
      entry.offset=integer(record[@"offset"]);entry.bytes=integer(record[@"length"]);
      entry.sourcePath=string(record[@"source_shard"]);entry.sourceOffset=integer(record[@"source_offset"]);
      require(entry.sourcePath==string(shard[@"source_path"]),"original source shard metadata differs");
      const uint64_t sourceBytes=integer(shard[@"source_bytes"]);
      require(entry.sourceOffset>=8&&entry.sourceOffset<=sourceBytes&&entry.bytes<=sourceBytes-entry.sourceOffset,"original source tensor range exceeds certified shard size");
      const auto canonical=std::filesystem::canonical(result.root/relative);
      const auto contained=canonical.lexically_relative(result.root);
      require(!contained.empty()&&!contained.is_absolute(),"source tensor is outside package");
      for(const auto &part:contained)require(part!="..","source tensor symlink escapes package");
      require(entry.offset%kAlignment==0&&entry.bytes%kAlignment==0&&entry.offset<=entry.fileBytes&&entry.bytes<=entry.fileBytes-entry.offset,"tensor range alignment/extent invalid");
      require(string(record[@"dtype"])==(field?"BF16":"U32"),"original tensor dtype differs");
      entry.dtype=field?flash::FlashDType::BF16:flash::FlashDType::U32;
      const uint64_t width=field?k/64:k/8;NSArray *shape=record[@"shape"];
      require([shape isKindOfClass:[NSArray class]]&&shape.count==3&&integer(shape[0])==512&&integer(shape[1])==n&&integer(shape[2])==width,"original tensor shape differs");
      entry.shape={512,n,width};require(entry.bytes==uint64_t(512)*n*width*(field?2:4),"original tensor logical bytes differ");
      result.originalBytes+=entry.bytes;
    }
    result.centeredBytes+=rounded(uint64_t(512)*n*(plane==2?384:1280));
  }
  for(uint32_t a=0;a<9;++a)for(uint32_t b=a+1;b<9;++b){
    const auto &aa=result.tensors[a],&bb=result.tensors[b];
    if(aa.path==bb.path)require(aa.offset+aa.bytes<=bb.offset||bb.offset+bb.bytes<=aa.offset,"selected native tensor ranges overlap");
    if(aa.sourcePath==bb.sourcePath)require(aa.sourceOffset+aa.bytes<=bb.sourceOffset||bb.sourceOffset+bb.bytes<=aa.sourceOffset,"selected original tensor ranges overlap");
  }
  require(result.originalBytes==1415577600ULL&&result.centeredBytes==1342177280ULL,"bounded one-layer source/centered ledger differs");
  return result;
}
class TensorMapping {
public:
  TensorMapping(const std::filesystem::path &path,uint64_t fileBytes,uint64_t offset,uint64_t length):length_(length) {
    fd_=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);require(fd_>=0,"cannot open readonly source tensor shard");
    if(::fstat(fd_,&before_)||!S_ISREG(before_.st_mode)||uint64_t(before_.st_size)!=fileBytes||(before_.st_mode&0222)) {
      ::close(fd_);fd_=-1;throw std::invalid_argument("one-layer W4A8 source shard must be exact-sized readonly regular file");
    }
    data_=::mmap(nullptr,length,PROT_READ,MAP_SHARED,fd_,static_cast<off_t>(offset));
    if(data_==MAP_FAILED){data_=nullptr;::close(fd_);fd_=-1;throw std::invalid_argument("one-layer W4A8 source tensor range mmap failed");}
  }
  ~TensorMapping(){if(data_)::munmap(data_,length_);if(fd_>=0)::close(fd_);}
  void *data()const{return data_;}
  void unchanged()const{
    struct stat after{};require(!::fstat(fd_,&after)&&after.st_dev==before_.st_dev&&after.st_ino==before_.st_ino&&after.st_size==before_.st_size&&
      after.st_mtimespec.tv_sec==before_.st_mtimespec.tv_sec&&after.st_mtimespec.tv_nsec==before_.st_mtimespec.tv_nsec&&
      after.st_ctimespec.tv_sec==before_.st_ctimespec.tv_sec&&after.st_ctimespec.tv_nsec==before_.st_ctimespec.tv_nsec&&!(after.st_mode&0222),"source tensor shard changed");
  }
private:int fd_=-1;void *data_=nullptr;uint64_t length_=0;struct stat before_{};
};
struct LayerPayload {
  LayerMetadata metadata;
  std::array<flash::FlashTensor,9> tensors;
  std::array<std::shared_ptr<TensorMapping>,9> mappings;
  std::array<MetalBuffer,3> centered;
  std::array<std::string,9> sourceHashes;
  std::array<std::string,3> centeredHashes;
  uint64_t plannedBytes=0,allocatedBytes=0,exactCodeBytesCertified=0;
  static LayerPayload load(MetalBackend &backend,const LayerMetadata &meta) {
    LayerPayload result;result.metadata=meta;result.plannedBytes=meta.originalBytes+meta.centeredBytes;
    const uint64_t before=backend.memoryStats().allocatedBytes;
    for(uint32_t field=0;field<9;++field) {
      const auto &record=meta.tensors[field];require(record.bytes<=backend.capabilities().maxBufferLengthBytes,"one-layer tensor exceeds buffer limit");
      auto mapping=std::make_shared<TensorMapping>(meta.root/record.path,record.fileBytes,record.offset,record.bytes);
      auto &tensor=result.tensors[field];tensor.dtype=record.dtype;tensor.shape=record.shape;tensor.logicalBytes=record.bytes;
      result.sourceHashes[field]=hash(mapping->data(),record.bytes);
      if(field%3) {
        const auto *values=static_cast<const uint16_t *>(mapping->data());
        for(uint64_t i=0;i<record.bytes/2;++i) {
          const float value=std::bit_cast<float>(uint32_t(values[i])<<16);
          require(std::isfinite(value)&&(field%3==2||value>0),"original scale/bias nonfinite or scale nonpositive");
        }
      }
      tensor.buffer=backend.wrapSharedMemory(mapping->data(),record.bytes,mapping,"readonly exact one-layer originalQ4 tensor");
      result.mappings[field]=mapping;
    }
    for(uint32_t plane=0;plane<3;++plane) {
      const uint32_t n=plane==2?2560:640,sourceStride=plane==2?320:1280,centerStride=plane==2?384:1280;
      auto &centered=result.centered[plane];centered=backend.allocateBuffer(uint64_t(512)*n*centerStride,metal::BufferStorage::Shared,"temporary one-layer centeredI4 plane");
      auto *to=static_cast<uint8_t *>(centered.contents());const auto *from=static_cast<const uint8_t *>(result.tensors[plane*3].buffer.contents());
      // Signed zero has nibble0. Padded tail channels are unused by logicalK640.
      std::memset(to,0,centered.sizeBytes());
      for(uint64_t row=0;row<uint64_t(512)*n;++row)for(uint32_t byte=0;byte<sourceStride;++byte) {
        const uint8_t original=from[row*sourceStride+byte];to[row*centerStride+byte]=original^0x88;
        require((to[row*centerStride+byte]^0x88)==original,"exact centered nibble inverse failed");result.exactCodeBytesCertified++;
      }
      result.centeredHashes[plane]=hash(centered);
    }
    result.allocatedBytes=metal::allocationDelta(before,backend.memoryStats().allocatedBytes);
    require(result.allocatedBytes<=result.plannedBytes,"one-layer source/centered load exceeds reservation");
    return result;
  }
  flash::FlashAffineProjection projection(uint32_t plane)const {
    require(plane<3,"bad plane");const uint32_t n=plane==2?2560:640,k=plane==2?640:2560;
    return{&tensors[plane*3],&tensors[plane*3+1],&tensors[plane*3+2],512,n,k,4,64,k/2,uint64_t(n)*k/2,k/32,uint64_t(n)*k/32};
  }
  void unchanged()const {
    for(uint32_t field=0;field<9;++field){mappings[field]->unchanged();require(hash(tensors[field].buffer)==sourceHashes[field],"original selected tensor bytes changed");}
    for(uint32_t plane=0;plane<3;++plane)require(hash(centered[plane])==centeredHashes[plane],"centered temporary codes changed");
  }
};
} // namespace splash::bench::w4a8
