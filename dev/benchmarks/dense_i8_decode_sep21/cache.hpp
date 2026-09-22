#pragma once
// Root runtime only: fit original F32 coefficient rows to signed-I8+F32 scale.
// BF16 activations and the mathematical F32 source remain immutable.
#include "metal/MetalBackend.hpp"
#include "dev/benchmarks/dense_i8_decode_sep21/precision.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dense_i8_decode_sep21 {
inline constexpr uint64_t kAlignment=16384,kGuardBytes=64;
inline uint64_t rounded(uint64_t value) {
  if (value>std::numeric_limits<uint64_t>::max()-kAlignment-kGuardBytes)
    throw std::invalid_argument("dense I8 allocation extent overflows");
  return (value+kAlignment-1)&~(kAlignment-1);
}
inline uint64_t guardedBytes(uint64_t value) {
  if(value>std::numeric_limits<uint64_t>::max()-kGuardBytes)throw std::invalid_argument("dense guard extent overflows");
  return rounded(value+kGuardBytes);
}
inline std::string hash(const void *address,uint64_t bytes) {
  CC_SHA256_CTX context{};
  if (!CC_SHA256_Init(&context)) throw std::runtime_error("dense I8 SHA initialization failed");
  const auto *cursor=static_cast<const uint8_t *>(address);
  while (bytes) {
    const CC_LONG step=CC_LONG(std::min<uint64_t>(bytes,1ULL<<30));
    if (!CC_SHA256_Update(&context,cursor,step)) throw std::runtime_error("dense I8 SHA update failed");
    cursor+=step;bytes-=step;
  }
  std::array<uint8_t,CC_SHA256_DIGEST_LENGTH> digest{};
  if (!CC_SHA256_Final(digest.data(),&context)) throw std::runtime_error("dense I8 SHA finalization failed");
  constexpr char hex[]="0123456789abcdef";std::string result;
  for(uint8_t value:digest){result+=hex[value>>4];result+=hex[value&15];}return result;
}
class ReadonlyMapping final {
 public:
  ReadonlyMapping(const std::string &path,uint64_t bytes):bytes_(bytes) {
    if (!bytes || bytes>std::numeric_limits<size_t>::max()) throw std::invalid_argument("dense source mapping extent invalid");
    fd_=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    if (fd_<0 || ::fstat(fd_,&before_) || !S_ISREG(before_.st_mode) ||
        before_.st_size<=0 || uint64_t(before_.st_size)!=bytes) {
      if(fd_>=0)::close(fd_);fd_=-1;throw std::invalid_argument("dense source payload extent/type changed");
    }
    address_=::mmap(nullptr,bytes,PROT_READ,MAP_SHARED,fd_,0);
    if(address_==MAP_FAILED){address_=nullptr;::close(fd_);fd_=-1;throw std::runtime_error("dense readonly source mapping failed");}
  }
  ~ReadonlyMapping(){if(address_)::munmap(address_,bytes_);if(fd_>=0)::close(fd_);}
  ReadonlyMapping(const ReadonlyMapping&)=delete;
  ReadonlyMapping& operator=(const ReadonlyMapping&)=delete;
  void *address()const{return address_;}
  void requireUnchanged()const {
    struct stat after{};
    if(::fstat(fd_,&after) || after.st_dev!=before_.st_dev || after.st_ino!=before_.st_ino ||
        after.st_size!=before_.st_size || after.st_mtimespec.tv_sec!=before_.st_mtimespec.tv_sec ||
        after.st_mtimespec.tv_nsec!=before_.st_mtimespec.tv_nsec ||
        after.st_ctimespec.tv_sec!=before_.st_ctimespec.tv_sec || after.st_ctimespec.tv_nsec!=before_.st_ctimespec.tv_nsec)
      throw std::runtime_error("dense readonly source/capture changed");
  }
 private:
  int fd_=-1;void *address_=nullptr;uint64_t bytes_;struct stat before_{};
};
struct Guarded final {
  splash::metal::MetalBuffer allocation,view;uint64_t logical=0;
  static Guarded allocate(splash::metal::MetalBackend &backend,uint64_t logical,const char *label) {
    Guarded result;result.logical=logical;
    result.allocation=backend.allocateBuffer(guardedBytes(logical),splash::metal::BufferStorage::Shared,label);
    result.view=backend.view(result.allocation,0,logical);
    std::memset(result.allocation.contents(),0xa5,logical);
    std::memset(static_cast<uint8_t *>(result.allocation.contents())+logical,0x5a,result.allocation.sizeBytes()-logical);
    return result;
  }
  bool clean()const {
    if(!allocation.contents() || allocation.sizeBytes()<logical+kGuardBytes)return false;
    const auto *begin=static_cast<const uint8_t *>(allocation.contents())+logical;
    return std::all_of(begin,begin+allocation.sizeBytes()-logical,[](uint8_t value){return value==0x5a;});
  }
};
struct RawSpan final {
  struct File final {
    int fd=-1;struct stat before{};
    explicit File(const std::string &path) {
      fd=::open(path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
      if(fd<0 || ::fstat(fd,&before) || !S_ISREG(before.st_mode)) {
        if(fd>=0)::close(fd);fd=-1;throw std::invalid_argument("dense selected raw source file invalid");
      }
    }
    ~File(){if(fd>=0)::close(fd);}
    void unchanged()const {
      struct stat after{};
      if(::fstat(fd,&after) || after.st_dev!=before.st_dev || after.st_ino!=before.st_ino ||
          after.st_size!=before.st_size || after.st_mtimespec.tv_sec!=before.st_mtimespec.tv_sec ||
          after.st_mtimespec.tv_nsec!=before.st_mtimespec.tv_nsec ||
          after.st_ctimespec.tv_sec!=before.st_ctimespec.tv_sec || after.st_ctimespec.tv_nsec!=before.st_ctimespec.tv_nsec)
        throw std::runtime_error("dense selected raw source file changed");
    }
  };
  std::shared_ptr<File> file;Guarded storage;std::string initialSHA;
  static RawSpan load(splash::metal::MetalBackend &backend,const std::string &path,uint64_t offset,uint64_t length) {
    RawSpan result;result.file=std::make_shared<File>(path);
    if(!length || offset>uint64_t(result.file->before.st_size) || length>uint64_t(result.file->before.st_size)-offset ||
        offset>uint64_t(std::numeric_limits<off_t>::max()) || length>uint64_t(std::numeric_limits<ssize_t>::max()))
      throw std::invalid_argument("dense selected raw tensor span exceeds source extent");
    result.storage=Guarded::allocate(backend,length,"private dense selected raw affine tensor span");
    auto *destination=static_cast<uint8_t *>(result.storage.view.contents());uint64_t done=0;
    while(done<length) {
      const size_t count=size_t(std::min<uint64_t>(length-done,64ULL<<20));
      const ssize_t read=::pread(result.file->fd,destination+done,count,off_t(offset+done));
      if(read<=0)throw std::runtime_error("dense selected raw tensor span read failed");done+=uint64_t(read);
    }
    result.file->unchanged();result.initialSHA=hash(destination,length);return result;
  }
  bool immutable()const {
    file->unchanged();return storage.clean() && hash(storage.view.contents(),storage.logical)==initialSHA;
  }
};
inline int roundEven(float value) {
  if (value>=127)return 127;if(value<=-127)return -127;
  if(!std::isfinite(value))throw std::invalid_argument("dense coefficient quantization ratio nonfinite");
  const double magnitude=std::abs(double(value)),base=std::floor(magnitude),fraction=magnitude-base;
  const int integral=int(base),roundedValue=integral+int(fraction>.5 || (fraction==.5 && (integral&1)));
  return std::signbit(value) ? -roundedValue : roundedValue;
}
struct CoefficientCache final {
  Guarded codes,scales;uint32_t outputs=0,inputs=0;
  static uint64_t plannedBytes(uint32_t outputs,uint32_t inputs) {
    if(!outputs || outputs>32768 || outputs%64 || !inputs || inputs>32768 || inputs%32)
      throw std::invalid_argument("dense I8 cache role geometry invalid");
    return guardedBytes(uint64_t(outputs)*inputs)+guardedBytes(uint64_t(outputs)*4);
  }
  static CoefficientCache fit(splash::metal::MetalBackend &backend,const float *source,
      uint32_t outputs,uint32_t inputs) {
    (void)plannedBytes(outputs,inputs);if(!source)throw std::invalid_argument("dense I8 cache has no original F32 source");
    CoefficientCache result;result.outputs=outputs;result.inputs=inputs;
    result.codes=Guarded::allocate(backend,uint64_t(outputs)*inputs,"private dense fitted I8 coefficient codes");
    result.scales=Guarded::allocate(backend,uint64_t(outputs)*4,"private dense fitted late F32 coefficient row scales");
    auto *codes=static_cast<int8_t *>(result.codes.view.contents());auto *scales=static_cast<float *>(result.scales.view.contents());
    for(uint32_t row=0;row<outputs;++row) {
      float maximum=0;
      for(uint32_t k=0;k<inputs;++k) {
        const float value=source[uint64_t(row)*inputs+k];
        if(!std::isfinite(value))throw std::invalid_argument("dense original F32 coefficient is nonfinite");
        maximum=std::max(maximum,std::abs(value));
      }
      const float scale=splash::flash::dense_i8_decode::precision::rowScale(maximum);
      if(!(scale>0) || !std::isfinite(scale))throw std::invalid_argument("dense coefficient F32 maxabs/127 scale invalid");
      scales[row]=scale;
      for(uint32_t k=0;k<inputs;++k)codes[uint64_t(row)*inputs+k]=int8_t(roundEven(source[uint64_t(row)*inputs+k]/scale));
    }
    return result;
  }
  bool guardsClean()const{return codes.clean() && scales.clean();}
};
inline void cpuSelfTest() {
  if(roundEven(.5f)!=0 || roundEven(1.5f)!=2 || roundEven(-.5f)!=0 || roundEven(-1.5f)!=-2 ||
      roundEven(2.5f)!=2 || roundEven(3.5f)!=4 || roundEven(200)!=127 || roundEven(-200)!=-127)
    throw std::logic_error("dense I8 coefficient CPU RNE tie/clamp golden differs");
  if(CoefficientCache::plannedBytes(2560,6144)!=guardedBytes(2560ULL*6144)+guardedBytes(2560ULL*4))
    throw std::logic_error("dense I8 governed cache accounting differs");
}
} // namespace dense_i8_decode_sep21
