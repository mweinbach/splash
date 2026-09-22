#pragma once
#include "policy.hpp"
#include "metal/MetalBackend.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace raw_large_R4_guard_pair_sep22 {
inline std::string hash(const void*address,uint64_t bytes) {
  CC_SHA256_CTX context{};
  if(!CC_SHA256_Init(&context))throw std::runtime_error("R4 SHA init");
  const auto*p=static_cast<const uint8_t*>(address);
  while(bytes) {
    const CC_LONG count=CC_LONG(std::min<uint64_t>(bytes,1ULL<<30));
    if(!CC_SHA256_Update(&context,p,count))throw std::runtime_error("R4 SHA update");
    p+=count;bytes-=count;
  }
  std::array<uint8_t,CC_SHA256_DIGEST_LENGTH>out{};
  if(!CC_SHA256_Final(out.data(),&context))throw std::runtime_error("R4 SHA final");
  constexpr char digits[]="0123456789abcdef";std::string result;
  for(uint8_t byte:out){result+=digits[byte>>4];result+=digits[byte&15];}
  return result;
}
struct Guarded {
  splash::metal::MetalBuffer base,view;
  uint64_t logical=0,offset=64;
  static Guarded allocate(splash::metal::MetalBackend&backend,uint64_t bytes,
      const char*label,uint64_t extraOffset=0) {
    if(!bytes||extraOffset>16)throw std::invalid_argument("R4 guarded allocation extent");
    Guarded result;result.logical=bytes;result.offset=64+extraOffset;
    result.base=backend.allocateBuffer(rounded(add(bytes,144)),
      splash::metal::BufferStorage::Shared,label);
    result.view=backend.view(result.base,result.offset,bytes);
    std::memset(result.base.contents(),0x5a,result.base.sizeBytes());
    std::memset(result.view.contents(),0xa5,bytes);
    return result;
  }
  void poison()const{std::memset(view.contents(),0xa5,logical);}
  bool clean()const {
    if(!base.contents()||base.sizeBytes()<offset+logical+64)return false;
    const auto*p=static_cast<const uint8_t*>(base.contents());
    return std::all_of(p,p+offset,[](uint8_t x){return x==0x5a;})&&
      std::all_of(p+offset+logical,p+base.sizeBytes(),[](uint8_t x){return x==0x5a;});
  }
};
struct RawSpan {
  struct File {
    int fd=-1;struct stat initial{};
    explicit File(const std::string&name) {
      fd=::open(name.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
      if(fd<0||::fstat(fd,&initial)||!S_ISREG(initial.st_mode)||initial.st_size<0) {
        if(fd>=0)::close(fd);fd=-1;
        throw std::invalid_argument("R4 selected native file invalid");
      }
    }
    ~File(){if(fd>=0)::close(fd);}
    void unchanged()const {
      struct stat now{};
      if(::fstat(fd,&now)||now.st_dev!=initial.st_dev||now.st_ino!=initial.st_ino||
          now.st_size!=initial.st_size||
          now.st_mtimespec.tv_sec!=initial.st_mtimespec.tv_sec||
          now.st_mtimespec.tv_nsec!=initial.st_mtimespec.tv_nsec||
          now.st_ctimespec.tv_sec!=initial.st_ctimespec.tv_sec||
          now.st_ctimespec.tv_nsec!=initial.st_ctimespec.tv_nsec)
        throw std::runtime_error("R4 immutable selected file changed");
    }
  };
  std::shared_ptr<File>file;Guarded data;std::string sha;
  static RawSpan load(splash::metal::MetalBackend&backend,const std::string&name,
      uint64_t offset,uint64_t bytes) {
    if(!bytes||bytes>selectedSourceLimit)throw std::invalid_argument("R4 bounded pread span");
    RawSpan result;result.file=std::make_shared<File>(name);
    const uint64_t size=uint64_t(result.file->initial.st_size);
    if(offset>size||bytes>size-offset||offset>INT64_MAX||bytes>INT64_MAX-offset)
      throw std::invalid_argument("R4 selected pread range outside file");
    result.data=Guarded::allocate(backend,bytes,"R4 bounded immutable native span");
    uint64_t done=0;
    while(done<bytes) {
      const ssize_t got=::pread(result.file->fd,
        static_cast<uint8_t*>(result.data.view.contents())+done,
        size_t(std::min<uint64_t>(bytes-done,1ULL<<20)),off_t(offset+done));
      if(got<=0)throw std::runtime_error("R4 selected bounded pread failed");
      done+=uint64_t(got);
    }
    result.file->unchanged();
    result.sha=hash(result.data.view.contents(),bytes);
    return result;
  }
  bool immutable()const {
    file->unchanged();return data.clean()&&hash(data.view.contents(),data.logical)==sha;
  }
};
} // namespace raw_large_R4_guard_pair_sep22
