#pragma once
// These routines execute only in the explicit Root GPU branch of oracle.mm.
#include "metal/MetalBackend.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace hc_r1_chronological_terms_sep22 {
inline std::string hash(const void *address,uint64_t bytes) {
  CC_SHA256_CTX c{};if(!CC_SHA256_Init(&c))throw std::runtime_error("HCchronology SHA init");
  auto *p=static_cast<const uint8_t*>(address);
  while(bytes){const CC_LONG n=CC_LONG(std::min<uint64_t>(bytes,1ULL<<30));if(!CC_SHA256_Update(&c,p,n))throw std::runtime_error("HCchronology SHA update");p+=n;bytes-=n;}
  std::array<uint8_t,CC_SHA256_DIGEST_LENGTH>d{};if(!CC_SHA256_Final(d.data(),&c))throw std::runtime_error("HCchronology SHA final");
  constexpr char h[]="0123456789abcdef";std::string s;for(auto b:d){s+=h[b>>4];s+=h[b&15];}return s;
}
inline uint64_t rounded(uint64_t n) {
  if(n>UINT64_MAX-16383)throw std::invalid_argument("HCchronology rounded extent overflow");
  return (n+16383)&~uint64_t{16383};
}
struct Guarded {
  splash::metal::MetalBuffer base,view;uint64_t logical=0,offset=64;
  static Guarded allocate(splash::metal::MetalBackend &b,uint64_t n,const char *label,uint64_t extraOffset=0) {
    if(!n||extraOffset>16||n>UINT64_MAX-144)throw std::invalid_argument("HCchronology guarded extent invalid");
    Guarded g;g.logical=n;g.offset=64+extraOffset;g.base=b.allocateBuffer(rounded(n+144),splash::metal::BufferStorage::Shared,label);
    g.view=b.view(g.base,g.offset,n);std::memset(g.base.contents(),0x5a,g.base.sizeBytes());std::memset(g.view.contents(),0xa5,n);return g;
  }
  bool clean()const {
    if(!base.contents()||base.sizeBytes()<offset+logical+64)return false;
    const auto*p=static_cast<const uint8_t*>(base.contents());
    return std::all_of(p,p+offset,[](uint8_t x){return x==0x5a;})&&
      std::all_of(p+offset+logical,p+base.sizeBytes(),[](uint8_t x){return x==0x5a;});
  }
  void poison()const{std::memset(view.contents(),0xa5,logical);}
};
struct RawSpan {
  struct File {
    int fd=-1;struct stat initial{};
    explicit File(const std::string &name){fd=::open(name.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);if(fd<0||::fstat(fd,&initial)||!S_ISREG(initial.st_mode)||initial.st_size<0){if(fd>=0)::close(fd);throw std::invalid_argument("HCchronology selected native source file invalid");}}
    ~File(){if(fd>=0)::close(fd);}
    void unchanged()const {struct stat s{};if(::fstat(fd,&s)||s.st_dev!=initial.st_dev||s.st_ino!=initial.st_ino||s.st_size!=initial.st_size||s.st_mtimespec.tv_sec!=initial.st_mtimespec.tv_sec||s.st_mtimespec.tv_nsec!=initial.st_mtimespec.tv_nsec||s.st_ctimespec.tv_sec!=initial.st_ctimespec.tv_sec||s.st_ctimespec.tv_nsec!=initial.st_ctimespec.tv_nsec)throw std::runtime_error("HCchronology selected native source changed");}
  };
  std::shared_ptr<File> file;Guarded data;std::string sha;
  static RawSpan load(splash::metal::MetalBackend &b,const std::string &name,uint64_t off,uint64_t n) {
    RawSpan r;r.file=std::make_shared<File>(name);
    if(!n||off>uint64_t(r.file->initial.st_size)||n>uint64_t(r.file->initial.st_size)-off||off>uint64_t(INT64_MAX)||n>uint64_t(INT64_MAX)-off)throw std::invalid_argument("HCchronology selected source span out of file");
    r.data=Guarded::allocate(b,n,"HCchronology selected native immutable Q4/BF16 span");uint64_t done=0;
    while(done<n){const ssize_t got=::pread(r.file->fd,static_cast<uint8_t*>(r.data.view.contents())+done,size_t(std::min<uint64_t>(n-done,64ULL<<20)),off_t(off+done));if(got<=0)throw std::runtime_error("HCchronology selected source read failed");done+=uint64_t(got);}
    r.file->unchanged();r.sha=hash(r.data.view.contents(),n);return r;
  }
  bool immutable()const{file->unchanged();return data.clean()&&hash(data.view.contents(),data.logical)==sha;}
};
} // namespace hc_r1_chronological_terms_sep22
