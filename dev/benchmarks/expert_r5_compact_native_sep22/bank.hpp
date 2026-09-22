#pragma once
// Root-only bounded real coefficient slices; CPU self-test never calls this.
#include <sys/stat.h>
#include <unistd.h>
namespace {
constexpr uint64_t R5ReservationBytes=384ULL<<20,R5SelectedBytes=246528000;
struct R5Resources {
  bool gpu=false,stopped=false,destroyed=false,pass=false;
  uint64_t initialOwned=0,initialDevice=0,ownedPeak=0,devicePeak=0,deviceCurrent=0,finalOwned=UINT64_MAX,
      denied=UINT64_MAX,reserved=UINT64_MAX,payloadRead=0;
  bool hostValid=false,growth=false;
  std::array<uint32_t,50> selected{};
  std::filesystem::path report,source;struct stat sourceStat{};
} r5Resources;
bool sameSource(const struct stat&a,const struct stat&b) {
  return S_ISREG(b.st_mode)&&!(b.st_mode&0222)&&a.st_dev==b.st_dev&&a.st_ino==b.st_ino&&a.st_size==b.st_size&&
    a.st_mtimespec.tv_sec==b.st_mtimespec.tv_sec&&a.st_mtimespec.tv_nsec==b.st_mtimespec.tv_nsec&&
    a.st_ctimespec.tv_sec==b.st_ctimespec.tv_sec&&a.st_ctimespec.tv_nsec==b.st_ctimespec.tv_nsec;
}
void sourceUnchanged() {
  const int fd=::open(r5Resources.source.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
  require(fd>=0,"final selected-source open");struct stat current{};const bool valid=!::fstat(fd,&current)&&sameSource(r5Resources.sourceStat,current);
  ::close(fd);require(valid,"selected coefficient source changed");
}
void boundedRanks(const one::OneLayerPayload&layer) {
  require(layer.ranks&&layer.ranks.storage()==BufferStorage::Shared&&layer.ranks.contents()&&layer.ranks.sizeBytes()==512*4&&reinterpret_cast<uintptr_t>(layer.ranks.contents())%4==0,"exact aligned compressed rank view");
  require(layer.base&&layer.base.storage()==BufferStorage::Shared&&layer.base.contents()&&reinterpret_cast<uintptr_t>(layer.base.contents())%8==0,"owned aligned compressed coefficient base");
  uint64_t offset=0;
  for(uint32_t plane=0;plane<3;++plane)for(uint32_t kind=0;kind<2;++kind) {
    offset=(offset+16383)&~uint64_t{16383};const uint64_t n=plane==2?2560:640,k=plane==2?640:2560,length=50*n*(kind?4:k);
    const auto view=kind?layer.scales[plane]:layer.codes[plane];
    require(view&&view.storage()==BufferStorage::Shared&&view.sizeBytes()==length&&view.contents()==static_cast<uint8_t*>(layer.base.contents())+offset&&offset+length<=layer.base.sizeBytes(),"exact source coefficient view ownership/extent");offset+=length;
  }
  require(offset==layer.base.sizeBytes(),"exact source bank physical view");
  const auto*r=static_cast<const uint32_t*>(layer.ranks.contents());
  for(uint32_t e=0;e<512;++e) {
    const bool selected=std::find(r5Resources.selected.begin(),r5Resources.selected.end(),e)!=r5Resources.selected.end();
    require(selected?(r[e]<50||r[e]>=512):r[e]==UINT32_MAX,"compressed bank rejects finite ranks50..511 before GPU");
  }
}
template<class Allocate>
one::OneLayerPayload selectedBank(MetalBackend&backend,const FlashInt8ExpertStoreLayer&entry,std::span<const int64_t>ids,Allocate allocate) {
  one::detail::validateEntry(entry);r5Resources.source=entry.path;
  std::array<bool,512> include{};include[7]=true;
  for(const auto id:ids){require(id>=0&&id<512,"healthy selected bank source IDs");include[size_t(id)]=true;}
  uint32_t count=0;for(const bool value:include)count+=value;require(count<=50,"source union exceeds50");
  for(uint32_t id=0;count<50&&id<512;++id)if(!include[id]){include[id]=true;++count;}
  count=0;for(uint32_t id=0;id<512;++id)if(include[id])r5Resources.selected[count++]=id;
  const int fd=::open(entry.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);require(fd>=0,"Root readonly selected coefficient open");
  one::OneLayerPayload layer;
  try {
    require(!::fstat(fd,&r5Resources.sourceStat)&&S_ISREG(r5Resources.sourceStat.st_mode)&&!(r5Resources.sourceStat.st_mode&0222)&&
        uint64_t(r5Resources.sourceStat.st_size)==entry.bytes,"canonical Full512 source stat");
    std::array<std::array<uint64_t,2>,3>offsets{};uint64_t total=0;
    for(uint32_t plane=0;plane<3;++plane)for(uint32_t kind=0;kind<2;++kind) {
      total=(total+16383)&~uint64_t{16383};offsets[plane][kind]=total;
      const uint64_t n=plane==2?2560:640,k=plane==2?640:2560;total+=50*n*(kind?4:k);
    }
    layer.base=allocate(total);layer.ranks=allocate(512*4);std::memset(layer.ranks.contents(),0xff,layer.ranks.sizeBytes());
    for(uint32_t e=0;e<50;++e)static_cast<uint32_t*>(layer.ranks.contents())[r5Resources.selected[e]]=e;
    for(uint32_t plane=0;plane<3;++plane) {
      const uint64_t n=plane==2?2560:640,k=plane==2?640:2560;
      layer.codes[plane]=backend.view(layer.base,offsets[plane][0],50*n*k);
      layer.scales[plane]=backend.view(layer.base,offsets[plane][1],50*n*4);
      for(uint32_t kind=0;kind<2;++kind)for(uint32_t e=0;e<50;++e) {
        const uint64_t length=n*(kind?4:k),sourceOffset=(kind?entry.scales[plane].offset:entry.codes[plane].offset)+uint64_t(r5Resources.selected[e])*length;
        auto*destination=static_cast<uint8_t*>((kind?layer.scales[plane]:layer.codes[plane]).contents())+e*length;
        uint64_t done=0;while(done<length){const auto count=::pread(fd,destination+done,length-done,off_t(sourceOffset+done));require(count>0,"bounded selected coefficient pread");done+=uint64_t(count);}
        r5Resources.payloadRead+=length;
        if(kind)for(uint64_t column=0;column<n;++column){float value;std::memcpy(&value,destination+column*4,4);require(value>0&&std::isfinite(value),"positive finite row scale");}
        else require(std::find(destination,destination+length,uint8_t{128})==destination+length,"symmetric signed code excludes -128");
      }
    }
    struct stat after{};require(!::fstat(fd,&after)&&sameSource(r5Resources.sourceStat,after),"source unchanged during bounded reads");
  }catch(...){::close(fd);throw;}
  ::close(fd);require(r5Resources.payloadRead==R5SelectedBytes,"exact50-expert selected read bound");boundedRanks(layer);return layer;
}
struct R5BackendLifetime {~R5BackendLifetime(){if(r5Resources.gpu)r5Resources.destroyed=true;}};
struct R5FinalLedger {
  MetalBackend&backend;splash::engine::MemoryGovernor&governor;
  ~R5FinalLedger() {
    const auto s=backend.memoryStats();const auto g=governor.snapshot();
    r5Resources.finalOwned=s.allocatedBytes>=r5Resources.initialOwned?s.allocatedBytes-r5Resources.initialOwned:UINT64_MAX;
    r5Resources.ownedPeak=s.peakAllocatedBytes>=r5Resources.initialOwned?s.peakAllocatedBytes-r5Resources.initialOwned:UINT64_MAX;
    r5Resources.deviceCurrent=s.deviceCurrentAllocatedBytes>=r5Resources.initialDevice?s.deviceCurrentAllocatedBytes-r5Resources.initialDevice:UINT64_MAX;
    r5Resources.devicePeak=s.devicePeakAllocatedBytes>=r5Resources.initialDevice?s.devicePeakAllocatedBytes-r5Resources.initialDevice:UINT64_MAX;
    r5Resources.denied=g.deniedReservations;r5Resources.reserved=g.reservedBytes;r5Resources.hostValid=g.hostMeasurementValid;r5Resources.growth=g.growthAllowed;
    r5Resources.pass=!r5Resources.finalOwned&&r5Resources.ownedPeak<=R5ReservationBytes&&r5Resources.deviceCurrent<=R5ReservationBytes&&
      r5Resources.devicePeak<=R5ReservationBytes&&!g.reservedBytes&&!g.deniedReservations&&g.hostMeasurementValid&&g.growthAllowed&&
      g.pressure==splash::engine::MemoryPressure::Normal&&g.systemPressure==splash::engine::MemoryPressure::Normal&&g.hostAvailableBytes>g.hostReserveBytes&&
      !s.sparseVirtualBytes&&!s.sparseResidentBytes&&!s.peakSparseResidentBytes&&backend.healthy()&&r5Resources.stopped;
  }
};
}
