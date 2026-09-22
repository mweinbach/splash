#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
namespace guard_metadata {
struct Counters{uint64_t contents=0,sizes=0;int firstFailure=-1,currentImmutable=-1;};
extern thread_local Counters counters;
extern thread_local bool countCalls;
namespace metal {
enum class BufferStorage:uint8_t{Shared,Private};
struct MetalBuffer{
  uintptr_t address=0;uint64_t length=0;bool present=true,shared=true;int immutableOrdinal=-1;
  explicit operator bool()const noexcept;
  BufferStorage storage()const noexcept;
  void *contents()const noexcept;
  uint64_t sizeBytes()const noexcept;
};
}
enum class FlashMoEBlockedTile:uint32_t{M16N64=16,M32N64=32,M64N64=64};
struct Buckets{
  metal::MetalBuffer counts,offsets,routeMap,canonicalToPacked,packedInputs,jobOffsets,jobCount,tileJobs;
  uint32_t rowCapacity=4,selectionCapacity=10,routeCapacity=40,jobCapacity=514;
};
struct FlashMoEBlockedScratch{Buckets buckets;metal::MetalBuffer packedActivated,scatteredDown;};
extern bool directA,wide;
bool flashMoEDirectAEnabled() noexcept;
FlashMoEBlockedTile flashMoEBlockedTile(uint32_t,bool) noexcept;
uint32_t moEBucketJobCapacity(uint32_t,uint32_t,uint32_t);
void requireBytes(const metal::MetalBuffer&,uint64_t);
void disjoint(const metal::MetalBuffer&,const metal::MetalBuffer&);
void allRowsScratch(const FlashMoEBlockedScratch&,metal::MetalBuffer,uint32_t,FlashMoEBlockedTile,uint32_t);
struct Original{
  struct Layer{metal::MetalBuffer base,ranks;};
  std::array<Layer,48> layers;
  void immutableDisjoint(const metal::MetalBuffer&)const;
};
struct Outcome{bool accepted=false;int category=0,firstFailure=-1;const char*message="";};
} // namespace guard_metadata
