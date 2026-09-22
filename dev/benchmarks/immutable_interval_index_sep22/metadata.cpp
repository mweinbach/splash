#include "metadata.hpp"
namespace guard_metadata {
thread_local Counters counters{};
thread_local bool countCalls=false;
bool directA=true,wide=true;
bool flashMoEDirectAEnabled()noexcept{return directA;}
FlashMoEBlockedTile flashMoEBlockedTile(uint32_t,bool)noexcept{return wide?FlashMoEBlockedTile::M64N64:FlashMoEBlockedTile::M32N64;}
namespace metal {
__attribute__((noinline)) MetalBuffer::operator bool()const noexcept{return present;}
__attribute__((noinline)) BufferStorage MetalBuffer::storage()const noexcept{return shared?BufferStorage::Shared:BufferStorage::Private;}
__attribute__((noinline)) void*MetalBuffer::contents()const noexcept{
  if(countCalls){++counters.contents;counters.currentImmutable=immutableOrdinal;}
  return shared?reinterpret_cast<void*>(address):nullptr;
}
__attribute__((noinline)) uint64_t MetalBuffer::sizeBytes()const noexcept{if(countCalls)++counters.sizes;return length;}
}
} // namespace guard_metadata
