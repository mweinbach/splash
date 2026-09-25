// Warm (just-read) vs cold re-read bandwidth by working-set size: does the SLC
// retain a layer's weights between dependent dispatches?
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>

static const char *kSource = R"(
#include <metal_stdlib>
using namespace metal;
kernel void rd(const device uint4 *src [[buffer(0)]], device uint *out [[buffer(1)]],
               constant uint &count [[buffer(2)]], uint gid [[thread_position_in_grid]],
               uint grid [[threads_per_grid]]) {
  uint4 acc = 0;
  for (uint i = gid; i < count; i += grid) acc ^= src[i];
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) out[0] = 1;
}
)";

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(kSource) options:nil error:&err];
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rd"] error:&err];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    const uint64_t total = 4ull << 30;
    id<MTLBuffer> src = [dev newBufferWithLength:total options:MTLResourceStorageModePrivate];
    id<MTLBuffer> out = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
      [b fillBuffer:src range:NSMakeRange(0, total) value:7];
      [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
    }
    uint64_t cursor = 0;
    for (uint64_t mb : {2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256}) {
      const uint64_t sz = mb << 20;
      const uint count = uint(sz / 16);
      double cold = 1e9, warm = 1e9;
      for (int t = 0; t < 5; ++t) {
        cursor = (cursor + sz + (64ull << 20)) % (total - sz);
        cursor &= ~uint64_t(16383);
        double times[2];
        for (int pass = 0; pass < 2; ++pass) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
          [e setComputePipelineState:p];
          [e setBuffer:src offset:cursor atIndex:0];
          [e setBuffer:out offset:0 atIndex:1];
          [e setBytes:&count length:4 atIndex:2];
          [e dispatchThreadgroups:MTLSizeMake(640, 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
          times[pass] = cb.GPUEndTime - cb.GPUStartTime;
        }
        cold = std::min(cold, times[0]);
        warm = std::min(warm, times[1]);
      }
      printf("%4llu MB  cold %7.1f us %7.1f GB/s   warm %7.1f us %7.1f GB/s\n", mb, cold * 1e6, sz / cold / 1e9,
             warm * 1e6, sz / warm / 1e9);
    }
  }
  return 0;
}
