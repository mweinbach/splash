#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <string>

int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"mpp_rate_tg.metallib"] error:&err];
    if (!L) { printf("lib %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> out = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
    id<MTLBuffer> a = [dev newBufferWithLength:1024 * 64 * 2 options:MTLResourceStorageModeShared];
    memset(a.contents, 0, a.length);
    for (NSString *name in [L.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
      int M, N, K, SG, COOP, ATG;
      sscanf(name.UTF8String, "tg_%dx%dx%d_sg%d_coop%d_a%d", &M, &N, &K, &SG, &COOP, &ATG);
      id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:name] error:&err];
      if (!p) { printf("pso %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); continue; }
      const uint32_t iters = 3000, groups = 80 * 4;
      double best = 1e9;
      for (int t = 0; t < 4; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e setBuffer:a offset:0 atIndex:2];
        [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(SG * 32, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (t) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
      }
      const double flops = 2.0 * M * N * K * double(iters) * groups * (SG / COOP);
      printf("%-30s %7.1f TFLOPS\n", name.UTF8String, flops / best / 1e12);
    }
  }
  return 0;
}
