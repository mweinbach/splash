#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <string>

int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"mpp_rate.metallib"] error:&err];
    if (!L) { printf("lib %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> out = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bsrc = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
    id<MTLBuffer> asrc = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
    memset(bsrc.contents, 0x11, 1 << 20);
    memset(asrc.contents, 0, 1 << 20);
    const char *filter = argc > 1 ? argv[1] : "";
    for (NSString *name in [L.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
      std::string n = name.UTF8String;
      if (*filter && n.find(filter) == std::string::npos) continue;
      int M = 0, N = 0, K = 0;
      sscanf(n.c_str() + n.rfind('_') + 1, "%dx%dx%d", &M, &N, &K);
      id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:name] error:&err];
      if (!p) { printf("pso fail %s: %s\n", n.c_str(), err.localizedDescription.UTF8String); continue; }
      const uint32_t iters = 4000, groups = 80 * 8;
      double best = 1e9;
      for (int t = 0; t < 3; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:out offset:0 atIndex:0];
        [e setBytes:&iters length:4 atIndex:1];
        [e setBuffer:bsrc offset:0 atIndex:2];
        [e setBuffer:asrc offset:0 atIndex:3];
        [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
      }
      const double flops = 2.0 * M * N * K * double(iters) * groups * 4;
      printf("%-30s %7.1f TFLOPS\n", n.c_str(), flops / best / 1e12);
    }
  }
  return 0;
}
