#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <algorithm>
#include <string>
struct P { uint32_t K, N, rows, xs, ys, row0; uint64_t a, b; };
int main(int argc, char **argv) {
  @autoreleasepool {
    const uint32_t K = atoi(argv[1]), N = atoi(argv[2]), rows = atoi(argv[3]);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"tallh.metallib"] error:&err];
    id<MTLBuffer> X = [dev newBufferWithLength:size_t(rows) * K * 2 * 8 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> W = [dev newBufferWithLength:size_t(64) * K * 2 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> Y = [dev newBufferWithLength:size_t(rows) * 64 * 2 options:MTLResourceStorageModePrivate];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (NSString *name in [L.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
      int M, NP, SG; sscanf(name.UTF8String, "tallh_m%d_np%d_sg%d", &M, &NP, &SG);
      if (NP < int(N) || (NP > 16 && NP / 2 >= int(N))) continue;
      id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:name] error:&err];
      P prm{K, N, rows, K, N, 0, 0, 0};
      double best = 1e9;
      for (int t = 0; t < 5; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p]; [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:Y offset:0 atIndex:4]; [e setBytes:&prm length:sizeof prm atIndex:5];
        for (int i = 0; i < 40; ++i) { [e setBuffer:X offset:size_t(i % 8) * rows * K * 2 atIndex:0];
          [e dispatchThreadgroups:MTLSizeMake((rows + M - 1) / M, 1, 1) threadsPerThreadgroup:MTLSizeMake(SG * 32, 1, 1)]; }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / 40);
      }
      printf("%-24s %7.2f us  (x %.1f GB/s)\n", name.UTF8String, best * 1e6, double(rows) * K * 2 / best / 1e9);
    }
  }
}
