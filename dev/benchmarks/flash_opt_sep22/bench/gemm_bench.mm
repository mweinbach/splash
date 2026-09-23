// Times gemm_bench.metal variants on the expert gate shape (K=2560, N=640).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <string>

int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"gemm_bench.metallib"] error:&err];
    const uint32_t E = 512, K = 2560, NOUT = 640;
    auto buf = [&](size_t n) { return [dev newBufferWithLength:n options:MTLResourceStorageModePrivate]; };
    id<MTLBuffer> A = buf(size_t(E) * 64 * K * 2), B = buf(size_t(E) * NOUT * K * 2),
                  S = buf(size_t(E) * NOUT * 40 * 4 * 2), O = buf(size_t(E) * 64 * NOUT * 2),
                  X = buf(size_t(E) * 64 * 40 * 4);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (NSString *name in [L.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
      std::string n = name.UTF8String;
      int M = 64, SG = 4;
      sscanf(n.substr(n.find("_m") + 2).c_str(), "%d_sg%d", &M, &SG);
      id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:name] error:&err];
      if (!p) { printf("pso %s\n", n.c_str()); continue; }
      double best = 1e9;
      for (int t = 0; t < 5; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:p];
        [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1];
        [e setBuffer:S offset:0 atIndex:2]; [e setBuffer:O offset:0 atIndex:3]; [e setBuffer:X offset:0 atIndex:4];
        // Rows per expert fixed at 64 regardless of M (M-tiles per expert = 64/M).
        [e dispatchThreadgroups:MTLSizeMake(NOUT / 64, E * (64 / M), 1) threadsPerThreadgroup:MTLSizeMake(SG * 32, 1, 1)];
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
      }
      const double flops = 2.0 * E * 64 * NOUT * double(K);
      printf("%-26s %7.3f ms  %6.1f TFLOPS\n", n.c_str(), best * 1e3, flops / best / 1e12);
    }
  }
  return 0;
}
