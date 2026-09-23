// Per-dispatch GPU time distribution of one kernel (one dispatch per command buffer).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
#include <vector>
struct P { uint32_t K, N, rows, xs, ys, row0; uint64_t wr, pr; };
int main(int argc, char **argv) {
  @autoreleasepool {
    const char *lib = argv[1], *fn = argv[2];
    const uint32_t K = atoi(argv[3]), N = atoi(argv[4]), bits = atoi(argv[5]), G = atoi(argv[6]), rows = atoi(argv[7]);
    const uint32_t gx = atoi(argv[8]), threads = atoi(argv[9]);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(lib)] error:&err];
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:@(fn)] error:&err];
    if (!p) { printf("pso fail\n"); return 1; }
    const uint64_t wr = uint64_t(K) * bits / 8, pr = uint64_t(K / G) * 2;
    id<MTLBuffer> X = [dev newBufferWithLength:uint64_t(K) * 32 * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> W = [dev newBufferWithLength:wr * N options:MTLResourceStorageModeShared];
    id<MTLBuffer> S = [dev newBufferWithLength:pr * N options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [dev newBufferWithLength:pr * N options:MTLResourceStorageModeShared];
    id<MTLBuffer> Y = [dev newBufferWithLength:uint64_t(N) * 32 * 2 options:MTLResourceStorageModeShared];
    uint16_t *x = (uint16_t *)X.contents; for (uint64_t i = 0; i < uint64_t(K) * 32; ++i) x[i] = 0x3f80 ^ (i & 0x7f);
    uint16_t *s = (uint16_t *)S.contents; for (uint64_t i = 0; i < pr * N / 2; ++i) s[i] = 0x3c00;
    uint8_t *w = (uint8_t *)W.contents; for (uint64_t i = 0; i < wr * N; ++i) w[i] = uint8_t(i * 37);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    P prm{K, N, rows, K, N, 0, wr, pr};
    std::vector<double> t;
    for (int it = 0; it < 400; ++it) {
      id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:p]; [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:W offset:0 atIndex:1];
      [e setBuffer:S offset:0 atIndex:2]; [e setBuffer:B offset:0 atIndex:3]; [e setBuffer:Y offset:0 atIndex:4];
      [e setBytes:&prm length:sizeof prm atIndex:5];
      [e dispatchThreadgroups:MTLSizeMake(gx, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      t.push_back((cb.GPUEndTime - cb.GPUStartTime) * 1e6);
    }
    std::sort(t.begin(), t.end());
    printf("%s p10 %.1f p50 %.1f p90 %.1f max %.1f us\n", fn, t[40], t[200], t[360], t.back());
  }
}
