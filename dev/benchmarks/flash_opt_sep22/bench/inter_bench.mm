// Interleaving cost: A = n x ALU kernel, B = n x MPP kernel, C = n x (ALU, MPP).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>
struct P { uint32_t K, N, rows, xs, ys, row0; uint64_t wr, pr; };
int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    id<MTLComputePipelineState> alu = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:@(argv[2])] error:&err];
    id<MTLComputePipelineState> mpp = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:@(argv[3])] error:&err];
    const uint32_t K = 2560, N = 640;
    const uint64_t wr = K, pr = K / 128 * 2;
    auto buf = [&](uint64_t n) { return [dev newBufferWithLength:n options:MTLResourceStorageModeShared]; };
    id<MTLBuffer> X = buf(K * 32 * 2), W = buf(wr * N), S = buf(pr * N), B = buf(pr * N), Y = buf(N * 32 * 2), Y2 = buf(N * 32 * 2);
    uint16_t *s = (uint16_t *)S.contents; for (uint64_t i = 0; i < pr * N / 2; ++i) s[i] = 0x3c00;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    P pa{K, N, 4, K, N, 0, wr, pr}, pm{K, N, 16, K, N, 0, wr, pr};
    auto run = [&](int mode) {
      double best = 1e9;
      for (int t = 0; t < 5; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:S offset:0 atIndex:2]; [e setBuffer:B offset:0 atIndex:3];
        for (int i = 0; i < 50; ++i) {
          if (mode != 1) { [e setComputePipelineState:alu]; [e setBuffer:Y offset:0 atIndex:4]; [e setBytes:&pa length:sizeof pa atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(320, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)]; }
          if (mode != 0) { [e setComputePipelineState:mpp]; [e setBuffer:Y2 offset:0 atIndex:4]; [e setBytes:&pm length:sizeof pm atIndex:5];
            [e dispatchThreadgroups:MTLSizeMake(40, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)]; }
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) * 1e6 / 50);
      }
      return best;
    };
    const double a = run(0), b = run(1), c = run(2);
    printf("ALU only %.1f us, MPP only %.1f us, interleaved pair %.1f us (sum %.1f)\n", a, b, c, a + b);
  }
}
