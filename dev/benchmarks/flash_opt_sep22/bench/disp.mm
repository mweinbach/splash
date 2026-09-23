#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <algorithm>
#include <initializer_list>
int main() {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"disp.metallib"] error:&err];
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:@"tiny"] error:&err];
    id<MTLBuffer> b = [dev newBufferWithLength:1 << 24 options:MTLResourceStorageModePrivate];
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (int groups : {1, 64, 1024}) {
      for (int mode = 0; mode < 3; ++mode) {
        double best = 1e9;
        for (int t = 0; t < 5; ++t) {
          id<MTLCommandBuffer> cb = [q commandBuffer];
          id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:mode == 0 ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent];
          [e setComputePipelineState:p];
          for (int i = 0; i < 1000; ++i) {
            [e setBuffer:b offset:(mode == 2 ? (i % 64) * 65536 * 4 : 0) atIndex:0];
            if (mode == 1 && i) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          }
          [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
          best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
        }
        printf("groups=%4d %-22s %.2f us/dispatch\n", groups, mode == 0 ? "serial" : mode == 1 ? "concurrent+barrier" : "concurrent-nobarrier", best * 1e6 / 1000);
      }
    }
  }
}
