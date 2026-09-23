#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"layout.metallib"] error:&err];
    for (NSString *n in @[@"lay_16_32_1", @"lay_16_64_1", @"lay_16_16_1", @"lay_8_32_1"]) {
      id<MTLFunction> f = [L newFunctionWithName:n]; if (!f) { printf("missing %s\n", n.UTF8String); continue; }
      id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
      if (!p) { printf("pso %s %s\n", n.UTF8String, err.localizedDescription.UTF8String); continue; }
      id<MTLBuffer> o = [dev newBufferWithLength:32 * 64 * 4 options:MTLResourceStorageModeShared];
      id<MTLBuffer> a = [dev newBufferWithLength:64 * 64 * 2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> b = [dev newBufferWithLength:64 * 64 options:MTLResourceStorageModeShared];
      id<MTLCommandQueue> q = [dev newCommandQueue]; id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; [e setComputePipelineState:p];
      [e setBuffer:o offset:0 atIndex:0]; [e setBuffer:a offset:0 atIndex:1]; [e setBuffer:b offset:0 atIndex:2];
      [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      int *r = (int *)o.contents; printf("%s cap=%d\n", n.UTF8String, r[0]);
      for (int l = 0; l < 4; ++l) { printf(" lane%d:", l); for (int i = 0; i < r[l * 64]; ++i) printf(" (%d,%d)", r[l * 64 + 1 + 2 * i], r[l * 64 + 2 + 2 * i]); printf("\n"); }
    }
  }
}
