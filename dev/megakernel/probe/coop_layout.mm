#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <string>

int main(int argc, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"coop_layout.metallib"] error:&err];
    if (!L) { printf("lib %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> out = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
    const char *which = argc > 1 ? argv[1] : "layout_bf16_u8_16x32x64";
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:@(which)] error:&err];
    if (!p) { printf("pso: %s\n", err.localizedDescription.UTF8String); return 1; }
    memset(out.contents, 0xff, 1 << 20);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p];
    [e setBuffer:out offset:0 atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    const int *o = (const int *)out.contents;
    printf("%s right capacity %d left capacity %d\n", which, o[0], o[1]);
    for (int part = 0; part < 3; ++part) {
      printf(part == 2 ? "DEST:\n" : part ? "LEFT (idx0, idx1) per lane:\n" : "RIGHT (idx0, idx1) per lane:\n");
      const int cap = o[part];
      for (int lane = 0; lane < 32; ++lane) {
        const int *r = o + 16 + part * 32 * 256 + lane * 256;
        printf("  lane %2d:", lane);
        for (int i = 0; i < cap; ++i) printf(" (%d,%d)", r[2 * i], r[2 * i + 1]);
        printf("\n");
      }
    }
  }
  return 0;
}
