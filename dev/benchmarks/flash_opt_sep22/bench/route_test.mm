#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal/abi/FlashMoE.h"
#include <cstdio>
#include <cstring>
#include <random>
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return uint16_t((u + ((u >> 16) & 1) + 0x7fff) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }
int main(int, char **argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    const uint32_t rows = 2048, E = 512, K = 10;
    std::mt19937 rng(3); std::normal_distribution<float> d(0, 2);
    id<MTLBuffer> lg = [dev newBufferWithLength:rows * E * 2 options:0];
    auto *p = (uint16_t *)lg.contents;
    for (uint32_t i = 0; i < rows * E; ++i) p[i] = f2bf(d(rng));
    for (uint32_t r = 0; r < rows; r += 7) p[r * E + 17] = p[r * E + 300];  // ties
    id<MTLBuffer> id0 = [dev newBufferWithLength:rows * K * 8 options:0], id1 = [dev newBufferWithLength:rows * K * 8 options:0];
    id<MTLBuffer> w0 = [dev newBufferWithLength:rows * K * 2 options:0], w1 = [dev newBufferWithLength:rows * K * 2 options:0];
    id<MTLBuffer> diag = [dev newBufferWithLength:16 options:0];
    FlashMoERouteParams rp{rows, E, K, 1};
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:[dev newComputePipelineStateWithFunction:[L newFunctionWithName:@"flash_moe_route"] error:&err]];
    [e setBuffer:lg offset:0 atIndex:0]; [e setBuffer:id0 offset:0 atIndex:1]; [e setBuffer:w0 offset:0 atIndex:2];
    [e setBuffer:diag offset:0 atIndex:3]; [e setBytes:&rp length:sizeof rp atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e setComputePipelineState:[dev newComputePipelineStateWithFunction:[L newFunctionWithName:@"opt_moe_route"] error:&err]];
    [e setBuffer:lg offset:0 atIndex:0]; [e setBuffer:id1 offset:0 atIndex:1]; [e setBuffer:w1 offset:0 atIndex:2];
    [e setBytes:&rp length:sizeof rp atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    const int64_t *a = (const int64_t *)id0.contents, *b = (const int64_t *)id1.contents;
    const uint16_t *wa = (const uint16_t *)w0.contents, *wb = (const uint16_t *)w1.contents;
    uint32_t idDiff = 0, wDiff = 0; float maxw = 0;
    for (uint32_t i = 0; i < rows * K; ++i) { idDiff += a[i] != b[i]; wDiff += wa[i] != wb[i]; maxw = std::max(maxw, std::fabs(bf2f(wa[i]) - bf2f(wb[i]))); }
    printf("ids differ %u / %u, weights differ %u (max %.4g), diag %u\n", idDiff, rows * K, wDiff, maxw, *(uint32_t *)diag.contents);
  }
}
