// Compare opt_hc_down / opt_hc_up_mix against flash_hc_fused_down / up_mix.
// Usage: hc_test <metallib> <down_bits> <inj_bits> <up_bits> <rows>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal/abi/FlashHCFused.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct OptHCParams {
  uint32_t rows, has_injection, inj_bits, inj_group;
  uint64_t dw, dp, iw, ip, uw, up;
  uint32_t row0, reserved;
};

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return uint16_t((u + ((u >> 16) & 1) + 0x7fff) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *lib = argv[1];
    const uint32_t db = atoi(argv[2]), ib = atoi(argv[3]), ub = atoi(argv[4]), rows = atoi(argv[5]);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(lib)] error:&err];
    std::mt19937 rng(7);
    auto buf = [&](size_t bytes) { return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; };
    auto fillBytes = [&](id<MTLBuffer> b) { auto *p = (uint8_t *)b.contents; for (size_t i = 0; i < b.length; ++i) p[i] = uint8_t(rng()); };
    auto fillBF = [&](id<MTLBuffer> b, float lo, float hi) {
      std::uniform_real_distribution<float> d(lo, hi); auto *p = (uint16_t *)b.contents;
      for (size_t i = 0; i < b.length / 2; ++i) p[i] = f2bf(d(rng)); };
    const uint64_t dRow = 10240 * db / 8, iRow = 10240 * ib / 8, uRow = 320 * ub / 8;
    const uint64_t dP = 10240 / 64 * 2, iP = 10240 / 64 * 2, uP = 320 / 64 * 2;
    id<MTLBuffer> x = buf(rows * 10240 * 2), dw = buf(320 * dRow), ds = buf(320 * dP), dbias = buf(320 * dP);
    id<MTLBuffer> iw = buf(4 * iRow), is = buf(4 * iP), ibias = buf(4 * iP);
    id<MTLBuffer> uw = buf(10240 * uRow), us = buf(10240 * uP), ubias = buf(10240 * uP);
    fillBF(x, -2, 2); fillBytes(dw); fillBF(ds, 0.001f, 0.01f); fillBF(dbias, -0.05f, 0.05f);
    fillBytes(iw); fillBF(is, 0.001f, 0.01f); fillBF(ibias, -0.05f, 0.05f);
    fillBytes(uw); fillBF(us, 0.001f, 0.05f); fillBF(ubias, -0.2f, 0.2f);
    id<MTLBuffer> act0 = buf(rows * 320 * 2), act1 = buf(rows * 320 * 2), g0 = buf(rows * 4 * 2), g1 = buf(rows * 4 * 2);
    id<MTLBuffer> mix0 = buf(rows * 2560 * 2), mix1 = buf(rows * 2560 * 2), rawdbg = buf(rows * 10240 * 2), diag = buf(16);
    FlashHCFusedParams fp{}; fp.rows = rows; fp.width = 2560; fp.streams = 4; fp.lowrank = 320; fp.has_injection = 1;
    fp.arithmetic_mode = 0; fp.simdgroups = 4; fp.norm_epsilon = 1e-6f;
    fp.down = {10240, 320, db, 64, dRow, dP}; fp.injection = {10240, 4, ib, 64, iRow, iP}; fp.up = {320, 10240, ub, 64, uRow, uP};
    OptHCParams op{rows, 1, ib, 64, dRow, dP, iRow, iP, uRow, uP, 0, 0};
    id<MTLCommandQueue> q = [dev newCommandQueue];
    auto pso = [&](NSString *name) { id<MTLFunction> f = [L newFunctionWithName:name]; if (!f) { printf("missing %s\n", name.UTF8String); exit(1);} return [dev newComputePipelineStateWithFunction:f error:&err]; };
    const uint32_t R = rows <= 5 ? rows : 5;
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    // reference down
    [e setComputePipelineState:pso([NSString stringWithFormat:@"flash_hc_fused_down_q%u_g64_s4", db])];
    for (int i = 0; i < 9; ++i) {}
    id<MTLBuffer> rb[] = {x, dw, ds, dbias, iw, is, ibias, act0, g0, diag};
    for (int i = 0; i < 10; ++i) [e setBuffer:rb[i] offset:0 atIndex:i];
    [e setBytes:&fp length:sizeof fp atIndex:10];
    [e dispatchThreadgroups:MTLSizeMake(81, rows, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    // reference up-mix
    [e setComputePipelineState:pso([NSString stringWithFormat:@"flash_hc_fused_up_mix_q%u_g64_s4", ub])];
    id<MTLBuffer> ub2[] = {x, act0, uw, us, ubias, mix0, rawdbg, diag};
    for (int i = 0; i < 8; ++i) [e setBuffer:ub2[i] offset:0 atIndex:i];
    [e setBytes:&fp length:sizeof fp atIndex:8];
    [e dispatchThreadgroups:MTLSizeMake(640, rows, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    // opt down
    [e setComputePipelineState:pso([NSString stringWithFormat:@"opt_hc_down_b%u_g64_r%u", db, R])];
    id<MTLBuffer> ob[] = {x, dw, ds, dbias, iw, is, ibias, act1, g1};
    for (int i = 0; i < 9; ++i) [e setBuffer:ob[i] offset:0 atIndex:i];
    [e setBytes:&op length:sizeof op atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(161, 1, 1) threadsPerThreadgroup:MTLSizeMake(512, 1, 1)];
    // opt up-mix consumes the REFERENCE activation so up-mix is isolated.
    [e setComputePipelineState:pso([NSString stringWithFormat:@"opt_hc_up_mix_b%u_g64_r%u", ub, R])];
    id<MTLBuffer> ou[] = {x, act0, uw, us, ubias, mix1};
    for (int i = 0; i < 6; ++i) [e setBuffer:ou[i] offset:0 atIndex:i];
    [e setBytes:&op length:sizeof op atIndex:6];
    [e dispatchThreadgroups:MTLSizeMake(640, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    // Timing: 400 back-to-back dispatches of each kernel.
    auto timeKernel = [&](NSString *name, int nb, id<MTLBuffer> __strong *bufs, const void *params, size_t plen, int pidx, MTLSize grid, MTLSize tg) {
      id<MTLComputePipelineState> ps = pso(name);
      double best = 1e9;
      for (int t = 0; t < 3; ++t) {
        id<MTLCommandBuffer> c = [q commandBuffer];
        id<MTLComputeCommandEncoder> en = [c computeCommandEncoder];
        [en setComputePipelineState:ps];
        for (int i = 0; i < nb; ++i) [en setBuffer:bufs[i] offset:0 atIndex:i];
        [en setBytes:params length:plen atIndex:pidx];
        for (int i = 0; i < 400; ++i) [en dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        [en endEncoding]; [c commit]; [c waitUntilCompleted];
        best = std::min(best, (c.GPUEndTime - c.GPUStartTime) / 400 * 1e6);
      }
      printf("  time %-36s %7.2f us\n", name.UTF8String, best);
    };
    timeKernel([NSString stringWithFormat:@"flash_hc_fused_down_q%u_g64_s4", db], 10, rb, &fp, sizeof fp, 10, MTLSizeMake(81, rows, 1), MTLSizeMake(128, 1, 1));
    timeKernel([NSString stringWithFormat:@"opt_hc_down_b%u_g64_r%u", db, R], 9, ob, &op, sizeof op, 9, MTLSizeMake(161, 1, 1), MTLSizeMake(512, 1, 1));
    timeKernel([NSString stringWithFormat:@"opt_hc_down_b%u_g64_r%u", db, R], 9, ob, &op, sizeof op, 9, MTLSizeMake(160, 1, 1), MTLSizeMake(512, 1, 1));
    timeKernel([NSString stringWithFormat:@"flash_hc_fused_up_mix_q%u_g64_s4", ub], 8, ub2, &fp, sizeof fp, 8, MTLSizeMake(640, rows, 1), MTLSizeMake(128, 1, 1));
    timeKernel([NSString stringWithFormat:@"opt_hc_up_mix_b%u_g64_r%u", ub, R], 6, ou, &op, sizeof op, 6, MTLSizeMake(640, 1, 1), MTLSizeMake(128, 1, 1));
    auto cmp = [&](const char *name, id<MTLBuffer> a, id<MTLBuffer> b, size_t n) {
      const uint16_t *pa = (const uint16_t *)a.contents, *pb = (const uint16_t *)b.contents;
      double maxd = 0, maxa = 0; size_t diff = 0, big = 0;
      for (size_t i = 0; i < n; ++i) {
        const double va = bf2f(pa[i]), vb = bf2f(pb[i]);
        maxd = std::max(maxd, std::fabs(va - vb)); maxa = std::max(maxa, std::fabs(va));
        if (pa[i] != pb[i]) ++diff;
        if (std::fabs(va - vb) > 0.02 * std::fabs(va) + 1e-3) ++big;
      }
      printf("%-10s n=%zu differ=%zu (%.2f%%) large=%zu maxdiff=%.4g maxabs=%.4g\n", name, n, diff, 100.0 * diff / n, big, maxd, maxa);
    };
    cmp("activated", act0, act1, rows * 320);
    cmp("gates", g0, g1, rows * 4);
    cmp("mixed", mix0, mix1, rows * 2560);
    printf("diag %u\n", *(uint32_t *)diag.contents);
  }
  return 0;
}
