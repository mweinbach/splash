#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
struct DParams { uint32_t K, N, rows, pad; uint64_t w_row_stride, p_row_stride; };
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t r = ((u >> 16) & 1) + 0x7fff; return uint16_t((u + r) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }
int main(int argc, char **argv) {
  @autoreleasepool {
    const uint32_t R = argc > 1 ? atoi(argv[1]) : 5;
    const uint32_t N = argc > 2 ? atoi(argv[2]) : 10240, K = argc > 3 ? atoi(argv[3]) : 2560;
    const int L = 48;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@"dense.metallib"] error:&err];
    if (!lib) { printf("lib %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    std::mt19937 rng(3); std::normal_distribution<float> nd(0.f, 1.f);
    const uint64_t wbytes = uint64_t(N) * K / 2, pcount = uint64_t(N) * (K / 64);
    std::vector<id<MTLBuffer>> W, S, Bi;
    for (int l = 0; l < L; ++l) {
      id<MTLBuffer> w = [dev newBufferWithLength:wbytes options:MTLResourceStorageModeShared];
      id<MTLBuffer> s = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      id<MTLBuffer> b = [dev newBufferWithLength:pcount * 2 options:MTLResourceStorageModeShared];
      uint32_t *wp = (uint32_t *)w.contents; for (uint64_t i = 0; i < wbytes / 4; ++i) wp[i] = rng();
      uint16_t *sp = (uint16_t *)s.contents, *bp = (uint16_t *)b.contents;
      for (uint64_t i = 0; i < pcount; ++i) { sp[i] = f2bf(0.004f + 0.001f * std::fabs(nd(rng))); bp[i] = f2bf(-0.03f + 0.003f * nd(rng)); }
      W.push_back(w); S.push_back(s); Bi.push_back(b);
    }
    id<MTLBuffer> x = [dev newBufferWithLength:16 * K * 2 options:MTLResourceStorageModeShared];
    memset(x.contents, 0, 16 * K * 2);
    for (uint32_t i = 0; i < R * K; ++i) ((uint16_t *)x.contents)[i] = f2bf(nd(rng) * 0.5f);
    id<MTLBuffer> y = [dev newBufferWithLength:16 * N * 2 options:MTLResourceStorageModeShared];
    DParams p{K, N, R, 0, K / 2, K / 64};
    for (NSString *name in [lib.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
      std::string n = name.UTF8String;
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
      if (!pso) { printf("pso %s: %s\n", n.c_str(), err.localizedDescription.UTF8String); continue; }
      uint32_t groups = 0, threads = 0;
      int nt = 0, sn = 0, sk = 0;
      if (sscanf(n.c_str(), "sgmv_q4_t%d_sn%d_sk%d", &nt, &sn, &sk) == 3) { groups = N / (8 * nt * sn); threads = 32 * sn * sk; }
      else if (sscanf(n.c_str(), "mppq_q4_nt%d_sk%d", &nt, &sk) == 2) { groups = N / nt; threads = 32 * sk; }
      if (threads > pso.maxTotalThreadsPerThreadgroup) { printf("%s: threads %u > max %lu\n", n.c_str(), threads, (unsigned long)pso.maxTotalThreadsPerThreadgroup); continue; }
      double best = 1e9;
      for (int t = 0; t < 4; ++t) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        [e setComputePipelineState:pso];
        [e setBuffer:x offset:0 atIndex:0]; [e setBuffer:y offset:0 atIndex:4]; [e setBytes:&p length:sizeof p atIndex:5];
        for (int l = 0; l < L; ++l) {
          [e setBuffer:W[l] offset:0 atIndex:1]; [e setBuffer:S[l] offset:0 atIndex:2]; [e setBuffer:Bi[l] offset:0 atIndex:3];
          if (l) [e memoryBarrierWithScope:MTLBarrierScopeBuffers];
          [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        }
        [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
        if (t) best = std::min(best, (cb.GPUEndTime - cb.GPUStartTime) / L);
      }
      // check last layer
      double maxerr = 0, maxref = 0;
      const uint8_t *wp = (const uint8_t *)W[L - 1].contents; const uint16_t *sp = (const uint16_t *)S[L - 1].contents, *bp = (const uint16_t *)Bi[L - 1].contents;
      for (int trial = 0; trial < 64; ++trial) {
        const uint32_t m = rng() % R, col = rng() % N;
        double acc = 0;
        for (uint32_t k = 0; k < K; ++k) {
          const uint32_t code = (wp[uint64_t(col) * (K / 2) + k / 2] >> ((k & 1) * 4)) & 15;
          acc += double(bf2f(((uint16_t *)x.contents)[m * K + k])) * (double(bf2f(sp[col * (K / 64) + k / 64])) * code + double(bf2f(bp[col * (K / 64) + k / 64])));
        }
        const double got = bf2f(((uint16_t *)y.contents)[uint64_t(m) * N + col]);
        maxerr = std::max(maxerr, std::fabs(got - acc)); maxref = std::max(maxref, std::fabs(acc));
      }
      printf("%-28s R=%u N=%u K=%u  %7.2f us  %6.1f GB/s  relerr %.2e\n", n.c_str(), R, N, K, best * 1e6,
             (wbytes + pcount * 4) / best / 1e9, maxerr / std::max(maxref, 1e-9));
    }
  }
  return 0;
}
