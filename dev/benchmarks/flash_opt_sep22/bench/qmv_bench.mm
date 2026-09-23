// Standalone timing/correctness harness for opt_qmv kernels.
// Usage: qmv_bench <metallib> K N bits group rows [kernel-filter]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct OptQmvParams {
  uint32_t K, N, rows, x_stride, y_stride, reserved;
  uint64_t w_row_stride, p_row_stride;
};

static uint16_t f2bf(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  uint32_t r = ((u >> 16) & 1) + 0x7fff;
  return uint16_t((u + r) >> 16);
}
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 7) { fprintf(stderr, "usage\n"); return 2; }
    const char *libpath = argv[1];
    const uint32_t K = atoi(argv[2]), N = atoi(argv[3]), bits = atoi(argv[4]), G = atoi(argv[5]), rows = atoi(argv[6]);
    const char *filter = argc > 7 ? argv[7] : "";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(libpath)] error:&err];
    if (!lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 1; }
    const uint64_t wrow = (uint64_t(K) * bits + 7) / 8;
    const uint64_t prow = uint64_t(K / G) * 2;
    const uint64_t wbytes = wrow * N, pbytes = prow * N;
    const uint64_t per = wbytes + 2 * pbytes;
    const uint32_t copies = std::max<uint64_t>(1, (1536ull << 20) / per + 1);
    std::mt19937 rng(1234);
    id<MTLBuffer> W = [dev newBufferWithLength:wbytes * copies options:MTLResourceStorageModeShared];
    id<MTLBuffer> S = [dev newBufferWithLength:pbytes * copies options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [dev newBufferWithLength:pbytes * copies options:MTLResourceStorageModeShared];
    const uint64_t xr = std::max<uint32_t>(rows, 8);
    id<MTLBuffer> X = [dev newBufferWithLength:uint64_t(K) * xr * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> Y = [dev newBufferWithLength:uint64_t(N) * xr * 2 * copies options:MTLResourceStorageModeShared];
    {
      uint8_t *w = (uint8_t *)W.contents;
      for (uint64_t i = 0; i < wbytes * copies; ++i) w[i] = uint8_t(rng());
      uint16_t *s = (uint16_t *)S.contents, *b = (uint16_t *)B.contents;
      std::uniform_real_distribution<float> us(0.001f, 0.02f), ub(-0.1f, 0.1f), ux(-1.f, 1.f);
      for (uint64_t i = 0; i < pbytes / 2 * copies; ++i) { s[i] = f2bf(us(rng)); b[i] = f2bf(ub(rng)); }
      uint16_t *x = (uint16_t *)X.contents;
      for (uint64_t i = 0; i < uint64_t(K) * xr; ++i) x[i] = f2bf(ux(rng));
    }
    // CPU reference for copy 0.
    std::vector<double> ref(uint64_t(rows) * N);
    {
      const uint8_t *w = (const uint8_t *)W.contents;
      const uint16_t *s = (const uint16_t *)S.contents, *b = (const uint16_t *)B.contents, *x = (const uint16_t *)X.contents;
      for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t r = 0; r < rows; ++r) {
          double acc = 0;
          for (uint32_t k = 0; k < K; ++k) {
            const uint64_t bit = uint64_t(k) * bits;
            const uint8_t *row = w + n * wrow;
            uint32_t v = row[bit >> 3] | (uint32_t((bit >> 3) + 1 < wrow ? row[(bit >> 3) + 1] : 0) << 8);
            const uint32_t q = (v >> (bit & 7)) & ((1u << bits) - 1);
            const double coef = double(bf2f(s[n * (prow / 2) + k / G])) * q + bf2f(b[n * (prow / 2) + k / G]);
            acc += coef * bf2f(x[uint64_t(r) * K + k]);
          }
          ref[uint64_t(r) * N + n] = acc;
        }
      }
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    const int Rs[] = {1, 2, 3, 4, 5, 8};
    int R = 0;
    for (int candidate : Rs) if (candidate >= int(rows)) { R = candidate; break; }
    if (getenv("TALL")) R = 4;
    char prefix[128];
    snprintf(prefix, sizeof prefix, "opt_qmv_b%u_g%u_r%d_", bits, G, R);
    for (NSString *name in [lib.functionNames sortedArrayUsingSelector:@selector(compare:)]) {
      std::string fn = name.UTF8String;
      char prefix2[128];
      snprintf(prefix2, sizeof prefix2, "opt_qmm8_b%u_g%u_", bits, G);
      const bool mm = fn.rfind(prefix2, 0) == 0;
      if ((!mm && fn.rfind(prefix, 0) != 0) || fn.find(filter) == std::string::npos) continue;
      if (mm && rows > 8) continue;
      int RN = 0, SN = 0, SK = 0, V = 0;
      if (mm) { int NT = 0; sscanf(fn.c_str() + strlen(prefix2), "nt%d_sk%d", &NT, &SK); RN = NT; SN = 1; V = 64; }
      else sscanf(fn.c_str() + strlen(prefix), "n%d_sn%d_sk%d_v%d", &RN, &SN, &SK, &V);
      if (K % (V) != 0) continue;
      id<MTLFunction> f = [lib newFunctionWithName:name];
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:f error:&err];
      if (!pso) { fprintf(stderr, "pso %s\n", fn.c_str()); continue; }
      const uint32_t outPerTG = uint32_t(RN * SN);
      const uint32_t rowBlocks = getenv("TALL") ? (rows + 3) / 4 : 1;
      const MTLSize grid = MTLSizeMake((N + outPerTG - 1) / outPerTG, rowBlocks, 1);
      const MTLSize tgs = MTLSizeMake(32 * SN * SK, 1, 1);
      OptQmvParams p{K, N, rows, K, N, 0, wrow, prow};
      auto run = [&](int iters) -> double {
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:X offset:0 atIndex:0];
        [enc setBytes:&p length:sizeof p atIndex:5];
        for (int i = 0; i < iters; ++i) {
          const uint64_t c = uint64_t(i) % copies;
          [enc setBuffer:W offset:c * wbytes atIndex:1];
          [enc setBuffer:S offset:c * pbytes atIndex:2];
          [enc setBuffer:B offset:c * pbytes atIndex:3];
          [enc setBuffer:Y offset:c * uint64_t(N) * xr * 2 atIndex:4];
          [enc dispatchThreadgroups:grid threadsPerThreadgroup:tgs];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return (cb.GPUEndTime - cb.GPUStartTime) / iters;
      };
      run(1);
      // correctness on copy 0
      double maxerr = 0, maxref = 0;
      const uint16_t *y = (const uint16_t *)Y.contents;
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t n = 0; n < N; ++n) {
          maxerr = std::max(maxerr, std::fabs(bf2f(y[uint64_t(r) * N + n]) - ref[uint64_t(r) * N + n]));
          maxref = std::max(maxref, std::fabs(ref[uint64_t(r) * N + n]));
        }
      run(copies);
      double best = 1e9;
      for (int t = 0; t < 5; ++t) best = std::min(best, run(std::max<uint32_t>(copies, 200)));
      const double gbs = double(wbytes + 2 * pbytes) / best / 1e9;
      printf("%-44s %8.2f us %7.1f GB/s relerr %.2e\n", fn.c_str(), best * 1e6, gbs, maxerr / std::max(maxref, 1e-9));
    }
  }
  return 0;
}
