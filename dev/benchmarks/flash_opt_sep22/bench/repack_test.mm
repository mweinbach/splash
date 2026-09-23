// Repack 5/6-bit MLX rows to bytes (as OptQmv::repackCodes) and check the
// 8-bit matrix-unit GEMV against a CPU reference on the original codes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
struct P { uint32_t K, N, rows, xs, ys, row0; uint64_t wr, pr; };
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); return uint16_t((u + ((u >> 16) & 1) + 0x7fff) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }
int main(int argc, char **argv) {
  @autoreleasepool {
    const uint32_t K = atoi(argv[2]), N = atoi(argv[3]), bits = atoi(argv[4]), G = atoi(argv[5]), rows = atoi(argv[6]);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *err = nil;
    id<MTLLibrary> L = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
    const std::string name = "opt_mppq_b8_g" + std::to_string(G) + "_nt32_sk8";
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[L newFunctionWithName:@(name.c_str())] error:&err];
    if (!p) { printf("missing %s\n", name.c_str()); return 1; }
    const uint64_t wr = (uint64_t(K) * bits + 7) / 8, pr = uint64_t(K / G) * 2;
    std::mt19937 rng(7);
    std::vector<uint8_t> packed(wr * N); for (auto &b : packed) b = uint8_t(rng());
    std::vector<uint16_t> s(N * K / G), bb(N * K / G), x(uint64_t(K) * 16);
    std::uniform_real_distribution<float> us(0.001f, 0.02f), ub(-0.1f, 0.1f), ux(-1, 1);
    for (auto &v : s) v = f2bf(us(rng)); for (auto &v : bb) v = f2bf(ub(rng)); for (auto &v : x) v = f2bf(ux(rng));
    std::vector<uint8_t> codes(uint64_t(N) * K);
    const uint32_t mask = (1u << bits) - 1u;
    for (uint32_t n = 0; n < N; ++n) for (uint32_t k = 0; k < K; ++k) {
      const uint64_t bit = uint64_t(k) * bits, byte = bit >> 3; const uint8_t *row = packed.data() + n * wr;
      uint32_t v = row[byte]; if (byte + 1 < wr) v |= uint32_t(row[byte + 1]) << 8;
      codes[uint64_t(n) * K + k] = uint8_t((v >> (bit & 7)) & mask);
    }
    auto buf = [&](const void *data, uint64_t bytes) { id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared]; if (data) memcpy(b.contents, data, bytes); return b; };
    id<MTLBuffer> X = buf(x.data(), x.size() * 2), W = buf(codes.data(), codes.size()), S = buf(s.data(), s.size() * 2),
                  B = buf(bb.data(), bb.size() * 2), Y = buf(nullptr, uint64_t(N) * 16 * 2);
    P prm{K, N, rows, K, N, 0, K, pr};
    id<MTLCommandQueue> q = [dev newCommandQueue]; id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:p]; [e setBuffer:X offset:0 atIndex:0]; [e setBuffer:W offset:0 atIndex:1]; [e setBuffer:S offset:0 atIndex:2];
    [e setBuffer:B offset:0 atIndex:3]; [e setBuffer:Y offset:0 atIndex:4]; [e setBytes:&prm length:sizeof prm atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(N / 32, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    double maxerr = 0, maxref = 0;
    const uint16_t *y = (const uint16_t *)Y.contents;
    for (uint32_t r = 0; r < rows; ++r) for (uint32_t n = 0; n < N; ++n) {
      double acc = 0;
      for (uint32_t k = 0; k < K; ++k) {  // reference from the ORIGINAL packed bits
        const uint64_t bit = uint64_t(k) * bits, byte = bit >> 3; const uint8_t *row = packed.data() + n * wr;
        uint32_t v = row[byte]; if (byte + 1 < wr) v |= uint32_t(row[byte + 1]) << 8;
        const uint32_t qv = (v >> (bit & 7)) & mask;
        acc += (double(bf2f(s[n * (K / G) + k / G])) * qv + bf2f(bb[n * (K / G) + k / G])) * bf2f(x[uint64_t(r) * K + k]);
      }
      maxerr = std::max(maxerr, std::fabs(acc - bf2f(y[uint64_t(r) * N + n]))); maxref = std::max(maxref, std::fabs(acc));
    }
    printf("K=%u N=%u bits=%u G=%u rows=%u relerr %.2e\n", K, N, bits, G, rows, maxerr / maxref);
  }
}
