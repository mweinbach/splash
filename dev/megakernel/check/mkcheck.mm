// Standalone correctness/timing harness for the megakernel decode phases.
// Maps the real package weights (no copies) and runs worker metallib kernels.
//   mkcheck moe <layer> <rows>      fused MoE vs CPU reference, then all-layer timing
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static const char *kPackage = "/Users/mweinbach/Projects/splash/install/local-models/Flash-Next-oQ4e-mtp-v1";
static const char *kLib = "/Users/mweinbach/Projects/splash/dev/benchmarks/flash_opt_sep22/worker/splash.metallib";

static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); if ((u & 0x7fffffff) > 0x7f800000) return 0x7fc0; uint32_t r = ((u >> 16) & 1) + 0x7fff; return uint16_t((u + r) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; memcpy(&f, &u, 4); return f; }
static float rb(float f) { return bf2f(f2bf(f)); }

struct Tensor {
  id<MTLBuffer> buffer;  // whole shard
  uint64_t offset = 0, length = 0;
  std::vector<uint64_t> shape;
  std::string dtype;
  const uint8_t *host = nullptr;
};

struct Affine {
  Tensor w, s, b;
  uint32_t N = 0, K = 0, bits = 0, group = 0, experts = 1;
  uint64_t wRow = 0, wExpert = 0, pRow = 0, pExpert = 0;  // p* in elements
};

struct Model {
  id<MTLDevice> dev;
  NSDictionary *tensors, *quant;
  std::map<std::string, id<MTLBuffer>> shards;
  std::map<std::string, const uint8_t *> hosts;
  explicit Model(id<MTLDevice> d) : dev(d) {
    NSData *m = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%s/manifest.json", kPackage]];
    NSDictionary *j = [NSJSONSerialization JSONObjectWithData:m options:0 error:nil];
    tensors = j[@"tensors"];
    NSData *c = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%s/config.json", kPackage]];
    quant = [NSJSONSerialization JSONObjectWithData:c options:0 error:nil][@"quantization_config"];
  }
  id<MTLBuffer> shard(const std::string &path, const uint8_t **host) {
    auto it = shards.find(path);
    if (it != shards.end()) { *host = hosts[path]; return it->second; }
    const std::string full = std::string(kPackage) + "/" + path;
    int fd = open(full.c_str(), O_RDONLY);
    if (fd < 0) { perror(full.c_str()); exit(1); }
    const off_t size = lseek(fd, 0, SEEK_END);
    void *p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); exit(1); }
    id<MTLBuffer> b = [dev newBufferWithBytesNoCopy:p length:size options:MTLResourceStorageModeShared deallocator:nil];
    if (!b) { fprintf(stderr, "no-copy buffer failed for %s\n", path.c_str()); exit(1); }
    shards[path] = b; hosts[path] = (const uint8_t *)p; *host = (const uint8_t *)p;
    return b;
  }
  Tensor tensor(const std::string &name) {
    NSDictionary *t = tensors[@(name.c_str())];
    if (!t) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
    Tensor r;
    const uint8_t *host = nullptr;
    r.buffer = shard([t[@"shard"] UTF8String], &host);
    r.offset = [t[@"offset"] unsignedLongLongValue];
    r.length = [t[@"length"] unsignedLongLongValue];
    for (NSNumber *n in t[@"shape"]) r.shape.push_back(n.unsignedLongLongValue);
    r.dtype = [t[@"dtype"] UTF8String];
    r.host = host + r.offset;
    return r;
  }
  Affine affine(const std::string &prefix) {
    Affine a;
    a.w = tensor(prefix + ".weight"); a.s = tensor(prefix + ".scales"); a.b = tensor(prefix + ".biases");
    NSDictionary *q = quant[@(prefix.c_str())];
    a.bits = q ? [q[@"bits"] unsignedIntValue] : [quant[@"bits"] unsignedIntValue];
    a.group = q ? [q[@"group_size"] unsignedIntValue] : [quant[@"group_size"] unsignedIntValue];
    const size_t rank = a.w.shape.size();
    a.experts = rank == 3 ? uint32_t(a.w.shape[0]) : 1;
    a.N = uint32_t(a.w.shape[rank - 2]);
    a.K = uint32_t(a.w.shape[rank - 1] * 32 / a.bits);
    a.wRow = a.w.shape[rank - 1] * 4;
    a.wExpert = a.wRow * a.N;
    a.pRow = a.s.shape[rank - 1];
    a.pExpert = a.pRow * a.N;
    return a;
  }
};

// CPU dequantized dot of row n (expert e) with x.
static double affineDot(const Affine &a, uint32_t e, uint32_t n, const float *x) {
  const uint8_t *w = a.w.host + e * a.wExpert + n * a.wRow;
  const uint16_t *s = (const uint16_t *)a.s.host + e * a.pExpert + n * a.pRow;
  const uint16_t *b = (const uint16_t *)a.b.host + e * a.pExpert + n * a.pRow;
  double acc = 0;
  for (uint32_t k = 0; k < a.K; ++k) {
    const uint64_t bit = uint64_t(k) * a.bits;
    uint32_t v = 0;
    for (uint32_t i = 0; i < a.bits; ++i) {
      const uint64_t bb = bit + i;
      v |= ((w[bb / 8] >> (bb % 8)) & 1u) << i;
    }
    acc += double(x[k]) * (double(bf2f(s[k / a.group])) * v + double(bf2f(b[k / a.group])));
  }
  return acc;
}

struct Ctx {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLLibrary> lib;
  std::map<std::string, id<MTLComputePipelineState>> psos;
  id<MTLComputePipelineState> pso(const std::string &name) {
    auto it = psos.find(name);
    if (it != psos.end()) return it->second;
    NSError *err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:@(name.c_str())];
    if (!f) { fprintf(stderr, "missing kernel %s\n", name.c_str()); exit(1); }
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) { fprintf(stderr, "pso %s: %s\n", name.c_str(), err.localizedDescription.UTF8String); exit(1); }
    psos[name] = p;
    return p;
  }
  id<MTLBuffer> buf(uint64_t bytes) { return [dev newBufferWithLength:std::max<uint64_t>(bytes, 16) options:MTLResourceStorageModeShared]; }
};

struct Bind { id<MTLBuffer> b; uint64_t off; };
static Bind B(const Tensor &t) { return {t.buffer, t.offset}; }
static Bind B(id<MTLBuffer> b) { return {b, 0}; }

static void dispatch(Ctx &c, id<MTLComputeCommandEncoder> e, const std::string &name, std::vector<Bind> binds,
                     const void *params, size_t psize, MTLSize groups, MTLSize threads) {
  [e setComputePipelineState:c.pso(name)];
  for (size_t i = 0; i < binds.size(); ++i) [e setBuffer:binds[i].b offset:binds[i].off atIndex:i];
  if (params) [e setBytes:params length:psize atIndex:binds.size()];
  [e dispatchThreadgroups:groups threadsPerThreadgroup:threads];
}

// ---- MoE ------------------------------------------------------------------
struct MkAffineP { uint64_t wRow, wExpert, pRow, pExpert; uint32_t bits, group, pad0, pad1; };
struct RouterParams { uint32_t rows, p0, p1, p2; MkAffineP sg, su, sl; };
struct MoEParams { uint32_t rows, p0, p1, p2; MkAffineP g, u, d, sd; };
struct CombineParams { uint32_t rows, hasNorm, normConvention; float eps; };
static MkAffineP P(const Affine &a) { return {a.wRow, a.wExpert, a.pRow, a.pExpert, a.bits, a.group, 0, 0}; }

struct MoELayer {
  Tensor router, norm;
  Affine sg, su, sd, sl, g, u, d;
};
static MoELayer loadMoE(Model &m, int layer) {
  const std::string mlp = "language_model.model.layers." + std::to_string(layer) + ".mlp";
  MoELayer L;
  L.router = m.tensor(mlp + ".gate.weight");
  L.sg = m.affine(mlp + ".shared_expert.gate_proj"); L.su = m.affine(mlp + ".shared_expert.up_proj");
  L.sd = m.affine(mlp + ".shared_expert.down_proj"); L.sl = m.affine(mlp + ".shared_expert_gate");
  L.g = m.affine(mlp + ".switch_mlp.gate_proj"); L.u = m.affine(mlp + ".switch_mlp.up_proj");
  L.d = m.affine(mlp + ".switch_mlp.down_proj");
  const std::string next = layer + 1 == 48 ? "language_model.model.hyper_connection_mixer"
                                            : "language_model.model.layers." + std::to_string(layer + 1) + ".attn_hyper_connection";
  L.norm = m.tensor(next + ".hc_norm.weight");
  return L;
}

struct MoEBuffers { id<MTLBuffer> x, gates, hyper, normalized, logits, sharedInter, sharedLogit, plan, inter, expertDown, sharedDown; };

static void encodeMoE(Ctx &c, id<MTLComputeCommandEncoder> e, const MoELayer &L, const MoEBuffers &b, uint32_t rows,
                      int only = -1) {
  RouterParams rp{rows, 0, 0, 0, P(L.sg), P(L.su), P(L.sl)};
  MoEParams mp{rows, 0, 0, 0, P(L.g), P(L.u), P(L.d), P(L.sd)};
  CombineParams cp{rows, 1, 0, 1e-6f};
  const uint32_t maxU = rows * 10;
  if (only < 0 || only == 0)
    dispatch(c, e, "mk_moe_router_mpp", {B(b.x), B(L.router), B(L.sg.w), B(L.sg.s), B(L.sg.b), B(L.su.w), B(L.su.s), B(L.su.b),
             B(L.sl.w), B(L.sl.s), B(L.sl.b), B(b.logits), B(b.sharedInter), B(b.sharedLogit)}, &rp, sizeof rp,
             MTLSizeMake(37, 1, 1), MTLSizeMake(256, 1, 1));
  if (only < 0 || only == 1)
    dispatch(c, e, "mk_moe_plan", {B(b.logits), B(b.plan)}, &mp, sizeof mp, MTLSizeMake(1, 1, 1), MTLSizeMake(256, 1, 1));
  if (only < 0 || only == 2)
    dispatch(c, e, "mk_moe_gate_up", {B(b.x), B(L.g.w), B(L.g.s), B(L.g.b), B(L.u.w), B(L.u.s), B(L.u.b), B(b.inter), B(b.plan)},
             &mp, sizeof mp, MTLSizeMake(20, maxU, 1), MTLSizeMake(128, 1, 1));
  if (only < 0 || only == 3)
    dispatch(c, e, "mk_moe_down", {B(b.inter), B(b.sharedInter), B(L.d.w), B(L.d.s), B(L.d.b), B(L.sd.w), B(L.sd.s), B(L.sd.b),
             B(b.expertDown), B(b.sharedDown), B(b.plan)}, &mp, sizeof mp, MTLSizeMake(80, maxU + 1, 1), MTLSizeMake(160, 1, 1));
  if (only < 0 || only == 4)
    dispatch(c, e, "mk_moe_combine", {B(b.expertDown), B(b.sharedDown), B(b.sharedLogit), B(b.plan), B(b.gates), B(b.hyper),
             B(L.norm), B(b.normalized)}, &cp, sizeof cp, MTLSizeMake(rows, 4, 1), MTLSizeMake(640, 1, 1));
}

static float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }

static int testMoE(Ctx &c, Model &m, int layer, uint32_t rows) {
  MoELayer L = loadMoE(m, layer);
  printf("layer %d: router %s %s, shared g/u Q%u G%u, down Q%u G%u, logit Q%u G%u, experts Q%u G%u\n", layer,
         L.router.dtype.c_str(), L.router.shape.size() == 2 ? "2d" : "?", L.sg.bits, L.sg.group, L.sd.bits, L.sd.group,
         L.sl.bits, L.sl.group, L.g.bits, L.g.group);
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  MoEBuffers b;
  b.x = c.buf(rows * 2560 * 2); b.gates = c.buf(rows * 4 * 2); b.hyper = c.buf(rows * 10240 * 2);
  b.normalized = c.buf(rows * 10240 * 2); b.logits = c.buf(8 * 512 * 2); b.sharedInter = c.buf(8 * 640 * 2);
  b.sharedLogit = c.buf(64); b.plan = c.buf(16384); b.inter = c.buf(80 * 16 * 640 * 2);
  b.expertDown = c.buf(80 * 2560 * 2); b.sharedDown = c.buf(8 * 2560 * 2);
  memset(b.inter.contents, 0, b.inter.length);
  std::vector<float> x(rows * 2560), hyper(rows * 10240), gates(rows * 4);
  for (auto &v : x) v = rb(nd(rng) * 0.6f);
  for (auto &v : hyper) v = getenv("MK_HYPER") ? rb(nd(rng) * 2.0f) : 0.f;
  for (auto &v : gates) v = rb(2.f * sigm(nd(rng)));
  for (size_t i = 0; i < x.size(); ++i) ((uint16_t *)b.x.contents)[i] = f2bf(x[i]);
  for (size_t i = 0; i < hyper.size(); ++i) ((uint16_t *)b.hyper.contents)[i] = f2bf(hyper[i]);
  for (size_t i = 0; i < gates.size(); ++i) ((uint16_t *)b.gates.contents)[i] = f2bf(gates[i]);
  {
    id<MTLCommandBuffer> cb = [c.q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    encodeMoE(c, e, L, b, rows);
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.error) { printf("GPU error %s\n", cb.error.localizedDescription.UTF8String); return 1; }
  }
  // CPU reference.
  int routeMismatch = 0;
  double maxErr = 0, maxRef = 0, maxNormErr = 0, sumSq = 0, sumErrSq = 0;
  const uint32_t *planWords = (const uint32_t *)b.plan.contents;
  const uint32_t *planRouteExpert = planWords + 4 + 80 + 640;
  printf("plan: rows %u unique %u\n", planWords[0], planWords[1]);
  for (uint32_t r = 0; r < rows; ++r) {
    const float *xr = &x[r * 2560];
    std::vector<float> logits(512);
    const uint16_t *rw = (const uint16_t *)L.router.host;
    float gpuLogitErr = 0;
    for (int e = 0; e < 512; ++e) {
      double a = 0; for (int k = 0; k < 2560; ++k) a += double(xr[k]) * bf2f(rw[e * 2560 + k]);
      logits[e] = rb(float(a));
      gpuLogitErr = std::max(gpuLogitErr, std::fabs(logits[e] - bf2f(((uint16_t *)b.logits.contents)[r * 512 + e])));
    }
    float peak = *std::max_element(logits.begin(), logits.end());
    double tot = 0; std::vector<float> pr(512);
    for (int e = 0; e < 512; ++e) { pr[e] = std::exp(logits[e] - peak); tot += pr[e]; }
    for (int e = 0; e < 512; ++e) pr[e] = rb(float(pr[e] / tot));
    std::vector<int> idx(512); for (int i = 0; i < 512; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int bb) { return pr[a] > pr[bb]; });
    float selSum = 0; for (int s = 0; s < 10; ++s) selSum = rb(selSum + pr[idx[s]]);
    std::vector<int> gpuSel(10);
    for (int s = 0; s < 10; ++s) gpuSel[s] = planRouteExpert[r * 10 + s];
    for (int s = 0; s < 10; ++s) if (gpuSel[s] != idx[s]) routeMismatch++;
    // shared expert
    std::vector<float> sinter(640);
    for (int n = 0; n < 640; ++n) {
      const float g = rb(float(affineDot(L.sg, 0, n, xr))), u = rb(float(affineDot(L.su, 0, n, xr)));
      const float sg = rb(1.f / (1.f + std::exp(-g)));
      sinter[n] = rb(rb(g * sg) * u);
    }
    const float slog = rb(float(affineDot(L.sl, 0, 0, xr)));
    std::vector<float> branch(2560, 0.f);
    std::vector<std::vector<float>> down(10, std::vector<float>(2560));
    for (int s = 0; s < 10; ++s) {
      const int ex = gpuSel[s];
      std::vector<float> inter(640);
      for (int n = 0; n < 640; ++n) {
        const float g = rb(float(affineDot(L.g, ex, n, xr))), u = rb(float(affineDot(L.u, ex, n, xr)));
        inter[n] = rb(rb(g * rb(sigm(g))) * u);
      }
      for (int n = 0; n < 2560; ++n) down[s][n] = rb(float(affineDot(L.d, ex, n, inter.data())));
    }
    const float sscale = rb(sigm(slog));
    for (int n = 0; n < 2560; ++n) {
      float routed = 0;
      for (int s = 0; s < 10; ++s) routed = rb(routed + rb(down[s][n] * rb(pr[gpuSel[s]] / selSum)));
      const float sd = rb(float(affineDot(L.sd, 0, n, sinter.data())));
      branch[n] = rb(routed + rb(sd * sscale));
    }
    for (int st = 0; st < 4; ++st) {
      double ss = 0; std::vector<float> vals(2560);
      for (int n = 0; n < 2560; ++n) {
        vals[n] = rb(hyper[(r * 4 + st) * 2560 + n] + rb(branch[n] * gates[r * 4 + st]));
        ss += double(vals[n]) * vals[n];
        const float got = bf2f(((uint16_t *)b.hyper.contents)[(r * 4 + st) * 2560 + n]);
        maxErr = std::max(maxErr, double(std::fabs(got - vals[n])));
        maxRef = std::max(maxRef, double(std::fabs(vals[n] - hyper[(r * 4 + st) * 2560 + n])));
        const double d = got - vals[n]; sumErrSq += d * d;
        const double inj = vals[n] - hyper[(r * 4 + st) * 2560 + n]; sumSq += inj * inj;
      }
      const float inv = 1.f / std::sqrt(float(ss / 2560) + 1e-6f);
      const uint16_t *nw = (const uint16_t *)L.norm.host;
      for (int n = 0; n < 2560; ++n) {
        const float ref = rb(vals[n] * inv * (1.f + bf2f(nw[st * 2560 + n])));
        const float got = bf2f(((uint16_t *)b.normalized.contents)[(r * 4 + st) * 2560 + n]);
        maxNormErr = std::max(maxNormErr, double(std::fabs(got - ref)));
      }
    }
    printf("row %u: logits max err %.3g, gpu top10 %d %d %d .. cpu %d %d %d, sgate %.4f\n", r, gpuLogitErr, gpuSel[0], gpuSel[1],
           gpuSel[2], idx[0], idx[1], idx[2], slog);
  }
  printf("route mismatches %d; hyper max err %.4g (max injection %.4g) rel-rms err %.4g; normalized max err %.4g\n",
         routeMismatch, maxErr, maxRef, std::sqrt(sumErrSq / std::max(sumSq, 1e-30)), maxNormErr);
  // Timing: all 48 layers in one command buffer, per kernel and fused.
  std::vector<MoELayer> layers;
  for (int l = 0; l < 48; ++l) layers.push_back(loadMoE(m, l));
  const char *names[5] = {"router", "plan", "gate_up", "down", "combine"};
  for (int only = -1; only < 5; ++only) {
    double best = 1e9;
    for (int t = 0; t < 4; ++t) {
      id<MTLCommandBuffer> cb = [c.q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      for (int l = 0; l < 48; ++l) encodeMoE(c, e, layers[l], b, rows, only);
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if (t) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
    }
    printf("timing %-8s %8.1f us/layer (48 layers %.3f ms)\n", only < 0 ? "all" : names[only], best * 1e6 / 48, best * 1e3);
  }
  return routeMismatch ? 1 : 0;
}

// ---- Dense projections: worker SIMD vs matrix-unit kernels on real weights.
struct QmvParams { uint32_t K, N, rows, xStride, yStride, row0; uint64_t wRow, pRowBytes; };
static int testDense(Ctx &c, Model &m, const std::string &which, uint32_t rows) {
  std::vector<Affine> mats;
  for (int l = 0; l < 48; ++l) {
    const std::string pre = "language_model.model.layers." + std::to_string(l);
    const bool gdn = (l + 1) % 4 != 0;
    std::string name;
    if (which == "qkv" && gdn) name = pre + ".linear_attn.in_proj_qkv";
    if (which == "z" && gdn) name = pre + ".linear_attn.in_proj_z";
    if (which == "out" && gdn) name = pre + ".linear_attn.out_proj";
    if (which == "q" && !gdn) name = pre + ".self_attn.q_proj";
    if (which == "o" && !gdn) name = pre + ".self_attn.o_proj";
    if (!name.empty()) mats.push_back(m.affine(name));
  }
  std::mt19937 rng(5); std::normal_distribution<float> nd(0.f, 1.f);
  const uint32_t Kmax = 6144, Nmax = 12288;
  id<MTLBuffer> x = c.buf(16 * Kmax * 2), y = c.buf(16 * Nmax * 2);
  for (uint32_t i = 0; i < 16 * Kmax; ++i) ((uint16_t *)x.contents)[i] = f2bf(nd(rng) * 0.5f);
  std::map<std::string, int> formats;
  for (auto &a : mats) formats["Q" + std::to_string(a.bits) + "G" + std::to_string(a.group)]++;
  printf("%s: %zu matrices", which.c_str(), mats.size());
  for (auto &f : formats) printf(" %s x%d", f.first.c_str(), f.second);
  printf("\n");
  auto run = [&](const char *label, auto pick) {
    double best = 1e9; int count = 0; double bytes = 0;
    for (int t = 0; t < 4; ++t) {
      id<MTLCommandBuffer> cb = [c.q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      count = 0; bytes = 0;
      for (auto &a : mats) {
        std::string pipeline; MTLSize groups, threads;
        if (!pick(a, pipeline, groups, threads)) continue;
        QmvParams p{a.K, a.N, rows, a.K, a.N, 0, a.wRow, a.pRow * 2};
        dispatch(c, e, pipeline, {B(x), B(a.w), B(a.s), B(a.b), B(y)}, &p, sizeof p, groups, threads);
        ++count; bytes += a.w.length + a.s.length + a.b.length;
      }
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if (t) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
    }
    if (count) printf("  %-44s %2d calls %7.2f us/call %6.1f GB/s\n", label, count, best * 1e6 / count, bytes / best / 1e9);
  };
  char name[128];
  if (getenv("MK_SWEEP")) {
    const int variants[][4] = {{4, 2, 1, 16}, {4, 1, 4, 16}, {4, 1, 8, 16}, {4, 2, 1, 8}, {4, 1, 4, 8}, {4, 1, 8, 8},
                               {2, 4, 1, 16}, {8, 2, 1, 16}, {4, 4, 1, 16}, {4, 2, 2, 16}, {4, 1, 16, 8}, {4, 1, 16, 16},
                               {4, 1, 2, 8}, {4, 1, 2, 16}, {2, 1, 8, 8}, {2, 1, 16, 8}, {4, 4, 2, 8}};
    for (auto &v : variants) {
      snprintf(name, sizeof name, "SIMD r%u n%d sn%d sk%d v%d (5/6-bit only)", rows, v[0], v[1], v[2], v[3]);
      run(name, [&](const Affine &a, std::string &pl, MTLSize &g, MTLSize &t) {
        if (a.bits != 5 && a.bits != 6) return false;
        pl = "opt_qmv_b" + std::to_string(a.bits) + "_g" + std::to_string(a.group) + "_r" + std::to_string(rows) + "_n" +
             std::to_string(v[0]) + "_sn" + std::to_string(v[1]) + "_sk" + std::to_string(v[2]) + "_v" + std::to_string(v[3]);
        const uint32_t per = v[0] * v[1];
        g = MTLSizeMake((a.N + per - 1) / per, 1, 1); t = MTLSizeMake(32 * v[1] * v[2], 1, 1); return true; });
    }
  }
  snprintf(name, sizeof name, "SIMD opt_qmv r%u n2 sn1 sk8 v8", rows);
  run(name, [&](const Affine &a, std::string &pl, MTLSize &g, MTLSize &t) {
    pl = "opt_qmv_b" + std::to_string(a.bits) + "_g" + std::to_string(a.group) + "_r" + std::to_string(rows) + "_n2_sn1_sk8_v8";
    g = MTLSizeMake((a.N + 1) / 2, 1, 1); t = MTLSizeMake(256, 1, 1); return true; });
  for (int sk : {4, 8}) {
    snprintf(name, sizeof name, "MPP opt_mppq nt32 sk%d (4/8-bit only)", sk);
    run(name, [&](const Affine &a, std::string &pl, MTLSize &g, MTLSize &t) {
      if (!((a.bits == 4 && a.group == 64) || a.bits == 8)) return false;
      pl = "opt_mppq_b" + std::to_string(a.bits) + "_g" + std::to_string(a.group) + "_nt32_sk" + std::to_string(sk);
      g = MTLSizeMake(a.N / 32, 1, 1); t = MTLSizeMake(32 * sk, 1, 1); return true; });
  }
  return 0;
}

// ---- Prefill MoE grouped GEMM over bucketed routes (2048 tokens x top-10).
struct FusedP { uint32_t rows, selections, in, out, experts, r0, r1, r2; uint64_t g[4], u[4]; };
struct GateP { FusedP a; uint32_t routeCap, jobCap, tileRows, reserved; };
struct DownFusedP { uint32_t rows, selections, in, out, experts, r0, r1, r2; uint64_t w[4]; };
struct DownP { DownFusedP a; uint32_t routeCap, jobCap, tileRows, reserved; };
static_assert(sizeof(GateP) == 112 && sizeof(DownP) == 80);

static int testPrefillMoE(Ctx &c, Model &m, bool skewed) {
  const uint32_t T = 2048, S = 10, R = T * S, E = 512;
  std::mt19937 rng(21);
  // Expert popularity: random permutation with a Zipf-like weight when skewed.
  std::vector<double> weight(E);
  std::vector<uint32_t> perm(E); for (uint32_t i = 0; i < E; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);
  for (uint32_t i = 0; i < E; ++i) weight[perm[i]] = skewed ? 1.0 / std::pow(i + 8.0, 0.9) : 1.0;
  std::vector<uint32_t> counts(E, 0);
  std::discrete_distribution<uint32_t> pick(weight.begin(), weight.end());
  for (uint32_t t = 0; t < T; ++t) {
    std::vector<uint32_t> chosen;
    while (chosen.size() < S) { const uint32_t e = pick(rng); if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) chosen.push_back(e); }
    for (uint32_t e : chosen) counts[e]++;
  }
  std::vector<uint32_t> offsets(E + 1, 0);
  for (uint32_t e = 0; e < E; ++e) offsets[e + 1] = offsets[e] + counts[e];
  const uint32_t maxc = *std::max_element(counts.begin(), counts.end()), minc = *std::min_element(counts.begin(), counts.end());
  printf("routing %s: counts min %u max %u mean %.1f\n", skewed ? "skewed" : "uniform", minc, maxc, R / double(E));
  auto buildJobs = [&](uint32_t tile, std::vector<uint32_t> &jobs) {
    jobs.clear();
    for (uint32_t e = 0; e < E; ++e)
      for (uint32_t r = 0; r < counts[e]; r += tile) { jobs.push_back(e); jobs.push_back(offsets[e] + r); }
    return uint32_t(jobs.size() / 2);
  };
  std::normal_distribution<float> nd(0.f, 1.f);
  id<MTLBuffer> packed = c.buf(uint64_t(R + 64) * 2560 * 2);
  for (uint64_t i = 0; i < uint64_t(R + 64) * 2560; ++i) ((uint16_t *)packed.contents)[i] = f2bf(i < uint64_t(R) * 2560 ? nd(rng) * 0.6f : 0.f);
  id<MTLBuffer> act = c.buf(uint64_t(R + 64) * 640 * 2);
  for (uint64_t i = 0; i < uint64_t(R + 64) * 640; ++i) ((uint16_t *)act.contents)[i] = f2bf(i < uint64_t(R) * 640 ? nd(rng) * 0.1f : 0.f);
  id<MTLBuffer> down = c.buf(uint64_t(R) * 2560 * 2);
  id<MTLBuffer> offs = c.buf((E + 1) * 4); memcpy(offs.contents, offsets.data(), (E + 1) * 4);
  id<MTLBuffer> map = c.buf(R * 4); for (uint32_t i = 0; i < R; ++i) ((uint32_t *)map.contents)[i] = i;
  id<MTLBuffer> diag = c.buf(16);
  id<MTLBuffer> sums = c.buf(uint64_t(R + 64) * 40 * 4);
  std::vector<std::array<Affine, 3>> L;
  for (int l = 0; l < 48; ++l) {
    const std::string mlp = "language_model.model.layers." + std::to_string(l) + ".mlp.switch_mlp";
    L.push_back({m.affine(mlp + ".gate_proj"), m.affine(mlp + ".up_proj"), m.affine(mlp + ".down_proj")});
  }
  struct Variant { const char *name; uint32_t tile, gx, threads; bool gate; };
  const Variant variants[] = {
      {"mk_moe_prefill_gate_up_pipe", 64, 10, 256, true}, {"mk_moe_prefill_down_pipe", 64, 20, 256, false},
      {"mk_abl_nofetch", 64, 10, 256, true}, {"mk_abl_nodequant", 64, 10, 256, true}, {"mk_abl_nomma", 64, 10, 256, true},
      {"mk_abl_mmaonly", 64, 10, 256, true}, {"mk_abl_nothing", 64, 10, 256, true},
      {"mk_moe_prefill_gate_up", 64, 10, 256, true}, {"mk_moe_prefill_gate_up_vec", 64, 10, 256, true},
      {"mk_moe_prefill_gate_up_nostage", 64, 10, 256, true}, {"mk_moe_prefill_gate_up_bk32", 64, 10, 256, true},
      {"mk_moe_prefill_gate_up_bk32a", 64, 10, 256, true}, {"mk_moe_prefill_gate_up_bk32a_nostage", 64, 10, 256, true},
      {"opt_moe_q4_gate_up_m32_n64", 32, 10, 128, true},
      {"opt_moe_q4_gate_up_m64_n64_sg8", 64, 10, 256, true}, {"opt_moe_q4d_gate_up_m64_n64_sg8", 64, 10, 256, true},
      {"mk_moe_prefill_down", 64, 20, 256, false}, {"opt_moe_q4_down_scatter_m32_n64", 32, 40, 128, false},
      {"opt_moe_q4_down_scatter_m64_n64_sg8", 64, 40, 256, false}, {"opt_moe_q4d_down_scatter_m64_n64_sg8", 64, 40, 256, false},
  };
  for (const auto &v : variants) {
    if (getenv("MK_ONLY") && !strstr(v.name, getenv("MK_ONLY"))) continue;
    std::vector<uint32_t> jobs;
    const uint32_t njobs = buildJobs(v.tile, jobs);
    const uint32_t jobCap = (R + v.tile - 1) / v.tile + 511;
    id<MTLBuffer> jb = c.buf(uint64_t(jobCap) * 8);
    memset(jb.contents, 0xff, jobCap * 8); memcpy(jb.contents, jobs.data(), jobs.size() * 4);
    id<MTLBuffer> jc = c.buf(16); ((uint32_t *)jc.contents)[0] = njobs;
    const bool direct = strstr(v.name, "q4d") != nullptr;
    double best = 1e9;
    for (int t = 0; t < 3; ++t) {
      id<MTLCommandBuffer> cb = [c.q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      for (int l = 0; l < 48; ++l) {
        const Affine &g = L[l][0], &u = L[l][1], &d = L[l][2];
        if (v.gate) {
          GateP p{{T, S, 2560, 640, E, 0, 0, 0, {g.wRow, g.wExpert, g.pRow * 2, g.pExpert * 2}, {u.wRow, u.wExpert, u.pRow * 2, u.pExpert * 2}},
                  R, jobCap, v.tile, 0};
          dispatch(c, e, v.name, {B(packed), B(g.w), B(g.s), B(g.b), B(u.w), B(u.s), B(u.b), B(offs), B(jb), B(jc), B(act),
                                  direct ? B(sums) : B(diag)}, &p, sizeof p, MTLSizeMake(v.gx, getenv("MK_EXACT") ? njobs : jobCap, 1), MTLSizeMake(v.threads, 1, 1));
        } else {
          DownP p{{T, S, 640, 2560, E, 0, 0, 0, {d.wRow, d.wExpert, d.pRow * 2, d.pExpert * 2}}, R, jobCap, v.tile, 0};
          dispatch(c, e, v.name, {B(act), B(d.w), B(d.s), B(d.b), B(offs), B(jb), B(jc), B(map), B(down),
                                  direct ? B(sums) : B(diag)}, &p, sizeof p, MTLSizeMake(v.gx, getenv("MK_EXACT") ? njobs : jobCap, 1), MTLSizeMake(v.threads, 1, 1));
        }
      }
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if (cb.error) { printf("%s GPU error %s\n", v.name, cb.error.localizedDescription.UTF8String); break; }
      if (t) best = std::min(best, cb.GPUEndTime - cb.GPUStartTime);
    }
    // Check layer 47 outputs for a few rows (gate_up: act; down: down[route]).
    double maxErr = 0, maxRef = 0;
    const Affine &g = L[47][0], &u = L[47][1], &d = L[47][2];
    for (int trial = 0; trial < (direct ? 0 : 24); ++trial) {
      const uint32_t row = rng() % R;
      uint32_t ex = 0; while (offsets[ex + 1] <= row) ++ex;
      if (v.gate) {
        std::vector<float> x(2560);
        for (int k = 0; k < 2560; ++k) x[k] = bf2f(((uint16_t *)packed.contents)[uint64_t(row) * 2560 + k]);
        const uint32_t n = rng() % 640;
        const float gv = rb(float(affineDot(g, ex, n, x.data()))), uv = rb(float(affineDot(u, ex, n, x.data())));
        const float ref = rb(rb(gv * rb(1.f / (1.f + std::exp(-gv)))) * uv);
        const float got = bf2f(((uint16_t *)act.contents)[uint64_t(row) * 640 + n]);
        maxErr = std::max(maxErr, double(std::fabs(got - ref))); maxRef = std::max(maxRef, double(std::fabs(ref)));
      } else {
        std::vector<float> x(640);
        for (int k = 0; k < 640; ++k) x[k] = bf2f(((uint16_t *)act.contents)[uint64_t(row) * 640 + k]);
        const uint32_t n = rng() % 2560;
        const float ref = float(affineDot(d, ex, n, x.data()));
        const float got = bf2f(((uint16_t *)down.contents)[uint64_t(row) * 2560 + n]);
        maxErr = std::max(maxErr, double(std::fabs(got - ref))); maxRef = std::max(maxRef, double(std::fabs(ref)));
      }
    }
    const double flops = 2.0 * R * 2560 * 640 * (v.gate ? 2 : 1);
    printf("%-40s jobs %4u  %7.3f ms/layer  %6.1f TFLOPS  relerr %.2e\n", v.name, njobs, best * 1e3 / 48,
           flops * 48 / best / 1e12, maxErr / std::max(maxRef, 1e-9));
  }
  return 0;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    Ctx c;
    c.dev = MTLCreateSystemDefaultDevice();
    c.q = [c.dev newCommandQueue];
    NSError *err = nil;
    c.lib = [c.dev newLibraryWithURL:[NSURL fileURLWithPath:@(kLib)] error:&err];
    if (!c.lib) { fprintf(stderr, "lib: %s\n", err.localizedDescription.UTF8String); return 1; }
    Model m(c.dev);
    const std::string test = argc > 1 ? argv[1] : "moe";
    if (test == "moe") return testMoE(c, m, argc > 2 ? atoi(argv[2]) : 4, argc > 3 ? atoi(argv[3]) : 5);
    if (test == "dense") return testDense(c, m, argc > 2 ? argv[2] : "qkv", argc > 3 ? atoi(argv[3]) : 5);
    if (test == "pmoe") return testPrefillMoE(c, m, argc > 2 && std::string(argv[2]) == "skewed");
    fprintf(stderr, "unknown test\n");
    return 2;
  }
}
