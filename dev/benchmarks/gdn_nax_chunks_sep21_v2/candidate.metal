// Private CPU-compiled experiment. Root is the sole GPU coordinator.
// Distinct from existing scalar triangular solve per value: prepare F32 W/U
// once per head/chunk, then apply state through SG4/8 device tensor products.
// BF16 q/k/v/beta and exposed output; F32 decay, transforms, and carried state.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "metal/abi/FlashGDN.h"
#pragma METAL fp math_mode(safe)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
using namespace metal;
using namespace mpp::tensor_ops;

inline void gwy_error(device atomic_uint &diagnostics, uint flag) {
  atomic_fetch_or_explicit(&diagnostics, flag, memory_order_relaxed);
}
inline bool gwy_valid(constant FlashGDNParams &p) {
  return p.rows && p.rows <= 2048 && p.lanes && p.lanes <= 32 &&
      p.key_heads == 16 && p.value_heads == 48 && p.key_dimension == 128 &&
      p.value_dimension == 128 && p.convolution_taps == 4 &&
      isfinite(p.norm_epsilon) && p.norm_epsilon > 0.0f &&
      p.convolution_lane_stride_bytes >= ulong(3) * 10240 * 2 &&
      !(p.convolution_lane_stride_bytes % 2) &&
      p.recurrent_lane_stride_bytes >= ulong(48) * 128 * 128 * 4 &&
      !(p.recurrent_lane_stride_bytes % 4);
}
template <ushort Time> constexpr uint gwy_stride() {
  return 3 * Time * 128 + Time * Time + Time;
}

template <ushort Time, ushort Groups>
inline void gwy_prepare(device const bfloat *mixed, device const float *decay,
    device const bfloat *beta, device float *prepared,
    device atomic_uint &diagnostics, constant FlashGDNParams &p,
    uint3 group, uint3 threads, uint tid,
    threadgroup float *gram, threadgroup float *inverse,
    threadgroup float *scaled, threadgroup float *alphas,
    threadgroup float *betas, threadgroup float *prefix) {
  const uint chunks = (p.rows + Time - 1) / Time;
  if (!gwy_valid(p) || threads.x != uint(Groups) * 32 ||
      threads.y != 1 || threads.z != 1 || group.x >= 48 ||
      group.y >= chunks || group.z >= p.lanes) {
    if (!tid) gwy_error(diagnostics, FlashGDNInvalidParameters);
    return;
  }
  const uint head = group.x, keyHead = head / 3, begin = group.y * Time;
  const uint batch = group.z, count = min(uint(Time), p.rows - begin);
  const ulong base = ((ulong(batch) * chunks + group.y) * 48 + head) * gwy_stride<Time>();
  const ulong source = (ulong(batch) * p.rows + begin) * 10240;
  for (uint i = tid; i < gwy_stride<Time>(); i += Groups * 32)
    prepared[base + i] = 0.0f;
  if (tid < Time) {
    const ulong gate = (ulong(batch) * p.rows + begin + tid) * 48 + head;
    alphas[tid] = tid < count ? decay[gate] : 1.0f;
    betas[tid] = tid < count ? float(beta[gate]) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  if (tid < Time) {
    float product = 1.0f;
    for (uint t = 0; t <= tid; ++t) product *= alphas[t];
    prefix[tid] = product;
    prepared[base + 3 * Time * 128 + Time * Time + tid] = tid < count ? product : 0.0f;
  }
  auto qt = tensor(const_cast<device bfloat *>(mixed) + source + keyHead * 128,
      dextents<int,2>{128, int(count)}, array<int,2>{1,10240});
  auto kt = tensor(const_cast<device bfloat *>(mixed) + source + 2048 + keyHead * 128,
      dextents<int,2>{128, int(count)}, array<int,2>{1,10240});
  auto vt = tensor(const_cast<device bfloat *>(mixed) + source + 4096 + head * 128,
      dextents<int,2>{128, int(count)}, array<int,2>{1,10240});
  auto q = qt.slice(0,0), k = kt.slice(0,0), v = vt.slice(0,0);
  constexpr auto gd = matmul2d_descriptor(Time,Time,128,false,true,false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<gd,execution_simdgroups<Groups>> gop;
  auto kk = gop.template get_destination_cooperative_tensor<decltype(k),decltype(k),float>();
  gop.run(k,k,kk);
#pragma unroll
  for (ushort i = 0; i < kk.get_capacity(); ++i) {
    if (!kk.is_valid_element(i)) continue;
    const auto ix = kk.get_multidimensional_index(i);
    gram[ix[1] * Time + ix[0]] = kk[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < Time * Time; i += Groups * 32) {
    const uint token = i / Time, previous = i % Time;
    float product = 1.0f;
    for (uint a = previous + 1; a <= token; ++a) product *= alphas[a];
    gram[i] = token < count && previous < token ?
        gram[i] * betas[token] * product : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // One thread owns each inverse column. Dependences never cross columns.
  if (tid < Time) {
    for (uint token = 0; token < Time; ++token) {
      float value = token == tid ? 1.0f : 0.0f;
      for (uint previous = 0; previous < token; ++previous)
        value -= gram[token * Time + previous] * inverse[previous * Time + tid];
      inverse[token * Time + tid] = value;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < Time * Time; i += Groups * 32)
    scaled[i] = inverse[i] * betas[i % Time] * prefix[i % Time];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto ft = tensor(scaled,dextents<int,2>{Time,Time},array<int,2>{1,Time});
  auto f = ft.template slice<Time,Time>(0,0);
  constexpr auto td = matmul2d_descriptor(Time,128,Time,false,false,false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<td,execution_simdgroups<Groups>> top;
  auto w = top.template get_destination_cooperative_tensor<decltype(f),decltype(k),float>();
  top.run(f,k,w);
#pragma unroll
  for (ushort i = 0; i < w.get_capacity(); ++i) {
    if (!w.is_valid_element(i)) continue;
    const auto ix = w.get_multidimensional_index(i);
    if (uint(ix[1]) < count) prepared[base + ix[1] * 128 + ix[0]] = w[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < Time * Time; i += Groups * 32)
    scaled[i] = inverse[i] * betas[i % Time];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto u = top.template get_destination_cooperative_tensor<decltype(f),decltype(v),float>();
  top.run(f,v,u);
#pragma unroll
  for (ushort i = 0; i < u.get_capacity(); ++i) {
    if (!u.is_valid_element(i)) continue;
    const auto ix = u.get_multidimensional_index(i);
    if (uint(ix[1]) < count) prepared[base + Time * 128 + ix[1] * 128 + ix[0]] = u[i];
  }
  auto qk = gop.template get_destination_cooperative_tensor<decltype(q),decltype(k),float>();
  gop.run(q,k,qk);
#pragma unroll
  for (ushort i = 0; i < qk.get_capacity(); ++i) {
    if (!qk.is_valid_element(i)) continue;
    const auto ix = qk.get_multidimensional_index(i);
    const uint token = ix[1], previous = ix[0];
    float product = 1.0f;
    for (uint a = previous + 1; a <= token; ++a) product *= alphas[a];
    if (token < count && previous <= token)
      prepared[base + 3 * Time * 128 + token * Time + previous] = qk[i] * product;
  }
  for (uint i = tid; i < Time * 128; i += Groups * 32) {
    const uint token = i / 128;
    if (token >= count) continue;
    float product = 1.0f;
    for (uint a = token + 1; a < count; ++a) product *= alphas[a];
    prepared[base + 2 * Time * 128 + i] = product * float(mixed[
        source + token * 10240 + 2048 + keyHead * 128 + i % 128]);
  }
}

template <ushort Values, ushort Time, ushort Groups, bool Audit>
inline void gwy_apply(device const bfloat *mixed, device const float *decay, device float *recurrent,
    device bfloat *output, device atomic_uint &diagnostics,
    constant FlashGDNParams &p, device const float *prepared,
    uint3 group, uint3 threads, uint tid, threadgroup float *delta,
    threadgroup float *stateQuery, device float *history,
    device float *deltaAudit, device float *outputAudit) {
  if (!gwy_valid(p) || threads.x != uint(Groups) * 32 ||
      threads.y != 1 || threads.z != 1 || group.x >= 48 ||
      group.y >= 128 / Values || group.z >= p.lanes || (Audit && group.x != 0)) {
    if (!tid) gwy_error(diagnostics,FlashGDNInvalidParameters);
    return;
  }
  const uint head = group.x, keyHead = head / 3, batch = group.z;
  const uint valueBegin = group.y * Values, chunks = (p.rows + Time - 1) / Time;
  const ulong stateBase = ulong(batch) * p.recurrent_lane_stride_bytes / 4 +
      (head * 128 + valueBegin) * 128;
  auto st = tensor(recurrent + stateBase,dextents<int,2>{128,Values},array<int,2>{1,128});
  auto s = st.template slice<128,Values>(0,0);
  auto dt = tensor(delta,dextents<int,2>{Time,Values},array<int,2>{1,Time});
  auto d = dt.template slice<Time,Values>(0,0);
  constexpr auto pd = matmul2d_descriptor(Values,Time,128,false,true,false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<pd,execution_simdgroups<Groups>> pop;
  constexpr auto od = matmul2d_descriptor(Values,Time,Time,false,true,false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<od,execution_simdgroups<Groups>> oop;
  constexpr auto ud = matmul2d_descriptor(Values,128,Time,false,false,false,
      matmul2d_descriptor::mode::multiply);
  matmul2d<ud,execution_simdgroups<Groups>> uop;
  for (uint begin = 0; begin < p.rows; begin += Time) {
    const uint count = min(uint(Time),p.rows-begin);
    const ulong base = ((ulong(batch) * chunks + begin / Time) * 48 + head) * gwy_stride<Time>();
    auto wt = tensor(const_cast<device float *>(prepared) + base,
        dextents<int,2>{128,Time},array<int,2>{1,128});
    auto w = wt.template slice<128,Time>(0,0);
    auto sw = pop.template get_destination_cooperative_tensor<decltype(s),decltype(w),float>();
    pop.run(s,w,sw);
#pragma unroll
    for (ushort i = 0; i < sw.get_capacity(); ++i) {
      if (!sw.is_valid_element(i)) continue;
      const auto ix = sw.get_multidimensional_index(i);
      delta[ix[1] * Time + ix[0]] = uint(ix[0]) < count ?
          prepared[base + Time * 128 + ix[0] * 128 + valueBegin + ix[1]] - sw[i] : 0.0f;
    }
    auto qt = tensor(const_cast<device bfloat *>(mixed) +
        (ulong(batch) * p.rows + begin) * 10240 + keyHead * 128,
        dextents<int,2>{128,int(count)},array<int,2>{1,10240});
    auto q = qt.slice(0,0);
    auto sq = pop.template get_destination_cooperative_tensor<decltype(s),decltype(q),float>();
    pop.run(s,q,sq);
#pragma unroll
    for (ushort i = 0; i < sq.get_capacity(); ++i) {
      if (!sq.is_valid_element(i)) continue;
      const auto ix = sq.get_multidimensional_index(i);
      stateQuery[ix[1] * Time + ix[0]] =
          prepared[base + 3 * Time * 128 + Time * Time + ix[0]] * sq[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto at = tensor(const_cast<device float *>(prepared) + base + 3 * Time * 128,
        dextents<int,2>{Time,Time},array<int,2>{1,Time});
    auto attention = at.template slice<Time,Time>(0,0);
    auto result = oop.template get_destination_cooperative_tensor<decltype(d),decltype(attention),float>();
    oop.run(d,attention,result);
#pragma unroll
    for (ushort i = 0; i < result.get_capacity(); ++i) {
      if (!result.is_valid_element(i)) continue;
      const auto ix = result.get_multidimensional_index(i);
      const uint token = ix[0], value = ix[1];
      if (token >= count) continue;
      const float y = stateQuery[value * Time + token] + result[i];
      output[(ulong(batch) * p.rows + begin + token) * 6144 + head * 128 + valueBegin + value] = bfloat(y);
      if (!isfinite(y)) gwy_error(diagnostics,FlashGDNNonFinite);
      if constexpr (Audit) {
        deltaAudit[(ulong(batch) * p.rows + begin + token) * 128 + valueBegin + value] = delta[value * Time + token];
        outputAudit[(ulong(batch) * p.rows + begin + token) * 128 + valueBegin + value] = y;
      }
    }
    if constexpr (Audit) {
      for (uint i = tid; i < Values * 128; i += Groups * 32) {
        const uint value = i / 128, dimension = i % 128;
        for (uint token = 0; token < count; ++token) {
          float y = prepared[base + 3 * Time * 128 + Time * Time + token] * recurrent[stateBase + i];
          for (uint previous = 0; previous <= token; ++previous) {
            float product = 1.0f;
            for (uint a = previous + 1; a <= token; ++a)
              product *= decay[(ulong(batch) * p.rows + begin + a) * 48 + head];
            y += product * delta[value * Time + previous] * float(mixed[
                (ulong(batch) * p.rows + begin + previous) * 10240 +
                2048 + keyHead * 128 + dimension]);
          }
          history[((ulong(batch) * p.rows + begin + token) * 128 + valueBegin + value) * 128 + dimension] = y;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto et = tensor(const_cast<device float *>(prepared) + base + 2 * Time * 128,
        dextents<int,2>{128,Time},array<int,2>{1,128});
    auto e = et.template slice<128,Time>(0,0);
    auto update = uop.template get_destination_cooperative_tensor<decltype(d),decltype(e),float>();
    uop.run(d,e,update);
    const float endPrefix = prepared[base + 3 * Time * 128 + Time * Time + count - 1];
#pragma unroll
    for (ushort i = 0; i < update.get_capacity(); ++i) {
      if (!update.is_valid_element(i)) continue;
      const auto ix = update.get_multidimensional_index(i);
      const uint item = ix[1] * 128 + ix[0];
      const float value = endPrefix * recurrent[stateBase + item] + update[i];
      recurrent[stateBase + item] = value;
      if (!isfinite(value)) gwy_error(diagnostics,FlashGDNNonFinite);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}

#define GWY_PREP_ENTRY(T,G) \
  [[max_total_threads_per_threadgroup(G*32)]] kernel void private_gdn_wy_prepare_t##T##_sg##G( \
      device const bfloat *mixed [[buffer(0)]], device const float *decay [[buffer(1)]], \
      device const bfloat *beta [[buffer(2)]], device float *prepared [[buffer(3)]], \
      device atomic_uint &diagnostics [[buffer(4)]], constant FlashGDNParams &p [[buffer(5)]], \
      uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
      uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup float gram[T*T], inverse[T*T], scaled[T*T], alphas[T], betas[T], prefix[T]; \
    gwy_prepare<T,G>(mixed,decay,beta,prepared,diagnostics,p,group,threads,tid, \
        gram,inverse,scaled,alphas,betas,prefix); \
  }
GWY_PREP_ENTRY(32,4)
GWY_PREP_ENTRY(32,8)
#undef GWY_PREP_ENTRY
#define GWY_APPLY_ARGS \
    device const bfloat *mixed [[buffer(0)]], device const float *decay [[buffer(1)]], \
    device const bfloat *beta [[buffer(2)]], device float *recurrent [[buffer(3)]], \
    device bfloat *output [[buffer(4)]], device atomic_uint &diagnostics [[buffer(5)]], \
    constant FlashGDNParams &p [[buffer(6)]], device const float *prepared [[buffer(10)]]
#define GWY_THREADS \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]
#define GWY_APPLY_ENTRY(V,T,G) \
  [[max_total_threads_per_threadgroup(G*32)]] kernel void private_gdn_wy_v##V##_t##T##_sg##G( \
      GWY_APPLY_ARGS, GWY_THREADS) { \
    (void)beta; \
    threadgroup float delta[V*T], stateQuery[V*T]; \
    gwy_apply<V,T,G,false>(mixed,decay,recurrent,output,diagnostics,p,prepared,group,threads,tid, \
        delta,stateQuery,nullptr,nullptr,nullptr); \
  } \
  [[max_total_threads_per_threadgroup(G*32)]] kernel void private_gdn_wy_audit_v##V##_t##T##_sg##G( \
      GWY_APPLY_ARGS, device float *history [[buffer(7)]], device float *deltaAudit [[buffer(8)]], \
      device float *outputAudit [[buffer(9)]], GWY_THREADS) { \
    (void)beta; \
    threadgroup float delta[V*T], stateQuery[V*T]; \
    gwy_apply<V,T,G,true>(mixed,decay,recurrent,output,diagnostics,p,prepared,group,threads,tid, \
        delta,stateQuery,history,deltaAudit,outputAudit); \
  }
GWY_APPLY_ENTRY(32,32,4)
GWY_APPLY_ENTRY(32,32,8)
#undef GWY_APPLY_ENTRY
#undef GWY_THREADS
#undef GWY_APPLY_ARGS
