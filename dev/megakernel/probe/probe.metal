#include <metal_stdlib>
using namespace metal;

// Streaming read bandwidth: each thread reads uint4 with a grid stride.
kernel void bw_read(device const uint4 *src [[buffer(0)]],
                    device uint *out [[buffer(1)]],
                    constant uint &count [[buffer(2)]],
                    uint gid [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]]) {
  uint4 acc = 0;
  for (uint i = gid; i < count; i += grid) acc ^= src[i];
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) out[0] = 1;
}

// Chunked read: each threadgroup reads a contiguous chunk (like a GEMV row block).
kernel void bw_read_chunk(device const uint4 *src [[buffer(0)]],
                          device uint *out [[buffer(1)]],
                          constant uint &count [[buffer(2)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint ntg [[threadgroups_per_grid]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint tsz [[threads_per_threadgroup]]) {
  const uint per = (count + ntg - 1) / ntg;
  const uint begin = tg * per;
  const uint end = min(count, begin + per);
  uint4 acc = 0;
  for (uint i = begin + lid; i < end; i += tsz) acc ^= src[i];
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) out[0] = 1;
}

// Every threadgroup reads the same `count` uint4 (broadcast activation reads).
kernel void bw_bcast(device const uint4 *src [[buffer(0)]],
                     device uint *out [[buffer(1)]],
                     constant uint &count [[buffer(2)]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint tsz [[threads_per_threadgroup]]) {
  uint4 acc = 0;
  for (uint i = lid; i < count; i += tsz) acc ^= src[i];
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) out[0] = 1;
}

kernel void tiny(device float *x [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  x[tid & 65535] = x[tid & 65535] * 1.0001f + 1.0f;
}

// Dependent phase chain as separate dispatches: each dispatch reads the
// previous dispatch's vector and writes a new one (checks visibility).
kernel void chain_step(device const uint *prev [[buffer(0)]],
                       device uint *next [[buffer(1)]],
                       constant uint &n [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid < n) next[gid] = prev[(gid * 7 + 1) % n] + 1;
}

struct GridBarrier {
  device atomic_uint *counter;
  uint groups;
  uint epoch;
};

inline bool grid_sync(device atomic_uint *counter, uint groups, thread uint &epoch,
                      uint lid, threadgroup uint *flag) {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  epoch += 1;
  if (lid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    const uint target = epoch * groups;
    uint spins = 0;
    bool ok = true;
    while (atomic_load_explicit(counter, memory_order_relaxed) < target) {
      if (++spins > (1u << 26)) { ok = false; break; }
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    flag[0] = ok ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  return flag[0] != 0;
}

// Persistent kernel: `phases` dependent phases separated by grid barriers.
// Phase p: every thread writes buf[(p+1)&1][i] = buf[p&1][perm(i)] + 1.
kernel void persistent_chain(device uint *buf [[buffer(0)]],
                             device atomic_uint *counter [[buffer(1)]],
                             constant uint &n [[buffer(2)]],
                             constant uint &phases [[buffer(3)]],
                             device uint *status [[buffer(4)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint ntg [[threadgroups_per_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint tsz [[threads_per_threadgroup]]) {
  threadgroup uint flag[1];
  uint epoch = 0;
  const uint stride = ntg * tsz;
  for (uint p = 0; p < phases; ++p) {
    device const uint *prev = buf + (p & 1) * n;
    device uint *next = buf + ((p + 1) & 1) * n;
    for (uint i = tg * tsz + lid; i < n; i += stride) next[i] = prev[(i * 7 + 1) % n] + 1;
    if (!grid_sync(counter + 1, ntg, epoch, lid, flag)) {
      if (lid == 0) atomic_fetch_add_explicit((device atomic_uint *)status, 1u, memory_order_relaxed);
      return;
    }
  }
}

// Last-arriver release barrier: counter[1] arrivals, counter[2] release epoch.
// All threads fence after release so no core keeps stale device lines.
inline bool grid_sync2(device atomic_uint *counter, uint groups, thread uint &epoch,
                       uint lid, threadgroup uint *flag, bool all_fence) {
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  epoch += 1;
  if (lid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    const uint ticket = atomic_fetch_add_explicit(counter + 1, 1u, memory_order_relaxed);
    bool ok = true;
    if (ticket + 1 == epoch * groups) {
      atomic_store_explicit(counter + 2, epoch, memory_order_relaxed);
    } else {
      uint spins = 0;
      while (atomic_load_explicit(counter + 2, memory_order_relaxed) < epoch) {
        if (++spins > (1u << 26)) { ok = false; break; }
      }
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    flag[0] = ok ? 1u : 0u;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (all_fence) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return flag[0] != 0;
}

kernel void persistent_chain2(device uint *buf [[buffer(0)]],
                              device atomic_uint *counter [[buffer(1)]],
                              constant uint &n [[buffer(2)]],
                              constant uint &phases [[buffer(3)]],
                              device uint *status [[buffer(4)]],
                              constant uint &all_fence [[buffer(5)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint ntg [[threadgroups_per_grid]],
                              uint lid [[thread_position_in_threadgroup]],
                              uint tsz [[threads_per_threadgroup]]) {
  threadgroup uint flag[1];
  uint epoch = 0;
  const uint stride = ntg * tsz;
  for (uint p = 0; p < phases; ++p) {
    device const uint *prev = buf + (p & 1) * n;
    device uint *next = buf + ((p + 1) & 1) * n;
    for (uint i = tg * tsz + lid; i < n; i += stride) next[i] = prev[(i * 7 + 1) % n] + 1;
    if (!grid_sync2(counter, ntg, epoch, lid, flag, all_fence != 0)) {
      if (lid == 0) atomic_fetch_add_explicit((device atomic_uint *)status, 1u, memory_order_relaxed);
      return;
    }
  }
}

// Same, but data accessed through atomics-free volatile-coherent path is not
// available; instead read prev via relaxed atomic loads (bypassing L1).
kernel void persistent_chain3(device atomic_uint *buf [[buffer(0)]],
                              device atomic_uint *counter [[buffer(1)]],
                              constant uint &n [[buffer(2)]],
                              constant uint &phases [[buffer(3)]],
                              device uint *status [[buffer(4)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint ntg [[threadgroups_per_grid]],
                              uint lid [[thread_position_in_threadgroup]],
                              uint tsz [[threads_per_threadgroup]]) {
  threadgroup uint flag[1];
  uint epoch = 0;
  const uint stride = ntg * tsz;
  for (uint p = 0; p < phases; ++p) {
    device atomic_uint *prev = buf + (p & 1) * n;
    device atomic_uint *next = buf + ((p + 1) & 1) * n;
    for (uint i = tg * tsz + lid; i < n; i += stride)
      atomic_store_explicit(next + i, atomic_load_explicit(prev + (i * 7 + 1) % n, memory_order_relaxed) + 1,
                            memory_order_relaxed);
    if (!grid_sync2(counter, ntg, epoch, lid, flag, false)) {
      if (lid == 0) atomic_fetch_add_explicit((device atomic_uint *)status, 1u, memory_order_relaxed);
      return;
    }
  }
}

kernel void persistent_chain4(device coherent(device) uint *buf [[buffer(0)]],
                              device atomic_uint *counter [[buffer(1)]],
                              constant uint &n [[buffer(2)]],
                              constant uint &phases [[buffer(3)]],
                              device uint *status [[buffer(4)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint ntg [[threadgroups_per_grid]],
                              uint lid [[thread_position_in_threadgroup]],
                              uint tsz [[threads_per_threadgroup]]) {
  threadgroup uint flag[1];
  uint epoch = 0;
  const uint stride = ntg * tsz;
  for (uint p = 0; p < phases; ++p) {
    device coherent(device) uint *prev = buf + (p & 1) * n;
    device coherent(device) uint *next = buf + ((p + 1) & 1) * n;
    for (uint i = tg * tsz + lid; i < n; i += stride) next[i] = prev[(i * 7 + 1) % n] + 1;
    if (!grid_sync2(counter, ntg, epoch, lid, flag, false)) {
      if (lid == 0) atomic_fetch_add_explicit((device atomic_uint *)status, 1u, memory_order_relaxed);
      return;
    }
  }
}

// Barrier-only persistent loop (no data).
kernel void persistent_barrier(device atomic_uint *counter [[buffer(1)]],
                               constant uint &phases [[buffer(3)]],
                               device uint *status [[buffer(4)]],
                               uint ntg [[threadgroups_per_grid]],
                               uint lid [[thread_position_in_threadgroup]]) {
  threadgroup uint flag[1];
  uint epoch = 0;
  for (uint p = 0; p < phases; ++p) {
    if (!grid_sync(counter + 1, ntg, epoch, lid, flag)) {
      if (lid == 0) atomic_fetch_add_explicit((device atomic_uint *)status, 1u, memory_order_relaxed);
      return;
    }
  }
}
