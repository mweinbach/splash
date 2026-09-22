#!/usr/bin/env python3
"""Generate private W8A8 numerical-activation-alternative MPP shaders.

Reads source code only; no GPU, model payload, activation or route-ID reads.
"""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'dev/benchmarks/prefill_moe_sep21/memory.metal'
DESTINATION = Path(__file__).with_name('candidate.metal')
HEADER = '''// Private W8A8 numerical activation alternative; not original-BF16-A math.
// Original persisted signed-I8 B and F32 late row scales remain unchanged.
// GPU row quantizers create new I8 A and per-row F32 activation scales;
// source BF16 inputs are never mutated. Include both quantizer commands in
// complete-chain timings. Decoder kernels and production sources are untouched.
// Quantizer ABI: buffer0 BF16 source [rows,K], 1 I8 A [rows,K], 2 F32 scales
// [rows], 3 diagnostics, 4 uint4{rows_including_padding,K,0,0}; one TG per row.
// Gate quantizer uses 256 threads/K2560; down uses 128 threads/K640. Up to
// original maximum routes+63 rows are allowed; zero padding has code0/scale1.
// Producers retain the native gate/down buffer ABI except input A is I8;
// activation scales are appended at gate12/down11. Timed kernels bind no
// audit buffers. Separate _audit kernels append scaled F32 G13/U14, D12,
// exact raw I32 G15/U16, D13. G/U are packed [route,640], D is canonical
// scattered [route,2560]. Audit commands are excluded from timing.
// I8 x I8 accumulates I32. Because B may contain -128, worst |dot| bounds are
// 127*128*2560=41,615,360 and 127*128*640=10,403,840, safely below INT32_MAX.
// Epilogues explicitly compute (float(intdot)*activationScale)*weightScale
// with contraction/reassociation disabled, then original BF16/SwiGLU stages.
'''
QUANTIZERS = r'''
// BF16 nonfinite values are sanitized to zero in the new quantized tensor.
// No nonfinite value participates in the maximum or conversion to integer.
template <ushort K, ushort Threads>
inline void prefill_moe_sep21_w8a8_quantize(
    device const bfloat *source, device int8_t *codes,
    device float *scales, device uint *diag, constant uint4 &p,
    uint3 group, uint3 threads, uint tid, threadgroup float *maxima) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
  constexpr uint MaximumRows = kFlashMoEBucketMaximumRows *
      kFlashMoEBucketMaximumSelections + 63;
  if (!p.x || p.x > MaximumRows || p.y != K || p.z || p.w ||
      group.x >= p.x || group.y || group.z || threads.x != Threads ||
      threads.y != 1 || threads.z != 1) {
    if (!tid) flash_mpp_error(diag, 2u); return;
  }
  const ulong origin = ulong(group.x) * K;
  float local_maximum = 0.0f;
  bool nonfinite = false;
  for (uint k = tid; k < K; k += Threads) {
    const float value = float(source[origin + k]);
    if (flash_mpp_finite(value)) local_maximum = max(local_maximum, abs(value));
    else nonfinite = true;
  }
  if (nonfinite) flash_mpp_error(diag, 4u);
  const float simd_maximum = simd_max(local_maximum);
  if (!(tid & 31u)) maxima[tid / 32] = simd_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float maximum = 0.0f;
#pragma unroll
  for (ushort i = 0; i < Threads / 32; ++i) maximum = max(maximum, maxima[i]);
  float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
  if (!(scale > 0.0f) || !flash_mpp_finite(scale)) {
    if (!tid) flash_mpp_error(diag, 4u);
    scale = 1.0f;
  }
  if (!tid) scales[group.x] = scale;
  for (uint k = tid; k < K; k += Threads) {
    const float value = float(source[origin + k]);
    const float sanitized = flash_mpp_finite(value) ? value : 0.0f;
    const float rounded = metal::rint(sanitized / scale);
    codes[origin + k] = int8_t(clamp(rounded, -127.0f, 127.0f));
  }
}

kernel void prefill_moe_sep21_w8a8_quantize_gate_t256(
    device const bfloat *source [[buffer(0)]], device int8_t *codes [[buffer(1)]],
    device float *scales [[buffer(2)]], device uint *diag [[buffer(3)]],
    constant uint4 &p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  prefill_moe_sep21_w8a8_quantize<2560, 256>(source, codes, scales, diag,
      p, group, threads, tid, maxima);
}
kernel void prefill_moe_sep21_w8a8_quantize_down_t128(
    device const bfloat *source [[buffer(0)]], device int8_t *codes [[buffer(1)]],
    device float *scales [[buffer(2)]], device uint *diag [[buffer(3)]],
    constant uint4 &p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float maxima[4];
  prefill_moe_sep21_w8a8_quantize<640, 128>(source, codes, scales, diag,
      p, group, threads, tid, maxima);
}
'''
MACROS = r'''
#define W8A8_GATE_COMMON \
    device int8_t *a [[buffer(0)]], device int8_t *g [[buffer(1)]], \
    device const float *gs [[buffer(2)]], device int8_t *u [[buffer(3)]], \
    device const float *us [[buffer(4)]], device const uint *ranks [[buffer(5)]], \
    device const uint *offsets [[buffer(6)]], device const FlashMoEBucketJob *jobs [[buffer(7)]], \
    device const uint *count [[buffer(8)]], device bfloat *out [[buffer(9)]], \
    device uint *diag [[buffer(10)]], constant FlashInt8ExpertStoreParams &p [[buffer(11)]], \
    device const float *activation_scales [[buffer(12)]]
#define W8A8_DOWN_COMMON \
    device int8_t *a [[buffer(0)]], device int8_t *w [[buffer(1)]], \
    device const float *s [[buffer(2)]], device const uint *ranks [[buffer(3)]], \
    device const uint *offsets [[buffer(4)]], device const FlashMoEBucketJob *jobs [[buffer(5)]], \
    device const uint *count [[buffer(6)]], device const uint *map [[buffer(7)]], \
    device bfloat *out [[buffer(8)]], device uint *diag [[buffer(9)]], \
    constant FlashInt8ExpertStoreParams &p [[buffer(10)]], \
    device const float *activation_scales [[buffer(11)]]
#define W8A8_POSITION \
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]], \
    uint tid [[thread_index_in_threadgroup]]
#define W8A8_GATE(NAME, M, SG) \
kernel void NAME(W8A8_GATE_COMMON, W8A8_POSITION) { \
  prefill_moe_sep21_w8a8_gate<M, SG, 0, false, false>(a, g, gs, u, us, ranks, offsets, \
      jobs, count, out, diag, p, activation_scales, nullptr, nullptr, nullptr, nullptr, group, threads, tid); \
}
#define W8A8_DOWN(NAME, M, SG) \
kernel void NAME(W8A8_DOWN_COMMON, W8A8_POSITION) { \
  prefill_moe_sep21_w8a8_down<M, SG, 0, false, false>(a, w, s, ranks, offsets, jobs, \
      count, map, out, diag, p, activation_scales, nullptr, nullptr, group, threads, tid); \
}
#define W8A8_GATE_AUDIT(NAME, M, SG) \
kernel void NAME(W8A8_GATE_COMMON, device float *raw_g [[buffer(13)]], \
    device float *raw_u [[buffer(14)]], device int *dot_g [[buffer(15)]], \
    device int *dot_u [[buffer(16)]], W8A8_POSITION) { \
  prefill_moe_sep21_w8a8_gate<M, SG, 0, false, true>(a, g, gs, u, us, ranks, offsets, \
      jobs, count, out, diag, p, activation_scales, raw_g, raw_u, dot_g, dot_u, group, threads, tid); \
}
#define W8A8_DOWN_AUDIT(NAME, M, SG) \
kernel void NAME(W8A8_DOWN_COMMON, device float *raw [[buffer(12)]], \
    device int *dot [[buffer(13)]], W8A8_POSITION) { \
  prefill_moe_sep21_w8a8_down<M, SG, 0, false, true>(a, w, s, ranks, offsets, jobs, \
      count, map, out, diag, p, activation_scales, raw, dot, group, threads, tid); \
}
'''
VARIANTS = (('m32_n64_sg4', 32, 4), ('m32_n64_sg2', 32, 2))


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError(f'W8A8 numerical shader source drift: {before!r}')
    return source.replace(before, after)


def generate(destination: Path = DESTINATION) -> str:
    source = SOURCE.read_text()
    body = source[source.index('#if __METAL_VERSION__ >= 410'):
                  source.index('\n#define PREFILL4K_INT8TILES_GATE')]
    body = body.replace('prefill_moe_sep21_memory_', 'prefill_moe_sep21_w8a8_')
    body = body.replace('template <ushort M, ushort SG, ushort K = 0, bool Static = false>',
                        'template <ushort M, ushort SG, ushort K = 0, bool Static = false, bool Audit = false>')
    body = body.replace('device bfloat *input', 'device int8_t *input')
    body = body.replace(', float>();', ', int>();')
    body = body.replace('gd[i] = 0.0f', 'gd[i] = 0').replace('ud[i] = 0.0f', 'ud[i] = 0')
    body = body.replace('dot[i] = 0.0f', 'dot[i] = 0')
    body = replace_once(body,
        '    device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,\n'
        '    uint3 group, uint3 threads, uint tid) {',
        '    device bfloat *output, device uint *diag, constant FlashInt8ExpertStoreParams &p,\n'
        '    device const float *activation_scales, device float *raw_g, device float *raw_u,\n'
        '    device int *dot_g, device int *dot_u, uint3 group, uint3 threads, uint tid) {\n'
        '#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)')
    body = replace_once(body,
        '    constant FlashInt8ExpertStoreParams &p, uint3 group, uint3 threads, uint tid) {',
        '    constant FlashInt8ExpertStoreParams &p, device const float *activation_scales,\n'
        '    device float *raw, device int *raw_dot, uint3 group, uint3 threads, uint tid) {\n'
        '#pragma clang fp contract(off)\n#pragma clang fp reassociate(off)')
    body = replace_once(body, '    const float gf = gd[i] * gs, uf = ud[i] * us;',
        '    const float activation_scale = activation_scales[ulong(begin + index[1])];\n'
        '    const float unscaled_g = float(gd[i]) * activation_scale;\n'
        '    const float unscaled_u = float(ud[i]) * activation_scale;\n'
        '    const float gf = unscaled_g * gs, uf = unscaled_u * us;\n'
        '    if (!(activation_scale > 0.0f) || !flash_mpp_finite(activation_scale))\n'
        '      flash_mpp_error(diag, 4u);\n'
        '    if constexpr (Audit) {\n'
        '      const ulong at = ulong(begin + index[1]) * 640 + n;\n'
        '      raw_g[at] = gf; raw_u[at] = uf;\n'
        '      dot_g[at] = gd[i]; dot_u[at] = ud[i];\n'
        '    }')
    body = replace_once(body, '    const float result = dot[i] * scale;',
        '    const float activation_scale = activation_scales[ulong(begin + index[1])];\n'
        '    const float unscaled = float(dot[i]) * activation_scale;\n'
        '    const float result = unscaled * scale;\n'
        '    if (!(activation_scale > 0.0f) || !flash_mpp_finite(activation_scale))\n'
        '      flash_mpp_error(diag, 4u);\n'
        '    if constexpr (Audit) {\n'
        '      const ulong at = ulong(route) * 2560 + n;\n'
        '      raw[at] = result; raw_dot[at] = dot[i];\n'
        '    }')
    kernels = []
    for suffix, rows, groups in VARIANTS:
        for producer, macro in (('gate_up', 'W8A8_GATE'), ('down_scatter', 'W8A8_DOWN')):
            name = f'prefill_moe_sep21_w8a8_{producer}_{suffix}'
            kernels.append(f'{macro}({name}, {rows}, {groups})')
            kernels.append(f'{macro}_AUDIT({name}_audit, {rows}, {groups})')
    result = HEADER + body + '\n' + QUANTIZERS + '\n' + MACROS + '\n' + '\n'.join(kernels)
    result += ('\n#undef W8A8_GATE_COMMON\n#undef W8A8_DOWN_COMMON\n#undef W8A8_POSITION\n'
               '#undef W8A8_GATE\n#undef W8A8_DOWN\n#undef W8A8_GATE_AUDIT\n#undef W8A8_DOWN_AUDIT\n#endif\n')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, nargs='?', default=DESTINATION)
    generate(parser.parse_args().destination)
