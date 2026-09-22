#!/usr/bin/env python3
"""CPU-only isolated R5 odd row-pair code generation and shader compilation.

Only program sources and immutable AIR/metallib artifacts are read. No GPU,
model, operand, capture, request, token or generated execution report is opened.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PARENT = ROOT / 'build/R5-integer-currentQ4-fixed4-sep22-worker-v2'
SOURCE = '4c271477b47d037cbe4018f73e6ba8f1a27dbfca4c9a0e98a6fce93b0155e73a'
AIR = 'efec83efc7c8aa89c69d1c322c91848bd446c434bd5dde3ca430e1fae544f5c4'
LIBRARY = 'dc1ab6f9178aac706bb408601fb734e9d508fb5c6c491732bc6ec4e36e6287e6'
FORMATS = ((4, 64), (5, 64), (5, 128), (6, 64))
OBSERVED = ((4, 64, 2560, 10240), (4, 64, 2560, 12288),
            (4, 64, 6144, 2560), (5, 64, 2560, 10240),
            (5, 128, 6144, 2560), (5, 128, 2560, 6144),
            (6, 64, 2560, 6144))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(text): return hashlib.sha256(text.encode()).hexdigest()
def once(text, old, new):
    if text.count(old) != 1: raise ValueError('nonunique literal source anchor: ' + repr(old))
    return text.replace(old, new)
def body(text, anchor):
    start = text.index('{', text.index(anchor)); depth = 1; end = start + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}'); end += 1
    return text[start + 1:end - 1]
def function(text, anchor):
    start = text.index(anchor); brace = text.index('{', start); depth = 1; end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}'); end += 1
    return text[start:end]

SIGNATURE = '''
kernel void NAME(
    const device bfloat *input [[buffer(0)]], const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]], const device uchar *biases [[buffer(3)]],
    const device long *expert_ids [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]], constant FlashAffineParams &p [[buffer(7)]],
    TAP
    uint3 group [[threadgroup_position_in_grid]], uint3 threads [[threads_per_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  (void)expert_ids;
  if (any(threads != uint3(64, 1, 1)) || simd_width != 32 || simd_group >= 2 || lane >= 32) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  CALL
}
'''

PROJECT = '''
template <ushort Bits, ushort GroupSize>
inline void project_pair(const device bfloat *input,
    const device uchar *weights, const device uchar *scales,
    const device uchar *biases, device bfloat *output,
    device atomic_uint *diagnostics, constant FlashAffineParams &p,
    uint3 group, uint simd_group, uint lane) {
  constexpr uint Values = Bits == 6 ? 8 : 16;
  constexpr uint Block = Values * 32;
  static_assert((Bits == 4 && GroupSize == 64) || Bits == 5 ||
                (Bits == 6 && GroupSize == 64));
  static_assert(GroupSize % Values == 0);
  const bool observed =
      (Bits == 4 && GroupSize == 64 &&
       ((p.input_size == 2560 && (p.output_size == 10240 || p.output_size == 12288)) ||
        (p.input_size == 6144 && p.output_size == 2560))) ||
      (Bits == 5 && GroupSize == 64 && p.input_size == 2560 && p.output_size == 10240) ||
      (Bits == 5 && GroupSize == 128 &&
       ((p.input_size == 6144 && p.output_size == 2560) ||
        (p.input_size == 2560 && p.output_size == 6144))) ||
      (Bits == 6 && GroupSize == 64 && p.input_size == 2560 && p.output_size == 6144);
  if (!observed || p.rows != 5 || p.selections != 1 || p.experts != 1 || p.flags ||
      p.bits != Bits || p.group_size != GroupSize || p.input_size % Block ||
      p.output_size % 8 || p.weight_row_stride_bytes < ulong(p.input_size) * Bits / 8 ||
      p.parameter_row_stride_bytes < ulong(p.input_size / GroupSize) * 2 ||
      p.parameter_row_stride_bytes % 2 || p.parameter_expert_stride_bytes % 2 ||
      !row_stride_bounds<Bits, GroupSize>(p)) {
    if (!lane) atomic_fetch_or_explicit(diagnostics, 2u, memory_order_relaxed);
    return;
  }
  const uint out_row = group.x * 8 + simd_group * 4;
  if (group.x >= p.output_size / 8 || group.y >= 3 || group.z || out_row >= p.output_size) return;
  if (group.y == 2) {
    // Literal original math body, explicit real fifth row. Calling the old
    // project driver with group.y2 would silently compute row2 and is forbidden.
    r5_raw_odd_literal::project_math<Bits, GroupSize, true>(
        input + ulong(4) * p.input_size, weights, scales, biases, output,
        diagnostics, p, out_row, 0, ulong(4), lane);
    return;
  }
  const ulong route0 = ulong(group.y) * 2, route1 = route0 + 1;
  const device bfloat *x0 = input + route0 * p.input_size;
  const device bfloat *x1 = input + route1 * p.input_size;
  thread float x_thread0[Values], x_thread1[Values];
  thread float result0[4] = {0}, result1[4] = {0};
  for (uint k = 0; k < p.input_size; k += Block) {
    const uint channel = k + lane * Values;
    const float sum0 = r5_raw_odd_literal::mlx_qmv_f32xsum_v1_load_vector<
        bfloat, float, Values, Bits>(x0 + channel, x_thread0);
    const float sum1 = r5_raw_odd_literal::mlx_qmv_f32xsum_v1_load_vector<
        bfloat, float, Values, Bits>(x1 + channel, x_thread1);
    const uint coefficient = channel / GroupSize;
    for (ushort row = 0; row < 4; ++row) {
      const device uchar *wl = weights +
          ulong(out_row + row) * p.weight_row_stride_bytes + ulong(channel) * Bits / 8;
      const ulong offset = ulong(out_row + row) * p.parameter_row_stride_bytes;
      const device bfloat *sl = reinterpret_cast<const device bfloat *>(scales + offset);
      const device bfloat *bl = reinterpret_cast<const device bfloat *>(biases + offset);
      const float s = float(sl[coefficient]);
      const float b = float(bl[coefficient]);
      float dot0, dot1;
      qdot_pair<Bits, Values>(wl, x_thread0, x_thread1, s, b, sum0, sum1, dot0, dot1);
      result0[row] += dot0;
      result1[row] += dot1;
    }
  }
  for (ushort row = 0; row < 4; ++row) {
    const float sum0 = simd_sum(result0[row]);
    if (lane == 0) {
      const bfloat value0 = bfloat(sum0);
      if (!r5_raw_odd_literal::finite(sum0) || !r5_raw_odd_literal::finite(float(value0)))
        atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
      output[route0 * p.output_size + out_row + row] = value0;
    }
    const float sum1 = simd_sum(result1[row]);
    if (lane == 0) {
      const bfloat value1 = bfloat(sum1);
      if (!r5_raw_odd_literal::finite(sum1) || !r5_raw_odd_literal::finite(float(value1)))
        atomic_fetch_or_explicit(diagnostics, 4u, memory_order_relaxed);
      output[route1 * p.output_size + out_row + row] = value1;
    }
  }
}
'''

def overflow_bounds_source():
    text = '''// Guard-only specialization of the original unsigned overflow test.
// Every constant is the unique floor((UINT64_MAX-rowBytes)/(N-1)).
// Unsupported geometry keeps the original generic expression; the outer
// observed-shape guard still refuses unsupported geometry before math.
template <ushort Bits, ushort GroupSize>
inline bool row_stride_bounds(constant FlashAffineParams &p) {
'''
    maximum = (1 << 64) - 1
    for bits, group, k, n in OBSERVED:
        weight = (maximum - k * bits // 8) // (n - 1)
        parameter = (maximum - (k // group) * 2) // (n - 1)
        text += f'''  if (Bits == {bits} && GroupSize == {group} && p.input_size == {k} && p.output_size == {n})
    return p.weight_row_stride_bytes <= {weight}ul &&
        p.parameter_row_stride_bytes <= {parameter}ul;
'''
    text += '''  if (p.output_size <= 1) return false;
  return p.weight_row_stride_bytes <=
      (~ulong(0) - ulong(p.input_size) * Bits / 8) / (p.output_size - 1) &&
      p.parameter_row_stride_bytes <=
      (~ulong(0) - ulong(p.input_size / GroupSize) * 2) / (p.output_size - 1);
}

'''
    return text

def guard_source_proof():
    old = ROOT / 'dev/benchmarks/R5_raw_odd_rowpair_sep22/kernel'
    expected = {'build.py': '4e60738d62ac9aac796cb209b16f4a32a33c2834f69a1f6a8c8cd97a57587d7e',
                'audit.py': 'f7bdb6fc123930752a27f7bf93256d978d738dccf010e45f1c4ffe6124652506'}
    for name, pin in expected.items():
        if sha(old / name) != pin: raise ValueError('qualified frozen old R5 source differs')
    maximum = (1 << 64) - 1; keys = set(); records = []; tests = 0; full_guard_tests = 0
    for bits, group, k, n in OBSERVED:
        key = (bits, group, k, n)
        if key in keys: raise ValueError('authorized shape/format bijection duplicated')
        keys.add(key)
        code_width = k * bits // 8; parameter_width = k // group * 2
        limits = []
        for width in (code_width, parameter_width):
            limit = (maximum - width) // (n - 1)
            if limit * (n - 1) + width > maximum or (limit + 1) * (n - 1) + width <= maximum:
                raise ValueError('overflow constant unique-floor proof failed')
            for stride in (0, width-1, width, limit-1, limit, limit+1, maximum):
                if (stride <= limit) != (stride * (n-1) + width <= maximum):
                    raise ValueError('old/new exact overflow refusal boundary mismatch')
                tests += 1
            limits.append(limit)
        for ws in (0, code_width-1, code_width, limits[0]-1, limits[0], limits[0]+1, maximum):
            for ps in (0, parameter_width-1, parameter_width, limits[1]-1, limits[1], limits[1]+1, maximum):
                for parameter_expert_stride in (0, 1, 2):
                    common = ws >= code_width and ps >= parameter_width and ps % 2 == 0 and parameter_expert_stride % 2 == 0
                    old = common and ws*(n-1)+code_width <= maximum and ps*(n-1)+parameter_width <= maximum
                    new = common and ws <= limits[0] and ps <= limits[1]
                    if old != new: raise ValueError('full min/even/overflow guard refusal equivalence failed')
                    full_guard_tests += 1
        if n % 8: raise ValueError('canonical N guard no longer proves complete column blocks')
        # All admitted x/SIMD/column positions, not a sample: maximal column
        # is N-1. Existing host refuses partial grids and non-SG32 pipelines.
        columns = [x*8 + sg*4 + c for x in range(n//8) for sg in (0,1) for c in range(4)]
        if len(columns) != n or set(columns) != set(range(n)): raise ValueError('removed column checks were not redundant')
        records.append({'bits': bits, 'group': group, 'K': k, 'N': n,
                        'code_width_bytes': code_width, 'parameter_width_bytes': parameter_width,
                        'maximum_weight_stride_bytes': limits[0], 'maximum_parameter_stride_bytes': limits[1],
                        'unique_floor_inequalities_proved': True, 'every_admitted_column_bounded_bijection': True})
    # The exact observed-shape predicate is unchanged. Every excluded tuple
    # remains excluded regardless of the retained generic overflow helper.
    excluded = []
    for bits, group, k, n in OBSERVED:
        for key in ((bits, group, k-1, n), (bits, group, k, n-1),
                    (bits, group, k, 0), (bits, group, k, 1),
                    (8, group, k, n)):
            if key in keys: raise ValueError('malformed geometry unexpectedly observed')
            excluded.append(key)
    proof = {'schema': 'R5-raw-guard-specialized-source-before-AIR-proof-v1', 'pass': True,
             'authorized_shape_format_bijection_count': len(keys), 'overflow_boundary_cases': tests,
             'full_min_even_overflow_refusal_cases': full_guard_tests, 'unsupported_geometry_refusal_cases': len(excluded),
             'source_review_GO_before_AIR_compile_required': True, 'AIR_compiled': False, 'GPU_work': False,
             'old_literal_qdot_FP_tail_and_tap_source_preserved': True,
             'newly_admitted_geometry': False, 'unsupported_geometry_generic_guard_retained': True,
             'column_proof_preconditions': 'shader threads64 + threads_per_simdgroup32 + simd_group<2 + lane<32; N%8, x<N/8; full canonical grid host refusal retained',
             'only_removed_checks': 'paired innerK/finalreduction out_row+row<N; literal original row4 retains all checks',
             'source_reference_sha256': expected, 'shape_bounds': records,
             'tensor_or_execution_payload_reads': 0}
    (HERE / 'source-guard-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    return proof

def qdot_pair(original):
    qdot = body(original, 'inline U mlx_qmv_f32xsum_v1_qdot(')
    parts = []
    restorations = []
    for bits in (4, 5, 6):
        branch = body(qdot, 'else if (bits == ' + str(bits) + ')')
        loop = body(branch, 'for (int i = 0;')
        old_fp = re.findall(r'accum\s*\+=\s*.*?;', loop, re.S)
        masks = list(dict.fromkeys(re.findall(r'\((?:packed|w\[\d\]) & 0x[0-9a-f]+\)', ''.join(old_fp))))
        pairs = {m: 'm' + str(i) for i, m in enumerate(masks)}
        if bits == 4:
            preamble = loop[:loop.index('accum +=')]
            # Original integer byte assembly remains literally unchanged.
            mutations = preamble
        else:
            preamble = loop[:loop.index('accum +=')]
            mutations = preamble.replace('x_thread +=', 'x0 +=').replace('      w +=', '      x1 += ' + str(8 if bits == 5 else 4) + ' * i;\n      w +=')
        mutations = mutations.rstrip() + '\n'
        declarations = ''.join('      const ' + ('uint16_t' if bits == 4 else 'int') + ' ' + name + ' = ' + mask[1:-1] + ';\n' for mask, name in pairs.items())
        row_bodies = []
        for row in (0, 1):
            fp = ''.join(old_fp).replace('accum', 'accum' + str(row)).replace('x_thread', 'x' + str(row))
            for mask, name in pairs.items(): fp = fp.replace(mask, name)
            restored = fp.replace('accum' + str(row), 'accum').replace('x' + str(row), 'x_thread')
            for mask, name in pairs.items(): restored = re.sub(r'\b' + name + r'\b', mask, restored)
            if restored != ''.join(old_fp): raise ValueError('per-row literal FP restoration failed')
            row_bodies.append('      ' + fp.replace(';', ';\n      ').rstrip() + '\n')
        header = re.search(r'for \(int i = 0;[^\n]+', branch).group(0).replace('values_per_thread', 'Values')
        parts.append(('  if' if bits == 4 else '  else if') + ' (Bits == ' + str(bits) + ') {\n    ' + header + '\n' + mutations + declarations + ''.join(row_bodies) + '    }\n  }\n')
        restorations.append({'bits': bits, 'per_row_FP_expression_inverse_byte_exact': True,
                             'original_FP_statements': old_fp, 'shared_integer_masks': pairs})
    result = '''template <ushort Bits, ushort Values>
inline void qdot_pair(const device uchar *w,
    const thread float *x0, const thread float *x1,
    float scale, float bias, float sum0, float sum1,
    thread float &dot0, thread float &dot1) {
  float accum0 = 0, accum1 = 0;
''' + ''.join(parts) + '''  dot0 = scale * accum0 + sum0 * bias;
  dot1 = scale * accum1 + sum1 * bias;
}
'''
    return result, restorations

def generate():
    guard_source_proof()
    original = PARENT / 'source/runtime/metal/kernels/shared/flash_affine_qmv_f32.metal'
    if sha(original) != SOURCE: raise ValueError('current frozen raw helper source differs')
    if sha(PARENT / 'splash.metallib') != LIBRARY: raise ValueError('current frozen shipping library differs')
    prior = ((ROOT / 'dev/benchmarks/raw_q4_rowpair_sep22/pair.metalh', '6b982d607cf07df189b956a3e4dd3ca2a79801989356fd098ab2c5793b714b10'),
             (ROOT / 'dev/benchmarks/raw_q5_rowpair_sep22/pair.metalh', '760939b191da28f26d1f313f3f0ab03a1618d4100d97b4172799c45831ce16af'))
    for path, pin in prior:
        if sha(path) != pin: raise ValueError('literal prior row-pair reference differs')
    src = original.read_text(); prefix = src.split('// Existing seven-buffer projection ABI;', 1)[0]
    literal = prefix.replace('splash_mlx_qmv_f32xsum_v1', 'r5_raw_odd_literal')
    if literal.replace('r5_raw_odd_literal', 'splash_mlx_qmv_f32xsum_v1') != prefix:
        raise ValueError('literal old helper inverse failed')
    (HERE / 'literal.metalh').write_text(literal)
    pair, row_inverse = qdot_pair(src)
    pair = '#include "literal.metalh"\n\nnamespace r5_raw_odd_rowpair_sep22 {\n' + overflow_bounds_source() + pair + PROJECT + '\n}\n'
    old_dir = ROOT / 'dev/benchmarks/R5_raw_odd_rowpair_sep22/kernel'
    old_tree = ast.parse((old_dir / 'build.py').read_text())
    old_project = next(ast.literal_eval(n.value) for n in old_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PROJECT' for t in n.targets))
    inverse_pair = once(pair, overflow_bounds_source(), '')
    inverse_pair = once(inverse_pair, PROJECT, old_project)
    if inverse_pair != (old_dir / 'pair.metalh').read_text():
        raise ValueError('guard-only delta does not inversely restore frozen old R5 pair source byte-exact')
    (HERE / 'pair.metalh').write_text(pair)
    control = literal.replace('r5_raw_odd_literal', 'r5_raw_odd_control_tap')
    control_edits = []
    def ce(old, new, count=1):
        nonlocal control
        if control.count(old) != count: raise ValueError('control tap anchor census differs')
        control = control.replace(old, new); control_edits.append({'before': old, 'after': new, 'count': count})
    ce('device bfloat *output, device atomic_uint *diagnostics,\n    constant FlashAffineParams &p, uint out_row, uint expert,',
       'device bfloat *output, device atomic_uint *diagnostics,\n    device float *raw_f32,\n    constant FlashAffineParams &p, uint out_row, uint expert,')
    ce('        output[route * p.output_size + out_row + row] = value;',
       '        output[route * p.output_size + out_row + row] = value;\n        raw_f32[route * p.output_size + out_row + row] = sum;')
    ce('device atomic_uint *diagnostics, constant FlashAffineParams &p,\n    uint3 group, uint simd_group, uint lane)',
       'device atomic_uint *diagnostics, device float *raw_f32,\n    constant FlashAffineParams &p, uint3 group, uint simd_group, uint lane)')
    ce('x, weights, scales, biases, output, diagnostics, p, out_row,',
       'x, weights, scales, biases, output, diagnostics, raw_f32, p, out_row,', 2)
    inverse = control
    for edit in reversed(control_edits): inverse = inverse.replace(edit['after'], edit['before'])
    if inverse.replace('r5_raw_odd_control_tap', 'r5_raw_odd_literal') != literal:
        raise ValueError('additive control tap inverse failed')
    (HERE / 'control_tap.metalh').write_text(control)
    tap = pair; edits = []
    def pe(old, new, count=1):
        nonlocal tap
        if tap.count(old) != count: raise ValueError('candidate tap literal anchor census differs')
        tap = tap.replace(old, new); edits.append({'before': old, 'after': new, 'count': count})
    pe('#include "literal.metalh"', '#include "control_tap.metalh"')
    pe('namespace r5_raw_odd_rowpair_sep22 {', 'namespace r5_raw_odd_rowpair_tap_sep22 {')
    pe('device atomic_uint *diagnostics, constant FlashAffineParams &p,',
       'device atomic_uint *diagnostics, device float *raw_f32,\n    constant FlashAffineParams &p,')
    pe('r5_raw_odd_literal::', 'r5_raw_odd_control_tap::', 7)
    pe('        diagnostics, p, out_row, 0, ulong(4), lane);',
       '        diagnostics, raw_f32, p, out_row, 0, ulong(4), lane);')
    for row in (0, 1):
        pe('      output[route' + str(row) + ' * p.output_size + out_row + row] = value' + str(row) + ';',
           '      output[route' + str(row) + ' * p.output_size + out_row + row] = value' + str(row) + ';\n      raw_f32[route' + str(row) + ' * p.output_size + out_row + row] = sum' + str(row) + ';')
    inverse = tap
    for edit in reversed(edits): inverse = inverse.replace(edit['after'], edit['before'])
    if inverse != pair: raise ValueError('coupled candidate tap inverse failed')
    (HERE / 'pair_tap.metalh').write_text(tap)
    shipping = '#include "pair.metalh"\n'; candidate_taps = '#include "pair_tap.metalh"\n'; control_taps = '#include "control_tap.metalh"\n'
    for bits, group in FORMATS:
        suffix = f'_q{bits}_g{group}'
        call = 'r5_raw_odd_rowpair_sep22::project_pair<' + str(bits) + ', ' + str(group) + '>(input, weights, scales, biases, output, diagnostics, p, group, simd_group, lane);'
        shipping += SIGNATURE.replace('NAME', 'r5_raw_odd_rowpair_sep22_timed' + suffix).replace('    TAP\n', '').replace('CALL', call)
        candidate_taps += SIGNATURE.replace('NAME', 'r5_raw_odd_rowpair_sep22_candidate_probe' + suffix).replace('TAP', 'device float *raw_f32 [[buffer(8)]],').replace('CALL', call.replace('r5_raw_odd_rowpair_sep22::', 'r5_raw_odd_rowpair_tap_sep22::').replace('diagnostics, p,', 'diagnostics, raw_f32, p,'))
        control_call = 'r5_raw_odd_control_tap::project<' + str(bits) + ', ' + str(group) + '>(input, weights, scales, biases, expert_ids, output, diagnostics, raw_f32, p, group, simd_group, lane);'
        control_taps += SIGNATURE.replace('NAME', 'r5_raw_odd_rowpair_sep22_control_probe' + suffix).replace('TAP', 'device float *raw_f32 [[buffer(8)]],').replace('CALL', control_call)
    (HERE / 'candidate.metal').write_text(shipping)
    (HERE / 'candidate_probe.metal').write_text(candidate_taps)
    (HERE / 'control_probe.metal').write_text(control_taps)
    journal = {'schema': 'R5-raw-guard-specialized-rowpair-literal-inverse-journal-v1', 'original_source_sha256': SOURCE,
               'guard_only_delta_restores_frozen_old_R5_pair_byte_exact': True,
               'guard_only_delta_before': old_project, 'guard_only_delta_after': PROJECT,
               'added_integer_guard_helper': overflow_bounds_source(),
               'original_helpers_inverse_byte_exact': True, 'control_tap_inverse_byte_exact': True,
               'candidate_tap_inverse_byte_exact': True, 'per_format_per_row_FP_inverse': row_inverse,
               'control_tap_changes': control_edits, 'candidate_tap_changes': edits,
               'tap_floating_operations_added': 0, 'tail': 'literal project_math(input+4*K,expert0,route4)',
               'observed_shape_format_combinations': OBSERVED, 'formats': FORMATS,
               'GPU_work': False, 'tensor_or_execution_report_payload_reads': 0,
               'source_guard_proof_sha256': sha(HERE / 'source-guard-proof.json'),
               'generated_source_sha256': {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.metal', '.metalh')}}
    (HERE / 'source-journal.json').write_text(json.dumps(journal, indent=2) + '\n')
    return journal

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE / '_cpu_build_v1'); args = parser.parse_args()
    journal = generate()
    if args.generate_only:
        print(json.dumps({'source_generated': True, 'formats': FORMATS, 'GPU_work': False})); return
    out = args.output.resolve()
    if not out.is_relative_to(HERE) or out.exists(): raise ValueError('fresh output under owned kernel directory required')
    out.mkdir(parents=True)
    native = PARENT / 'reused/air/074-flash_affine_qmv_f32.air'
    if sha(native) != AIR: raise ValueError('current immutable untapped shipping control AIR differs')
    shutil.copy2(native, out / 'native-qmv.air')
    metal = ['xcrun', '-sdk', 'macosx', 'metal', '-std=metal4.1', '-O3', '-Wall', '-Wextra', '-Werror',
             '-mmacosx-version-min=27.0', '-DSPLASH_INT8_EXPERIMENT=1',
             '-I' + str(PARENT / 'source/runtime'), '-I' + str(HERE)]
    commands = []
    for name in ('candidate', 'control_probe', 'candidate_probe'):
        command = metal + ['-c', str(HERE / (name + '.metal')), '-o', str(out / (name + '.air'))]
        subprocess.run(command, cwd=ROOT, check=True); commands.append(command)
    for name in ('native-qmv', 'candidate', 'control_probe', 'candidate_probe'):
        command = ['xcrun', 'air-opt', '-S', str(out / (name + '.air')), '-o', str(out / (name + '.ll'))]
        subprocess.run(command, cwd=ROOT, check=True); commands.append(command)
    command = ['xcrun', '-sdk', 'macosx', 'metallib'] + [str(out / (n + '.air')) for n in ('native-qmv', 'candidate', 'control_probe', 'candidate_probe')] + ['-o', str(out / 'component.metallib')]
    subprocess.run(command, cwd=ROOT, check=True); commands.append(command)
    command = [sys.executable, str(HERE / 'audit.py'), '--build', str(out)]
    subprocess.run(command, cwd=ROOT, check=True); commands.append(command)
    manifest = {'schema': 'R5-raw-odd-rowpair-CPU-shader-build-v1', 'GPU_work': False, 'tensor_payload_reads': 0,
                'current_parent_library_sha256': LIBRARY, 'shipping_native_control_AIR_sha256': AIR,
                'original_source_sha256': SOURCE, 'source_journal_sha256': sha(HERE / 'source-journal.json'),
                'shipping_native_control_recompiled': False, 'compiler_commands': commands,
                'artifacts': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                'strict_actual_SSA_audit_complete': True,
                'strict_actual_SSA_audit_sha256': sha(out / 'strict-SSA-audit.json'),
                'SSA_audit_scope': 'finite-path actual per-row dependency/load/attribute proof; explicit finite-mask and same-address chunk operand commutations only; no reassociation; no nonfinite/GPU-bit proof',
                'source_files': {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py', '.metal', '.metalh', '.json')},
                'Root_GPU_or_worker_qualification': False}
    (out / 'build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'CPU_shader_build_pass': True, 'output': str(out), 'native_control_recompiled': False,
                      'GPU_work': False, 'strict_SSA_audit_complete': True}))

if __name__ == '__main__': main()
