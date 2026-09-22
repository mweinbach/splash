#!/usr/bin/env python3
"""CPU-only literal original-Q5-helper extraction and additive F32-tap journal."""
from pathlib import Path
import argparse,hashlib,json

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1:raise ValueError(f"one literal anchor required:{a!r}")
    return s.replace(a,b)

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('fresh generated source directory required')
    a.output.mkdir(parents=True)
    src=a.original.read_text();pin='4c271477b47d037cbe4018f73e6ba8f1a27dbfca4c9a0e98a6fce93b0155e73a'
    if sha(src)!=pin:raise ValueError('native original QMV source differs')
    prefix=src.split('// Existing seven-buffer projection ABI;',1)[0]
    literal=prefix.replace('splash_mlx_qmv_f32xsum_v1','raw_q5_literal')
    if literal.replace('raw_q5_literal','splash_mlx_qmv_f32xsum_v1')!=prefix:raise ValueError('literal source restoration failed')
    (a.output/'literal.metalh').write_text(literal)
    changes=[]
    control=prefix.replace('splash_mlx_qmv_f32xsum_v1','raw_q5_control_tap')
    def edit(s,x,y):
        changes.append({'before':x,'after':y});return once(s,x,y)
    control=edit(control,'device bfloat *output, device atomic_uint *diagnostics,\n    constant FlashAffineParams &p, uint out_row, uint expert,',
        'device bfloat *output, device atomic_uint *diagnostics,\n    device float *raw_f32,\n    constant FlashAffineParams &p, uint out_row, uint expert,')
    control=edit(control,'        output[route * p.output_size + out_row + row] = value;',
        '        output[route * p.output_size + out_row + row] = value;\n        raw_f32[route * p.output_size + out_row + row] = sum;')
    control=edit(control,'device atomic_uint *diagnostics, constant FlashAffineParams &p,\n    uint3 group, uint simd_group, uint lane)',
        'device atomic_uint *diagnostics, device float *raw_f32,\n    constant FlashAffineParams &p, uint3 group, uint simd_group, uint lane)')
    x='x, weights, scales, biases, output, diagnostics, p, out_row,'
    y='x, weights, scales, biases, output, diagnostics, raw_f32, p, out_row,'
    if control.count(x)!=2:raise ValueError('control driver call anchors differ')
    control=control.replace(x,y);changes.append({'before':x,'after':y,'count':2})
    restored=control
    for c in reversed(changes):restored=restored.replace(c['after'],c['before'])
    if restored.replace('raw_q5_control_tap','splash_mlx_qmv_f32xsum_v1')!=prefix:raise ValueError('control F32 tap does not restore original bytes')
    signature='''\n
kernel void raw_q5_rowpair_sep22_control_probe(
    const device bfloat *input [[buffer(0)]], const device uchar *weights [[buffer(1)]],
    const device uchar *scales [[buffer(2)]], const device uchar *biases [[buffer(3)]],
    const device long *expert_ids [[buffer(4)]], device bfloat *output [[buffer(5)]],
    device atomic_uint *diagnostics [[buffer(6)]], constant FlashAffineParams &p [[buffer(7)]],
    device float *raw_f32 [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
  raw_q5_control_tap::project<5,128>(input,weights,scales,biases,expert_ids,output,
      diagnostics,raw_f32,p,group,simd_group,lane);
}
'''
    (a.output/'control_probe.metal').write_text(control+signature)
    pair=(a.source/'pair.metalh').read_text();tap=pair;pair_changes=[]
    def pe(x,y):
        nonlocal tap
        tap=once(tap,x,y);pair_changes.append({'before':x,'after':y})
    pe('namespace raw_q5_rowpair_sep22 {','namespace raw_q5_rowpair_tap_sep22 {')
    pe('device atomic_uint *diagnostics, constant FlashAffineParams &p,',
       'device atomic_uint *diagnostics, device float *raw_f32,\n    constant FlashAffineParams &p,')
    pe('      output[route0 * p.output_size + out_row + row] = value0;',
       '      output[route0 * p.output_size + out_row + row] = value0;\n      raw_f32[route0 * p.output_size + out_row + row] = sum0;')
    pe('      output[route1 * p.output_size + out_row + row] = value1;',
       '      output[route1 * p.output_size + out_row + row] = value1;\n      raw_f32[route1 * p.output_size + out_row + row] = sum1;')
    restored=tap
    for c in reversed(pair_changes):restored=restored.replace(c['after'],c['before'])
    if restored!=pair:raise ValueError('candidate F32 tap does not restore timed source bytes')
    cs=(a.source/'candidate.metal').read_text();cs=cs.replace('#include "pair.metalh"','')
    cs=once(cs,'raw_q5_rowpair_sep22_timed','raw_q5_rowpair_sep22_candidate_probe')
    cs=once(cs,'constant FlashAffineParams &p [[buffer(7)]],','constant FlashAffineParams &p [[buffer(7)]],\n    device float *raw_f32 [[buffer(8)]],')
    cs=once(cs,'raw_q5_rowpair_sep22::project_pair','raw_q5_rowpair_tap_sep22::project_pair')
    cs=once(cs,'diagnostics, p, group, simd_group, lane);','diagnostics, raw_f32, p, group, simd_group, lane);')
    (a.output/'candidate_probe.metal').write_text(tap+cs)
    journal={'schema':'splash-raw-q5-rowpair-source-taps-v1','original_source_sha256':sha(src),
        'original_helpers_restored_byte_exact':True,'control_tap_restored_byte_exact':True,'candidate_tap_restored_byte_exact':True,
        'control_tap_changes':changes,'candidate_tap_changes':pair_changes,'floating_operations_added_by_taps':0,
        'tap_position':'after original BF16 value/finite diagnostics/output store','generated':{p.name:sha(p.read_text())for p in sorted(a.output.iterdir())},
        'GPU_register_equality_proved':False,'GPU_work':False,'payload_read':False}
    (a.output/'source-journal.json').write_text(json.dumps(journal,indent=2)+'\n')
    print(json.dumps({k:journal[k]for k in ['original_helpers_restored_byte_exact','control_tap_restored_byte_exact','candidate_tap_restored_byte_exact','GPU_work','payload_read']}))
if __name__=='__main__':main()
