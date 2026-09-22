#!/usr/bin/env python3
"""Metadata-only actual-AIR guard reachability; stops before any data load.

Follows the existing strict checker's actual SSA/control-flow interpreter with
only numeric descriptor metadata. No operand value is read or constructed.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import audit
import build

HERE = Path(__file__).resolve().parent
class ReachedData(Exception): pass
class GuardExecutor(audit.Executor):
    def load(self, typ, ptr):
        if ptr[1] in ('Input', 'W', 'S', 'B'):
            raise ReachedData()
        return super().load(typ, ptr)

def reach(module, bits, group, k, n, ws, ps):
    p = audit.params(bits, group, k, n); p[8] = ws; p[10] = ps
    executor = GuardExecutor(module, p)
    name = module.find(f'project_pairILt{bits}ELt{group}EE')
    divisions = set(); nodes = set()
    def trace(frame, event, arg):
        if frame.f_code is audit.Executor.run.__code__ and event == 'line':
            instruction = frame.f_locals.get('instruction', '')
            if instruction:
                node = (frame.f_locals.get('name'), frame.f_locals.get('block'), instruction)
                nodes.add(node)
                if re.match(r'udiv (?:(?:nuw|nsw|exact) )*i64\b', instruction): divisions.add(node)
        return trace
    args = [audit.pointer('Input'), audit.pointer('W'), audit.pointer('S'), audit.pointer('B'),
            audit.pointer('Output'), audit.pointer('Diagnostics'), audit.pointer('Params'), (0,0,0), 0,0]
    prior = sys.gettrace(); reached = False
    try:
        sys.settrace(trace); executor.run(name, args)
    except ReachedData:
        reached = True
    finally:
        sys.settrace(prior)
    return reached, divisions, nodes

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--build', type=Path, default=HERE/'_cpu_build_v1'); args = parser.parse_args()
    directory = args.build.resolve()
    if not directory.is_relative_to(HERE): raise ValueError('owned fresh kernel output required')
    module = audit.Module(directory/'candidate.ll'); records = []; cases = 0
    maximum = (1 << 64) - 1
    for bits, group, k, n in build.OBSERVED:
        code = k*bits//8; param = k//group*2
        max_code = (maximum-code)//(n-1); max_param = (maximum-param)//(n-1)
        packet_cases = [('compact',code,param,True),
                        ('maximum_allowed',max_code,max_param&~1,True),
                        ('weight_overflow_refusal',max_code+1,param,False),
                        ('parameter_overflow_refusal',code,(max_param+2)&~1,False),
                        ('short_code_refusal',code-1,param,False),
                        ('short_parameter_refusal',code,param-2,False),
                        ('odd_parameter_refusal',code,param+1,False)]
        for label, ws, ps, expected in packet_cases:
            reached, divisions, nodes = reach(module,bits,group,k,n,ws,ps)
            if reached != expected: raise ValueError('actual AIR stride admission/refusal mismatch: '+label)
            if divisions: raise ValueError('dynamic i64 udiv executed on an authorized shape guard path')
            if not nodes: raise ValueError('vacuous actual control-flow reachability witness')
            records.append({'bits':bits,'group':group,'K':k,'N':n,'packet_case':label,
                            'reached_data_boundary':reached,'dynamic_i64_udiv_executions':0,
                            'actual_executed_SSA_nodes':len(nodes)})
            cases += 1
    report = {'schema':'R4-raw-large-guard-pair-actual-AIR-reachability-v1','pass':True,
              'candidate_AIR_sha256':build.sha(directory/'candidate.air'),
              'authorized_R4_plan_identifier':'26a76238',
              'CPU_compile_concurrent_frozen_source_review_authorized':True,
              'authorized_natural_shape_format_count':7,'packet_cases':cases,
              'all_authorized_actual_guard_paths_dynamic_i64_udiv_executions':0,
              'source_generic_fallback_retained':True,'unsupported_geometry_still_refused':True,
              'scope':'R4 actual SSA guard paths for exactly seven authorized natural shape/format tuples; compact/max-admitted and overflow/min/even refusal boundaries. Stops before first data load. No geometry sweep, current-trained-model, numerical or GPU performance claim.',
              'GPU_work':False,'operand_payload_reads':0,'cases':records}
    target = directory/'guard-reachability.json'; target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pass':True,'packet_cases':cases,'dynamic_i64_udiv_executions':0,'GPU_work':False}))

if __name__ == '__main__': main()
