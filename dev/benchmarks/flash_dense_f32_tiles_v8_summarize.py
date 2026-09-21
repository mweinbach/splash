"""Summarize Root-generated dense tile screen reports; performs no GPU work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    matrices={}
    reports=[]
    for path in args.reports:
        report=json.loads(path.read_text())
        reports.append({'path':str(path.resolve()),'pass':report.get('pass'),
                        'library_sha256':report.get('loaded_metallib_sha256'),
                        'cases':len(report.get('cases',report.get('completed_cases',[]))),
                        'precision_traps':len(report.get('precision_traps',[]))})
        for case in report.get('cases',report.get('completed_cases',[])):
            key=(case['projection'],case['rows'])
            matrices.setdefault(key,[]).append(case)
    winners=[]
    all_strict=True
    for (prefix,rows),cases in matrices.items():
        eligible=[c for c in cases if c.get('strict_relative_l2_pass') and c.get('accuracy_pass')]
        all_strict &= len(eligible)==len(cases)
        candidates=[c for c in eligible if c['tile'] not in ('m8n64s4','m8n128s4','m16n64s4','m16n128s4')]
        candidate=min(candidates,key=lambda c:c['small_timing']['median_gpu_seconds']) if candidates else None
        best=min(eligible,key=lambda c:c['small_timing']['median_gpu_seconds']) if eligible else None
        winners.append(dict(projection=prefix,rows=rows,bits=cases[0]['bits'],group_size=cases[0]['group_size'],
            candidate_tile=candidate['tile'] if candidate else None,
            candidate_gpu_seconds=candidate['small_timing']['median_gpu_seconds'] if candidate else None,
            speedup_vs_selected_gpu=candidate['speedup_vs_selected_policy_gpu'] if candidate else None,
            speedup_vs_selected_wall=candidate['speedup_vs_selected_policy_wall'] if candidate else None,
            strict_candidate_relative_l2=candidate['small_vs_selected_policy']['relative_l2'] if candidate else None,
            any_tile_winner=best['tile'] if best else None,
            strict_failures=[dict(tile=c['tile'],raw_l2=c['small_vs_raw_f32_coefficients']['relative_l2'],
                selected_l2=c['small_vs_selected_policy']['relative_l2']) for c in cases if c not in eligible]))
    result=dict(schema='flash-private-original-f32-dense-tiles-v8-summary-v1',reports=reports,
        all_cases_strict=all_strict,per_matrix_row_winners=winners,gpu_commands_executed=0,
        source_weights_modified=False,production_routes_changed=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':str(args.output.resolve()),'matrix_row_groups':len(winners),'all_cases_strict':all_strict}))


if __name__=='__main__': main()
