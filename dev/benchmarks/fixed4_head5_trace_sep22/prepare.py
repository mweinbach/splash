#!/usr/bin/env python3
"""CPU-only trace proposal: no prompt, token, model or output payload reads."""
import argparse
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from dev.benchmarks.fixed4_head5_trace_sep22 import probe

ROOT=probe.ROOT;HERE=Path(__file__).resolve().parent
PARENT=ROOT/'build/fixed4-R5-Root-bound-normal-sep22-v2/root-command.json'
PARENT_SHA='40a9946dacc45a8e3a77f97c5c18fbaae7d308e913153fa7b3536e8fd702e9ab'

def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--output',type=Path,default=ROOT/'build/fixed4-head5-single-trace-sep22-v1')
    args=parser.parse_args();out=args.output.resolve()
    if ROOT/'build'not in out.parents or out.exists():raise ValueError('Fresh private diagnostic preparation required')
    if probe.sha(PARENT)!=PARENT_SHA:raise ValueError('Qualified actual canonical command changed')
    base=json.loads(PARENT.read_text());out.mkdir()
    pins=dict(base['pins']);pins[str(PARENT)]=PARENT_SHA
    for file in (HERE/'probe.py',HERE/'prepare.py',HERE/'test_probe.py'):pins[str(file)]=probe.sha(file)
    # Inherited plan/model/tokenizer provenance pins are copied as metadata.
    # Root alone verifies them before execution; preparation opens no payload.
    trace=ROOT/'build/release/flash/sep22-fixed4-head5-single64-trace-v1.jsonl'
    report=ROOT/'build/release/flash/sep22-fixed4-head5-single64-runtime-v1.json'
    c={'schema':'Root-fixed4-head5-single-trace-command-v1','Root_GPU_only':True,'parent_command':str(PARENT),
       'parent_argv':base['argv'],'parent_environment':base['environment'],'pins':pins,'budget':64,'width':1,'requests':1,
       'warmup':0,'port':8051,'trace':str(trace),'report':str(report),'server_log':str(report)+'.server.log',
       'source_runtime_or_helper_changes':False,'performance_claim':False,'full_Head_cache_bitproof':False,
       'diagnostic_only_changes':['one request/output64 instead of canonical trials256; no warmup; privateport8051',probe.TRACE_FLAG+'='+str(trace)],
       'no_rows5_event_outcome':'inconclusive; no automatic output-budget expansion or invented proof'}
    path=out/'Root-command.json';path.write_text(json.dumps(c,indent=2)+'\n')
    script=out/'run-root.sh';script.write_text('#!/bin/sh\nset -eu\nexec '+shlex.join([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'probe.py'),'--command',str(path),'--command-sha256',probe.sha(path),'--run-root-gpu'])+'\n')
    print(json.dumps({'command':str(path),'sha256':probe.sha(path),'GPU_work':False,'payload_reads':0}))

if __name__=='__main__':main()
