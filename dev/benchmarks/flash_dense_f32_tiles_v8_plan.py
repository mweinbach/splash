"""Prepare private dense F32 tile screens without creating a Metal device."""
from __future__ import annotations
import argparse
import hashlib
import json
import shlex
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path('install/local-models/Flash-Next-oQ4e-mtp-v1'))
    parser.add_argument('--operand-store', type=Path, default=Path('install/local-models/Flash-Next-operands-v1'))
    parser.add_argument('--build', type=Path, default=Path('build/flash-dense-f32-tiles-v8'))
    parser.add_argument('--output-dir', type=Path, default=Path('build/release/flash/dense-f32-tiles-v8'))
    args = parser.parse_args()
    package, store, build, output = [p.resolve() for p in (args.package,args.operand_store,args.build,args.output_dir)]
    raw = (package/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    tensors, quant = manifest['tensors'], manifest['quantization']
    groups = {}
    for layer in range(48):
        for role in ('linear_attn.in_proj_qkv','self_attn.q_proj','self_attn.o_proj'):
            prefix = f'language_model.model.layers.{layer}.{role}'
            if prefix+'.weight' not in tensors: continue
            q = quant.get(prefix,quant)
            n = tensors[prefix+'.weight']['shape'][0]
            k = tensors[prefix+'.scales']['shape'][1]*q['group_size']
            key=(role,n,k,q['bits'],q['group_size'])
            groups.setdefault(key,dict(prefix=prefix,role=role,n=n,k=k,bits=q['bits'],group_size=q['group_size']))
    expanded=list(groups.values())
    quick=[g for g in expanded if g['prefix'] in (
        'language_model.model.layers.0.linear_attn.in_proj_qkv',
        'language_model.model.layers.3.self_attn.q_proj',
        'language_model.model.layers.15.self_attn.o_proj')]
    oracle=build/'flash-dense-f32-tiles-v8-oracle'
    library=build/'flash-dense-f32-tiles-v8.metallib'
    flags={'SPLASH_FLASH_QMV_F32':'1','SPLASH_FLASH_OPERAND_STORE':str(store),
           'FLASH_FLOAT_CACHE_ROWS':'4,8,16','FLASH_FLOAT_CACHE_TILES':'0,2,4,5,6,7',
           'FLASH_FLOAT_CACHE_REPEATS':'6','FLASH_FLOAT_CACHE_CPU_COLUMNS':'19',
           'FLASH_DENSE_F32_TILES_V8_BOUNDARY_AUDIT':'0',
           'FLASH_FLOAT_CACHE_CONTINUE_ON_ACCURACY_FAILURE':'1'}
    commands={}
    output.mkdir(parents=True,exist_ok=True)
    for label, selected in (('quick',quick),('expanded',expanded)):
        assignments=flags|{'FLASH_FLOAT_CACHE_PREFIXES':','.join(g['prefix'] for g in selected)}
        command=shlex.join(['env',*[f'{key}={value}' for key,value in assignments.items()],str(oracle),str(library),str(package),str(output/f'{label}-screen.json')])
        commands[label]=command
        (output/f'run-{label}-screen.sh').write_text('#!/bin/sh\n# Root alone executes GPU work; preparer never executes this.\nexec '+command+'\n')
    commands['traps']=shlex.join([str(oracle),'--trap-only',str(library),str(output/'precision-traps.json')])
    report=dict(schema='flash-private-original-f32-dense-tiles-v8-plan-v1',
        source_identity_sha256=manifest['source_identity_sha256'],manifest_sha256=hashlib.sha256(raw).hexdigest(),
        oracle_path=str(oracle),oracle_sha256=hashlib.sha256(oracle.read_bytes()).hexdigest(),
        library_path=str(library),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        command_timing_bytes=200,quick=quick,expanded=expanded,flags=flags,commands=commands,
        candidate_tiles={'4':'M8N32 SIMD2','5':'M8N32 SIMD4','6':'M16N64 SIMD8','7':'M8N64 SIMD2'},
        acceptance='strict relative L2 below 1e-4 against raw original-F32 affine and selected production policy; no boundary exception enabled',
        source_weights_modified=False,production_routes_changed=False,gpu_commands_executed=0)
    (output/'plan.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(plan=str(output/'plan.json'),quick_groups=len(quick),expanded_groups=len(expanded),gpu_commands_executed=0)))


if __name__=='__main__': main()
