#!/usr/bin/env python3
"""Freeze only the original native MoE M64 threshold change over a qualified private build."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[3]


def sha(data): return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'build/prefill-moe-sep21-native-m64-worker-v2')
    args=parser.parse_args();base=args.base.resolve();output=args.output.resolve()
    if base ==output or ROOT/'build' not in output.parents:
        raise ValueError('Native M64 policy experiment requires a distinct private build')
    baseManifestPath=base/'overlay-manifest.json'
    parent=json.loads(baseManifestPath.read_text())
    if parent.get('omitted_original_target_tensor_count') !=432 or not parent.get('gathered_mpp_composed'):
        raise ValueError('Expected private Full512 original-target-omitted gathered+bulk source')
    manifest=copy.deepcopy(parent)
    manifest.update({
        'route':'private-allrows-full512-bulk-gathered-native-m64-at-2k-sep21-v1',
        'native_moe_m64_minimum_rows':2048,'original_native_moe_m64_minimum_rows':4096,
        'native_moe_m64_policy':True,'native_bucket_job_list_used':True,
        'additional_hit_job_prefix_dispatches':0,'coefficient_scale_and_bf16_boundaries_unchanged':True,
        'base_build':str(base),'base_manifest_sha256':sha(baseManifestPath.read_bytes()),
        'gpu_executed':False,'payload_bytes_read':0,
        'private_transform_sha256':sha(Path(__file__).read_bytes()),'files':[]})
    changed=0
    for record in parent['files']:
        relative=record['path'];data=(base/'source'/relative).read_bytes()
        if sha(data) !=record['overlay_sha256']:
            raise ValueError(f'Base source drift: {relative}')
        if relative =='runtime/flash/FlashMoEBlocked.cpp':
            text=data.decode();before='wide && rows >= 4096 && !hasHotExpertCache'
            if text.count(before) !=1:raise ValueError('Native row policy source drift')
            data=text.replace(before,'wide && rows >= 2048 && !hasHotExpertCache').encode();changed +=1
        if relative =='runtime/flash/FlashMoEBlocked.hpp':
            text=data.decode()
            if text.count('rows4096plus') !=2:raise ValueError('Native M64 semantics drift')
            data=text.replace('rows4096plus','rows2048plus-private-sep21-native-policy').encode();changed +=1
        destination=output/'source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists() or destination.read_bytes() !=data:destination.write_bytes(data)
        manifest['files'].append({**record,'native_m64_changed':relative in ('runtime/flash/FlashMoEBlocked.cpp','runtime/flash/FlashMoEBlocked.hpp'),
                                  'base_overlay_sha256':record['overlay_sha256'],'overlay_sha256':sha(data)})
    if changed !=2:raise ValueError('Expected policy implementation and explicit private header marker')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('splash.metallib','splash.metallib.config'):
        shutil.copy2(base/name,output/name)
    (output/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'base-build.txt').write_text(str(base)+'\n')
    print(json.dumps({'prepared':str(output),'source_files_checked':len(manifest['files']),'changed_files':changed,
                      'gpu_executed':False,'payload_bytes_read':0,'native_job_tile_for_2048':64}))


if __name__=='__main__':main()
