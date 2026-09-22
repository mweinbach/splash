#!/usr/bin/env python3
"""Freeze opt-in low-SIMD/register INT8 expert prefill over an original-target-omitted build."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/prefill_moe_sep21')
EXTRAS=('bridge.hpp','memory.metal','register.metal')


def sha(data):return hashlib.sha256(data).hexdigest()


def transform(text):
    before='#include "FlashInt8ExpertStore.hpp"'
    if text.count(before) !=1:raise ValueError('Store include contract drift')
    text=text.replace(before,before +'\n#include "dev/benchmarks/prefill_moe_sep21/bridge.hpp"\n#include <cstdlib>')
    point='constexpr uint64_t kAlignment = 16384;'
    addition=r'''
const prefill_moe_sep21::Variant *privatePrefillMoEVariant(uint32_t rows) {
  const auto selected=prefill_moe_sep21::requestedVariantIndex();
  return selected &&rows ==32 ? &prefill_moe_sep21::variants[selected -1] :nullptr;
}
'''
    if text.count(point) !=1:raise ValueError('Store namespace contract drift')
    text=text.replace(point,point +addition)
    before='  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;'
    if text.count(before) !=2:raise ValueError('Store launch contract drift')
    text=text.replace(before,'  const auto *variant=privatePrefillMoEVariant(p.tile_rows);\n  const uint32_t threads=variant ? variant->sg *32 :tile ==FlashMoEBlockedTile::M64N64 ? 256 :128;')
    for phase, gate in [('gate_up',True),('down_scatter',False)]:
        before=f'graph.add(pipeline("{phase}", tile),'
        if text.count(before) !=1:raise ValueError('Store producer naming drift')
        text=text.replace(before,f'graph.add(variant ? prefill_moe_sep21::pipelineName(*variant,{str(gate).lower()}) :pipeline("{phase}", tile),')
    point='    numericalIdentity = hash(derivative.data(), derivative.size());'
    if text.count(point) !=1:raise ValueError('Store numerical identity contract drift')
    text=text.replace(point,'    const auto prefillPolicy=prefill_moe_sep21::policyIdentity();\n    if (!prefillPolicy.empty()) derivative +="prefill_moe_policy=" +prefillPolicy +"\\n";\n' +point)
    return text


def transform_blocked(text):
    text='#include "dev/benchmarks/prefill_moe_sep21/bridge.hpp"\n' +text
    point='const char *flashMoEBlockedRouteSemantics() {\n'
    if text.count(point) !=1:raise ValueError('Blocked route identity contract drift')
    addition=r'''  if (prefill_moe_sep21::requestedVariantIndex()) {
    static const std::string semantics=[] {
      const char *original=directAEnabled()
        ? (m64Enabled() ? kFlashMoEBlockedDirectAM64Semantics :kFlashMoEBlockedDirectASemantics)
        :m64Enabled() ? kFlashMoEBlockedQ4x8M64Semantics
        :q4x8Enabled() ? kFlashMoEBlockedQ4x8Semantics :kFlashMoEBlockedSemantics;
      return std::string(original) +";" +prefill_moe_sep21::policyIdentity();
    }();
    return semantics.c_str();
  }
'''
    return text.replace(point,point +addition)


def transform_worker(text):
    text='#include "dev/benchmarks/prefill_moe_sep21/bridge.hpp"\n' +text
    point='      (void)gathered_mpp::requestedMaximumRows();'
    if text.count(point) !=1:raise ValueError('Worker early flag validation contract drift')
    text=text.replace(point,'      (void)prefill_moe_sep21::requestedVariantIndex();\n' +point)
    point='      << R"(,"target_gathered_mpp_numerical_policy":)"'
    if text.count(point) !=1:raise ValueError('Worker status policy contract drift')
    addition=r'''      << R"(,"target_prefill_moe_variant":)" <<prefill_moe_sep21::requestedVariantIndex()
      << R"(,"target_prefill_moe_policy":)" <<(prefill_moe_sep21::requestedVariantIndex() ? json::quote(prefill_moe_sep21::policyIdentity()) :"null")
'''
    return text.replace(point,addition +point)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'build/prefill-moe-sep21-worker-v2')
    args=parser.parse_args();base=args.base.resolve();output=args.output.resolve()
    if base ==output or ROOT/'build' not in output.parents:
        raise ValueError('Prefill arithmetic experiment requires a distinct private build')
    parentPath=base/'overlay-manifest.json';parent=json.loads(parentPath.read_text())
    if parent.get('omitted_original_target_tensor_count') !=432 or not parent.get('gathered_mpp_composed'):
        raise ValueError('Expected private Full512 original-target-omitted gathered+bulk source')
    manifest=copy.deepcopy(parent)
    manifest.update({'route':'private-allrows-full512-bulk-gathered-low-simd-register-sep21-v1',
      'prefill_moe_required_environment':'SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT=1..11',
      'prefill_moe_native_job_tile_required':32,'prefill_moe_other_job_tiles_original':True,
      'prefill_moe_original_signed_i8_and_scales':True,'prefill_moe_numerical_qualification_pending':True,
      'prefill_moe_base_build':str(base),'prefill_moe_base_manifest_sha256':sha(parentPath.read_bytes()),
      'prefill_moe_transform_sha256':sha(Path(__file__).read_bytes()),'gpu_executed':False,'payload_bytes_read':0,'files':[]})
    changed=0
    for record in parent['files']:
        relative=record['path'];data=(base/'source'/relative).read_bytes()
        if sha(data) !=record['overlay_sha256']:raise ValueError(f'Base source drift:{relative}')
        transforms={'runtime/flash/FlashInt8ExpertStore.mm':transform,
                    'runtime/flash/FlashMoEBlocked.cpp':transform_blocked,
                    'runtime/flash/FlashWorker.mm':transform_worker}
        if relative in transforms:data=transforms[relative](data.decode()).encode();changed +=1
        target=output/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or target.read_bytes() !=data:target.write_bytes(data)
        manifest['files'].append({**record,'prefill_moe_changed':relative in transforms,
                                 'base_overlay_sha256':record['overlay_sha256'],'overlay_sha256':sha(data)})
    if changed !=3:raise ValueError('Expected Store arithmetic, blocked route identity and worker status')
    for name in EXTRAS:
        relative=str(PRIVATE/name);data=(ROOT/relative).read_bytes();target=output/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        manifest['files'].append({'path':relative,'new_private_file':True,'patched':True,'overlay_sha256':sha(data)})
    output.mkdir(parents=True,exist_ok=True)
    (output/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'base-build.txt').write_text(str(base)+'\n')
    print(json.dumps({'prepared':str(output),'checked_base_files':len(parent['files']),'changed_files':changed,
                      'private_added_files':len(EXTRAS),'gpu_executed':False,'payload_bytes_read':0}))


if __name__=='__main__':main()
