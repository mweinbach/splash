#!/usr/bin/env python3
"""Freeze packed-V singleton worker from the actual authenticated HCv3 closure."""
import argparse,copy,hashlib,importlib.util,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/prefill_qsa_twopass_sep21')
DEFAULT_BASE=ROOT/'build/prefill-hc-inject-norm-sep21-worker-v3'
DEFAULT_BUILD=ROOT/'build/prefill-qsa-twopass-sep21-worker-v1'
QUALIFIED=ROOT/'build/prefill-qsa-twopass-sep21'
FLAG='SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21'
CHANGED={'runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm','dev/benchmarks/prefill4k_attribution.mm'}
TOOLS=('worker_overlay.py','worker_transform.py','worker_witness.py','worker.mk','worker_README.md')
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def transform(relative,text,path=None):return load(path or Path(__file__).with_name('worker_transform.py'),'qsa_worker_transform').transform(relative,text)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=DEFAULT_BASE);p.add_argument('--output',type=Path,default=DEFAULT_BUILD)
    args=p.parse_args();base=args.base.resolve();output=args.output.resolve()
    if output.exists() or ROOT/'build' not in output.parents or output==base:raise ValueError('Choose a fresh distinct private build')
    parentPath=base/'overlay-manifest.json';parent=json.loads(parentPath.read_text())
    if not parent.get('prefill_hc_composed'):raise ValueError('Expected strict HCv3 parent')
    parentHelper=base/'machinery/parent_overlay.py';parentMake=base/'machinery/worker.mk'
    if sha(parentHelper.read_bytes())!=parent['prefill_hc_parent_helper_sha256']:raise ValueError('Authenticated parent helper drift')
    helper=load(parentHelper,'qsa_authenticated_parent_helper');closure=helper.effective_closure(base,parentMake)
    seals,unsealed=helper.parent_input_seals(base,parent,closure);witness=helper.parent_artifact_witness(base)
    records={r['path']:r for r in parent['files']};deps=helper.frozen_dependency_closure(base,closure['objects'],records)
    cpuSealPath=QUALIFIED/'cpu-seal.json';seal=json.loads(cpuSealPath.read_text());sourceManifestPath=QUALIFIED/'source-manifest.json';qualified=json.loads(sourceManifestPath.read_text())
    if not seal['pass'] or seal['gpu_executed'] or sha(sourceManifestPath.read_bytes())!=seal['source_manifest_sha256']:raise ValueError('Qualified immutable CPU seal is invalid')
    for relative,digest in qualified['source_sha256'].items():
        if sha((QUALIFIED/'source'/relative).read_bytes())!=digest:raise ValueError(f'Qualified source drift: {relative}')
    for relative,digest in seal['artifact_sha256'].items():
        if sha((QUALIFIED/relative).read_bytes())!=digest:raise ValueError(f'Qualified artifact drift: {relative}')
    reportPath=ROOT/'build/release/flash/sep21-qsa-twopass-packed-v-v1.json';report=json.loads(reportPath.read_text())
    required=('original_prefix_prepared_Q_index_selection_exact','five_cache_planes_and_four_future_sparse_appends_exact','physical_KV_cache_padding_NaN_preserved',
        'finite_future_KV_causal_anchor_pass','source_active_selection_rejection_and_inactive_cells_pass','underflow_tie_and_exact_cancellation_anchors_pass',
        'nonfinite_fresh_V_source_numeric_rejection_preserved','external_guards_pass','real_governor_reservation_before_allocations','all_pack_Q_QK_softmax_pack_V_PV_unpack_gate_costs_inclusive')
    if not report['execution_complete'] or not report['packed_V'] or report['candidate_dispatches']!=9 or report['baseline_dispatches']!=6 or report['warm_gpu_seconds']<.150 or report['warm_pairs']<8 or any(not report[k] for k in required):raise ValueError('Root packed-V qualification is incomplete')
    cross=[]
    for relative in ('runtime/metal/MetalBackend.hpp','runtime/metal/CommandGraph.hpp','runtime/flash/FlashQSA.hpp','runtime/flash/FlashQSAFast.hpp',
                     'runtime/flash/FlashQSAMPP.hpp','runtime/metal/abi/FlashQSAFast.h','runtime/metal/abi/FlashQSA.h','runtime/metal/abi/FlashQSAMPP.h',
                     'runtime/flash/FlashQSA.cpp','runtime/flash/FlashQSAFast.cpp','runtime/flash/FlashQSAMPP.cpp',
                     'dev/benchmarks/prefill4k_attention/coalesced.cpp','dev/benchmarks/prefill4k_attention/bulk.cpp'):
        qualifiedPath=QUALIFIED/'source'/relative if relative.startswith('runtime/') else QUALIFIED/'source/base'/Path(relative).name
        q=qualifiedPath.read_bytes();b=(base/'source'/relative).read_bytes()
        if q!=b:raise ValueError(f'Qualified component ABI/header mismatch: {relative}')
        cross.append({'path':relative,'sha256':sha(q)})
    if (QUALIFIED/'source/base/coalesced.hpp').read_bytes()!=(base/'source/dev/benchmarks/prefill4k_attention/coalesced.hpp').read_bytes():raise ValueError('Qualified prepared-plane declaration differs')
    manifest=copy.deepcopy(parent);manifest.update({'route':'private-hcv3-packedV-global-f32P-qsa-fresh2048-singleton-v1','prefill_qsa_composed':True,
        'prefill_qsa_base_build':str(base),'prefill_qsa_base_manifest_sha256':sha(parentPath.read_bytes()),'prefill_qsa_base_make_path':str(parentMake),
        'prefill_qsa_base_make_sha256':sha(parentMake.read_bytes()),'prefill_qsa_parent_helper_sha256':sha(parentHelper.read_bytes()),
        'prefill_qsa_effective_parent_objects':[str(x) for x in closure['objects']],'prefill_qsa_effective_parent_airs':[str(x) for x in closure['airs']],
        'prefill_qsa_parent_input_seals':seals,'prefill_qsa_parent_owned_unsealed_inputs':unsealed,'prefill_qsa_parent_artifact_witness':witness,
        'prefill_qsa_parent_dependencies':deps,'prefill_qsa_cross_pins':cross,'prefill_qsa_component_cpu_seal_sha256':sha(cpuSealPath.read_bytes()),
        'prefill_qsa_component_report_sha256':sha(reportPath.read_bytes()),'prefill_qsa_flag':FLAG,'prefill_qsa_default':0,
        'prefill_qsa_added_workspace_bytes':478150656,'prefill_qsa_reused_prepared_bytes':31457280,'prefill_qsa_numerical_change_flag1_only':True,
        'prefill_qsa_scope':'singleton main fresh begin0 rows2048 nonverification packed-V only','prefill_qsa_tools':[],'prefill_qsa_link_inputs':[],
        'prefill_qsa_qualification_snapshots':[],'files':[],'gpu_executed':False,'payload_bytes_read':0,'whole_model_qualified':False})
    changed=[]
    for relative,record in records.items():
        original=(base/'source'/relative).read_bytes()
        if sha(original)!=record['overlay_sha256']:raise ValueError(f'Parent source drift: {relative}')
        data=transform(relative,original.decode()).encode()
        if data!=original:changed.append(relative)
        write(output/'source'/relative,data);manifest['files'].append({**record,'prefill_qsa_changed':data!=original,'prefill_qsa_parent_source_sha256':sha(original),'overlay_sha256':sha(data)})
    if set(changed)!=CHANGED:raise ValueError(f'Unexpected transformed files: {changed}')
    qualifiedNames=('twopass.hpp','twopass.cpp','candidate.metal')
    for name in qualifiedNames:
        relative=PRIVATE/name;source=QUALIFIED/'source/experiment'/name;data=source.read_bytes();write(output/'source'/relative,data)
        manifest['files'].append({'path':relative.as_posix(),'new_prefill_qsa_file':True,'qualified_component_source':str(source),'overlay_sha256':sha(data)})
    for name in ('worker_bridge.hpp','worker_policy_cpu.cpp'):
        relative=PRIVATE/name;data=(ROOT/relative).read_bytes()
        if name=='worker_bridge.hpp':data=data.replace(b'QUALIFIED_SHADER_SHA',sha((QUALIFIED/'source/experiment/candidate.metal').read_bytes()).encode()).replace(b'QUALIFIED_HOST_SHA',sha((QUALIFIED/'source/experiment/twopass.cpp').read_bytes()).encode())
        write(output/'source'/relative,data);manifest['files'].append({'path':relative.as_posix(),'new_prefill_qsa_file':True,'repository_source':str(ROOT/relative),'repository_sha256':sha((ROOT/relative).read_bytes()),'overlay_sha256':sha(data)})
    inputs={'REUSED':[],'CORE':[],'AIRS':[]}
    def freeze(path,category,relative):
        data=path.read_bytes();write(output/relative,data);inputs[category].append(relative.as_posix());manifest['prefill_qsa_link_inputs'].append({'source_path':str(path),'private_path':relative.as_posix(),'category':category,'sha256':sha(data)})
    for path in closure['objects']:
        if path.stem not in ('FlashForward','FlashWorker'):freeze(path,'CORE' if path in closure['core'] else 'REUSED',Path('reused/base')/path.relative_to(base))
    for path in closure['airs']:freeze(path,'AIRS',Path('reused/base')/path.relative_to(base))
    freeze(QUALIFIED/'metal/candidate.air','AIRS',Path('reused/qualified-qsa/candidate.air'))
    make='\n'.join(f'{key} := '+' '.join('$(BUILD)/'+relative for relative in value) for key,value in inputs.items())+'\n';write(output/'link-inputs.mk',make.encode())
    manifest['prefill_qsa_link_make_sha256']=sha(make.encode());manifest['prefill_qsa_changed_files']=changed
    for record in deps:
        if record['dependency_metadata_present']:
            path=Path(record['source_path']);relative=Path('qualification/parent-dependencies')/path.relative_to(base);write(output/relative,path.read_bytes());record['private_path']=relative.as_posix()
    for name in TOOLS:
        source=ROOT/PRIVATE/name;relative=Path('machinery')/name;data=source.read_bytes();write(output/relative,data);manifest['prefill_qsa_tools'].append({'source_path':str(source),'private_path':relative.as_posix(),'sha256':sha(data)})
    write(output/'machinery/parent_overlay.py',parentHelper.read_bytes())
    for name,path in {'base-manifest.json':parentPath,'base-worker.mk':parentMake,'base-cpu-witness.json':Path(witness['source_path']),
                      'component-cpu-seal.json':cpuSealPath,'component-source-manifest.json':sourceManifestPath,'packed-v-component-report.json':reportPath,
                      'independent-component-review.json':ROOT/'build/release/flash/sep21-qsa-packed-v-independent-component-review-v1.json'}.items():
        relative=Path('qualification')/name;data=path.read_bytes();write(output/relative,data);manifest['prefill_qsa_qualification_snapshots'].append({'source_path':str(path),'private_path':relative.as_posix(),'sha256':sha(data)})
    for name in ('splash-flash.config','splash.metallib.config'):write(output/name,(base/name).read_bytes().rstrip(b'\n')+b'-private-packedV-global-f32P-qsa-r2048-singleton-sep21-v1\n')
    write(output/'base-build.txt',(str(base)+'\n').encode());write(output/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    print(json.dumps({'prepared':str(output),'frozen_sources':len(manifest['files']),'parent_objects':len(closure['objects']),'parent_airs':len(closure['airs']),'new_single_arena_bytes':478150656,'gpu_executed':False}))
if __name__=='__main__':main()
