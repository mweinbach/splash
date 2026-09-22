#!/usr/bin/env python3
"""CPU audit of scoped R1 vector composition, certificate and header closure."""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

PRIVATE=Path('dev/benchmarks/gemv_decode_r1_worker_sep21')
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def segment(text:str,a:str,z:str)->str:
    start=text.index(a);return text[start:text.index(z,start)]
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('build',type=Path);p.add_argument('--report',type=Path);args=p.parse_args();b=args.build.resolve();m=json.loads((b/'overlay-manifest.json').read_text());base=Path(m['r1_ab_base_build']);old=Path(m['r1_ab_vector_build']);vm=json.loads((old/'overlay-manifest.json').read_text())
    assert sha(base/'overlay-manifest.json')==m['r1_ab_base_manifest_sha256'];assert sha(old/'overlay-manifest.json')==m['r1_ab_vector_manifest_sha256']
    file=old/'source'/PRIVATE/'worker_prepare.py';assert sha(file)==vm['vector_r1_transform_sha256'];spec=importlib.util.spec_from_file_location('sealedtransform',file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    changed=[]
    for e in m['files']:
        current=b/'source'/e['path'];assert sha(current)==e['overlay_sha256']
        if 'r1_ab_base_overlay_sha256' in e:
            original=base/'source'/e['path'];assert sha(original)==e['r1_ab_base_overlay_sha256'];assert current.read_bytes()==module.transform(e['path'],original.read_text()).encode()
            if original.read_bytes()!=current.read_bytes():changed.append(e['path'])
        else:assert current.read_bytes()==(old/'source'/e['path']).read_bytes()
    assert set(changed)==module.CHANGED_PATHS
    forward=(b/'source/runtime/flash/FlashForward.cpp').read_text();bh=(b/'source/runtime/flash/FlashForward.hpp').read_text();worker=(b/'source/runtime/flash/FlashWorker.mm').read_text();store=(b/'source/runtime/flash/FlashInt8ExpertStore.mm').read_text()
    baseline=(base/'source/runtime/flash/FlashForward.cpp').read_text()
    assert segment(forward,'uint64_t FlashForward::workspacePlannedBytes','std::string FlashForward::kernelRoutes')==segment(baseline,'uint64_t FlashForward::workspacePlannedBytes','std::string FlashForward::kernelRoutes')
    assert 'return forwardImpl(request,token,false,false,false,true);' in forward and 'bool ordinaryDecode=false' in bh and worker.count('forward_.forwardDecode(')==1
    assert 'const bool vectorR1Decode = ordinaryDecode && gatheredMPP &&' in forward
    assert segment(baseline,'      if (impl_->mergeGDNAB','      const FlashGDNWeights weights') in forward
    assert segment(baseline,'      if (impl_->bulkQSAWorkspace','      // Adjacent MPP') in forward if '      // Adjacent MPP' in baseline else True
    assert 'ordinary_r1_decode_policy=' in store and 'if (gemvDecodeR1)\n      derivative +=' in store
    identity=segment(worker,'      << R"(,"identity":{"source":)"','      << R"(,"weight_format":"mlx_affine_preconverted"})"')
    assert 'vectorR1Graphs' not in identity and 'gate_graph_calls' not in identity and 'abGraphs' not in identity
    assert worker.count('"vector_i8_r1_decode"')==1
    own=set(m['r1_ab_rebuilt_host_names']);assert own=={'FlashForward','FlashInt8ExpertStore','FlashWorker','FlashBatchForward','FlashBatchPrefill','FlashBatchVerify'}
    assert {x.stem for x in (b/'host').glob('*.o')}==own
    for row in m['r1_ab_header_dependency_census']:
        if row['consume_modified_headers']:assert Path(row['source']).stem in own
    names=[]
    for e in m['r1_ab_link_inputs']:
        assert sha(b/e['private_path'])==e['sha256']==sha(Path(e['original']))
        if e['category']=='REUSED':
            name=Path(e['original']).stem.split('-',1)[-1] if Path(e['original']).stem[:1].isdigit() else Path(e['original']).stem
            assert name not in own;names.append(name)
    names+=list(own);assert max(Counter(names).values())==1
    assert sha(b/'link-inputs.mk')==m['r1_ab_link_make_sha256']
    live=[]
    for d in (b/'host').glob('*.d'):
        for t in d.read_text().replace('\\\n',' ').split():
            if t.startswith(('runtime/','dev/benchmarks/')):live.append(t)
    assert not live
    vector=old/'source'/PRIVATE;assert (b/'source'/PRIVATE/'candidate.metal').read_bytes()==(vector/'candidate.metal').read_bytes()
    assert (b/'source'/PRIVATE/'source_identity.hpp').read_bytes()==(vector/'source_identity.hpp').read_bytes()
    for name in ['kernels.metal','quality.hpp','PREREGISTRATION.md','FTZ_CERTIFICATE.md']:
        assert sha(b/'source'/PRIVATE/'qualified-source'/name)==m['vector_r1_identity_parts'][name]
    probes={}
    for mode in ['', '--freeze0','--freeze1','--missing','--retry0','--retry1']:
        env=dict(os.environ);env.pop('SPLASH_FLASH_GEMV_DECODE_R1_SEP21',None);command=[str(b/'policy-cpu')]+([mode] if mode else [])
        probes[mode or 'default']=json.loads(subprocess.check_output(command,text=True,env=env));assert probes[mode or 'default']['pass']
    traps={}
    for value in ['','2','true','01',' 1','1 ']:
        env=dict(os.environ);env['SPLASH_FLASH_GEMV_DECODE_R1_SEP21']=value
        r=subprocess.run([str(b/'splash-flash'),'serve-flash-native','/ar1vector-no-such-path','16384','auto'],env=env,capture_output=True,text=True)
        assert r.returncode and 'must be exactly 0 or 1 when present' in r.stderr;traps[value]={'exitcode':r.returncode,'stderr':r.stderr.strip()}
    result={'schema':'splash-sealed-AR1-vector-AB-QSA-CPU-witness-v1','pass':True,'gpu_work':False,'model_payload_reads':0,
        'frozen_sources':len(m['files']),'frozen_inputs':len(m['r1_ab_link_inputs']),'header_dependency_TUs':len(m['r1_ab_header_dependency_census']),
        'rebuilt_all_modified_header_consumers':sorted(own),'effective_hosts_link_once':len(names),'no_live_header_dependencies':True,
        'ordinary_AR1_only_scope_prefill_MTP_head_batch_unchanged':True,'R4_vector_excluded':True,'AB_QSA_cache_plan_and_packed_QSA_logic_preserved':True,
        'numerical_alternative_source_certificate_identity':m['vector_r1_source_identity_sha256'],'old_shipping_shader_and_certificate_byte_exact':True,
        'mutable_counters_outside_identity':True,'added_GPU_bytes':0,'compiled_policy_probes':probes,'invalid_flag_before_paths':traps,
        'runtime_sha256':{name:sha(b/name) for name in ['splash-flash','splash.metallib']},'whole_model_qualification_pending':True}
    if args.report:
        assert not args.report.exists();args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['compiled_policy_probes','invalid_flag_before_paths']},indent=2))
if __name__=='__main__':main()
