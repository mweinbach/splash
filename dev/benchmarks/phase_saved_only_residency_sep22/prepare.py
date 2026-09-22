#!/usr/bin/env python3
"""CPU-only saved-only startup lease variant; all phase arithmetic stays intact."""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess
ROOT=Path(__file__).resolve().parents[3]
FLAG='SPLASH_FLASH_PHASE_Q4_SAVED_ONLY_RESIDENCY_SEP22'
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def once(text,before,after):
    if text.count(before)!=1:raise ValueError('Resource variant source anchor differs:'+before[:100])
    return text.replace(before,after,1)
def transform(text):
    text=once(text,'      if(!phase_q4_sep21::requested())throw std::invalid_argument("private phase worker requires PREFILL_I8_DECODE_Q4=1");',
        '''      const char *savedOnlyValue=std::getenv("SPLASH_FLASH_PHASE_Q4_SAVED_ONLY_RESIDENCY_SEP22");
      const bool savedOnlyResourceRequested=!savedOnlyValue || std::string_view(savedOnlyValue)=="0"?false:
          std::string_view(savedOnlyValue)=="1"?true:throw std::invalid_argument("saved-only resource flag must be 0 or 1");
      if(savedOnlyResourceRequested && environmentSwitch("SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT"))
        throw std::invalid_argument("saved-only resource flag requires HYBRID_Q4_EXPERT_RESIDENT=0");
      if(!phase_q4_sep21::requested())throw std::invalid_argument("private phase worker requires PREFILL_I8_DECODE_Q4=1");''')
    text=once(text,'          !environmentSwitch("SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT") ||',
        '          (!savedOnlyResourceRequested && !environmentSwitch("SPLASH_FLASH_HYBRID_Q4_EXPERT_RESIDENT")) ||')
    text=once(text,'        if(savedResidency.hybridExpertAdded && savedResidencyLease &&',
        '''        if(savedOnlyResourceRequested && savedResidencyLease &&
            (savedResidencyLease.bufferCount()!=807 || savedResidencyLease.byteCount()!=131005546496ULL))
          throw std::logic_error("saved-only startup lease requires807owners/131005546496bytes");
        if(savedResidency.hybridExpertAdded && savedResidencyLease &&''')
    # Flag freezes once before any paths or backend; its marker is immutable.
    text=once(text,'      << R"(,"target_phase_policy_schema":)" << json::quote(phase_q4_sep21::schema)',
        '''      << R"(,"phase_resource_profile":)" << json::quote(savedResidency_.hybridExpertRequested?
          "phase-q4-expert-composite-startup832-v1":"phase-saved-only-startup807-existing-q4-backing-direct-transient-v1")
      << R"(,"phase_resource_variant_source_policy":"explicit saved-only flag1 requires Q4expert lease0; flag0 preserves original composite guard; mathematical derivative unchanged; all backing and reservations retained")"
      << R"(,"target_phase_policy_schema":)" << json::quote(phase_q4_sep21::schema)''')
    return text
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT/'build/prefill-i8-decode-q4-teacher-sep22-worker-v5');p.add_argument('--output',type=Path,default=ROOT/'build/phase-saved-only-residency-sep22-worker-v1');a=p.parse_args();base=a.base.resolve();out=a.output.resolve()
    if out.exists()or ROOT/'build'not in out.parents:raise ValueError('Choose fresh private output')
    mp=base/'overlay-manifest.json';parent=json.loads(mp.read_text());sp=base/'compiled-cpu-seal.json';seal=json.loads(sp.read_text())
    if not seal['pass']or sha(mp.read_bytes())!=seal['source_manifest_sha256']:raise ValueError('Sealed phase-v5 parent required')
    for rel,digest in seal['source_sha256'].items():
        if sha((base/'source'/rel).read_bytes())!=digest:raise ValueError('Parent source drift:'+rel)
    for rel,digest in seal['artifact_sha256'].items():
        if sha((base/rel).read_bytes())!=digest:raise ValueError('Parent artifact drift:'+rel)
    files=[]
    for r in parent['files']:
        rel=r['path'];data=(base/'source'/rel).read_bytes();changed=transform(data.decode()).encode()if rel=='runtime/flash/FlashWorker.mm'else data
        write(out/'source'/rel,changed);files.append({'path':rel,'sha256':sha(changed),'parent_sha256':sha(data),'changed':changed!=data})
    reused=[];frozen=[]
    for rel,digest in seal['artifact_sha256'].items():
        if not rel.endswith('.o')or Path(rel).stem=='FlashWorker':continue
        dest=Path('reused')/rel;data=(base/rel).read_bytes();write(out/dest,data);reused.append(dest.as_posix());frozen.append({'path':dest.as_posix(),'sha256':digest,'parent_path':rel})
    if len(reused)!=53:raise ValueError('Expected exactly53 unchanged effective objects')
    write(out/'splash.metallib',(base/'splash.metallib').read_bytes());write(out/'link-inputs.mk',('REUSED := '+' '.join('$(BUILD)/'+x for x in reused)+'\n').encode())
    for name in ('prepare.py','worker.mk'):write(out/'machinery'/name,(Path(__file__).parent/name).read_bytes())
    manifest={'schema':'private-phase-saved-only-startup-residency-v1','base':str(base),'base_source_manifest_sha256':sha(mp.read_bytes()),'base_compiled_seal_sha256':sha(sp.read_bytes()),'metallib_sha256':sha((base/'splash.metallib').read_bytes()),'files':files,'reused_objects':frozen,'only_recompiled_TU':'runtime/flash/FlashWorker.mm','no_headers_changed':True,'no_math_or_weight_or_workspace_or_governor_reservation_changed':True,'new_lease_owner_count':807,'new_lease_bytes':131005546496,'excluded_existing_Q4_owners':25,'excluded_lease_only_bytes':69363302400,'no_backing_freed':True,'native_numeric_semantic_performance_qualified':False,'GPU_executed':False,'model_payload_bytes_read':0}
    write(out/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps({'output':str(out),'sources':len(files),'reused_objects':len(reused),'GPU_executed':False}))
if __name__=='__main__':main()
