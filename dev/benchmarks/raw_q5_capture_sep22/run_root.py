#!/usr/bin/env python3
"""Root-only sealed actual-input capture invocation. Preparers never execute it."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if len(sys.argv)!=3:raise ValueError('Root runner COMMAND_JSON EXACT_METADATA_SHA required')
    path=Path(sys.argv[1]);pin=sys.argv[2]
    if sha(path)!=pin:raise ValueError('sealed capture command metadata differs')
    c=json.loads(path.read_text())
    if c['schema']!='raw-q5-current-input-capture-root-command-v1' or c['capture_scope']!='current VerifyR4 layer0 GDN output ordinary target greedy inputs':raise ValueError('registered capture command scope differs')
    for p,h in c['artifact_pins'].items():
        if sha(p)!=h:raise ValueError('Root capture code/artifact pin differs:'+p)
    argv=c['argv'];env=dict(os.environ)
    for k in list(env):
        if k.startswith('SPLASH_'):del env[k]
    env.update(c['environment'])
    for k in ['SPLASH_FLASH_COMPACT_NATIVE_R4_VERIFY_SEP22','SPLASH_FLASH_COMPACT_R4_PREFLIGHT_BUNDLE_SEP22','SPLASH_FLASH_HC_PAD_VERIFY_R4_SEP22','SPLASH_FLASH_GUARD_HC_FAST_COMPOSITE_SEP22']:
        if env.get(k)!='1':raise ValueError('strict current composite required:'+k)
    if argv[1]!='--gpu' or len(argv)!=7 or Path(argv[5]).exists() or Path(argv[6]).exists() or Path(argv[6]+'.failure.json').exists():raise ValueError('capture Root argv/fresh paths differ')
    subprocess.run(argv,cwd=c['cwd'],env=env,check=True)
    # Validate only bounded result/provenance declarations here. Payload checks
    # and hashing happen inside the completed Root capture executable.
    report=json.loads(Path(argv[6]).read_text());metadata=Path(argv[5])/'layer0-GDNout-VerifyR4.json';m=json.loads(metadata.read_text())
    for j in [report,m]:
        if not j.get('pass') or not j.get('current_input_capture_preservation_proved') or j.get('rowpair_math_qualified') or j.get('payload_disk_bytes')!=49152 or j.get('state_tape_payload_disk_bytes')!=0 or j.get('actual_capture_owned_buffer_guard_cases')!=6 or not j['allocation']['backend_destroyed']:raise ValueError('completed current-input preservation declaration differs')
        if j['actual_capture_clone_executable_sha256']!=c['artifact_pins'][argv[0]] or j['actual_capture_library_sha256']!=c['artifact_pins'][argv[2]]:raise ValueError('actual capture executable/library receipt differs')
    print(json.dumps({'pass':True,'scope':c['capture_scope'],'capture_metadata':str(metadata),'report':argv[6],'payload_disk_bytes':49152,'actual_MTP_proposal_capture':False,'rowpair_math_qualified':False}))
if __name__=='__main__':main()
