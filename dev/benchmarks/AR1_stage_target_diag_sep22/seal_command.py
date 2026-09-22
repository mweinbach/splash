#!/usr/bin/env python3
"""CPU-only command envelope. Canonical token files are never opened here."""
import argparse,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[3]
HERE=pathlib.Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def need(value,message):
    if not value:raise ValueError(message)
def main():
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument('--build',type=pathlib.Path,required=True)
    parser.add_argument('--suffix',default='v1');args=parser.parse_args();build=args.build.resolve()
    ready=json.loads((build/'CPU_READY.json').read_text());need(ready['pass'],'Complete CPU closure required')
    for record in ready['compiled54_objects']:need(sha(build/record['path'])==record['sha256'],'Compiled object drift')
    for record in ready['files']:need(sha(build/'source'/record['path'])==record['sha256'],'Source/header drift')
    need(sha(build/'splash-flash')==ready['exe_sha256'] and sha(build/'splash.metallib')==ready['library_sha256'],'Current executable/library drift')
    baseline=ROOT/'build/currentQ4-4k-standard-B1-Root-sep22-v1/root-command-v2.json'
    base=json.loads(baseline.read_text());argv=base['argv'];overrides={}
    for index,value in enumerate(argv):
        if value=='--env':
            key,val=argv[index+1].split('=',1);overrides[key]=val
    profile=json.loads((ROOT/'.splash-local-profile.json').read_text())
    environment=dict(profile['environment']);environment.update(overrides)
    environment.update({'SPLASH_FLASH_MTP':'0','SPLASH_FLASH_BATCH_MTP':'0','SPLASH_FLASH_BATCH_MTP_PREFILL':'0',
        'SPLASH_FLASH_MTP_TEACHER_CACHE_ONLY':'0','SPLASH_FLASH_GPU_PREFILL_COPY':'0',
        'SPLASH_FLASH_SINGLETON_TEACHER_BULK2048_SEP21':'0','SPLASH_FLASH_TEACHER_BULK_QA_PAUSE_MS':'0',
        'SPLASH_FLASH_DIAG_AR1_TARGET_STAGE_SEP22':'1'})
    evidence=ROOT/'build/release/flash'
    output=evidence/f'sep22-current-standard-AR1-target-stage-native-evidence-{args.suffix}.json'
    diagnostic=evidence/f'sep22-current-standard-AR1-target-stage-profile-{args.suffix}.json'
    log=evidence/f'sep22-current-standard-AR1-target-stage-native-{args.suffix}.log'
    environment['SPLASH_FLASH_DIAG_AR1_STAGE_OUTPUT_SEP22']=str(diagnostic)
    needed=('SPLASH_FLASH_DENSE_W8A8_PREFILL_SEP21','SPLASH_FLASH_GDN_PREFILL_FMA_SEP21',
            'SPLASH_FLASH_PREFILL_QSA_TWOPASS_SEP21','SPLASH_FLASH_PREFILL_HC_INJECT_NORM_SEP21')
    need(all(environment[k]=='1' for k in needed),'All four current prefill switches ON')
    command=build/'root-command.json';need(not command.exists(),'Fresh command envelope required')
    for path in (output,diagnostic,log):need(not path.exists(),'Fresh Root output required')
    programs=[HERE/'run_root.py',HERE/'bridge.hpp',HERE/'overlay.py',HERE/'prepare.py',HERE/'seal_command.py',
        HERE/'PLAN.json',HERE/'test_runner.py',HERE/'cpu_test.cpp',build/'CPU_READY.json',
        ROOT/'server/runtime.py',ROOT/'server/protocol.py',ROOT/'.splash-local-profile.json',baseline]
    value={'schema':'Root-current-standard-AR1-stage-one-native-command-v1','Root_GPU_only':True,
        'build':str(build),'package':str(ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1'),
        'context':2048,'max_context':16384,'output_tokens':64,'request_id':1,'generation':1,'requests':1,
        'tokens':str(ROOT/'build/release/flash/prefill4k-fixture/code2048.tokens.json'),
        'tokens_file_sha256':'72e5a23f0504ba862d43c22e01e820fc4007b3b939ec4cbc8518aaee499832fb',
        'tokens_u32le_sha256':'55cf1a355b4a2c97012c752b87955198ef3bb1f1b992b3fb48d35ff7659f3795',
        'exe_sha256':ready['exe_sha256'],'library_sha256':ready['library_sha256'],
        'baseline_source_identity_sha256':ready['parent_source_identity_sha256'],'diagnostic_source_id':ready['diagnostic_source_id'],
        'environment':environment,'report':str(output),'profile':str(diagnostic),'native_log':str(log),
        'program_pins':{str(path):sha(path) for path in programs},
        'canonical_throughput_claim':False,'SourceWorld_qualification_claim':False,'baseline_rerun_requested':False}
    command.write_text(json.dumps(value,indent=2)+'\n');digest=sha(command)
    line=f'.venv/bin/python -B {HERE.relative_to(ROOT)}/run_root.py --command {command} --command-sha256 {digest} --run-root-gpu'
    (build/'root-command.txt').write_text(line+'\n')
    print(json.dumps({'Root_command':line,'command_sha256':digest,'GPU_work':False,'tokenfile_payload_reads':0}))
if __name__=='__main__':main()
