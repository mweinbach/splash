#!/usr/bin/env python3
"""Freeze standalone component sources and exact current Core4, never payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/Users/mweinbach/Projects/splash')
PARENT=ROOT/'build/rawQ4-GDN26-VerifyR4-composite-sep22-worker-v2'
CODE=ROOT/'dev/benchmarks/main_prefill_rhs_tile64_sep22'
CORE={'047-MetalBackend.o':'f22b2da0996210e43d264feccd677ba5de89ac3e5eee0ccabed230ff5505dbd0',
      '048-DeviceCapabilities.o':'acffc16c907d19ea249c02bbd7531b9dc5159532eed4d0f2ffc4717667a43756',
      '049-Protocol.o':'2ee48752656ee597c3f12cada0fd97c4dcc7261d1c33b9a7fcb0b78042e061aa',
      '050-MemoryGovernor.o':'ebd872a379edc28f779cde5ed7b257c4c6d2f129e719e52bd5c06ea9d34e0ada'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v:raise ValueError(m)
def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    destination=a.destination.resolve();require(not destination.exists() and destination.is_relative_to(ROOT/'build'),'Fresh private component source directory')
    require(sha(PARENT/'splash-flash')=='663663067a6b696811980c5afa3d2cca2dd1b0b28629e6d9b326a7973d084438','Exact current parent worker')
    require(sha(PARENT/'splash.metallib')=='7540286fde20ea7032f1aadbeeb0107920dfc9c42aed05feb7bb3c9373cde7c8','Exact current parent original library')
    source=destination/'source';shutil.copytree(PARENT/'source',source)
    local=source/'dev/benchmarks/main_prefill_rhs_tile64_sep22';local.mkdir(parents=True,exist_ok=True)
    for file in CODE.iterdir():
        if file.is_file():shutil.copy2(file,local/file.name)
    for name in ('prefill4k_allrows_qmv_one_layer.hpp','flash_expert_int8_bucket_reference.hpp'):
        origin=ROOT/'dev/benchmarks'/name;target=source/'dev/benchmarks'/name
        if target.exists():require(sha(target)==sha(origin),'Current parent/private inline helper source differs:'+name)
        else:shutil.copy2(origin,target)
    supplemental=[]
    for name in ('runtime/metal/kernels/shared/flash_moe_buckets.metal','runtime/metal/kernels/shared/flash_moe_direct_a.metal'):
        target=source/name
        if not target.exists():
            origin=ROOT/name;shutil.copy2(origin,target);supplemental.append({'path':name,'origin':str(origin),'sha256':sha(origin),
                'reason':'Current archive omits original leaf; fresh source compiled against frozen current-parent ABI/common headers. Exact integer and BF16 bit-copy output is checked independently.',
                'historical_original_AIR_source_lineage_claimed':False})
    reused=destination/'core';reused.mkdir()
    core=[]
    for name,digest in CORE.items():
        origin=PARENT/'reused/core'/name;require(sha(origin)==digest,'Authenticated current Core4 object drift:'+name)
        target=reused/name;shutil.copy2(origin,target);core.append({'path':str(target),'sha256':sha(target),'source':str(origin)})
    cpp=['runtime/flash/FlashInt8ExpertStoreMetadata.mm','runtime/flash/FlashDescriptor.mm','runtime/flash/FlashMoEBuckets.cpp','runtime/flash/FlashMoEBlocked.cpp','dev/benchmarks/main_prefill_rhs_tile64_sep22/oracle.mm']
    metal=['runtime/metal/kernels/shared/flash_moe_buckets.metal','runtime/metal/kernels/shared/flash_moe_direct_a.metal','dev/benchmarks/moe_pointwise_sep21/candidate.metal','dev/benchmarks/main_prefill_rhs_tile64_sep22/candidate.metal']
    for name in cpp+metal:require((source/name).is_file(),'Frozen dependency source missing:'+name)
    flags='-std=c++20 -O3 -Wall -Wextra -Werror -Wno-deprecated-declarations -fobjc-arc -mmacosx-version-min=27.0 -DSPLASH_INT8_EXPERIMENT=1'
    mk=['BUILD := '+str(destination),'CXX := xcrun -sdk macosx clang++','METAL := xcrun -sdk macosx metal','CPPFLAGS := '+flags+' -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/main_prefill_rhs_tile64_sep22',
        'METALFLAGS := -std=metal4.1 -O3 -Wall -Wextra -Werror -mmacosx-version-min=27.0 -I$(BUILD)/source -I$(BUILD)/source/runtime -I$(BUILD)/source/dev/benchmarks/main_prefill_rhs_tile64_sep22']
    objects=[]
    for index,name in enumerate(cpp):
        obj=f'$(BUILD)/host/{index:02d}.o';objects.append(obj);mk += [obj+': $(BUILD)/source/'+name,'\tmkdir -p $(BUILD)/host','\t$(CXX) $(CPPFLAGS) -MMD -MP -c $< -o $@']
    airs=[]
    for index,name in enumerate(metal):
        air=f'$(BUILD)/air/{index:02d}.air';airs.append(air);mk += [air+': $(BUILD)/source/'+name,'\tmkdir -p $(BUILD)/air','\t$(METAL) $(METALFLAGS) -c $< -o $@']
    mk += ['.PHONY: all cpu-self-test','all: $(BUILD)/oracle $(BUILD)/rhs-main-prefill.metallib','$(BUILD)/oracle: '+' '.join(objects)+' '+' '.join(str(reused/n) for n in CORE),
           '\t$(CXX) $(CPPFLAGS) $^ -framework Foundation -framework Metal -framework IOKit -o $@','$(BUILD)/rhs-main-prefill.metallib: '+' '.join(airs),
           '\txcrun -sdk macosx metallib $^ -o $@','cpu-self-test: $(BUILD)/oracle','\t$(BUILD)/oracle --cpu-self-test','-include '+' '.join(o[:-2]+'.d' for o in objects)]
    (destination/'component.mk').write_text('\n'.join(mk)+'\n')
    records=[{'path':str(file.relative_to(source)),'sha256':sha(file)} for file in sorted(source.rglob('*')) if file.is_file()]
    manifest={'schema':'current-main-prefill-physical-RHS-tile64-component-source-plan-v1','parent':str(PARENT),'parent_source_seal_sha256':sha(PARENT/'compiled-cpu-seal.json'),
        'parent_source_identity_sha256':'162b01e610d480552c3e005c3ec77566163f4648730c22d303cfa7665d3c810a','source_files':records,'fresh_CXX_TUs':cpp,'fresh_Metal_leafs':metal,'authenticated_reused_Core4':core,
        'opaque_parent_host_or_AIR_reused':False,'original_48layer_store_or_worker_linked':False,'component_mk_sha256':sha(destination/'component.mk'),
        'explicit_original_integer_bitcopy_leaf_source_supplements':supplemental,
        'initial_ROI_input':'explicitly synthetic per-row RMS BF16 A/spread routes/unequal BF16 weights; Root-selected trained I8 coefficients',
        'actual_current_fixture_manifest_required_before_actual_input_label_or_integration':True,'synthetic_or_old_fixture_relabelled_actual':False,'build_GPU_model_payload_read_or_hashed':False,
        'source_review_GO_required_before_compile':True,'native_budget_source':'packing.hpp::nativePlannedBytes','host_budget_bytes':512<<20}
    path=destination/'SOURCE_READY.json';path.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'SOURCE_READY':str(path),'sha256':sha(path),'source_objects':len(records),'GPU_work':False,'compile_started':False}))
if __name__=='__main__':main()
