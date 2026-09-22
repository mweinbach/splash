#!/usr/bin/env python3
"""Create a separate prelaunch-corrected W4A8 source snapshot.

No model payload is read; no existing oracle, shader, or model permission is
changed. The optional Full512 parser is copied from its qualified private
source, while the v1 GPU library is retained byte-for-byte.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[3]


def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def once(text:str,before:str,after:str)->str:
    if text.count(before)!=1:raise ValueError(f'Sealed source anchor drift:{before[:100]}')
    return text.replace(before,after,1)


def loader(text:str)->str:
    text=once(text,'uint64_t(before_.st_size)!=fileBytes||(before_.st_mode&0222)',
        'uint64_t(before_.st_size)!=fileBytes')
    text=once(text,'source shard must be exact-sized readonly regular file',
        'source shard must be exact-sized regular file opened readonly')
    return once(text,'after.st_ctimespec.tv_nsec==before_.st_ctimespec.tv_nsec&&!(after.st_mode&0222)',
        'after.st_ctimespec.tv_nsec==before_.st_ctimespec.tv_nsec&&after.st_mode==before_.st_mode')


def oracle(text:str)->str:
    before='    if(argc==4&&std::string_view(argv[1])=="--plan")'
    after='''    if(argc==3&&std::string_view(argv[1])=="--check-i8-metadata") {
      const auto m=loadFlashInt8ExpertStoreMetadata(argv[2],w4::kSourceIdentity,
          "edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0",NormConvention::OnePlusWeight);
      need(m.identitySha256=="ba22514a30a41d5ddc734ad0aea0a67ce527a3a5c22b95031c1c5810972363f1","certified Full512 metadata differs");
      for(const auto &layer:m.layers)need(one::oneLayerPlannedBytes(layer)==2524463104ULL,"Full512 one-layer plan differs");
      std::cout<<"{\\\"valid\\\":true,\\\"gpu_work\\\":false,\\\"payload_bytes_read\\\":0,\\\"layers\\\":48,\\\"experts_per_layer\\\":512,\\\"one_layer_planned_bytes\\\":2524463104}\\n";
      return 0;
    }
    if(argc==4&&std::string_view(argv[1])=="--plan")'''
    return once(text,before,after)


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'build/prefill-w4a8-sep21')
    parser.add_argument('--metadata-base',type=Path,default=ROOT/'build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'build/prefill-w4a8-sep21-v2')
    args=parser.parse_args();base=args.base.resolve();output=args.output.resolve();metadata_base=args.metadata_base.resolve()
    if ROOT/'build' not in output.parents or output.exists() or output==base:raise ValueError('Choose fresh separate v2 output')
    sealed=json.loads((base/'cpu-source-build-manifest.json').read_text());files={};records=[]
    for name in ('abi.h','candidate.metal','loader.hpp','precision.hpp','oracle.mm'):
        path=ROOT/'dev/benchmarks/prefill_w4a8_sep21'/name;data=path.read_bytes()
        if sha(data)!=sealed['source_hashes'][name]:raise ValueError(f'V1 source drift:{name}')
        transformed={'loader.hpp':loader,'oracle.mm':oracle}.get(name)
        result=transformed(data.decode()).encode() if transformed else data
        files[Path('source')/name]=result;records.append({'path':name,'v1_sha256':sha(data),'v2_sha256':sha(result),'changed':data!=result})
    metadata_manifest=json.loads((metadata_base/'overlay-manifest.json').read_text())
    for name in ('FlashInt8ExpertStoreMetadata.hpp','FlashInt8ExpertStoreMetadata.mm'):
        relative='runtime/flash/'+name;record=next(x for x in metadata_manifest['files'] if x['path']==relative)
        data=(metadata_base/'source'/relative).read_bytes()
        if sha(data)!=record['overlay_sha256']:raise ValueError(f'Qualified Full512 metadata source drift:{name}')
        files[Path('source/private-runtime/flash')/name]=data
        records.append({'path':str(Path('private-runtime/flash')/name),'qualified_source_sha256':sha(data),'v2_sha256':sha(data)})
    for name in ('candidate.air','candidate.metallib'):
        data=(base/name).read_bytes()
        if sha(data)!=sealed['compiled_hashes'][name]:raise ValueError(f'V1 GPU artifact drift:{name}')
        files[Path(name)]=data
    manifest={'schema':'splash-one-layer-W4A8-v2-prelaunch-corrections','files':records,'gpu_executed':False,'model_payload_bytes_read':0,
        'model_permissions_changed':False,'v1_source_and_artifacts_modified':False,'shader_source_and_GPU_library_preserved_byte_exact':True,
        'source_mapping_policy':'O_RDONLY|O_NOFOLLOW + PROT_READ; filesystem writable mode permitted, unchanged mode/inode/size/timestamps + whole selected-range hashes required',
        'Full512_optional_parser':'qualified private inventory32/64/128/256/512 parser; maps only one certified payload when Root runs GPU',
        'GPU_library_sha256':sha(files[Path('candidate.metallib')]),'generator_sha256':sha(Path(__file__).read_bytes())}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prefill-w4a8-v2-',dir=output.parent) as temporary:
        staged=Path(temporary)/'output';staged.mkdir()
        for name,data in files.items():
            path=staged/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (staged/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');staged.rename(output)
    print(json.dumps({'prepared':str(output),**manifest}))


if __name__=='__main__':main()
