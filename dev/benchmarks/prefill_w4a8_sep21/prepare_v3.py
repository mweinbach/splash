#!/usr/bin/env python3
"""Snapshot the proved signed/zero affine-scale loader correction.

Root's bounded census confirms negative original weight scales. All source
parameters still require finite BF16; activation quantizer scales stay positive.
No model payload reads, model writes, or v1/v2 edits happen in this generator.
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
    if text.count(before)!=1:raise ValueError(f'V2 source anchor drift:{before[:100]}')
    return text.replace(before,after,1)


def loader(text:str)->str:
    return once(text,
        '          require(std::isfinite(value)&&(field%3==2||value>0),"original scale/bias nonfinite or scale nonpositive");',
        '''          // Original affine scales are signed BF16, not fitted positive
          // row scales. q*s+b and centered c*s+(b+8*s) accept any finite s.
          if (!std::isfinite(value)) {
            const uint64_t groups=record.shape[2],outputs=record.shape[1];
            throw std::invalid_argument("one-layer W4A8 nonfinite original BF16 parameter: "+record.name+
                " expert="+std::to_string(i/(groups*outputs))+" output="+std::to_string((i/groups)%outputs)+
                " G64="+std::to_string(i%groups)+" rawBits="+std::to_string(values[i]));
          }''')


def oracle(text:str)->str:
    before='constexpr uint32_t kSticky=0x80000000u;'
    after='''constexpr uint32_t kSticky=0x80000000u;
void signedAndZeroWeightScaleCPU() {
  std::array<uint16_t,64>x;std::array<int8_t,64>a;
  std::array<uint8_t,32>q{},centered{};std::array<int32_t,1>dot{},sum{};
  for(uint32_t k=0;k<64;++k){a[k]=int8_t(k==0?127:k==1?-127:int(k%19)-9);x[k]=w4::bf16(float(a[k]));
    const uint8_t code=(k*11+3)%16;q[k/2]|=uint8_t(code<<((k%2)*4));dot[0]+=int(a[k])*(int(code)-8);sum[0]+=a[k];}
  for(uint32_t byte=0;byte<32;++byte)centered[byte]=w4::centeredByte(q[byte]);
  for(uint16_t s:{uint16_t(0),uint16_t(0x8000),w4::bf16(-.03125f),w4::bf16(.03125f)})
    for(float bias:{-.0625f,0.0f,.0625f}){
      const std::array<uint16_t,1>ss{s},bb{w4::bf16(bias)};
      const std::array<float,1>corrected{w4::roundedAdd(bias,w4::roundedMultiply(8,w4::number(s)))};
      const float linear=w4::roundedAdd(w4::roundedMultiply(w4::number(s),float(dot[0])),w4::roundedMultiply(corrected[0],float(sum[0])));
      const w4::SourceRowView source{x,q,ss,bb,centered};
      const w4::ObservedRowView observed{1,a,dot,sum,corrected,linear};
      if(!w4::certifyRow(source,observed).pass())throw std::invalid_argument("signed/zero original scale certificate failed");
      auto bad=source;const std::array<uint16_t,1>nf{uint16_t(0x7fc1)};bad.groupScaleBf16=nf;
      if(w4::certifyRow(bad,observed).pass())throw std::invalid_argument("nonfinite original scale accepted");
    }
}'''
    text=once(text,before,after)
    return once(text,'{w4::cpuSelfTest();need(sizeof(PrefillW4A8QuantParams)',
        '{w4::cpuSelfTest();signedAndZeroWeightScaleCPU();need(sizeof(PrefillW4A8QuantParams)')


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'build/prefill-w4a8-sep21-v2')
    parser.add_argument('--output',type=Path,default=ROOT/'build/prefill-w4a8-sep21-v3')
    parser.add_argument('--census',type=Path,default=ROOT/'build/release/flash/sep21-w4-original-gate-scale-census-v1.json')
    args=parser.parse_args();base=args.base.resolve();output=args.output.resolve()
    if ROOT/'build' not in output.parents or output.exists() or output==base:raise ValueError('Choose fresh v3 private output')
    parent=json.loads((base/'source-manifest.json').read_text());census_raw=args.census.read_bytes();census=json.loads(census_raw)
    first=census['results'][0]
    if first['nonfinite'] or first['finite_negative']!=6596895 or first['finite_positive']!=6510305:raise ValueError('Root source census proof differs')
    files={};records=[]
    for record in parent['files']:
        name=record['path'];data=(base/'source'/name).read_bytes()
        if sha(data)!=record['v2_sha256']:raise ValueError(f'Sealed v2 source drift:{name}')
        transform={'loader.hpp':loader,'oracle.mm':oracle}.get(name)
        changed=transform(data.decode()).encode() if transform else data;files[Path('source')/name]=changed
        records.append({'path':name,'v2_sha256':sha(data),'v3_sha256':sha(changed),'changed':data!=changed})
    for name in ('candidate.air','candidate.metallib'):files[Path(name)]=(base/name).read_bytes()
    if sha(files[Path('candidate.metallib')])!=parent['GPU_library_sha256']:raise ValueError('V2 GPU library changed')
    manifest={'schema':'splash-one-layer-W4A8-v3-signed-affine-source-scales','files':records,'model_payload_bytes_read_by_agents':0,
        'gpu_executed':False,'model_permissions_or_bytes_modified':False,'prior_artifacts_modified':False,
        'shader_source_and_GPU_library_preserved_byte_exact':True,'GPU_library_sha256':sha(files[Path('candidate.metallib')]),
        'original_weight_scale_contract':'any finite signed BF16 including +/-zero; bias finite; activation row quantization scale independently remains strictlypositive',
        'Root_census_sha256':sha(census_raw),'Root_actual_gate_scale_counts':first,
        'centering_identity':'q*s+b==(q-8)*s+(b+8*s), exact dyadic coefficient proof for signed/zero s; F32 correctedbias rounding bounded separately',
        'deterministic_CPU_signed_zero_negative_cases_added':12,'nonfinite_original_parameter_checks_retained':True,
        'first_nonfinite_exception_reports_source_tensor_and_expert_output_G64_rawbits':True}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prefill-w4a8-signed-v3-',dir=output.parent) as temporary:
        staged=Path(temporary)/'output';staged.mkdir()
        for name,data in files.items():
            path=staged/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (staged/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');staged.rename(output)
    print(json.dumps({'prepared':str(output),**manifest}))


if __name__=='__main__':main()
