#!/usr/bin/env python3
"""Seal an ordinary R1 decode-only vector I8 worker; CPU source/binary copies only."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/gemv_decode_r1_worker_sep21')
QUALIFIED=Path('dev/benchmarks/gemv_decode_sep21_v1b')
FLAG='SPLASH_FLASH_GEMV_DECODE_R1_SEP21'
OWN_NAMES={'FlashInt8ExpertStore','FlashForward','FlashWorker'}
CHANGED_PATHS={'runtime/flash/FlashInt8ExpertStore.hpp','runtime/flash/FlashInt8ExpertStore.mm',
               'runtime/flash/FlashForward.hpp','runtime/flash/FlashForward.cpp',
               'runtime/flash/FlashWorker.mm','dev/benchmarks/prefill4k_attribution.mm'}

def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def replace(text,before,after,count=1):
    if text.count(before)!=count:raise ValueError(f'R1 sealed-source anchor drift: {before!r}')
    return text.replace(before,after)

def shipping_shader(original):
    # Keep the qualified arithmetic helpers and exactly the two L32 producers.
    end=original.index('GEMV_DECODE_GATE(gemv_decode_sep21_v4_l16_o8_gate_up,16)')
    text=original[:end]
    text=replace(text,'#include "flash/FlashGatheredI8QMV.hpp"',f'#include "{PRIVATE}/abi.hpp"')
    text=replace(text,'#include "prefill4k_allrows_qmv_probe.h"\n','')
    text=text.replace('FlashGatheredI8QMVParams','FlashGEMVDecodeR1Params')
    text=replace(text,'if (!p.rows || p.rows>16 || p.selections!=10','if (p.rows!=1 || p.selections!=10')
    return text

def methods(text):
    start=text.index('void FlashInt8ExpertStore::addGatheredMPPGateUp(')
    end=text.index('} // namespace splash::flash',start)
    result=text[start:end]
    result=result.replace('addGatheredMPPGateUp','addGEMVDecodeR1GateUp').replace('addGatheredMPPDown','addGEMVDecodeR1Down')
    result=replace(result,'if (!gatheredMPPEnabled()) fail("private gathered I8 MPP gate/up requires frozen flag1");',
        'if (!gemvDecodeR1Enabled()) fail("private vector R1 decode gate/up requires frozen flag1");')
    result=replace(result,'if (!gatheredMPPEnabled()) fail("private gathered I8 MPP down requires frozen flag1");',
        'if (!gemvDecodeR1Enabled()) fail("private vector R1 decode down requires frozen flag1");')
    result=replace(result,'if (rows > gatheredMPPMaximumRows()) fail("private gathered MPP rows exceed frozen route cap");',
        'if (rows!=1 || selections!=10) fail("private vector decode only canonical physical R1");',2)
    result=result.replace('flash_gathered_mpp_gate_up_m16_n64_sg4','gemv_decode_sep21_v4_l32_o4_gate_up')
    result=result.replace('flash_gathered_mpp_down_m16_n64_sg4','gemv_decode_sep21_v4_l32_o4_down')
    result=result.replace('FlashGatheredMPPParams','FlashGEMVDecodeR1Params')
    result=result.replace('{g.gateColumnGroups, rows, selections}','{160, rows, selections}')
    result=result.replace('{g.downColumnGroups, rows, selections}','{640, rows, selections}')
    result=result.replace('impl_->gatheredMPPGate','impl_->gemvDecodeR1Gate').replace('impl_->gatheredMPPDown','impl_->gemvDecodeR1Down')
    return result

def transform(relative,text):
    if relative in CHANGED_PATHS:
        text=f'#include "{PRIVATE}/bridge.hpp"\n'+text
    if relative=='runtime/flash/FlashInt8ExpertStore.hpp':
        declarations='''  [[nodiscard]] bool gemvDecodeR1Enabled() const;
  [[nodiscard]] gemv_decode_r1_sep21::Counters gemvDecodeR1Counters() const;
  void addGEMVDecodeR1GateUp(metal::CommandGraph &graph,uint32_t layer,
      metal::MetalBuffer input,metal::MetalBuffer originalExpertIDs,
      metal::MetalBuffer canonicalIntermediate,metal::MetalBuffer diagnostics,
      uint32_t rows,uint32_t selections=10) const;
  void addGEMVDecodeR1Down(metal::CommandGraph &graph,uint32_t layer,
      metal::MetalBuffer canonicalIntermediate,metal::MetalBuffer originalExpertIDs,
      metal::MetalBuffer canonicalExpertDown,metal::MetalBuffer diagnostics,
      uint32_t rows,uint32_t selections=10) const;
'''
        text=replace(text,'private:\n  struct Impl;',declarations+'private:\n  struct Impl;')
    if relative=='runtime/flash/FlashInt8ExpertStore.mm':
        clones=methods(text)
        text=f'#include "{PRIVATE}/abi.hpp"\n'+text
        text=replace(text,'  const bool gatheredMPP = gathered_mpp::requested();',
            '''  const bool gemvDecodeR1 = gemv_decode_r1_sep21::requested();
  mutable std::atomic<uint64_t> gemvDecodeR1GateCalls{0},gemvDecodeR1GateRows{0},gemvDecodeR1DownCalls{0},gemvDecodeR1DownRows{0};
  const bool gatheredMPP = gathered_mpp::requested();''')
        text=replace(text,'    numericalIdentity = hash(derivative.data(), derivative.size());',
            '''    if (gemvDecodeR1)
      derivative += std::string("ordinary_r1_decode_policy=")+gemv_decode_r1_sep21::implementationMarker()+"\\n";
    numericalIdentity = hash(derivative.data(), derivative.size());''')
        additions='''bool FlashInt8ExpertStore::gemvDecodeR1Enabled() const {
  if (!impl_) fail("private vector R1 decode Store was moved or disposed");
  if (gemv_decode_r1_sep21::requested()!=impl_->gemvDecodeR1)
    fail("private vector R1 decode flag changed after Store construction");
  if (impl_->gemvDecodeR1&&!gatheredMPPEnabled())
    fail("private vector R1 decode requires the original gathered route enabled");
  return impl_->gemvDecodeR1;
}
gemv_decode_r1_sep21::Counters FlashInt8ExpertStore::gemvDecodeR1Counters() const {
  return {gemvDecodeR1Enabled(),impl_->gemvDecodeR1GateCalls.load(std::memory_order_relaxed),
      impl_->gemvDecodeR1GateRows.load(std::memory_order_relaxed),impl_->gemvDecodeR1DownCalls.load(std::memory_order_relaxed),
      impl_->gemvDecodeR1DownRows.load(std::memory_order_relaxed)};
}
'''+clones
        text=replace(text,'} // namespace splash::flash',additions+'} // namespace splash::flash')
    if relative=='runtime/flash/FlashForward.hpp':
        text=replace(text,'  // Incoming window is [anchor,draft1..draft7], 1..8 tokens.',
            '''  // Explicit ordinary singleton decode; forward() keeps every prefill
  // window and MTP seed unchanged, including physical singleton tails.
  [[nodiscard]] FlashForwardResult
  forwardDecode(FlashRequestState &state,std::span<const uint32_t> token);
  // Incoming window is [anchor,draft1..draft7], 1..8 tokens.''')
        text=replace(text,'bool returnAllLogits, bool captureHidden, bool verification);',
            'bool returnAllLogits, bool captureHidden, bool verification, bool ordinaryDecode=false);')
    if relative=='runtime/flash/FlashForward.cpp':
        text=replace(text,'      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +',
            '''      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +
      gemv_decode_r1_sep21::implementationMarker() +''')
        text=replace(text,'FlashForwardResult FlashForward::verify(FlashRequestState &request,',
            '''FlashForwardResult FlashForward::forwardDecode(FlashRequestState &request,
                                              std::span<const uint32_t> token) {
  if (token.size()!=1 || !request.logicalLength())
    throw std::invalid_argument("private ordinary decode requires one token after a nonempty prompt");
  return forwardImpl(request,token,false,false,false,true);
}

FlashForwardResult FlashForward::verify(FlashRequestState &request,''')
        text=replace(text,'                                             bool verification) {',
            '                                             bool verification, bool ordinaryDecode) {')
        text=replace(text,'    if (gatheredMPP) {',
            '''    const bool vectorR1Decode = ordinaryDecode && gatheredMPP &&
        gemv_decode_r1_sep21::eligible(rows,verification) && impl_->int8ExpertStore->gemvDecodeR1Enabled();
    if (vectorR1Decode) {
      impl_->int8ExpertStore->addGEMVDecodeR1GateUp(graph,layer,mixed,ids,
          bf(Scratch::ExpertIntermediate,kSelections*640),diag,rows,kSelections);
      impl_->int8ExpertStore->addGEMVDecodeR1Down(graph,layer,
          bf(Scratch::ExpertIntermediate,kSelections*640),ids,
          bf(Scratch::ExpertDown,kSelections*kWidth),diag,rows,kSelections);
    } else if (gatheredMPP) {''')
    if relative=='runtime/flash/FlashWorker.mm':
        text=replace(text,'      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.',
            '''      if (gemv_decode_r1_sep21::requested()&&!gathered_mpp::requested())
        throw std::invalid_argument("SPLASH_FLASH_GEMV_DECODE_R1_SEP21=1 requires gathered MPP enabled");
      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.''')
        text=replace(text,'      const auto result = forward_.forward(*request.state, token);',
            '      const auto result = forward_.forwardDecode(*request.state, token);')
        text=replace(text,'  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};',
            '''  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};
  const auto vectorR1Graphs = persistedExperts ? persistedExperts->gemvDecodeR1Counters() : gemv_decode_r1_sep21::Counters{};''')
        text=replace(text,'      << R"(,"gdn_verification_storage":{"lazy_enabled":)"',
            '''      << R"(,"vector_i8_r1_decode":{"scope":"ordinary singleton autoregressive call only; graph construction not GPU completion; synthetic R1 component qualified only","enabled":)"
      <<(vectorR1Graphs.enabled?"true":"false")<<R"(,"source_identity_sha256":)"
      <<json::quote(gemv_decode_r1_sep21::kSourceIdentitySha256)
      <<R"(,"numerical_alternative":true,"full_model_quality_qualified":false,"additional_gpu_allocation_bytes":0,"gate_graph_calls":)"
      <<vectorR1Graphs.gateCalls<<R"(,"gate_graph_rows":)"<<vectorR1Graphs.gateRows
      <<R"(,"down_graph_calls":)"<<vectorR1Graphs.downCalls<<R"(,"down_graph_rows":)"<<vectorR1Graphs.downRows<<'}'
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
    if relative=='dev/benchmarks/prefill4k_attribution.mm':
        text=replace(text,'      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.',
            '''      (void)gemv_decode_r1_sep21::requested(); // Freeze ordinary decode selection before backend creation.
      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.''')
    return text

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=ROOT/'build/adaptive-expert-tail-sg2k128-fma-sep21-worker-v1')
    p.add_argument('--output',type=Path,default=ROOT/'build/gemv-decode-r1-pointwise-sg2tail-sep21-worker-v1')
    p.add_argument('--r1-report',type=Path,default=ROOT/'build/release/flash/sep21-vector-i8-gemv-r1-v1b.json')
    args=p.parse_args();base=args.base.resolve();out=args.output.resolve()
    if out.exists() or base==out or ROOT/'build' not in out.parents:raise ValueError('NEW private build directory required')
    parent_path=base/'overlay-manifest.json';parent=json.loads(parent_path.read_text())
    if parent.get('omitted_original_target_tensor_count')!=432 or not parent.get('pointwise_composed') or not parent.get('gathered_mpp_composed'):
        raise ValueError('Sealed pure I8 pointwise/gathered base required')
    report=json.loads(args.r1_report.read_text())
    vector=next(v for v in report['variants'] if v['name']=='vectorGEMVL32O4')
    if report['rows']!=1 or not vector['numerical_alternative_qualified'] or report['model_quality_qualified']:
        raise ValueError('Only Root-qualified synthetic R1 component evidence is admitted')
    originals={name:(ROOT/QUALIFIED/name).read_bytes() for name in ('kernels.metal','quality.hpp','PREREGISTRATION.md','FTZ_CERTIFICATE.md')}
    candidate=shipping_shader(originals['kernels.metal'].decode()).encode()
    identity_parts={name:sha(data) for name,data in originals.items()}
    identity_parts.update({'shipping_candidate.metal':sha(candidate),'abi.hpp':sha((ROOT/PRIVATE/'abi.hpp').read_bytes()),
                          'bridge.hpp':sha((ROOT/PRIVATE/'bridge.hpp').read_bytes()),'prepare_schema':'ordinary-R1-only-vector4-L32-O4-nonverify-no-prefill-v1'})
    identity=sha(json.dumps(identity_parts,sort_keys=True,separators=(',',':')).encode())
    manifest=copy.deepcopy(parent)
    manifest.update({'route':'private-r1-ordinary-decode-vector-i8-pointwise-sg2tail-sep21-v1',
        'vector_r1_base_build':str(base),'vector_r1_base_manifest_sha256':sha(parent_path.read_bytes()),
        'vector_r1_flag':FLAG,'vector_r1_source_identity_sha256':identity,'vector_r1_identity_parts':identity_parts,
        'vector_r1_transform_sha256':sha(Path(__file__).read_bytes()),'vector_r1_added_gpu_bytes':0,
        'vector_r1_scope':'ordinary autoregressive singleton decode only; every forward()/verify()/batch/prefill path remains original',
        'vector_r1_numerical_alternative':True,'vector_r1_flag0_base_graphs_identity':True,
        'vector_r1_component_report_sha256':sha(args.r1_report.read_bytes()),'vector_r1_model_quality_qualified':False,
        'gpu_executed':False,'payload_bytes_read':0,'files':[]})
    changed=[]
    for record in parent['files']:
        relative=record['path'];original=(base/'source'/relative).read_bytes()
        if sha(original)!=record['overlay_sha256']:raise ValueError(f'Base source drift: {relative}')
        data=transform(relative,original.decode()).encode();write(out/'source'/relative,data)
        if data!=original:changed.append(relative)
        manifest['files'].append({**record,'vector_r1_base_sha256':sha(original),'vector_r1_changed':data!=original,'overlay_sha256':sha(data)})
    if set(changed)!=CHANGED_PATHS:raise ValueError(f'Expected six isolated host/source changes: {changed}')
    def extra(relative,data,**metadata):
        write(out/'source'/relative,data);manifest['files'].append({'path':str(relative),'new_vector_r1_file':True,'overlay_sha256':sha(data),**metadata})
    for name in ('bridge.hpp','policy_cpu.cpp','abi.hpp','worker_prepare.py','worker.mk','worker_witness.py'):
        data=(ROOT/PRIVATE/name).read_bytes();extra(PRIVATE/name,data,repository_sha256=sha(data))
    extra(PRIVATE/'candidate.metal',candidate,qualified_kernel_sha256=identity_parts['kernels.metal'])
    generated=f'#pragma once\nnamespace splash::flash::gemv_decode_r1_sep21 {{ inline constexpr char kSourceIdentitySha256[]="{identity}"; }}\n'.encode()
    extra(PRIVATE/'source_identity.hpp',generated,generated_source_identity=True)
    for name,data in originals.items():extra(PRIVATE/'qualified-source'/name,data,qualified_source=True)
    write(out/'qualification/r1-component.json',args.r1_report.read_bytes())
    inputs={'REUSED':[],'CORE':[],'AIRS':[]};manifest['vector_r1_link_inputs']=[]
    def freeze(path,relative,category):
        data=path.read_bytes();write(out/relative,data);inputs[category].append(relative.as_posix())
        manifest['vector_r1_link_inputs'].append({'source_path':str(path),'private_path':relative.as_posix(),'category':category,'sha256':sha(data)})
    for line in (base/'link-inputs.mk').read_text().splitlines():
        category,raw=line.split(' := ',1)
        if category not in inputs:raise ValueError('Unexpected frozen link category')
        for token in raw.split():
            if not token.startswith('$(BUILD)/'):raise ValueError('Unsealed link token')
            relative=Path(token[len('$(BUILD)/'):]);freeze(base/relative,Path('reused/parent')/relative,category)
    for path in sorted((base/'host').glob('*.o')):
        if path.stem not in OWN_NAMES:freeze(path,Path('reused/parent-host')/path.name,'REUSED')
    for path in sorted(base.glob('*.air')):freeze(path,Path('reused/parent-air')/path.name,'AIRS')
    frozen='\n'.join(f'{cat} := '+' '.join('$(BUILD)/'+name for name in names) for cat,names in inputs.items())+'\n'
    write(out/'link-inputs.mk',frozen.encode());manifest['vector_r1_link_make_sha256']=sha(frozen.encode())
    manifest['vector_r1_changed_files']=changed
    write(out/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    for name in ('splash-flash.config','splash.metallib.config'):
        write(out/name,(base/name).read_bytes().rstrip(b'\n')+b'-gemv-r1-ordinary-numerical-alt-v1\n')
    print(json.dumps({'prepared':str(out),'source_identity_sha256':identity,'changed_files':changed,
                      'added_gpu_bytes':0,'gpu_work':False,'payload_bytes_read':0}))

if __name__=='__main__':main()
