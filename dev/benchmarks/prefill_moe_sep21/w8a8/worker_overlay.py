#!/usr/bin/env python3
"""Freeze W8A8 main-prefill selectors over sealed exact pointwise worker inputs."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
PRIVATE=Path('dev/benchmarks/prefill_moe_sep21/w8a8')
OWN={'FlashInt8ExpertStore','FlashForward','FlashWorker'}
POINTWISE={'FlashMoE','FlashMoEBlocked','FlashExpertDenseCache'}
INCLUDE='#include "dev/benchmarks/prefill_moe_sep21/w8a8/worker.hpp"\n'


def sha(data):return hashlib.sha256(data).hexdigest()


def replace(text,before,after,count=1):
    if text.count(before)!=count:raise ValueError(f'W8A8 source seal anchor drift:{before!r}')
    return text.replace(before,after)


def transform(relative,text):
    if relative in ('runtime/flash/FlashForward.hpp','runtime/flash/FlashInt8ExpertStore.hpp',
                    'runtime/flash/FlashForward.cpp','runtime/flash/FlashInt8ExpertStore.mm',
                    'runtime/flash/FlashWorker.mm','dev/benchmarks/prefill4k_attribution.mm'):
        text=INCLUDE+text
    if relative=='runtime/flash/FlashInt8ExpertStore.hpp':
        point='  // Store methods keep immutable base mappings and rank allocations private.'
        addition='''  // Explicit numerical main-prefill producers. Existing BF16-A methods and
  // their ABI remain unchanged for decoding, verification and other shapes.
  void addW8A8PrefillGateUp(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,
      uint32_t rows,FlashMoEBlockedTile tile,const prefill_w8a8_sep21::Workspace &workspace,
      uint32_t selections=10) const;
  void addW8A8PrefillDownScatter(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,
      uint32_t rows,FlashMoEBlockedTile tile,const prefill_w8a8_sep21::Workspace &workspace,
      uint32_t selections=10) const;

'''
        text=replace(text,point,addition+point)
    if relative=='runtime/flash/FlashInt8ExpertStore.mm':
        point='    numericalIdentity = hash(derivative.data(), derivative.size());'
        text=replace(text,point,'''    if (prefill_w8a8_sep21::mode())
      derivative +=std::string("prefill_w8a8_policy=") +prefill_w8a8_sep21::policy() +"\\n";
'''+point)
        begin=text.index('void FlashInt8ExpertStore::addGateUp(')
        end=text.index('\nnamespace {\nvoid gatheredMPPViews',begin)
        methods=text[begin:end]
        methods=methods.replace('FlashInt8ExpertStore::addGateUp(', 'FlashInt8ExpertStore::addW8A8PrefillGateUp(')
        methods=methods.replace('FlashInt8ExpertStore::addDownScatter(', 'FlashInt8ExpertStore::addW8A8PrefillDownScatter(')
        methods=replace(methods,'    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {',
            '    uint32_t rows,FlashMoEBlockedTile tile,const prefill_w8a8_sep21::Workspace &workspace,uint32_t selections) const {\n'
            '  if (!prefill_w8a8_sep21::eligible(rows,tile,false) ||selections !=10) fail("W8A8 producer only canonical main R2048/M32");\n'
            '  workspace.validate(rows);\n'
            '  for (const auto &buffer :workspace.buffers()) impl_->immutableDisjoint(buffer);',2)
        methods=replace(methods,'  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;',
            '  const uint32_t threads=prefill_w8a8_sep21::producerThreads();',2)
        methods=replace(methods,'  const auto p = allRowsParams(rows, selections, tile);',
            '  workspace.addQuantGate(graph,s.buckets.packedInputs,diagnostics,rows);\n  const auto p = allRowsParams(rows, selections, tile);',1 if methods.count('  const auto p = allRowsParams(rows, selections, tile);')==1 else 2)
        # The second inserted quant call belongs to the activated-down source.
        down=methods.index('void FlashInt8ExpertStore::addW8A8PrefillDownScatter(')
        methods=methods[:down]+methods[down:].replace('workspace.addQuantGate(graph,s.buckets.packedInputs,diagnostics,rows);',
            'workspace.addQuantDown(graph,s.packedActivated,diagnostics,rows);',1)
        methods=replace(methods,'graph.add(pipeline("gate_up", tile), {s.buckets.packedInputs,',
            'graph.add(prefill_w8a8_sep21::producerName(true), {workspace.qGate,')
        methods=replace(methods,'s.buckets.jobCount, s.packedActivated, diagnostics}, p,',
            's.buckets.jobCount, s.packedActivated, diagnostics,workspace.gateScales}, p,')
        methods=replace(methods,'graph.add(pipeline("down_scatter", tile), {s.packedActivated,',
            'graph.add(prefill_w8a8_sep21::producerName(false), {workspace.qDown,')
        methods=replace(methods,'s.buckets.routeMap, s.scatteredDown, diagnostics}, p,',
            's.buckets.routeMap, s.scatteredDown, diagnostics,workspace.downScales}, p,')
        text=text[:end]+'\n'+methods+text[end:]
    if relative=='runtime/flash/FlashForward.hpp':
        point='  [[nodiscard]] const FlashInt8ExpertStore *batchInt8ExpertStore() const noexcept;'
        text=replace(text,point,point+'\n  [[nodiscard]] prefill_w8a8_sep21::Counters w8a8PrefillCounters() const noexcept;')
    if relative=='runtime/flash/FlashForward.cpp':
        text=replace(text,'  FlashMoEBlockedScratch blockedScratch;',
            '  FlashMoEBlockedScratch blockedScratch;\n  std::optional<prefill_w8a8_sep21::Workspace> w8a8Workspace;')
        point='    if (bulkQSAPrefill && maximumRows >= 2048) {'
        text=replace(text,point,'''    if (prefill_w8a8_sep21::mode() &&maximumRows >=2048) {
      if (!allRowsInt8Target ||!blockMoE ||!int8ExpertStore)
        throw std::invalid_argument("W8A8 main-prefill requires original-target-omitted Full512 store");
      w8a8Workspace.emplace(backend,maximumRows);
    }
'''+point)
        point='  if (privateBulkPrefillEnabled && maximumRows >= 2048)\n    total += prefill4k::bulkExactPlannedBytes();'
        text=replace(text,point,point+'\n  if (prefill_w8a8_sep21::mode() &&maximumRows >=2048)\n    total +=prefill_w8a8_sep21::Workspace::plannedBytes(maximumRows);')
        point='      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +'
        text=replace(text,point,point+'\n      std::string(prefill_w8a8_sep21::marker()) +')
        point='const FlashInt8ExpertStore *FlashForward::batchInt8ExpertStore() const noexcept {'
        begin=text.index(point);end=text.index('\n}',begin)+2
        text=text[:end]+'''

prefill_w8a8_sep21::Counters FlashForward::w8a8PrefillCounters() const noexcept {
  return impl_ &&impl_->w8a8Workspace ? impl_->w8a8Workspace->counters() :prefill_w8a8_sep21::Counters{};
}
'''+text[end:]
        before='''        impl_->int8ExpertStore->addGateUp(graph, layer, impl_->blockedScratch, diag, rows, tile);
        impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);'''
        after='''        if (impl_->w8a8Workspace &&prefill_w8a8_sep21::eligible(rows,tile,verification)) {
          impl_->int8ExpertStore->addW8A8PrefillGateUp(graph,layer,impl_->blockedScratch,diag,rows,tile,*impl_->w8a8Workspace);
          impl_->int8ExpertStore->addW8A8PrefillDownScatter(graph,layer,impl_->blockedScratch,diag,rows,tile,*impl_->w8a8Workspace);
        } else {
          impl_->int8ExpertStore->addGateUp(graph, layer, impl_->blockedScratch, diag, rows, tile);
          impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);
        }'''
        text=replace(text,before,after)
    if relative=='runtime/flash/FlashWorker.mm':
        point='      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.'
        text=replace(text,point,point+'\n      (void)prefill_w8a8_sep21::mode(); // Numeric policy rejects before paths, metadata or backend.')
        point='      << R"(,"gdn_verification_storage":{"lazy_enabled":)"'
        addition='''      << R"(,"prefill_w8a8_route_counters":{"scope":"main nonverification physical rows2048/M32; graph construction not GPU completion; numerical activation alternative","enabled":)"
      << (forward_.w8a8PrefillCounters().enabled ? "true" :"false")
      << R"(,"mode":)" <<prefill_w8a8_sep21::mode()
      << R"(,"maximum_rows":)" <<forward_.w8a8PrefillCounters().maxRows
      << R"(,"planned_bytes":)" <<forward_.w8a8PrefillCounters().plannedBytes
      << R"(,"logical_bytes":)" <<forward_.w8a8PrefillCounters().logicalBytes
      << R"(,"actual_allocated_bytes":)" <<forward_.w8a8PrefillCounters().actualAllocatedBytes
      << R"(,"gate_graph_calls":)" <<forward_.w8a8PrefillCounters().gateCalls
      << R"(,"gate_graph_rows":)" <<forward_.w8a8PrefillCounters().gateRows
      << R"(,"down_graph_calls":)" <<forward_.w8a8PrefillCounters().downCalls
      << R"(,"down_graph_rows":)" <<forward_.w8a8PrefillCounters().downRows
      << R"(,"quantizer_dispatches":)" <<forward_.w8a8PrefillCounters().quantDispatches
      << R"(,"quality_qualification":"pending whole-model semantics and MTP acceptance"})"
'''
        text=replace(text,point,addition+point)
    if relative=='dev/benchmarks/prefill4k_attribution.mm':
        point='      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.'
        text=replace(text,point,point+'\n      (void)prefill_w8a8_sep21::mode();')
    return text


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or path.read_bytes()!=data:path.write_bytes(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'build/moe-pointwise-sep21-worker-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'build/prefill-moe-sep21-w8a8-pointwise-worker-v1')
    args=parser.parse_args();base=args.base.resolve();output=args.output.resolve()
    if output==base or ROOT/'build' not in output.parents:raise ValueError('Distinct private W8A8 snapshot required')
    parentPath=base/'overlay-manifest.json';parent=json.loads(parentPath.read_text())
    if not parent.get('pointwise_composed') or not parent.get('gathered_mpp_composed') or parent.get('omitted_original_target_tensor_count')!=432:
        raise ValueError('Expected sealed pointwise/gathered/bulk Full512 omitted-original-target worker')
    manifest=copy.deepcopy(parent);manifest.update({'route':'private-full512-pointwise-bulk-gathered-w8a8-main-prefill-sep21-v1',
      'w8a8_required_environment':'SPLASH_FLASH_PREFILL_MOE_W8A8=0|1|2','w8a8_flag0_original_graphs_and_identity':True,
      'w8a8_numerical_alternative':True,'w8a8_model_semantic_and_mtp_quality_qualification':'pending',
      'w8a8_eligible_call':'main nonverification physical rows2048, canonical Full512 nativeM32, K10,W2560/640',
      'w8a8_base_build':str(base),'w8a8_base_manifest_sha256':sha(parentPath.read_bytes()),
      'w8a8_transform_sha256':sha(Path(__file__).read_bytes()),'gpu_executed':False,'payload_bytes_read':0,'files':[]})
    changed=[]
    for record in parent['files']:
        relative=record['path'];original=(base/'source'/relative).read_bytes()
        if sha(original)!=record['overlay_sha256']:raise ValueError(f'Sealed pointwise source drift:{relative}')
        data=transform(relative,original.decode()).encode();write(output/'source'/relative,data)
        if data!=original:changed.append(relative)
        manifest['files'].append({**record,'w8a8_changed':data!=original,'w8a8_base_overlay_sha256':sha(original),'overlay_sha256':sha(data)})
    for name in ('worker.hpp','worker_policy_cpu.cpp'):
        relative=str(PRIVATE/name);data=(ROOT/relative).read_bytes();write(output/'source'/relative,data)
        manifest['files'].append({'path':relative,'new_w8a8_private_file':True,'overlay_sha256':sha(data)})
    relative=str(PRIVATE/'candidate.metal');data=(ROOT/relative).read_bytes()
    text=data.decode()
    gateBegin=text.index('#define W8A8_GATE_COMMON')
    downBegin=text.index('#define W8A8_DOWN_COMMON',gateBegin)
    positionBegin=text.index('#define W8A8_POSITION',downBegin)
    gateBlock=text[gateBegin:downBegin]
    downBlock=text[downBegin:positionBegin]
    gateBlock=replace(gateBlock,'constant FlashInt8ExpertStoreParams &p [[buffer(11)]]','constant FlashInt8ExpertStoreParams &p [[buffer(12)]]')
    gateBlock=replace(gateBlock,'device const float *activation_scales [[buffer(12)]]','device const float *activation_scales [[buffer(11)]]')
    downBlock=replace(downBlock,'constant FlashInt8ExpertStoreParams &p [[buffer(10)]]','constant FlashInt8ExpertStoreParams &p [[buffer(11)]]')
    downBlock=replace(downBlock,'device const float *activation_scales [[buffer(11)]]','device const float *activation_scales [[buffer(10)]]')
    text=text[:gateBegin]+gateBlock+downBlock+text[positionBegin:]
    text='// Worker ABI gate activationScale11/params12; down activationScale10/params11; primitive arithmetic unchanged.\n'+text
    for role in ('gate_up','down_scatter'):
        text=text.replace('prefill_moe_sep21_w8a8_'+role+'_m32','private_w8a8_worker_'+role+'_m32')
    workerData=text.encode();write(output/'source'/relative,workerData)
    manifest['files'].append({'path':relative,'new_w8a8_private_file':True,'primitive_source_sha256':sha(data),
      'worker_abi_parameter_slots':'gate scales11/params12, down scales10/params11; quantizer unchanged',
      'overlay_sha256':sha(workerData)})
    inputs={'REUSED':[],'CORE':[],'AIRS':[]};records=[]
    def freeze(path,relative,category,expected=None):
        data=path.read_bytes()
        if expected and sha(data)!=expected:raise ValueError(f'Sealed link input drift:{path}')
        write(output/relative,data);inputs[category].append(relative.as_posix())
        records.append({'source_path':str(path),'private_path':relative.as_posix(),'category':category,'sha256':sha(data)})
    for record in parent['pointwise_link_inputs']:
        freeze(base/record['private_path'],Path(record['private_path']),record['category'],record['sha256'])
    for name in sorted(POINTWISE):freeze(base/'host'/f'{name}.o',Path('reused/pointwise-host')/f'{name}.o','REUSED')
    freeze(base/'pointwise.air',Path('reused/pointwise.air'),'AIRS')
    frozenMake='\n'.join(f'{key} := '+' '.join('$(BUILD)/'+path for path in paths) for key,paths in inputs.items())+'\n'
    write(output/'link-inputs.mk',frozenMake.encode());manifest['w8a8_link_inputs']=records
    manifest['w8a8_link_inputs_make_sha256']=sha(frozenMake.encode());manifest['w8a8_changed_files']=changed
    write(output/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    for name in ('splash-flash.config','splash.metallib.config'):
        write(output/name,(base/name).read_bytes().rstrip(b'\n')+b'-w8a8-main-prefill-sep21-v1\n')
    print(json.dumps({'prepared':str(output),'checked_sources':len(parent['files']),'changed_sources':changed,
      'frozen_link_inputs':len(records),'gpu_executed':False,'payload_bytes_read':0}))


if __name__=='__main__':main()
