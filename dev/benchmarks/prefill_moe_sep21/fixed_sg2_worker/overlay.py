#!/usr/bin/env python3
"""Freeze exact-intended SG2/K128 main-prefill scheduling over sealed pointwise sources."""
from pathlib import Path
import argparse
import copy
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]
PRIVATE=Path('dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker')
INCLUDE='#include "dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy.hpp"\n'
INHERITED_HOST=('FlashMoE','FlashMoEBlocked','FlashExpertDenseCache')


def sha(data):return hashlib.sha256(data).hexdigest()


def replace(text,before,after,count=1):
    if text.count(before)!=count:raise ValueError(f'SG2 fixedK128 source anchor drift:{before!r}')
    return text.replace(before,after)


def transform(relative,text):
    if relative in ('runtime/flash/FlashInt8ExpertStore.hpp','runtime/flash/FlashInt8ExpertStore.mm',
                    'runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm','dev/benchmarks/prefill4k_attribution.mm'):
        text=INCLUDE+text
    if relative=='runtime/flash/FlashInt8ExpertStore.hpp':
        point='  // Store methods keep immutable base mappings and rank allocations private.'
        addition='''  // Component-bit-exact qualified scheduling; actual-model equality pending.
  void addFixedSG2PrefillGateUp(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,
      uint32_t rows,FlashMoEBlockedTile tile,uint32_t selections=10) const;
  void addFixedSG2PrefillDownScatter(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,
      uint32_t rows,FlashMoEBlockedTile tile,uint32_t selections=10) const;
  [[nodiscard]] fixed_sg2_prefill_sep21::Counters fixedSG2PrefillCounters() const noexcept;

'''
        text=replace(text,point,addition+point)
    if relative=='runtime/flash/FlashInt8ExpertStore.mm':
        point='  std::string numericalIdentity;'
        text=replace(text,point,point+'\n  mutable std::atomic<uint64_t> fixedSG2GateCalls{0},fixedSG2GateRows{0},fixedSG2DownCalls{0},fixedSG2DownRows{0};')
        begin=text.index('void FlashInt8ExpertStore::addGateUp(')
        end=text.index('\nnamespace {\nvoid gatheredMPPViews',begin)
        methods=text[begin:end]
        methods=methods.replace('FlashInt8ExpertStore::addGateUp(', 'FlashInt8ExpertStore::addFixedSG2PrefillGateUp(')
        methods=methods.replace('FlashInt8ExpertStore::addDownScatter(', 'FlashInt8ExpertStore::addFixedSG2PrefillDownScatter(')
        methods=replace(methods,'    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {',
            '    uint32_t rows,FlashMoEBlockedTile tile,uint32_t selections) const {\n'
            '  if (!fixed_sg2_prefill_sep21::eligible(rows,tile,false) ||selections !=10) fail("fixedSG2 only main R2048/M32 canonical source");',2)
        methods=replace(methods,'  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;',
            '  const uint32_t threads=fixed_sg2_prefill_sep21::producerThreads();',2)
        methods=replace(methods,'graph.add(pipeline("gate_up", tile),','graph.add(fixed_sg2_prefill_sep21::producerName(true),')
        methods=replace(methods,'graph.add(pipeline("down_scatter", tile),','graph.add(fixed_sg2_prefill_sep21::producerName(false),')
        methods=replace(methods,'  impl_->recordGraph(false, rows, 512);',
            '  impl_->recordGraph(false, rows, 512);\n  impl_->fixedSG2GateCalls.fetch_add(1,std::memory_order_relaxed);\n  impl_->fixedSG2GateRows.fetch_add(rows,std::memory_order_relaxed);')
        methods=replace(methods,'  impl_->recordGraph(true, rows, 512);',
            '  impl_->recordGraph(true, rows, 512);\n  impl_->fixedSG2DownCalls.fetch_add(1,std::memory_order_relaxed);\n  impl_->fixedSG2DownRows.fetch_add(rows,std::memory_order_relaxed);')
        methods+='''
fixed_sg2_prefill_sep21::Counters FlashInt8ExpertStore::fixedSG2PrefillCounters() const noexcept {
  return {fixed_sg2_prefill_sep21::requested(),impl_->fixedSG2GateCalls.load(std::memory_order_relaxed),
      impl_->fixedSG2GateRows.load(std::memory_order_relaxed),impl_->fixedSG2DownCalls.load(std::memory_order_relaxed),
      impl_->fixedSG2DownRows.load(std::memory_order_relaxed)};
}
'''
        text=text[:end]+'\n'+methods+text[end:]
    if relative=='runtime/flash/FlashForward.cpp':
        point='      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +'
        text=replace(text,point,point+'\n      std::string(fixed_sg2_prefill_sep21::marker()) +')
        before='''        impl_->int8ExpertStore->addGateUp(graph, layer, impl_->blockedScratch, diag, rows, tile);
        impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);'''
        after='''        if (fixed_sg2_prefill_sep21::eligible(rows,tile,verification)) {
          impl_->int8ExpertStore->addFixedSG2PrefillGateUp(graph,layer,impl_->blockedScratch,diag,rows,tile);
          impl_->int8ExpertStore->addFixedSG2PrefillDownScatter(graph,layer,impl_->blockedScratch,diag,rows,tile);
        } else {
          impl_->int8ExpertStore->addGateUp(graph, layer, impl_->blockedScratch, diag, rows, tile);
          impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);
        }'''
        text=replace(text,before,after)
    if relative=='runtime/flash/FlashWorker.mm':
        point='      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.'
        text=replace(text,point,point+'\n      (void)fixed_sg2_prefill_sep21::selection(); // Validate before paths, metadata, backend or model.')
        point='  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};'
        text=replace(text,point,point+'\n  const auto fixedSG2Graphs=persistedExperts ? persistedExperts->fixedSG2PrefillCounters() :fixed_sg2_prefill_sep21::Counters{};')
        point='      << R"(,"gdn_verification_storage":{"lazy_enabled":)"'
        addition='''      << R"(,"fixed_sg2_prefill_route_counters":{"scope":"main nonverification physical rows2048/M32; graph construction not GPU completion; component BF16 exact, actual-model equality pending","selected_variant":)"
      <<fixed_sg2_prefill_sep21::selection()
      << R"(,"enabled":)" <<(fixedSG2Graphs.enabled ? "true" :"false")
      << R"(,"additional_allocation_bytes":0,"numerical_derivative_changed":false,"gate_graph_calls":)" <<fixedSG2Graphs.gateCalls
      << R"(,"gate_graph_rows":)" <<fixedSG2Graphs.gateRows
      << R"(,"down_graph_calls":)" <<fixedSG2Graphs.downCalls
      << R"(,"down_graph_rows":)" <<fixedSG2Graphs.downRows <<'}'
'''
        text=replace(text,point,addition+point)
    if relative=='dev/benchmarks/prefill4k_attribution.mm':
        point='      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.'
        text=replace(text,point,point+'\n      (void)fixed_sg2_prefill_sep21::selection();')
    return text


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists() or path.read_bytes()!=data:path.write_bytes(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'build/moe-pointwise-sep21-worker-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'build/prefill-moe-sg2-k128-pointwise-sep21-worker-v1')
    args=parser.parse_args();base=args.base.resolve();output=args.output.resolve()
    if output==base or ROOT/'build' not in output.parents:raise ValueError('Distinct private SG2 scheduling build required')
    parentPath=base/'overlay-manifest.json';parent=json.loads(parentPath.read_text())
    if not parent.get('pointwise_composed') or not parent.get('gathered_mpp_composed') or parent.get('omitted_original_target_tensor_count')!=432:
        raise ValueError('Expected sealed pointwise/gathered/bulk Full512 source')
    manifest=copy.deepcopy(parent);manifest.update({'route':'private-full512-pointwise-bulk-gathered-fixedSG2K128-main-prefill-sep21-v1',
      'fixed_sg2_prefill_environment':'SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT=0|7','fixed_sg2_flag0_original_graphs_and_identity':True,
      'fixed_sg2_additional_allocation_bytes':0,'fixed_sg2_numerical_derivative_changed':False,
      'fixed_sg2_component_qualification':'Root complete expert-chain synthetic BF16 exact; actual model equality/throughput pending',
      'fixed_sg2_scope':'main nonverification R2048/M32 canonical Full512; other rows, batch executor, decode/verification unchanged',
      'fixed_sg2_base_build':str(base),'fixed_sg2_base_manifest_sha256':sha(parentPath.read_bytes()),
      'fixed_sg2_transform_sha256':sha(Path(__file__).read_bytes()),'gpu_executed':False,'payload_bytes_read':0,'files':[]})
    changed=[]
    for record in parent['files']:
        relative=record['path'];original=(base/'source'/relative).read_bytes()
        if sha(original)!=record['overlay_sha256']:raise ValueError(f'Sealed pointwise source drift:{relative}')
        data=transform(relative,original.decode()).encode();write(output/'source'/relative,data)
        if data!=original:changed.append(relative)
        manifest['files'].append({**record,'fixed_sg2_changed':data!=original,'fixed_sg2_base_sha256':sha(original),'overlay_sha256':sha(data)})
    for name in ('policy.hpp','policy_cpu.cpp'):
        relative=str(PRIVATE/name);data=(ROOT/relative).read_bytes();write(output/'source'/relative,data)
        manifest['files'].append({'path':relative,'new_fixed_sg2_private_file':True,'overlay_sha256':sha(data)})
    shader=(ROOT/'dev/benchmarks/prefill_moe_sep21/memory.metal').read_text()
    lines=shader.splitlines(keepends=True)
    shader=''.join(line for line in lines if not line.startswith('PREFILL4K_INT8TILES_') or
                   '_m32_n64_k128_sg2' in line)
    relative=str(PRIVATE/'candidate.metal');data=shader.encode();write(output/'source'/relative,data)
    manifest['files'].append({'path':relative,'new_fixed_sg2_private_file':True,'primitive_source_sha256':sha((ROOT/'dev/benchmarks/prefill_moe_sep21/memory.metal').read_bytes()),'overlay_sha256':sha(data)})
    inputs={'REUSED':[],'CORE':[],'AIRS':[]};records=[]
    def freeze(path,relative,category,expected=None):
        data=path.read_bytes()
        if expected and sha(data)!=expected:raise ValueError(f'Sealed link input drift:{path}')
        write(output/relative,data);inputs[category].append(relative.as_posix())
        records.append({'source_path':str(path),'private_path':relative.as_posix(),'category':category,'sha256':sha(data)})
    for r in parent['pointwise_link_inputs']:freeze(base/r['private_path'],Path(r['private_path']),r['category'],r['sha256'])
    for name in INHERITED_HOST:freeze(base/'host'/f'{name}.o',Path('reused/pointwise-host')/f'{name}.o','REUSED')
    freeze(base/'pointwise.air',Path('reused/pointwise.air'),'AIRS')
    make='\n'.join(f'{key} := '+' '.join('$(BUILD)/'+p for p in paths) for key,paths in inputs.items())+'\n'
    write(output/'link-inputs.mk',make.encode());manifest['fixed_sg2_link_inputs']=records
    manifest['fixed_sg2_link_inputs_make_sha256']=sha(make.encode());manifest['fixed_sg2_changed_files']=changed
    write(output/'overlay-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    for name in ('splash-flash.config','splash.metallib.config'):
        write(output/name,(base/name).read_bytes().rstrip(b'\n')+b'-fixed-sg2-k128-main-prefill-sep21-v1\n')
    print(json.dumps({'prepared':str(output),'source_seals_checked':len(parent['files']),'changed_sources':changed,
      'frozen_link_inputs':len(records),'additional_allocation_bytes':0,'gpu_work':False,'payload_bytes_read':0}))


if __name__=='__main__':main()
