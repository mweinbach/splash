#!/usr/bin/env python3
"""CPU-only sealed Full512 batch exact bulk QSA/cached dense snapshot.

The default-off private flag controls both per-lane exact SG8 QSA and wide
cached dense eligibility. Decode, MTP and singleton numerical bodies are kept.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[3]
FLAG="SPLASH_FLASH_BATCH_QSA_BULK_PREFILL"
EXTRA=234356736


def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def once(text:str,before:str,after:str)->str:
    if text.count(before)!=1:raise ValueError(f"Sealed source anchor drift: {before[:100]}")
    return text.replace(before,after,1)


def batch(text:str)->str:
    text=once(text,'#include "flash/FlashBatchPrefill.hpp"',
        '#include "flash/FlashBatchPrefill.hpp"\n#include "bulk.hpp"')
    text=once(text,'#include <mutex>','#include <mutex>\n#include <optional>')
    text=once(text,'struct FlashBatchPrefill::Impl final {',
        '''bool flashPrivateBatchBulkPrefillEnabled() {
  if (!enabled("SPLASH_FLASH_BATCH_QSA_BULK_PREFILL")) return false;
  if (!enabled("SPLASH_FLASH_QSA_BULK_PREFILL") ||
      !enabled("SPLASH_FLASH_QSA_BULK_PREFILL_SG8") ||
      !enabled("SPLASH_FLASH_QSA_F32") || !enabled("SPLASH_FLASH_QSA_MPP") ||
      !enabled("SPLASH_FLASH_QSA_ROW_TILES"))
    throw std::invalid_argument("private batch bulk requires exact singleton bulk SG8 and its QSA dependencies");
  return true;
}

struct FlashBatchPrefill::Impl final {''')
    text=once(text,'  const bool gpuGreedy = enabled("SPLASH_FLASH_GPU_GREEDY");',
        '  const bool gpuGreedy = enabled("SPLASH_FLASH_GPU_GREEDY");\n'
        '  const bool bulkQSAPrefill = flashPrivateBatchBulkPrefillEnabled();')
    text=once(text,'  FlashQSAFastWorkspace qsaFast;',
        '  FlashQSAFastWorkspace qsaFast;\n  std::optional<prefill4k::BulkExactWorkspace> bulkQSA;')
    text=once(text,'    const uint64_t before = backend.memoryStats().allocatedBytes;',
        '''    if (bulkQSAPrefill && !route(";private-qsa-bulk-prefill-begin0-r2048-w128-"))
      throw std::invalid_argument("private batch bulk requires its frozen exact source trunk");
    const uint64_t before = backend.memoryStats().allocatedBytes;''')
    text=once(text,'    allocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);',
        '''    if (bulkQSAPrefill && maximumRows == 2048 && capacity >= 2048) {
      bulkQSA.emplace(prefill4k::allocateBulkExactWorkspace(backend));
      const auto &bulk = *bulkQSA;
      const uint64_t extent = bulk.prepared.queries.sizeBytes() + bulk.prepared.indexQueries.sizeBytes() +
          bulk.prepared.selectedBlocks.sizeBytes() + bulk.partials.partitionStatistics.sizeBytes() +
          bulk.partials.partitionValues.sizeBytes();
      if (extent != prefill4k::bulkExactPlannedBytes())
        throw std::logic_error("private batch bulk five-plane allocation/admission mismatch");
    }
    allocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);''')
    text=once(text,'      (impl_->batchGDNILP ? kFlashGDNBatchILPRoute : "");',
        '      (impl_->batchGDNILP ? kFlashGDNBatchILPRoute : "") +\n'
        '      (impl_->bulkQSA ? ";private-batch-lane2k-begin0-exact-bulk-sg8-one-fully-consumed-reused-workspace-v1" : "");')
    text=once(text,'    total += FlashPLESSD::plannedBytes(lanes, rows);\n  return total;',
        '    total += FlashPLESSD::plannedBytes(lanes, rows);\n'
        '  if (flashPrivateBatchBulkPrefillEnabled() && rows == 2048 && capacity >= 2048)\n'
        '    total += prefill4k::bulkExactPlannedBytes();\n  return total;')
    text=once(text,
        '      for (uint32_t lane = 0; lane < lanes; ++lane) {\n'
        '        for (uint32_t offset = 0; offset < rows;) {',
        '''      for (uint32_t lane = 0; lane < lanes; ++lane) {
        if (impl_->bulkQSA && rows == 2048 && states[lane]->length == 0) {
          const auto laneSlice = [&](const metal::MetalBuffer &buffer, uint32_t width) {
            return impl_->backend.view(buffer, uint64_t{lane} * rows * width * 2,
                uint64_t{rows} * width * 2);
          };
          const FlashQSAFastInputs inputs{laneSlice(q,12288), laneSlice(k,512), laneSlice(v,512),
              laneSlice(index,640), &impl_->weights.tensor(qNorm), &impl_->weights.tensor(kNorm),
              &impl_->weights.tensor(iqNorm), &impl_->weights.tensor(ikNorm),
              laneSlice(attentionOutput,6144), diag, {}, impl_->weights.normConvention(qNorm),
              impl_->weights.normConvention(kNorm), impl_->weights.normConvention(iqNorm),
              impl_->weights.normConvention(ikNorm), impl_->descriptor.normEpsilon, impl_->descriptor.rotaryTheta};
          // Each six-dispatch lane graph fully consumes its prepared queries and
          // F32 partials in the reducer before the next lane reuses those planes.
          // Independent request caches and lane output views are never reused.
          prefill4k::addBulkExactQSA(impl_->backend, graph, inputs, states[lane]->qsa[layer],
              impl_->qsa, impl_->qsaFast, *impl_->bulkQSA, 0, rows, true);
          continue;
        }
        for (uint32_t offset = 0; offset < rows;) {''')
    return text


def batch_header(text:str)->str:
    return once(text,'namespace splash::flash {',
        'namespace splash::flash {\n[[nodiscard]] bool flashPrivateBatchBulkPrefillEnabled();')


def dense_header(text:str)->str:
    anchor='  if (rows != 2048) return {};'
    if text.count(anchor)!=2:raise ValueError("Both sealed dense geometry/role guards required")
    text=text.replace(anchor,'  if (rows != 2048 && rows != 4096 && rows != 8192) return {};')
    return once(text,'namespace splash::flash {',
        'namespace splash::flash {\n[[nodiscard]] bool flashPrivateBatchBulkPrefillEnabled();')


def dense_caller(text:str)->str:
    text=once(text,'  const auto prefillPlan = flashPrefillDenseTilesEnabled() && weight.shape.size() == 2',
        '  auto prefillPlan = flashPrefillDenseTilesEnabled() && weight.shape.size() == 2')
    return once(text,'      : FlashPrefillDenseTilePlan{};\n  addDenseBF16WholeKImpl',
        '      : FlashPrefillDenseTilePlan{};\n'
        '  if ((rows == 4096 || rows == 8192) && !flashPrivateBatchBulkPrefillEnabled())\n'
        '    prefillPlan = {};\n  addDenseBF16WholeKImpl')


def forward(text:str)->str:
    return once(text,'      (impl_->allRowsInt8Target ? ";private-allrows-full512-target-m16-below256-v1" : "") +',
        '      (impl_->allRowsInt8Target ? ";private-allrows-full512-target-m16-below256-v1" : "") +\n'
        '      (flashPrivateBatchBulkPrefillEnabled() ? ";private-batch-exact-bulk-sg8-and-cached-dense-flat4096or8192-m128-v1" : "") +')


def worker(text:str)->str:
    text=once(text,'      (void)gathered_mpp::requestedMaximumRows();',
        '      (void)flashPrivateBatchBulkPrefillEnabled(); // Strict default-off batch policy before backend creation.\n'
        '      (void)gathered_mpp::requestedMaximumRows();')
    return once(text,
        '      << R"(,"batch_prefill_semantics":)" << (batchPrefill_ ? json::quote(kFlashBatchPrefillSemantics) : "null")',
        '      << R"(,"batch_prefill_semantics":)" << (batchPrefill_ ? json::quote(kFlashBatchPrefillSemantics) : "null")\n'
        '      << R"(,"batch_prefill_kernel_routes":)" << (batchPrefill_ ? json::quote(batchPrefill_->kernelRoutes()) : "null")')


def shader(text:str)->str:
    return once(text,'  if (p.rows != 2048 || !p.input_size',
        '  if ((p.rows != 2048 && p.rows != 4096 && p.rows != 8192) || !p.input_size')


def partition_audit()->dict:
    cases=0
    for lanes in range(1,5):
        for rows in (1,127,128,129,512,1024,2047,2048):
            for enabled in (False,True):
                for fresh_bits in range(1<<lanes):
                    all_rows=[];events=[]
                    for lane in range(lanes):
                        fresh=bool(fresh_bits&(1<<lane));begin=0 if fresh else 4096
                        if enabled and rows==2048 and fresh:
                            events.extend((lane,phase) for phase in ('prepare','pool','select','early','temporalSG8','reduce'))
                            covered=list(range(lane*rows,(lane+1)*rows))
                        else:
                            covered=[lane*rows+offset+r for offset in range(0,rows,128) for r in range(min(128,rows-offset))]
                        if covered!=list(range(lane*rows,(lane+1)*rows)):raise AssertionError("Lane slice gap/overlap")
                        all_rows.extend(covered)
                    if all_rows!=list(range(lanes*rows)):raise AssertionError("Cross-lane row alias")
                    for i,event in enumerate(events):
                        if event[1]=='prepare' and i and events[i-1][1]!='reduce':raise AssertionError("Workspace reused before consume")
                    cases+=1
    return {"lane_schedule_combinations_checked":cases,"lane_outputs_disjoint_and_exact_coverage":True,
        "one_shared_workspace_consumed_before_next_lane":True,"ordinary_nonfresh_or_short_fallback_retained":True}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'build/prefill4k-qsa-bulk-gathered-mpp-sep21-v1')
    parser.add_argument('--output',type=Path,default=ROOT/'build/prefill4k-batch-bulk-gathered-sep21-v1')
    args=parser.parse_args();base=args.base.resolve();output=args.output.resolve()
    if ROOT/'build' not in output.parents or output==base or output.exists():raise ValueError('Choose fresh separate build output')
    raw=(base/'overlay-manifest.json').read_bytes();parent=json.loads(raw)
    if not parent.get('gathered_mpp_composed') or not parent.get('qsa_bulk_composed') or not parent.get('qsa_bulk_sg8_available'):
        raise ValueError('Sealed exactbulk cappedgathered base required')
    transforms={'runtime/flash/FlashBatchPrefill.cpp':batch,'runtime/flash/FlashBatchPrefill.hpp':batch_header,
        'runtime/flash/FlashPrefillDenseTiles.hpp':dense_header,'runtime/flash/FlashDenseCache.cpp':dense_caller,
        'runtime/flash/FlashForward.cpp':forward,'runtime/flash/FlashWorker.mm':worker}
    records=[];files={};changed=[]
    for record in parent['files']:
        relative=Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe manifest path')
        original=(base/'source'/relative).read_bytes()
        if sha(original)!=record['overlay_sha256']:raise ValueError(f'Sealed source drift:{relative}')
        function=transforms.get(str(relative));data=function(original.decode()).encode() if function else original
        files[relative]=data;r=dict(record);r.update({'batch_bulk_input_sha256':sha(original),
            'batch_bulk_changed':data!=original,'overlay_sha256':sha(data)})
        if data!=original:r['patched']=True;changed.append(str(relative))
        records.append(r)
    if set(changed)!=set(transforms):raise AssertionError('Unexpected changed path set')
    relative=Path('runtime/metal/kernels/shared/flash_dense_cache_prefill.metal')
    original=(ROOT/relative).read_bytes();data=shader(original.decode()).encode();files[relative]=data
    records.append({'path':str(relative),'new_private_file':True,'patched':True,'batch_bulk_changed':True,
        'original_sha256':sha(original),'overlay_sha256':sha(data),'shader_change':'Only accepted row guard widened; numericalbody unchanged'})
    manifest=copy.deepcopy(parent);manifest.update({'route':'private-full512-cappedgathered-batch-lane2k-exactsg8-and-widecached-dense-defaultoff-v1',
        'batch_bulk_composed':True,'batch_bulk_base_build':str(base),'batch_bulk_input_manifest_sha256':sha(raw),
        'batch_bulk_flag':FLAG,'batch_bulk_flag_default':False,'batch_bulk_single_workspace_extra_bytes':EXTRA,
        'batch_bulk_geometry':'Each mainprefill lane rows2048 begin0; workspace consumed before nextlane; generic fallback otherwise',
        'batch_bulk_dense_flat_rows':[4096,8192],'batch_bulk_dense_requires_same_private_flag':True,
        'batch_bulk_generator_sha256':sha(Path(__file__).read_bytes()),'normal_sources_modified':False,
        'gpu_executed':False,'payload_bytes_read':0,'batch_model_quality_qualified':False,'files':records})
    audit={'schema':'splash-prefill-batch-bulk-cpu-source-v1','source_files_hash_verified':len(parent['files']),
        'changed_paths':changed+[str(relative)],'extra_workspace_bytes':EXTRA,
        'existing_worker_reservation_calls_updated_static_planner_before_constructor':True,
        'existing_worker_checks_actual_delta_against_reservation':True,'new_flag_validated_before_backend_creation':True,
        'bulk_helpers_and_kernels_byte_unchanged':all(not r.get('batch_bulk_changed') for r in records if r['path'].startswith('dev/benchmarks/prefill4k_attention/')),
        'decode_and_mtp_executor_bytes_unchanged':all(not r.get('batch_bulk_changed') for r in records if any(n in r['path'] for n in ('FlashBatchForward','FlashBatchVerify','FlashMTP','FlashBatchMTP','FlashInt8ExpertStore','flash_gathered_mpp'))),
        'singleton_arithmetic_body_unchanged':True,'dense_descriptor_and_numericalbody_unchanged':True,
        'flag0_wide_dense_route_is_original_generic':True,'flag0_trunk_kernel_routes_are_original':True,
        'gpu_executed':False,'payload_bytes_read':0,**partition_audit()}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='prefill-batch-bulk-',dir=output.parent) as temporary:
        staged=Path(temporary)/'output'
        for relative,data in files.items():
            p=staged/'source'/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        (staged/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (staged/'batch-bulk-cpu-source-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        staged.rename(output)
    print(json.dumps({'prepared':str(output),**audit}))


if __name__=='__main__':main()
