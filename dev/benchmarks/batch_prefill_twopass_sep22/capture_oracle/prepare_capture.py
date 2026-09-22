#!/usr/bin/env python3
"""CPU-only private actual batch capture TU preparation, original runtime retained."""
import argparse,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
 if out.exists():raise ValueError('fresh capture source directory required')
 parent=ROOT/'build/batch-prefill-restored-teacher-clock-sep22-worker-v1';seal=json.loads((parent/'compiled-cpu-seal.json').read_text());out.mkdir(parents=True)
 original=parent/'source/runtime/flash/FlashBatchPrefill.cpp';text=original.read_text();text='#include "capture.hpp"\n'+text
 projection_call='''          prefill4k::addBulkExactQSA(impl_->backend, graph, inputs, states[lane]->qsa[layer],
              impl_->qsa, impl_->qsaFast, *impl_->bulkQSA, 0, rows, true);'''
 capture_call='''          batch_qsa_capture::beforeQSA(graph,inputs,states[lane]->qsa[layer],layer,lane,lanes,rows,
              states[lane]->length,impl_->weights.sourceIdentity());
          prefill4k::addBulkExactQSA(impl_->backend, graph, inputs, states[lane]->qsa[layer],
              impl_->qsa, impl_->qsaFast, *impl_->bulkQSA, 0, rows, true);
          batch_qsa_capture::afterQSA(graph,inputs.output,layer,lane);'''
 if text.count(projection_call)!=1:raise ValueError('actual original Bulk6DSP source position drift')
 text=text.replace(projection_call,capture_call)
 preflight='''  if (batch_qsa_capture::activeSession) {
    std::vector<uint64_t> actualLengths; actualLengths.reserve(lanes);
    for (uint32_t lane=0; lane<lanes; ++lane) actualLengths.push_back(states[lane]->length);
    std::vector<metal::MetalBuffer> others;
    for (const auto &buffer : impl_->guardedBases) others.push_back(buffer);
    for (const auto &buffer : {impl_->qsa.queries, impl_->qsa.indexQueries, impl_->qsa.blockScores,
        impl_->qsa.selectedBlocks, impl_->qsa.attentionScores, impl_->qsa.probabilities,
        impl_->qsaFast.partitionStatistics, impl_->qsaFast.partitionValues,
        impl_->greedyWorkspace.partials, impl_->greedyResults}) others.push_back(buffer);
    if (impl_->pleSSD) for (const auto &buffer : impl_->pleSSD->scratchBuffers()) others.push_back(buffer);
    if (impl_->bulkQSA) for (const auto &buffer : {impl_->bulkQSA->prepared.queries,
        impl_->bulkQSA->prepared.indexQueries, impl_->bulkQSA->prepared.selectedBlocks,
        impl_->bulkQSA->partials.partitionStatistics, impl_->bulkQSA->partials.partitionValues}) others.push_back(buffer);
    const auto &outerBlocked = impl_->blocked;
    for (const auto &buffer : {outerBlocked.buckets.counts, outerBlocked.buckets.offsets, outerBlocked.buckets.routeMap,
        outerBlocked.buckets.canonicalToPacked, outerBlocked.buckets.packedInputs, outerBlocked.buckets.jobOffsets,
        outerBlocked.buckets.jobCount, outerBlocked.buckets.tileJobs, outerBlocked.packedActivated, outerBlocked.scatteredDown}) others.push_back(buffer);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      others.push_back(states[lane]->pleHistory); others.push_back(states[lane]->pleConvolution);
      for (uint32_t layer = 0; layer < impl_->descriptor.layers; ++layer) {
        const auto &gdn = states[lane]->gdn[layer];
        if (gdn.recurrent) { others.push_back(gdn.convolution); others.push_back(gdn.recurrent); }
        else { const auto &qsa = states[lane]->qsa[layer];
          for (const auto &buffer : {qsa.keys,qsa.values,qsa.rawIndexKeys,qsa.pooledKeys,qsa.indexPositions}) others.push_back(buffer); }
      }
    }
    for (const auto &buffer : impl_->weights.immutableWeightBuffers()) others.push_back(buffer);
    if (hiddenDestination) others.push_back(hiddenDestination);
    batch_qsa_capture::validateWholeCohort(impl_->backend,lanes,rows,impl_->capacity,actualLengths,others);
  }
'''
 preflight_anchor='  for (uint32_t token : tokens)\n'
 if text.count(preflight_anchor)!=1:raise ValueError('complete preflight insertion source drift')
 text=text.replace(preflight_anchor,preflight+preflight_anchor)
 publication='  for (uint32_t lane = 0; lane < lanes; ++lane) lengths[lane] = states[lane]->length += rows;'
 completion='\n  batch_qsa_capture::completed(impl_->backend.healthy());'
 if text.count(publication)!=1:raise ValueError('completed native result publication source drift')
 text=text.replace(publication,publication+completion)
 (out/'FlashBatchPrefill_capture.cpp').write_text(text);shutil.copy2(HERE/'capture.hpp',out/'capture.hpp')
 normalization=text.removeprefix('#include "capture.hpp"\n').replace(capture_call,projection_call).replace(preflight,'').replace(completion,'')
 if normalization!=original.read_text():raise ValueError('debug capture changes original computation/binding/channel source')
 d={'schema':'actual-batch-qsa-capture-source-v1','pass':True,'GPU_executed':False,'model_or_capture_payload_read':False,'parent':str(parent),'parent_source_sha256':sha(original),'parent_seal_sha256':sha(parent/'compiled-cpu-seal.json'),'exact_original_body_after_only_debughook_normalization':True,'public_header_or_layout_changes':False,'whole_cohort_fresh_preflight_before_any_token_or_diag_write':True,'all_capture_owners_disjoint_from_batch_state_cache_immutable_shared_views':True,'only_one_host_TU_modified':'FlashBatchPrefill.cpp','default_off':'thread_local activeSession=nullptr; normalWorker neverinstalls','input_owned_bytes':57147392,'mandatory_output_owned_bytes':25165824,'guarded_norm_and_buffers_Gov_reservation':84<<20,'selected_actual_source':'realBulkBQSA cohortr2048/B2B4 lane0layer3 begin0 actualcache/norms/projectionviews','files':[{'path':str(q),'sha256':sha(q)} for q in out.glob('*') if q.is_file()]}
 (out/'capture-source-receipt.json').write_text(json.dumps(d,indent=2)+'\n');print(out)
if __name__=='__main__':main()
