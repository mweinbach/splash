#!/usr/bin/env python3
"""Isolated CPU-only current-parent batch QSA transfer. Never loads model data."""
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def once(text, before, after):
    if text.count(before) != 1: raise ValueError('Source anchor differs: ' + before[:100])
    return text.replace(before, after, 1)

def transform_batch(text):
    text = '#include "dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp"\n' + text
    text = once(text, '  std::optional<prefill4k::BulkExactWorkspace> bulkQSA;',
        '  std::optional<prefill4k::BulkExactWorkspace> bulkQSA;\n  std::optional<batch_prefill_twopass_sep22::Workspace> twoPassQSA;')
    text = once(text, '    allocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);',
        '''    if (batch_prefill_twopass_sep22::extraBytes(capacity, maximumRows, batch_prefill_twopass_sep22::requested())) {
      if (!bulkQSA || !onlineQSA || !mppQSA || qsa.maximumRows != 128 ||
          qsaFast.maximumRows != 128 || qsaFast.maximumPartitions != 32 ||
          descriptor.normEpsilon != 1e-6 || descriptor.rotaryTheta != 1e7)
        throw std::logic_error("batch two-pass source workspace/model constants differ");
      twoPassQSA.emplace(backend);
    }
    allocatedBytes = metal::allocationDelta(before, backend.memoryStats().allocatedBytes);''')
    text = once(text, '  return total;\n}\n\nFlashBatchPrefillResult',
        '  total += batch_prefill_twopass_sep22::extraBytes(capacity, rows, batch_prefill_twopass_sep22::requested());\n  return total;\n}\n\nFlashBatchPrefillResult')
    text = once(text, '      (impl_->bulkQSA ? ";private-batch-lane2k-begin0-exact-bulk-sg8-one-fully-consumed-reused-workspace-v1" : "");',
        '      (impl_->bulkQSA ? ";private-batch-lane2k-begin0-exact-bulk-sg8-one-fully-consumed-reused-workspace-v1" : "") +\n      (impl_->twoPassQSA ? ";private-batch-real2or4-allfresh2048-existing-packedV-twopass-v1" : "");')
    text = once(text, '  for (uint32_t token : tokens)\n    if (token >= impl_->descriptor.vocabularySize)',
        '''  bool allFresh = true;
  for (uint32_t lane = 0; lane < lanes; ++lane) allFresh = allFresh && states[lane]->length == 0;
  const bool useTwoPass = impl_->twoPassQSA &&
      batch_prefill_twopass_sep22::eligible(lanes, rows, allFresh, true);
  if (useTwoPass) {
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
    batch_prefill_twopass_sep22::validateWholeCohortArena(*impl_->twoPassQSA, others);
  }
  for (uint32_t token : tokens)
    if (token >= impl_->descriptor.vocabularySize)''')
    text = once(text, '    const auto &blocked = impl_->blocked;\n    for (const auto &buffer : {blocked.buckets.counts',
        '''    if (impl_->twoPassQSA) {
      for (const auto &buffer : impl_->twoPassQSA->planes())
        if (flashBatchPrefillCopyRangesOverlap(hiddenDestination.contents(), bytes, buffer.contents(), buffer.sizeBytes()))
          throw std::invalid_argument("Flash batch prefill feature destination overlaps batch two-pass QSA workspace");
      if (flashBatchPrefillCopyRangesOverlap(hiddenDestination.contents(), bytes,
          impl_->twoPassQSA->arena.contents(), impl_->twoPassQSA->arena.sizeBytes()))
        throw std::invalid_argument("Flash batch prefill feature destination overlaps batch two-pass QSA workspace");
    }
    const auto &blocked = impl_->blocked;
    for (const auto &buffer : {blocked.buckets.counts''')
    text = once(text, '          prefill4k::addBulkExactQSA(impl_->backend, graph, inputs, states[lane]->qsa[layer],\n              impl_->qsa, impl_->qsaFast, *impl_->bulkQSA, 0, rows, true);',
        '''          if (useTwoPass) {
            prefill4k::addTwoPassQSA(impl_->backend, graph, inputs, states[lane]->qsa[layer],
                impl_->qsa, impl_->qsaFast, impl_->twoPassQSA->qualified, 0, rows, false, true);
            batch_prefill_twopass_sep22::counters().layers.fetch_add(1);
            batch_prefill_twopass_sep22::counters().lanes.fetch_add(1);
          } else prefill4k::addBulkExactQSA(impl_->backend, graph, inputs, states[lane]->qsa[layer],
              impl_->qsa, impl_->qsaFast, *impl_->bulkQSA, 0, rows, true);''')
    text = once(text, '  for (uint32_t lane = 0; lane < lanes; ++lane) lengths[lane] = states[lane]->length += rows;',
        '  for (uint32_t lane = 0; lane < lanes; ++lane) lengths[lane] = states[lane]->length += rows;\n  if (useTwoPass) batch_prefill_twopass_sep22::counters().forwards.fetch_add(1);')
    # This inspector is metadata only and returns existing owned views.
    text += '''
namespace splash::flash {
std::array<metal::MetalBuffer,7> FlashBatchPrefill::inspectTwoPassQSAWorkspaceForOracle() const {
  if (!impl_) throw std::logic_error("Flash batch prefill is not initialized");
  std::scoped_lock lock(impl_->mutex, impl_->trunk.batchMutex());
  if (!impl_->twoPassQSA) return {};
  const auto p = impl_->twoPassQSA->planes();
  return {impl_->twoPassQSA->arena,p[0],p[1],p[2],p[3],p[4],p[5]};
}
}
'''
    return text

def transform_worker(text):
    text = '#include "dev/benchmarks/batch_prefill_twopass_sep22/policy.hpp"\n' + text
    text = once(text, '      (void)adaptive_expert_tail_sg2k128_sep21::requested();',
        '      (void)batch_prefill_twopass_sep22::requested(); // Freeze strict batch scope before any config/path/backend.\n      (void)adaptive_expert_tail_sg2k128_sep21::requested();')
    text = once(text, '      << R"(,"batch_prefill_semantics":)"',
        '''      << R"(,"batch_prefill_twopass_requested":)" << (batch_prefill_twopass_sep22::requested()?"true":"false")
      << R"(,"batch_prefill_twopass_schema":)" << (batch_prefill_twopass_sep22::requested()?json::quote(batch_prefill_twopass_sep22::schema):"null")
      << R"(,"batch_prefill_twopass_policy":)" << (batch_prefill_twopass_sep22::requested()?json::quote(batch_prefill_twopass_sep22::policy):"null")
      << R"(,"batch_prefill_twopass_source_sha256":)" << (batch_prefill_twopass_sep22::requested()?json::quote(batch_prefill_twopass_sep22::sourceSHA):"null")
      << R"(,"batch_prefill_twopass_shader_sha256":)" << (batch_prefill_twopass_sep22::requested()?json::quote(batch_prefill_twopass_sep22::shaderSHA):"null")
      << R"(,"batch_prefill_twopass_host_sha256":)" << (batch_prefill_twopass_sep22::requested()?json::quote(batch_prefill_twopass_sep22::hostSHA):"null")
      << R"(,"batch_prefill_twopass_numerical_identity":)" << (batch_prefill_twopass_sep22::requested()?json::quote(batch_prefill_twopass_sep22::numericalIdentity(forward_.kernelRoutes())):"null")
      << R"(,"batch_prefill_twopass_arena_plan_bytes":)" << (batch_prefill_twopass_sep22::requested()?batch_prefill_twopass_sep22::plannedBytes:0)
      << R"(,"batch_prefill_semantics":)"''')
    text = once(text, '      << R"(,"persisted_operands":{"schema":"splash-local-affine-operands-v1",',
        '''      << R"(,"batch_prefill_twopass_counters":{"scope":"arena and encoded fields count graph construction; completed_native_forwards counts healthy API completion after diagnostics/state publication","constructed_arenas":)" << batch_prefill_twopass_sep22::counters().arenas.load()
      << R"(,"constructed_arena_bytes":)" << batch_prefill_twopass_sep22::counters().arenaBytes.load()
      << R"(,"encoded_QSA_lane_calls":)" << batch_prefill_twopass_sep22::counters().lanes.load()
      << R"(,"encoded_QSA_lane_layer_calls":)" << batch_prefill_twopass_sep22::counters().layers.load()
      << R"(,"completed_native_forwards":)" << batch_prefill_twopass_sep22::counters().forwards.load() << '}'
      << R"(,"persisted_operands":{"schema":"splash-local-affine-operands-v1",''')
    return text

def main():
    p = argparse.ArgumentParser(); p.add_argument('--base',type=Path,default=ROOT/'build/batch-prefill-restored-teacher-clock-sep22-worker-v1'); p.add_argument('--output',type=Path,default=ROOT/'build/batch-prefill-twopass-restored-sep22-worker-v1'); a=p.parse_args()
    base,out=a.base.resolve(),a.output.resolve()
    if out.exists() or ROOT/'build' not in out.parents: raise ValueError('fresh isolated output required')
    seal=json.loads((base/'compiled-cpu-seal.json').read_text())
    if not seal['pass'] or seal['metallib_sha256']!='1e34d3b01907acd7ad8d42532c072212b0d276336c23dc0fd49b8663711dfbaf': raise ValueError('exact current restored parent required')
    records=[]
    for e in seal['source_files']:
        rel=e['path']; before=(base/'source'/rel).read_bytes()
        if hashlib.sha256(before).hexdigest()!=e['sha256']: raise ValueError('parent source drift:'+rel)
        text=before.decode()
        if rel=='runtime/flash/FlashBatchPrefill.cpp': text=transform_batch(text)
        if rel=='runtime/flash/FlashWorker.mm': text=transform_worker(text)
        if rel=='runtime/flash/FlashBatchPrefill.hpp':
            text=once(text,'  [[nodiscard]] std::string kernelRoutes() const;', '  [[nodiscard]] std::string kernelRoutes() const;\n  [[nodiscard]] std::array<metal::MetalBuffer,7> inspectTwoPassQSAWorkspaceForOracle() const;')
        path=out/'source'/rel; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
        records.append({'path':rel,'sha256':sha(path),'parent_sha256':e['sha256'],'changed':before!=text.encode()})
    for name in ['policy.hpp']:
        path=out/'source/dev/benchmarks/batch_prefill_twopass_sep22'/name; path.parent.mkdir(parents=True,exist_ok=True)
        raw=(HERE/name).read_bytes(); material=raw+(base/'source/runtime/flash/FlashBatchPrefill.cpp').read_bytes()+(base/'source/dev/benchmarks/prefill_qsa_twopass_sep21/twopass.cpp').read_bytes()
        policy_sha=hashlib.sha256(material).hexdigest();path.write_bytes(raw.replace(b'BATCH_SOURCE_SHA_PLACEHOLDER',policy_sha.encode()));records.append({'path':str(path.relative_to(out/'source')),'sha256':sha(path),'new':True})
    cores=[]
    for e in seal['core_objects']:
        if sha(base/e['path'])!=e['sha256']: raise ValueError('parent Core drift')
        path=out/e['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((base/e['path']).read_bytes());cores.append(e)
    for e in seal['compiled_objects']:
        if sha(base/e['object'])!=e['sha256']: raise ValueError('parent host drift')
    if len(seal['compiled_objects'])!=50 or len(cores)!=4:raise ValueError('expected50host/Core4')
    (out/'splash.metallib').write_bytes((base/'splash.metallib').read_bytes())
    if sha(out/'splash.metallib')!=seal['metallib_sha256']:raise ValueError('parent library drift')
    (out/'machinery').mkdir()
    for name in ['prepare.py','policy.hpp']: (out/'machinery'/name).write_bytes((HERE/name).read_bytes())
    manifest={'schema':'private-current-batch-packedV-twopass-full509-v1','base':str(base),'base_seal_sha256':sha(base/'compiled-cpu-seal.json'),'files':records,'rebuild':seal['compiled_objects'],'Core':cores,'metallib_sha256':seal['metallib_sha256'],'source_policy_sha256':policy_sha,'additional_arena_bytes':509607936,'legacy_arena_bytes_retained':234356736,'GPU_executed':False,'model_payload_read':False,'GPU_qualified':False}
    (out/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'prepared':str(out),'sources':len(records),'GPU_work':False}))
if __name__=='__main__':main()
