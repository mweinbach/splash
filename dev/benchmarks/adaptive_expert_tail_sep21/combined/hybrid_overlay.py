#!/usr/bin/env python3
"""Freeze SG2/K128 tails and optional GDN prefill FMA on resident Q4/I8 hybrid v4.

Original Q4 decode/verify, strong aliases, fixed-R4 caches and residency guards
remain sealed. The exact I8 tail certificate is independent of FMA rounding.
No GPU backend, model mapping, payload read or promotion occurs here.
"""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
from worker_overlay import ROOT, PRIVATE, FMA_PRIVATE, SG2_POLICY, sha, write, replace, qualify_tail, fma_module

OWN_NAMES = {'FlashInt8ExpertStore', 'FlashForward', 'FlashWorker', 'FlashGDNStaged', 'FlashBatchPrefill'}
SG2_INCLUDE = '#include "dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/policy.hpp"\n'
BASE_IDENTITY = '1a00dd45649f14de4ad48bafa32ff67f1207aaf27b201de01bc6641a210134e2'
EXPECTED_CHANGED = {'runtime/flash/FlashInt8ExpertStore.hpp', 'runtime/flash/FlashInt8ExpertStore.mm',
                    'runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm',
                    'runtime/flash/FlashGDNStaged.cpp', 'runtime/flash/FlashBatchPrefill.cpp',
                    'dev/benchmarks/prefill4k_attribution.mm'}


def cloned_methods(text):
    begin = text.index('void FlashInt8ExpertStore::addGateUp(')
    end = text.index('} // namespace splash::flash', begin)
    methods = text[begin:end]
    methods = methods.replace('FlashInt8ExpertStore::addGateUp(', 'FlashInt8ExpertStore::addFixedSG2PrefillGateUp(')
    methods = methods.replace('FlashInt8ExpertStore::addDownScatter(', 'FlashInt8ExpertStore::addFixedSG2PrefillDownScatter(')
    methods = replace(methods, '    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {',
        '''    uint32_t rows, FlashMoEBlockedTile tile, uint32_t selections) const {
  if (!fixed_sg2_prefill_sep21::eligible(rows,tile,false) || selections !=10)
    fail("hybrid SG2 prefill only canonical main R2048/M32");''', 2)
    methods = replace(methods, '  const auto &layer = impl_->layer(index);',
        '''  const auto &layer = impl_->layer(index);
  if (impl_->metadata.layers[index].selectedIDs.size() !=512)
    fail("hybrid SG2 prefill requires immutable Full512 inventory");''', 2)
    methods = replace(methods, '  const auto hitPipeline = pipeline("gate_up", tile);',
                       '  const auto hitPipeline = fixed_sg2_prefill_sep21::producerName(true);')
    methods = replace(methods, '  const auto hitPipeline = pipeline("down_scatter", tile);',
                       '  const auto hitPipeline = fixed_sg2_prefill_sep21::producerName(false);')
    methods = replace(methods, '  const uint32_t threads = tile == FlashMoEBlockedTile::M64N64 ? 256 : 128;',
                       '  const uint32_t threads = fixed_sg2_prefill_sep21::producerThreads();', 2)
    methods = replace(methods, '  impl_->recordGraph(false, rows, miss.stored_experts);',
        '''  impl_->recordGraph(false, rows, miss.stored_experts);
  impl_->fixedSG2GateCalls.fetch_add(1,std::memory_order_relaxed);
  impl_->fixedSG2GateRows.fetch_add(rows,std::memory_order_relaxed);''')
    methods = replace(methods, '  impl_->recordGraph(true, rows, miss.stored_experts);',
        '''  impl_->recordGraph(true, rows, miss.stored_experts);
  impl_->fixedSG2DownCalls.fetch_add(1,std::memory_order_relaxed);
  impl_->fixedSG2DownRows.fetch_add(rows,std::memory_order_relaxed);''')
    return methods


def tail_transform(relative, text):
    if relative in ('runtime/flash/FlashInt8ExpertStore.hpp', 'runtime/flash/FlashInt8ExpertStore.mm',
                    'runtime/flash/FlashForward.cpp', 'runtime/flash/FlashWorker.mm', 'dev/benchmarks/prefill4k_attribution.mm'):
        text = SG2_INCLUDE + text
    if relative == 'runtime/flash/FlashInt8ExpertStore.hpp':
        text = replace(text, 'private:\n  struct Impl;', '''  // Exact component-certified ordered SG2/K128 prefill; native decode/verify unchanged.
  void addFixedSG2PrefillGateUp(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,
      uint32_t rows,FlashMoEBlockedTile tile,uint32_t selections=10) const;
  void addFixedSG2PrefillDownScatter(metal::CommandGraph &graph,uint32_t layer,
      const FlashMoEBlockedScratch &scratch,metal::MetalBuffer diagnostics,
      uint32_t rows,FlashMoEBlockedTile tile,uint32_t selections=10) const;
  [[nodiscard]] fixed_sg2_prefill_sep21::Counters fixedSG2PrefillCounters() const noexcept;
private:
  struct Impl;''')
    if relative == 'runtime/flash/FlashInt8ExpertStore.mm':
        methods = cloned_methods(text)
        text = replace(text, '  uint64_t allocated = 0;',
            '''  uint64_t allocated = 0;
  mutable std::atomic<uint64_t> fixedSG2GateCalls{0},fixedSG2GateRows{0},fixedSG2DownCalls{0},fixedSG2DownRows{0};''')
        text = replace(text, '} // namespace splash::flash', methods + '''
fixed_sg2_prefill_sep21::Counters FlashInt8ExpertStore::fixedSG2PrefillCounters() const noexcept {
  return {fixed_sg2_prefill_sep21::requested(),impl_->fixedSG2GateCalls.load(std::memory_order_relaxed),
      impl_->fixedSG2GateRows.load(std::memory_order_relaxed),impl_->fixedSG2DownCalls.load(std::memory_order_relaxed),
      impl_->fixedSG2DownRows.load(std::memory_order_relaxed)};
}
std::string combinedSG2TailPipelineForCPU(bool gate) { return fixed_sg2_prefill_sep21::producerName(gate); }
std::string combinedSG2TailFallbackPipelineForCPU(const char *phase,uint32_t tileRows) {
  return pipeline(phase,static_cast<FlashMoEBlockedTile>(tileRows));
}
bool combinedSG2TailEligibleForCPU(uint32_t rows,uint32_t tileRows,bool verification) {
  return fixed_sg2_prefill_sep21::eligible(rows,static_cast<FlashMoEBlockedTile>(tileRows),verification);
}
} // namespace splash::flash''')
    if relative == 'runtime/flash/FlashForward.cpp':
        text = replace(text, '      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +',
            '''      std::string(pointwise_sep21::marker(pointwise_sep21::requested())) +
      std::string(fixed_sg2_prefill_sep21::marker()) +
      std::string(adaptive_expert_tail_sg2k128_sep21::markerFor(
          fixed_sg2_prefill_sep21::selection(),adaptive_expert_tail_sg2k128_sep21::requested())) +''')
        text = replace(text, '''        impl_->int8ExpertStore->addGateUp(graph, layer, impl_->blockedScratch, diag, rows, tile);
        impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);''',
            '''        if (fixed_sg2_prefill_sep21::eligible(rows,tile,verification)) {
          impl_->int8ExpertStore->addFixedSG2PrefillGateUp(graph,layer,impl_->blockedScratch,diag,rows,tile);
          impl_->int8ExpertStore->addFixedSG2PrefillDownScatter(graph,layer,impl_->blockedScratch,diag,rows,tile);
        } else {
          impl_->int8ExpertStore->addGateUp(graph, layer, impl_->blockedScratch, diag, rows, tile);
          impl_->int8ExpertStore->addDownScatter(graph, layer, impl_->blockedScratch, diag, rows, tile);
        }''')
    if relative == 'runtime/flash/FlashWorker.mm':
        text = replace(text, '      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.',
            '''      (void)adaptive_expert_tail_sg2k128_sep21::requested();
      (void)fixed_sg2_prefill_sep21::selection(); // Freeze before paths/metadata/backend, no ALLROWS dependency.
      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.''')
        text = replace(text, '  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};',
            '''  const auto persistedExpertGraphs = persistedExperts ? persistedExperts->graphCounters() : FlashInt8ExpertStoreGraphCounters{};
  const auto fixedSG2Graphs = persistedExperts ? persistedExperts->fixedSG2PrefillCounters() : fixed_sg2_prefill_sep21::Counters{};''')
        text = replace(text, '      << R"(,"gdn_verification_storage":{"lazy_enabled":)"',
            '''      << R"(,"fixed_sg2_prefill_route_counters":{"scope":"hybrid I8 main nonverify R2048/M32 only; graph construction, not GPU completion","selected_variant":)"
      <<fixed_sg2_prefill_sep21::selection()<< R"(,"tail_enabled":)"
      <<(adaptive_expert_tail_sg2k128_sep21::requested() ? "true" : "false")
      << R"(,"enabled":)"<<(fixedSG2Graphs.enabled ? "true" : "false")
      << R"(,"additional_allocation_bytes":0,"tail_numerical_derivative_changed":false,"gate_graph_calls":)"<<fixedSG2Graphs.gateCalls
      << R"(,"gate_graph_rows":)"<<fixedSG2Graphs.gateRows
      << R"(,"down_graph_calls":)"<<fixedSG2Graphs.downCalls
      << R"(,"down_graph_rows":)"<<fixedSG2Graphs.downRows<<'}'
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
    if relative == 'dev/benchmarks/prefill4k_attribution.mm':
        text = replace(text, '      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.',
            '''      (void)adaptive_expert_tail_sg2k128_sep21::requested();
      (void)fixed_sg2_prefill_sep21::selection();
      (void)pointwise_sep21::requested(); // Freeze exact pointwise policy before backend creation.''')
        text = replace(text, '    if (name.starts_with("flash_moe") || name.starts_with("flash_expert") || name.starts_with("flash_int8_expert")) return "moe";',
            '    if (name.starts_with("flash_moe") || name.starts_with("flash_expert") || name.starts_with("flash_int8_expert") || name.starts_with("prefill_moe_sep21_memory_fixed_") || name.starts_with("adaptive_expert_tail_sg2k128_sep21_")) return "moe";')
    return text


def transform(relative, text):
    text = tail_transform(relative, text)
    if relative != 'runtime/flash/FlashWorker.mm':
        return fma_module().transform(relative, text)
    # Hybrid target identity is an intentional static phase fingerprint, not an
    # all-row Store derivative. Bind FMA to THAT existing hybrid identity.
    text = '#include "dev/benchmarks/gdn_chunk_sep21/worker_bridge.hpp"\n' + text
    text = replace(text, '      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.',
        '''      (void)gdn_prefill_fma_sep21::requested();
      (void)pointwise_sep21::requested(); // Freeze/validate before paths, metadata, backend or model.''')
    before = '      << R"(,"target_numerical_derivative_sha256":)" << json::quote("' + BASE_IDENTITY + '")'
    after = '''      << R"(,"target_numerical_derivative_sha256":)" << json::quote(
          gdn_prefill_fma_sep21::numericalIdentity("''' + BASE_IDENTITY + '''",gdn_prefill_fma_sep21::requested()))
      << R"(,"target_base_numerical_derivative_sha256":)" << json::quote("''' + BASE_IDENTITY + '''")
      << R"(,"gdn_prefill_fma_enabled":)" << (gdn_prefill_fma_sep21::requested() ? "true" : "false")
      << R"(,"gdn_prefill_fma_numerical_policy":)" << (gdn_prefill_fma_sep21::requested()
          ? json::quote(std::string(gdn_prefill_fma_sep21::kPolicy)) : "null")
      << R"(,"gdn_prefill_fma_kernel_sha256":)" << (gdn_prefill_fma_sep21::requested()
          ? json::quote(std::string(gdn_prefill_fma_sep21::kKernelSourceSHA256)) : "null")
      << R"(,"gdn_prefill_fma_scope":)" << json::quote("staged prefill rows64..2048, lanes1..32; Q4 decode/verify/replay unchanged")'''
    return replace(text, before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / 'build/hybrid-q4-i8-expert-residency-sep21-v4')
    parser.add_argument('--certified', type=Path, default=ROOT / 'build/adaptive-expert-tail-sg2k128-fma-sep21-worker-v1')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/hybrid-sg2-tail-fma-sep21-worker-v1')
    args = parser.parse_args()
    base, certified, output = (p.resolve() for p in (args.base, args.certified, args.output))
    if output in (base, certified) or ROOT / 'build' not in output.parents:
        raise ValueError('Hybrid composition requires a distinct private build')
    parent_path, certificate_path = base / 'overlay-manifest.json', certified / 'overlay-manifest.json'
    parent, certificate = json.loads(parent_path.read_text()), json.loads(certificate_path.read_text())
    if parent['target_numerical_derivative_sha256'] != BASE_IDENTITY or parent['expert_residency_metadata_census']['composite_owner_count'] != 748:
        raise ValueError('Expected source-sealed resident hybrid v4 and unchanged phase identity')
    manifest = copy.deepcopy(parent)
    manifest.update({'route': 'private-resident-hybrid-q4decode-i8prefill-sg2k128-m16tail-optional-gdn-fma-sep21-v1',
        'hybrid_combined_composed': True, 'hybrid_combined_base_build': str(base),
        'hybrid_combined_base_manifest_sha256': sha(parent_path.read_bytes()),
        'hybrid_combined_transform_sha256': sha(Path(__file__).read_bytes()),
        'hybrid_combined_certificate_build': str(certified), 'hybrid_combined_certificate_manifest_sha256': sha(certificate_path.read_bytes()),
        'hybrid_combined_controls': {'sg2': 'SPLASH_FLASH_PREFILL_MOE_SEP21_VARIANT=0|7',
                                    'tail': 'SPLASH_FLASH_ADAPTIVE_EXPERT_TAIL_SG2K128_SEP21=0|1',
                                    'fma': certificate['combined_fma_environment']},
        'hybrid_combined_defaults': {'sg2': 0, 'tail': 0, 'fma': 0},
        'hybrid_combined_flag0_original_v4_graphs_and_identity': True,
        'hybrid_combined_allrows_flags_required': False,
        'hybrid_combined_scope': 'I8 main nonverify R2048/M32 only; Q4<256 and all singleton verification unchanged; original M16/M64/native I8 rows retained',
        'hybrid_combined_tail_certificate': certificate['combined_tail_exact_certificate'],
        'hybrid_combined_tail_source_sha256': certificate['combined_tail_source_sha256'],
        'hybrid_combined_fma_certificate_separate': True,
        'hybrid_combined_fma_source_sha256': certificate['combined_fma_source_sha256'],
        'hybrid_combined_fma_new_identity': 'SHA256(existing hybrid phase fingerprint + newline + FMA policy + newline + FMA kernel SHA)',
        'hybrid_combined_added_backing_and_workspace_bytes': 0,
        'hybrid_combined_model_qualification': 'pending root model benchmark/suite; no promotion',
        'model_payload_bytes_read': 0, 'gpu_execution': False, 'files': [], 'hybrid_combined_link_inputs': []})
    changed = []
    for record in parent['files']:
        relative = record['path']
        original = (base / 'source' / relative).read_bytes()
        if sha(original) != record['overlay_sha256']:
            raise ValueError(f'Hybrid v4 source seal drift: {relative}')
        data = transform(relative, original.decode()).encode()
        write(output / 'source' / relative, data)
        if data != original:
            changed.append(relative)
        manifest['files'].append({**record, 'hybrid_combined_changed': data != original,
                                  'hybrid_combined_base_sha256': sha(original), 'overlay_sha256': sha(data)})
    if set(changed) != EXPECTED_CHANGED:
        raise ValueError(f'Unexpected hybrid composition changes: {changed}')
    extras = [PRIVATE / 'worker_bridge.hpp', PRIVATE / 'hybrid_policy_cpu.cpp', PRIVATE / 'adaptive.metal',
              SG2_POLICY, FMA_PRIVATE / 'worker_bridge.hpp', FMA_PRIVATE / 'worker_policy_cpu.cpp', FMA_PRIVATE / 'scalar_fma.metal']
    certified_records = {r['path']: r for r in certificate['files']}
    for relative in extras:
        if relative.name == 'hybrid_policy_cpu.cpp':
            data = (ROOT / relative).read_bytes()
        else:
            data = (certified / 'source' / relative).read_bytes()
            if sha(data) != certified_records[relative.as_posix()]['overlay_sha256']:
                raise ValueError(f'Qualified composed source drift: {relative}')
        write(output / 'source' / relative, data)
        manifest['files'].append({'path': relative.as_posix(), 'new_hybrid_combined_file': True,
                                  'certified_source': relative.name != 'hybrid_policy_cpu.cpp', 'overlay_sha256': sha(data)})
    inputs = {'REUSED': [], 'CORE': [], 'AIRS': []}
    def freeze(path, relative, category, expected=None):
        data = path.read_bytes()
        if expected and sha(data) != expected:
            raise ValueError(f'Effective hybrid link input drift: {path}')
        write(output / relative, data)
        inputs[category].append(relative.as_posix())
        manifest['hybrid_combined_link_inputs'].append({'source_path': str(path), 'private_path': relative.as_posix(),
                                                      'category': category, 'sha256': sha(data)})
    for record in parent['frozen_link_inputs']:
        stem = Path(record['private_path']).stem
        if record['category'] == 'REUSED' and (stem in OWN_NAMES or stem == 'FlashWeights'):
            continue
        freeze(base / record['private_path'], Path('reused/hybrid-v4') / record['private_path'], record['category'], record['sha256'])
    for path in sorted((base / 'host').glob('*.o')):
        if path.stem not in OWN_NAMES:
            freeze(path, Path('reused/hybrid-v4-host') / path.name, 'REUSED')
    # v3/v4 reused the whole metallib. Restore omitted top-level v2 AIRs for
    # a complete relink, matching each sealed v4 source to the v2 source.
    v2 = Path(parent['root_built_v2_parent'])
    for name, relative in (('pointwise', Path('dev/benchmarks/moe_pointwise_sep21/candidate.metal')),
                           ('bulk-attention', Path('dev/benchmarks/prefill4k_attention/bulk_attention_sg8.metal')),
                           ('dense-prefill', Path('runtime/metal/kernels/shared/flash_dense_cache_prefill.metal'))):
        if (base / 'source' / relative).read_bytes() != (v2 / 'source' / relative).read_bytes():
            raise ValueError(f'Restored v2 AIR source differs from sealed v4: {relative}')
        freeze(v2 / f'{name}.air', Path('reused/restored-v2-metal') / f'{name}.air', 'AIRS')
    for record in certificate['combined_link_inputs']:
        if Path(record['private_path']).name in ('fixed-sg2.air', 'gdn-prefill-fma.air'):
            freeze(certified / record['private_path'], Path('reused/qualified-private') / Path(record['private_path']).name,
                   'AIRS', record['sha256'])
    make = '\n'.join(f'{key} := ' + ' '.join('$(BUILD)/' + p for p in paths) for key, paths in inputs.items()) + '\n'
    write(output / 'link-inputs.mk', make.encode())
    manifest['hybrid_combined_link_make_sha256'] = sha(make.encode())
    manifest['hybrid_combined_changed_files'] = changed
    for path in sorted((certified / 'qualification').glob('*')):
        write(output / 'qualification' / path.name, path.read_bytes())
    write(output / 'overlay-manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    write(output / 'base-build.txt', (str(base) + '\n').encode())
    for name in ('splash-flash.config', 'splash.metallib.config'):
        if (base / name).exists():
            write(output / name, (base / name).read_bytes().rstrip(b'\n') + b'-hybrid-sg2k128-exact-m16tail-optional-qualified-fma-sep21-v1\n')
    print(json.dumps({'prepared': str(output), 'frozen_sources': len(manifest['files']),
                      'changed_sources': changed, 'frozen_link_inputs': len(manifest['hybrid_combined_link_inputs']),
                      'gpu_work': False, 'payload_bytes_read': 0, 'added_backing_and_workspace_bytes': 0,
                      'original_q4_strong_aliases_and_residency_preserved': True}))


if __name__ == '__main__':
    main()
