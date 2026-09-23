#!/usr/bin/env python3
"""CPU source/dependency closure witness; never opens a model or a GPU device."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, re

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path('dev/benchmarks/expert_r4_compact_verify_worker_sep22')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build/compact-native-r4-verify-teacher-sep22-worker-v1b')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    build = a.build.resolve()
    m = json.loads((build / 'overlay-manifest.json').read_text())
    base = Path(m['base'])
    parent = json.loads((base / 'overlay-manifest.json').read_text())
    checks = []

    def require(value, name):
        checks.append({'name': name, 'pass': bool(value)})
        if not value:
            raise ValueError('CPU source witness failed: ' + name)

    require(sha((base / 'overlay-manifest.json').read_bytes()) == m['parent_manifest_sha256'], 'sealed parent manifest unchanged')
    require(sha((build / 'link-inputs.mk').read_bytes()) == m['link_make_sha256'], 'sealed link make unchanged')
    for record in m['files']:
        require(sha((build / 'source' / record['path']).read_bytes()) == record['sha256'], 'sealed source ' + record['path'])
    for record in m['frozen_inputs']:
        require(sha((build / record['path']).read_bytes()) == record['sha256'], 'sealed binary link input ' + record['path'])
    spec = importlib.util.spec_from_file_location('compact_worker_prepare', build / 'source' / PRIVATE / 'worker_prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    changed = []
    for record in parent['files']:
        rel = record['path']
        original = (base / 'source' / rel).read_bytes()
        require(sha(original) == (record.get('sha256') or record.get('overlay_sha256')), 'parent source ' + rel)
        actual = (build / 'source' / rel).read_bytes()
        require(actual == module.transform(rel, original.decode()).encode(), 'only registered transformation ' + rel)
        if actual != original:
            changed.append(rel)
    require(sorted(changed) == m['changed_paths'] == sorted(module.CHANGED), 'exact four changed host sources')
    require(sha(json.dumps(m['identity_parts'], sort_keys=True, separators=(',', ':')).encode()) == m['source_identity_sha256'], 'source identity reconstruction')
    identity = (build / 'source' / PRIVATE / 'source_identity.hpp').read_text()
    require('kSourceIdentitySha256[]="' + m['source_identity_sha256'] + '"' in identity, 'compiled source identity matches manifest')
    q = json.loads((ROOT / 'build/expert-r4-compact-native-sep22-component-v2/source-seal.json').read_text())
    plan = (build / 'source' / PRIVATE / 'plan.metal').read_bytes()
    qualified_plan = next(record for record in q['sources'] if record['path'] == 'dev/benchmarks/expert_r4_compact_native_parallel_sep22/plan.metal')
    require(sha(plan) == qualified_plan['sha256'] == m['identity_parts']['qualified_plan'], 'planner literal qualified parallel v2')
    require(q['source_identity_sha256'] == m['qualified_parallel_source_identity'], 'qualified parallel v2 identity bound')
    source = build / 'source'
    forward = (source / 'runtime/flash/FlashForward.cpp').read_text()
    store = (source / 'runtime/flash/FlashInt8ExpertStore.mm').read_text()
    worker = (source / 'runtime/flash/FlashWorker.mm').read_text()
    require('const bool compactR4Verify = verification && impl_->allRowsInt8Target && impl_->int8ExpertStore &&' in forward
            and 'compact_native_r4_verify_sep22::eligible(rows,verification)' in forward, 'selector requires singleton verification and Full512 Store')
    require('(compactR4Verify || (blocked && !gatheredMPP)) ? impl_->blockedScratch.scatteredDown' in forward, 'compact combine reads native canonical scatter')
    require('addGateUp(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);' in store
            and 'addDownScatter(graph,index,s,diagnostics,rows,FlashMoEBlockedTile::M16N64,selections);' in store, 'original literal M16 consumers called')
    require('FlashMoEBucketParams{4,10,2560,512,40,16,514,0},{1,1,1},{256,1,1}' in store
            and 'FlashMoEBucketParams{4,10,2560,512,40,0,0,0},{103,1,1},{256,1,1}' in store, 'qualified fixed ABI and original pack dispatch')
    require('additional_gpu_allocation_bytes":0' in worker and 'dispatches_per_layer":6' in worker
            and 'base_gather_dispatches_per_layer":2' in worker
            and 'full_model_quality_qualified":false' in worker, 'status honestly separates graph calls and pending model quality')
    require(worker.index('compact_native_r4_verify_sep22::validateDependencies(') < worker.index('(void)pointwise_sep21::requested();'), 'qualified dependencies checked before model construction')
    require('if (compactR4Verify)\n      derivative +=' in store and 'singleton_r4_verify_execution_policy=' in store, 'default off retains inherited derivative identity')
    for rel in ('runtime/flash/FlashBatchForward.cpp', 'runtime/flash/FlashBatchVerify.cpp', 'runtime/flash/FlashMTP.cpp',
                'runtime/flash/FlashMTPDepth.cpp', 'runtime/flash/FlashInt8Head.cpp', 'runtime/flash/FlashBF16Q8Head.cpp',
                'runtime/flash/FlashBatchPrefill.cpp', 'dev/benchmarks/mtp_teacher_bulk_sep21/bulk.cpp'):
        require((source / rel).read_bytes() == (base / 'source' / rel).read_bytes(), 'batch/head/teacher/state source unchanged ' + rel)
    link = (build / 'link-inputs.mk').read_text()
    names = next(line.split(' := ', 1)[1].split() for line in link.splitlines() if line.startswith('REBUILD_NAMES := '))
    require(len(names) == len(set(names)) == len(parent['rebuild']) + 1 and 'teacher_bulk' in names, 'all non-core host consumers rebuilt once')
    dependencies = []
    objects = []
    bridge_consumers = []
    for name in names:
        obj = build / 'host' / (name + '.o')
        dep = obj.with_suffix('.d')
        require(obj.is_file() and dep.is_file(), 'rebuilt object and dependency file ' + name)
        # Clang's first make rule supplies dependencies; trailing phony rules
        # generated by -MP are deliberately ignored.
        rule = dep.read_text().replace('\\\n', '').split('\n', 1)[0]
        require(': ' in rule, 'clang dependency rule ' + name)
        paths = [Path(value).resolve() for value in rule.split(': ', 1)[1].split()]
        private_paths = [value for value in paths if ROOT in value.parents]
        require(all(source in value.parents for value in private_paths), 'no live workspace dependency ' + name)
        if source / PRIVATE / 'bridge.hpp' in paths:
            bridge_consumers.append(name)
        dependencies.extend(str(value.relative_to(source)) for value in private_paths)
        objects.append({'path': str(obj.relative_to(build)), 'sha256': sha(obj.read_bytes()), 'bytes': obj.stat().st_size})
    require({'FlashForward', 'FlashWorker', '002-FlashInt8ExpertStore'}.issubset(bridge_consumers), 'changed public-header consumers compile new policy')
    artifacts = []
    for name in ('splash-flash', 'splash.metallib', 'compact-plan.air', 'policy-cpu'):
        path = build / name
        require(path.is_file() and path.stat().st_size > 0, 'complete CPU build artifact ' + name)
        artifacts.append({'path': name, 'sha256': sha(path.read_bytes()), 'bytes': path.stat().st_size})
    report = {'schema': 'compact-native-R4-verify-private-worker-CPU-closure-v1', 'build': str(build),
              'source_identity_sha256': m['source_identity_sha256'], 'checks': checks, 'pass': True,
              'host_tus_rebuilt': len(names), 'new_bridge_consumers': bridge_consumers,
              'private_compiler_dependencies': sorted(set(dependencies)), 'compiled_objects': objects,
              'artifacts': artifacts, 'additional_gpu_allocation_bytes': 0, 'gpu_work': False,
              'model_payload_reads': False, 'capture_payload_reads': False, 'full_model_quality_qualified': False}
    output = a.output or build / 'cpu-source-witness.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'checks': len(checks), 'host_tus_rebuilt': len(names), 'pass': True, 'gpu_work': False}))


if __name__ == '__main__':
    main()
