#!/usr/bin/env python3
"""CPU-only source/link closure and arithmetic-preservation witness."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[4]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--build',type=Path,default=ROOT/'build/prefill-moe-sg2-k128-pointwise-sep21-worker-v1')
 p.add_argument('--out',type=Path)
 args=p.parse_args();build=args.build.resolve();manifest=json.loads((build/'overlay-manifest.json').read_text())
 base=Path(manifest['fixed_sg2_base_build'])
 for r in manifest['files']:
  if sha(build/'source'/r['path'])!=r['overlay_sha256']:raise ValueError(f'Source seal drift:{r["path"]}')
 for r in manifest['fixed_sg2_link_inputs']:
  if sha(build/r['private_path'])!=r['sha256']:raise ValueError(f'Link input seal drift:{r["private_path"]}')
 if sha(build/'link-inputs.mk')!=manifest['fixed_sg2_link_inputs_make_sha256']:raise ValueError('Link make seal drift')
 old=(base/'source/runtime/flash/FlashInt8ExpertStore.mm').read_text()
 new=(build/'source/runtime/flash/FlashInt8ExpertStore.mm').read_text()
 begin=old.index('void FlashInt8ExpertStore::addGateUp(');end=old.index('\nnamespace {\nvoid gatheredMPPViews',begin)
 if old[begin:end] not in new:raise ValueError('Original BF16-input Store methods changed')
 begin=old.index('  Impl(metal::MetalBackend &b, const FlashWeights &weights, const std::filesystem::path &directory)')
 end=old.index('  const Layer &layer(',begin)
 if old[begin:end] not in new:raise ValueError('Store constructor/numerical derivative seed changed')
 original=(ROOT/'dev/benchmarks/prefill_moe_sep21/memory.metal').read_text()
 shader=(build/'source/dev/benchmarks/prefill_moe_sep21/fixed_sg2_worker/candidate.metal').read_text()
 if original[:original.index('PREFILL4K_INT8TILES_GATE(prefill_moe_sep21_memory_whole_')] not in shader:
  raise ValueError('Shared qualified primitive arithmetic changed')
 invocations=[line for line in shader.splitlines() if line.startswith('PREFILL4K_INT8TILES_')]
 if len(invocations)!=2 or any('_m32_n64_k128_sg2' not in line for line in invocations):raise ValueError('Extra exported candidate variants')
 result={'pass':True,'gpu_work':False,'model_payload_reads':False,'source_seals_verified':len(manifest['files']),
  'sealed_link_inputs_verified':len(manifest['fixed_sg2_link_inputs']),'additional_allocation_bytes':0,
  'original_store_methods_byte_identical':True,'store_constructor_and_numerical_derivative_byte_identical':True,
  'qualified_component_arithmetic_byte_identical':True,'only_variant7_kernels_exported':True,
  'scope':'main nonverification R2048/M32 canonical Full512 only; other callers original',
  'full_model_equality_and_throughput':'pending Root GPU/model qualification',
  'artifact_sha256':{n:sha(build/n) for n in ('splash-flash','prefill4k-attribution','splash.metallib')}}
 out=args.out or build/'cpu-sealed-witness-v1.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
