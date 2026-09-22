#!/usr/bin/env python3
"""Prepare actual dense-I8 role metadata without reading any coefficient/input payload."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_ROLES=(
 'language_model.model.layers.0.linear_attn.in_proj_qkv',
 'language_model.model.layers.0.linear_attn.in_proj_z',
 'language_model.model.layers.0.linear_attn.out_proj',
 'language_model.model.layers.1.ple.value_proj',
 'language_model.model.layers.3.self_attn.q_proj',
 'language_model.model.layers.3.self_attn.k_proj',
 'language_model.model.layers.3.self_attn.indexer.index_qk_proj')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--store',type=Path,default=ROOT/'install/local-models/Flash-Next-operands-v1')
 p.add_argument('--source',type=Path,default=ROOT/'install/local-models/Flash-Next-oQ4e-mtp-v1')
 p.add_argument('--capture-manifest',type=Path,default=ROOT/'build/prefill4k-dense/actual-activations-v1/manifest.json')
 p.add_argument('--out',type=Path,default=ROOT/'build/dense-i8-decode-sep21/actual-role-fixtures.json')
 args=p.parse_args();store=args.store.resolve();source=args.source.resolve();capture=args.capture_manifest.resolve()
 operands=json.loads((store/'manifest.json').read_text());original=json.loads((source/'manifest.json').read_text())
 captures=json.loads(capture.read_text());byformat={(e['projection'],e['format']):e for e in operands['entries']}
 bycapture={c['projection']:c for c in captures['cases']}
 cases=[]
 for role in DEFAULT_ROLES:
  f32=byformat[(role,'F32')];bf16=byformat.get((role,'BF16'));c=bycapture[role];geometry=f32['source']
  n,k=f32['shape']
  if geometry['experts']!=1 or n!=c['output_size'] or k!=c['input_size'] or k%32 or n%64:
   raise ValueError('Actual dense fixture source geometry drift')
  raw={}
  for name in ('weight','scales','biases'):
   # The installed source names packed code matrices as .weight.
   tensor=original['tensors'][role+'.'+name]
   raw[name]={'file':str(source/tensor['shard']),'offset_bytes':tensor['offset'],
              'length_bytes':tensor['length'],'dtype':tensor['dtype'],'shape':tensor['shape']}
  entry={'projection':role,'input_size':k,'output_size':n,'captured_rows':c['rows'],
   'input_file':c['input_file'],'input_sha256':c['input_sha256'],
   'captured_expected_file':c['expected_file'],'captured_expected_sha256':c['expected_sha256'],
   'f32_weights_file':str(store/f32['file']),'f32_weights_logical_bytes':f32['logical_bytes'],
   'f32_weights_allocated_bytes':f32['allocated_bytes'],'f32_weights_sha256':f32['payload_sha256'],
   'f32_operand_math':f32['operand_math'],'source_bits':geometry['bits'],'source_group_size':geometry['group_size'],
   'source_weight_row_stride_bytes':geometry['weight_row_stride_bytes'],
   'source_weight_expert_stride_bytes':geometry['weight_expert_stride_bytes'],
   'source_parameter_row_stride_bytes':geometry['parameter_row_stride_bytes'],
   'source_parameter_expert_stride_bytes':geometry['parameter_expert_stride_bytes'],
   'raw_source_tensors':raw,'inputs_are_actual_prefill_capture_slices':True,'live_decode_activation_capture':False}
  if bf16:
   entry.update({'bf16_weights_file':str(store/bf16['file']),'bf16_weights_logical_bytes':bf16['logical_bytes'],
      'bf16_weights_allocated_bytes':bf16['allocated_bytes'],'bf16_weights_sha256':bf16['payload_sha256']})
  cases.append(entry)
 d={'schema':'splash-private-dense-i8-f32-row-fit-fixtures-sep21-v1','model_payload_reads':False,'gpu_work':False,
   'weight_requantization':True,'activation_quantization':False,'rows':[1,4,8,16],
   'input_scope':'first R actual BF16 rows of captured2048-token prefill; not live decode states',
   'excluded_initial_roles':['vocabulary','router','HC fused down/up','trained MTP'],
   'source_identity_sha256':operands['source_identity_sha256'],
   'weights_manifest_fingerprint':operands['weights_manifest_fingerprint'],
   'source_manifest_sha256':digest(source/'manifest.json'),'operand_manifest_sha256':digest(store/'manifest.json'),
   'capture_manifest_sha256':digest(capture),'prepare_source_sha256':digest(Path(__file__)),
   'cases':cases}
 args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(d,indent=2)+'\n')
 print(json.dumps({'prepared':str(args.out),'roles':len(cases),'rows':[1,4,8,16],'model_payload_reads':False,'gpu_work':False}))


if __name__=='__main__':main()
