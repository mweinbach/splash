#!/usr/bin/env python3
"""CPU only: ONE old batch TU QSA-emitter override; all other arithmetic exact."""
import argparse,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();assert not out.exists();out.mkdir(parents=True)
 parent=ROOT/'build/batch-prefill-restored-teacher-clock-sep22-worker-v1';original=parent/'source/runtime/flash/FlashBatchPrefill.cpp';baseline=original.read_text();s='#include "bulk.hpp"\n#include "reference_qsa.hpp"\n'+baseline
 old='''          prefill4k::addBulkExactQSA(impl_->backend, graph, inputs, states[lane]->qsa[layer],
              impl_->qsa, impl_->qsaFast, *impl_->bulkQSA, 0, rows, true);'''
 new='''          batch_qsa_intended_reference::emit(impl_->backend,graph,inputs,states[lane]->qsa[layer],
              impl_->qsa,impl_->qsaFast,*impl_->bulkQSA,layer,lane,lanes,rows,states[lane]->length);'''
 assert s.count(old)==1;s=s.replace(old,new)
 # Exact old complete batch/state/cache inventory from the independently sealed
 # actual capture hook. Add original trunk's complete source-cache guards too.
 capture=(ROOT/'build/batch-qsa-actual-capture-sep22-source-v2/FlashBatchPrefill_capture.cpp').read_text();begin=capture.index('  if (batch_qsa_capture::activeSession) {');end=capture.index('  for (uint32_t token : tokens)',begin);pre=capture[begin:end]
 pre=pre.replace('batch_qsa_capture::activeSession','batch_qsa_intended_reference::active').replace('    batch_qsa_capture::validateWholeCohort(impl_->backend,lanes,rows,impl_->capacity,actualLengths,others);','    for (const auto &plane : batch_qsa_intended_reference::active->planes())\n      impl_->trunk.batchValidateExternalDestination(plane);\n    batch_qsa_intended_reference::beforeWhole(impl_->backend,lanes,rows,impl_->capacity,actualLengths,others,impl_->weights.sourceIdentity());')
 anchor='  for (uint32_t token : tokens)\n';assert s.count(anchor)==1;s=s.replace(anchor,pre+anchor)
 publish='  for (uint32_t lane = 0; lane < lanes; ++lane) lengths[lane] = states[lane]->length += rows;';completion='\n  batch_qsa_intended_reference::completed(impl_->backend.healthy());';assert s.count(publish)==1;s=s.replace(publish,publish+completion)
 inverse=s.removeprefix('#include "bulk.hpp"\n#include "reference_qsa.hpp"\n').replace(new,old).replace(pre,'').replace(completion,'');assert inverse==baseline
 (out/'FlashBatchPrefill_intended_reference.cpp').write_text(s);shutil.copy2(HERE/'reference_qsa.hpp',out/'reference_qsa.hpp')
 d={'schema':'test-only-oldbatch-existing-singleton-QSA-emitter-golden-source-v1','pass':True,'GPU_executed':False,'model_or_capture_payload_read':False,'parent':str(parent),'parent_seal_sha256':sha(parent/'compiled-cpu-seal.json'),'original_BatchPrefill_sha256':sha(original),'exact_original_body_after_only_intended_QSA_emitter_inverse':True,'all_projection_HC_GDN_MoE_PLE_head_logit_greedy_AR_ownership_source_unchanged':True,'public_header_or_layout_changes':False,'only_runtime_TU_modified':'FlashBatchPrefill.cpp','reference_arena_own_separate_bytes':509607936,'legacy_arena234356736_retained':True,'selection':'TEST_ONLY installed scope alwaysusesUNCHANGEDexistingSinglepackedV helper onactualfresh2048 real2/4 oldbatch projection/norm/cache source;defaultTLSnull oldSG8 unchanged','normal_service_or_Worker_changed':False,'numeric_risk':'116/2048layer3 oldSG8 everyROW comparisons FAIL;oldGLOBAL andF64 PASS;newarithmetic EXACTexistingSingle SAMEinputs;thissource isintendedarithmeticgolden andNOT oldSG8numeric certificate','source_files':[{'path':str(q),'sha256':sha(q)}for q in out.glob('*')if q.is_file()]}
 (out/'source-receipt.json').write_text(json.dumps(d,indent=2)+'\n');print(out)
if __name__=='__main__':main()
