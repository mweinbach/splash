#!/usr/bin/env python3
"""Freeze exact-current-kernel teacher bulk component, CPU/source/artifacts only."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil
ROOT=Path(__file__).resolve().parents[3]
PRIVATE=Path('dev/benchmarks/mtp_teacher_bulk_sep21')
def sha(data):return hashlib.sha256(data).hexdigest()
def write(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
def once(text,before,after):
    if text.count(before)!=1:raise ValueError('Frozen teacher source drift: '+before[:90])
    return text.replace(before,after,1)
def transform(relative,text):
    if relative=='runtime/flash/FlashMTP.hpp':
        text=once(text,'\nclass FlashMTPForward;\n','\nclass FlashMTPForward;\nclass FlashMTPTeacherBulkForward;\n')
        text=text.replace('  friend class FlashBatchMTPForward;','  friend class FlashBatchMTPForward;\n  friend class FlashMTPTeacherBulkForward;')
        if text.count('friend class FlashMTPTeacherBulkForward;')!=2:raise ValueError('Expected both owner friends')
        return once(text,'  [[nodiscard]] const FlashDenseCache *batchDenseCache() const;', '''  [[nodiscard]] const FlashDenseCache *batchDenseCache() const;
  [[nodiscard]] uint32_t teacherBulkMaximumRows() const;
  void teacherBulkAddPrefix(metal::CommandGraph &graph, FlashQSAFastInputs input,
      FlashQSAState &state, uint32_t begin, uint32_t rows);
  void teacherBulkAppendPrefix(metal::CommandGraph &graph,
      const metal::CommandGraph &prefix);''')
    if relative=='runtime/flash/FlashMTP.cpp':
        anchor='std::vector<metal::MetalBuffer> FlashMTPForward::cachedOperandsOnly() const {'
        methods='''uint32_t FlashMTPForward::teacherBulkMaximumRows() const {
  if (!impl_) throw std::logic_error("teacher bulk head is not initialized");
  return impl_->maximumRows;
}
void FlashMTPForward::teacherBulkAddPrefix(metal::CommandGraph &graph,
    FlashQSAFastInputs input, FlashQSAState &state,uint32_t begin,uint32_t rows) {
  if(!impl_ || rows!=128 || rows>impl_->maximumRows || begin>state.capacity || rows>state.capacity-begin)
    throw std::invalid_argument("teacher bulk cache prefix shape/capacity differs");
  input.output=impl_->bf(Scratch::AttentionOutput,rows,6144);
  addTeacherQSACache(graph,input,state,impl_->qsaWorkspace,impl_->qsaFastWorkspace,
      begin,rows,impl_->attentionPolicy);
}
void FlashMTPForward::teacherBulkAppendPrefix(metal::CommandGraph &graph,
    const metal::CommandGraph &prefix) {
  for(const auto &d:prefix.dispatches()) {
    const auto &name=d.pipelineName;
    const bool preparation=name=="flash_qsa_fast_prepare" || name.starts_with("flash_qsa_norm_rope_") ||
        name=="flash_qsa_append_aux" || name.starts_with("flash_qsa_pool_rope_");
    if(!preparation || d.bytes.size()!=1 || d.bytes[0].index!=d.buffers.size())
      throw std::logic_error("teacher bulk prefix ABI changed");
    std::vector<metal::MetalBuffer> buffers;
    for(uint32_t i=0;i<d.buffers.size();++i) {
      if(d.buffers[i].index!=i) throw std::logic_error("teacher bulk prefix binding order changed");
      buffers.push_back(d.buffers[i].buffer);
    }
    if(d.bytes[0].sizeBytes==sizeof(FlashQSAFastParams)) {
      FlashQSAFastParams params{};std::memcpy(&params,d.bytes[0].data,sizeof(params));
      graph.add(name,std::move(buffers),params,d.threadgroups,d.threadsPerThreadgroup);
    } else if(d.bytes[0].sizeBytes==sizeof(FlashQSAParams)) {
      FlashQSAParams params{};std::memcpy(&params,d.bytes[0].data,sizeof(params));
      graph.add(name,std::move(buffers),params,d.threadgroups,d.threadsPerThreadgroup);
    } else throw std::logic_error("teacher bulk cache prefix parameter ABI changed");
  }
}
'''
        return once(text,anchor,methods+anchor)
    return text
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,default=ROOT/'build/prefill-qsa-twopass-sep21-worker-v1')
    ap.add_argument('--output',type=Path,default=ROOT/'build/mtp-teacher-bulk-sep21-v3')
    args=ap.parse_args();base=args.base.resolve();out=args.output.resolve()
    if out.exists() or ROOT/'build' not in out.parents:raise ValueError('Choose fresh private output')
    parent=json.loads((base/'overlay-manifest.json').read_text())
    sealPath=base/'compiled-closure-seal.json';compiled=json.loads(sealPath.read_text())
    if not compiled.get('pass'):raise ValueError('Parent complete compiled seal required')
    if sha((base/'overlay-manifest.json').read_bytes())!=compiled['source_manifest_sha256']:
        raise ValueError('Parent compiled manifest authentication differs')
    for rel,digest in compiled['source_sha256'].items():
        if sha((base/'source'/rel).read_bytes())!=digest:raise ValueError('Parent compiled source authentication differs: '+rel)
    for rel,digest in compiled['compiled_and_evidence_sha256'].items():
        if sha((base/rel).read_bytes())!=digest:raise ValueError('Parent compiled artifact authentication differs: '+rel)
    hp=base/'machinery/parent_overlay.py'
    spec=importlib.util.spec_from_file_location('teacher_parent',hp);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    closure=helper.effective_closure(base,base/'machinery/worker.mk')
    seals,unsealed=helper.parent_input_seals(base,parent,closure)
    witness=helper.parent_artifact_witness(base)
    records={x['path']:x for x in parent['files']}
    deps=helper.frozen_dependency_closure(base,closure['objects'],records)
    files=[]
    for rel,r in records.items():
        data=(base/'source'/rel).read_bytes()
        if sha(data)!=r['overlay_sha256']:raise ValueError('Parent source seal differs: '+rel)
        changed=transform(rel,data.decode()).encode()
        write(out/'source'/rel,changed);files.append({'path':rel,'parent_sha256':sha(data),'sha256':sha(changed),'changed':changed!=data})
    for name in ('bulk.hpp','bulk.cpp','oracle.mm','policy_cpu.cpp','policy.hpp'):
        rel=PRIVATE/name;data=(ROOT/rel).read_bytes();write(out/'source'/rel,data)
        files.append({'path':rel.as_posix(),'sha256':sha(data),'new':True})
    inputs={'REUSED':[],'CORE':[]};frozen=[];rebuild=[]
    aliases={'Prefill4kQSABulk':'dev/benchmarks/prefill4k_attention/bulk.cpp',
             'Prefill4kQSACoalesced':'dev/benchmarks/prefill4k_attention/coalesced.cpp'}
    for obj in closure['objects']:
        # Many inherited objects intentionally have no adjacent.d. Recompile
        # every non-core host TU against the single modified class definition,
        # rather than guessing which header consumers are ABI-safe to reuse.
        if obj not in closure['core']:
            if obj.stem=='FlashWorker':continue # Component does not link Worker/main.
            src=aliases.get(obj.stem) or next((x for x in records if Path(x).stem==obj.stem and x.endswith(('.cpp','.mm'))),None)
            if not src:raise ValueError('Cannot resolve affected source '+str(obj))
            rebuild.append({'object':obj.stem,'source':src,'parent_object':str(obj)});continue
        if obj.stem=='FlashWorker':continue
        rel=Path('reused')/obj.relative_to(base);data=obj.read_bytes();write(out/rel,data)
        kind='CORE' if obj in closure['core'] else 'REUSED';inputs[kind].append(rel.as_posix())
        frozen.append({'path':rel.as_posix(),'sha256':sha(data),'source':str(obj)})
    lib=(base/'splash.metallib').read_bytes();write(out/'splash.metallib',lib)
    for name in ('prepare.py','component.mk','README.md'):write(out/'machinery'/name,(ROOT/PRIVATE/name).read_bytes())
    link='\n'.join(k+' := '+' '.join('$(BUILD)/'+p for p in v)for k,v in inputs.items())+'\n'
    link+='REBUILD_NAMES := '+' '.join(x['object']for x in rebuild)+'\n'
    for r in rebuild:link+='SRC_'+r['object']+' := $(BUILD)/source/'+r['source']+'\n'
    write(out/'link-inputs.mk',link.encode())
    man={'schema':'singleton-teacher-exact-bulk2048-component-v1','base':str(base),'parent_compiled_seal_sha256':sha(sealPath.read_bytes()),'parent_source_manifest_sha256':sha((base/'overlay-manifest.json').read_bytes()),'parent_helper_sha256':sha(hp.read_bytes()),'parent_input_seals':seals,'parent_unsealed_owned_inputs':unsealed,'parent_artifact_witness':witness,'parent_dependencies':deps,'rebuild':rebuild,'frozen_objects':frozen,'metallib_sha256':sha(lib),'files':files,'all_noncore_host_sources_recompiled':True,'additional_workspace_planned_bytes':310181888,'gpu_executed':False,'model_payload_bytes_read':0,'head_proposal_and_batch_paths_changed':False}
    write(out/'source-manifest.json',(json.dumps(man,indent=2)+'\n').encode())
    print(json.dumps({'output':str(out),'source_count':len(files),'rebuild':rebuild,'frozen_objects':len(frozen),'payload_bytes_read':0}))
if __name__=='__main__':main()
