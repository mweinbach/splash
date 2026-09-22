"""Supplemental strict service audit over an immutable composition adapter.

No global monkeypatch, source/version mutation, device/model/tokenizer import,
or original22 body/budget/grader change. Always invoke the frozen ORIGINAL22
common status/coverage/task gates too; this layer audits composition routes.
"""
import importlib.util
from pathlib import Path

def load_inner(binding):
    path=Path(binding['inner_adapter_path']).resolve()
    from hashlib import sha256
    if sha256(path.read_bytes()).hexdigest()!=binding['inner_adapter_sha256']:
        raise ValueError('immutable inner composition adapter changed')
    spec=importlib.util.spec_from_file_location('_sealed_b4_composition_inner',path)
    inner=importlib.util.module_from_spec(spec);spec.loader.exec_module(inner)
    return inner

def status_errors(status,binding,mode,bqsa_enabled=True,integer_enabled=True):
    inner=load_inner(binding)
    errors=inner.status_errors(status,binding,mode,bqsa_enabled,integer_enabled)
    if mode=='mtp3':
        fields={'capabilities.mtp':True,'capabilities.batch_mtp':True,
                'capabilities.batch_mtp_prefill':True,'mtp.enabled':True,
                'mtp.singleton_maximum_draft_tokens':3,
                'mtp.teacher_cache_only_requested':True,
                'mtp.batch_teacher_cache_only_requested':True,
                'batch_prefill.enabled':True}
        for path,wanted in fields.items():
            actual=inner.get(status,path)
            if type(actual)is not type(wanted)or actual!=wanted:
                errors.append('actual MTP3 composition prerequisite differs:'+path)
    if binding.get('frozen_original22_common_status_and_coverage_required')is not True:
        errors.append('frozen original22 common status/coverage gates must also run')
    return errors

def coverage(before,after,case,width,mode,binding,bqsa_enabled=True,integer_enabled=True):
    inner=load_inner(binding)
    record,errors=inner.coverage(before,after,case,width,mode,binding,bqsa_enabled,integer_enabled)
    for side,status in (('before',before),('after',after)):
        errors += [side+': '+e for e in status_errors(status,binding,mode,bqsa_enabled,integer_enabled)]
    if width==2:
        for role in ('plan','gate','down'):
            for suffix in ('graph_calls','graph_rows'):
                path=f'compact_native_batch_verify.r16.{role}_{suffix}'
                value=inner.counter_delta(before,after,path,errors)
                if value is not None and value!=0:
                    errors.append('physicalR16 cannot arise from a width2-only cohort:'+path)
    # Width4 may legitimately encode R8 after two peers naturally finish.
    record['frozen_common_quality_gate_required']=True
    record['width2_physicalR16_delta_must_be_zero']=width==2
    record['no_original22_task_or_mode_qualification_inherited']=True
    return record,errors
