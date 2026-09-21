"""CPU-only native window audit; no payload reads, mapping or Metal imports."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ALIGNMENT=16384
def rounded(value): return (value+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT

def audit(package):
    raw=(package/'manifest.json').read_bytes()
    manifest=json.loads(raw)
    assert hashlib.sha256(raw).hexdigest()=='0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402'
    assert manifest['source_identity_sha256']=='ca9b5afd950d122c0b1bce90735886e39d3c20fb1d085c3d5245fb566779033e'
    grouped=defaultdict(list)
    for name,tensor in manifest['tensors'].items(): grouped[tensor['shard']].append((tensor['offset'],name,tensor))
    bytes_by_category=Counter();counts=Counter();rows=[];norm_examples=[]
    for shard in manifest['shards']:
        tensors=sorted(grouped[shard['path']]);previous=0;total=0
        for offset,name,tensor in tensors:
            length=tensor['length'];native=rounded(length)
            assert offset%ALIGNMENT==0 and offset==previous and offset+native<=shard['bytes']
            previous=offset+native;total+=native
            category='vision' if name.startswith('vision_tower.') else 'ple' if '.ple.'in name else 'mtp' if name.startswith('mtp.') else 'text'
            bytes_by_category[category]+=native;counts[category]+=1
            if '.hc_norm.weight'in name and len(norm_examples)<4:
                norm_examples.append({'name':name,'logical_bytes':length,'per_tensor_native_bytes':native,'whole_shard_native_bytes':shard['bytes'],'wiring_amplification':shard['bytes']/native})
        assert previous==total==shard['bytes']
        rows.append({'path':shard['path'],'bytes':shard['bytes'],'native_tensor_count':len(tensors),'rounded_tensor_bytes':total,'extra_bytes':0})
    dense=json.loads((package.parent/'Flash-Next-operands-v1'/'manifest.json').read_bytes())
    prefixes={entry['projection']for entry in dense['entries']if entry['format']=='BF16'}
    dense_names={prefix+'.'+suffix for prefix in prefixes for suffix in('weight','scales','biases')}
    body_names={name for name in dense_names if not name.startswith('language_model.lm_head.')}
    select=lambda names:sum(rounded(manifest['tensors'][name]['length'])for name in names)
    logical=sum(tensor['length']for tensor in manifest['tensors'].values());padded=sum(s['bytes']for s in manifest['shards'])
    result={'schema':'splash-private-per-tensor-native-window-audit-v1','scope':'CPU metadata only; zero payload reads and GPU/Metal operations','source_identity_sha256':manifest['source_identity_sha256'],'aligned_manifest_sha256':hashlib.sha256(raw).hexdigest(),'loaded_model_layout_sha256':'edc4ffe67a2479bad999fd7cd26d05cc16549d52f79659577ad21ff1d8d310a0','source_mapping_count':len(rows),'old_native_base_count':len(rows),'new_native_base_count':len(manifest['tensors']),'logical_tensor_bytes':logical,'padded_native_bytes':padded,'padding_bytes':padded-logical,'extra_native_bytes':0,'native_window_overlaps':0,'native_window_gaps':0,'category_native_bytes':dict(bytes_by_category),'category_native_count':dict(counts),'potential_vision_lazy_map_saving_bytes':bytes_by_category['vision'],'original_dense_body_coefficient_native_bytes':select(body_names),'original_dense_including_vocabulary_coefficient_native_bytes':select(dense_names),'dense_bytes_elided_in_first_experiment':0,'vision_bytes_elided_in_first_experiment':0,'dense_elision_caveat':'Singleton/raw routes still use original coefficients; no lazy elision in this granularity-only experiment.','norm_examples':norm_examples,'shards':rows,'numerical_bytes_or_manifest_changed':False,'original_text_residency_mutually_excluded':True,'performance_gain':'unmeasured hypothesis; per-tensor resource wiring may reduce original-shard driver work.'}
    return result

if __name__=='__main__':
    result=audit(Path(sys.argv[1]));Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key]for key in('source_mapping_count','old_native_base_count','new_native_base_count','padded_native_bytes','padding_bytes','extra_native_bytes','potential_vision_lazy_map_saving_bytes','original_dense_body_coefficient_native_bytes')}))
