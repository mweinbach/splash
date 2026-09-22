#!/usr/bin/env python3
"""Root-only bounded CPU census of selected original BF16 affine parameters.

No Metal/MLX import or GPU action. Each requested tensor is capped at25MiB.
The default reads only layer0 gate scales; offsets/types come from the sealed
native manifest. Reports signed/zero/nonfinite counts and first coordinates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import numpy as np

QUALIFIED_MANIFEST_SHA='0cf9f8641fc97eae6ae4bf80d1ac5615a7674a1466006841dd72b6a5332a9402'
MAX_TENSOR_BYTES=26214400


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('install/local-models/Flash-Next-oQ4e-mtp-v1'))
    parser.add_argument('--layer',type=int,default=0)
    parser.add_argument('--roles',default='gate_proj')
    parser.add_argument('--fields',default='scales')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Choose a fresh census report')
    if not 0<=args.layer<48:raise ValueError('Layer outside0..47')
    roles=args.roles.split(',');fields=args.fields.split(',')
    if not set(roles)<={'gate_proj','up_proj','down_proj'} or not set(fields)<={'scales','biases'}:raise ValueError('Unsupported role/field')
    raw=(args.model/'manifest.json').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=QUALIFIED_MANIFEST_SHA:raise ValueError('Qualified original metadata seal differs')
    manifest=json.loads(raw);shards={s['path']:s for s in manifest['shards']};results=[];total=0
    for role in roles:
      for field in fields:
        name=f'language_model.model.layers.{args.layer}.mlp.switch_mlp.{role}.{field}'
        record=manifest['tensors'][name];shape=record['shape'];length=record['length'];offset=record['offset'];path=args.model/record['shard']
        if record['dtype']!='BF16' or len(shape)!=3 or shape[0]!=512 or length!=2*math.prod(shape) or not 0<length<=MAX_TENSOR_BYTES:raise ValueError('Bounded parameter extent/type differs')
        fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
        try:
          before=os.fstat(fd)
          if before.st_size!=shards[record['shard']]['bytes'] or offset+length>before.st_size:raise ValueError('Source file extent differs')
          payload=os.pread(fd,length,offset)
          if len(payload)!=length:raise ValueError('Short parameter read')
          after=os.fstat(fd)
          if (before.st_dev,before.st_ino,before.st_size,before.st_mode,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mode,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Source changed during census')
        finally:os.close(fd)
        total+=length;words=np.frombuffer(payload,dtype='<u2');nf=(words&0x7f80)==0x7f80;zero=(words&0x7fff)==0;negative=(words&0x8000)!=0
        finite_negative=negative&~zero&~nf;finite_positive=~negative&~zero&~nf
        coordinate=lambda index:{'flat_index':index,'expert':index//(shape[1]*shape[2]),'output_column':(index//shape[2])%shape[1],'group64':index%shape[2],'file_byte_offset':offset+index*2,'bf16_bits':f'{int(words[index]):04x}','f32_value':value if math.isfinite(value:=struct.unpack('<f',struct.pack('<I',int(words[index])<<16))[0]) else None}
        first={}
        for kind,mask in [('finite_negative',finite_negative),('zero',zero),('negative_zero',zero&negative),('nonfinite',nf)]:
          indexes=np.flatnonzero(mask);first[kind]=coordinate(int(indexes[0])) if len(indexes) else None
        results.append({'tensor':name,'dtype':'BF16','shape':shape,'selected_bytes_read':length,'source_mode_octal':oct(before.st_mode&0o777),'source_range_sha256_observed':hashlib.sha256(payload).hexdigest(),'values':len(words),'finite_positive':int(np.count_nonzero(finite_positive)),'finite_negative':int(np.count_nonzero(finite_negative)),'positive_zero':int(np.count_nonzero(zero&~negative)),'negative_zero':int(np.count_nonzero(zero&negative)),'nonfinite':int(np.count_nonzero(nf)),'first_coordinates':first})
    report={'schema':'splash-original-q4-affine-parameter-census-v1','gpu_work':False,'root_only_selected_payload_read':True,'model_files_modified':False,'qualified_manifest_sha256':digest,'total_selected_parameter_bytes_read':total,'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
