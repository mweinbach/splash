#!/usr/bin/env python3
"""Seal code, dependency headers and frozen link inputs without payload reads."""
import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,default=ROOT/'build/prefill-m64-low-sg-sep21')
    a=p.parse_args();build=a.build.resolve();directory=Path(__file__).parent
    host=ROOT/'build/prefill4k-wide-fullcache';base=ROOT/'build/flash-next'
    control=ROOT/'build/adaptive-expert-tail-sg2k128-sep21'
    files=set(directory.glob('*'));files={f for f in files if f.is_file()}
    files.update(build/f for f in ['oracle','oracle.mm','oracle.o','oracle.d','candidate.metal','candidate.air','splash.metallib','source-manifest.json'])
    deps=(build/'oracle.d').read_text().replace('\\\n',' ')
    for item in shlex.split(deps.split(':',1)[1]):
        path=Path(item);path=path if path.is_absolute() else ROOT/path
        if path.is_relative_to(ROOT):files.add(path)
    files.update(f for f in (host/'host').glob('*.o') if f.name!='FlashWorker.o')
    files.update(base/f for f in ['engine/metal/MetalBackend.o','engine/metal/DeviceCapabilities.o','engine/engine/Protocol.o','engine/engine/MemoryGovernor.o'])
    files.update(f for f in (base/'metal').glob('*/*.air') if f.name!='flash_int8_expert_store.air')
    files.update([host/'metal/flash_int8_expert_store.air',control/'adaptive.air',control/'adaptive.metal',control/'oracle.mm',control/'shader-manifest.json'])
    # Candidate/control MSL recursively consumes only these repo include roots.
    pending=[build/'candidate.metal',control/'adaptive.metal']
    while pending:
        path=pending.pop()
        for name in re.findall(r'^\s*#include\s+"([^"]+)"',path.read_text(),re.M):
            candidates=[path.parent/name,ROOT/'runtime'/name,ROOT/name]
            included=next((f for f in candidates if f.is_file()),None)
            if included is None:raise RuntimeError('Missing MSL include: '+name)
            if included not in files:files.add(included);pending.append(included)
    entries={str(f.relative_to(ROOT)):sha(f) for f in sorted(files)}
    evidence=dict(schema='private-native-m64-low-sg-input-seal-v1',gpu_executed=False,payload_bytes_read=0,
                  code_and_frozen_inputs=len(entries),sha256=entries,
                  host_object_scope='frozen prefill4k-wide-fullcache objects excluding Worker; four frozen Core objects',
                  shader_scope='candidate low-SIMD M64 + qualified SG2/K128 M16-tail control; inherited original AIRs',
                  native_m64_gpu_exactness='pending root execution')
    path=build/'input-seal.json';path.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(sealed=str(path),files=len(entries),gpu_executed=False,payload_bytes_read=0)))


if __name__=='__main__':main()
