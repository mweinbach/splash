#!/usr/bin/env python3
"""CPU-only private actual-QSA capture overlays; normal source untouched."""
import argparse
import hashlib
import json
import os
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder',type=Path)
parser.add_argument('--all-rows',action='store_true',help='Record the full 2048-row capture plan; runtime also needs PREFILL4K_ATTENTION_CAPTURE_ALL_ROWS=1')
args=parser.parse_args()
raw_all_rows=os.environ.get('PREFILL4K_ATTENTION_CAPTURE_ALL_ROWS','0')
if raw_all_rows not in ('','0','1'):
 parser.error('PREFILL4K_ATTENTION_CAPTURE_ALL_ROWS must be absent/0/1')
all_rows=args.all_rows or raw_all_rows=='1'
capture_rows=2048 if all_rows else 128
query_offset=0 if all_rows else 1920
capture_planned_bytes=3*(capture_rows*12288*2+capture_rows*6144*2+2048*512*2*2)
folder=args.folder
source=Path('runtime/flash/FlashForward.cpp').read_text()
start='      const std::string ikNorm = attention + ".indexer.k_layernorm.weight";'
end='      affine(graph, attention + ".o_proj", attentionOutput, branch);'
if source.count(start)!=1 or source.count(end)!=1:
 raise RuntimeError('authoritative QSA capture anchors drifted')
first=source.index(start);last=source.index(end)
if first>=last:raise RuntimeError('authoritative QSA capture anchor order drifted')
section=source[first:last]
window_end='        offset += count;'
if section.count(window_end)!=1:
 raise RuntimeError('authoritative QSA per-window capture anchor drifted')
section=section.replace(start,start+'\n      prefill4k_attention::CaptureScope privateAttentionCapture(impl_->backend,graph,layer,state.qsa[layer],q,begin,rows);')
section=section.replace(window_end,'        privateAttentionCapture.preparedWindow(impl_->qsaWorkspace.queries,offset,count);\n'+window_end)
modified='#include "capture.hpp"\n'+source[:first]+section+source[last:]
original=Path('dev/benchmarks/prefill4k_attribution.mm').read_text()
anchor='      auto reservation = governor.tryReserve(planned);'
if original.count(anchor)!=1: raise RuntimeError('capture reservation anchor drifted')
attribution=original.replace(anchor,'      planned += prefill4k_attention::capturePlannedBytes();\n'+anchor)
folder.mkdir(parents=True,exist_ok=True)
(folder/'FlashForward.cpp').write_text(modified)
(folder/'capture_attribution.mm').write_text(attribution)
(folder/'manifest.json').write_text(json.dumps({'schema':'splash-private-qsa-activation-capture-overlay-v2','gpu_executed':False,'normal_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'overlay_source_sha256':hashlib.sha256(modified.encode()).hexdigest(),'capture_planned_bytes':capture_planned_bytes,'layers':[3,27,47],'query_offset':query_offset,'rows':capture_rows,'visible_cache_rows':2048,'all_rows':all_rows,'mode_control':'runtime PREFILL4K_ATTENTION_CAPTURE_ALL_ROWS=1 selects all rows; one generated overlay supports both modes','reservation':'dynamic capturePlannedBytes before governor.tryReserve/target construction','query_capture':'each original prepared window copied before ordinary scratch reuse'},indent=2)+'\n')
print(json.dumps({'gpu_executed':False,'folder':str(folder),'rows':capture_rows,'query_offset':query_offset,'capture_planned_bytes':capture_planned_bytes}))
