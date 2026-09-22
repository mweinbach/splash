#!/usr/bin/env python3
"""Private GDN layer-zero capture overlay generation; CPU-only."""
from pathlib import Path
import sys

source=Path("runtime/flash/FlashForward.cpp").read_text()
old="""          addGDNStagedPrefill(graph, weights, buffers, state.gdn[layer], rows, 1,
              FlashGDNStageTile::Values16Time16,
              static_cast<float>(impl_->descriptor.normEpsilon));"""
if source.count(old)!=1:
    raise SystemExit("GDN capture prefill callsite drift")
new="""          private_gdn_capture::add(impl_->backend,graph,weights,buffers,state.gdn[layer],
              rows,layer,static_cast<float>(impl_->descriptor.normEpsilon));"""
destination=Path(sys.argv[1]);destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text('#include "gdn_capture.hpp"\n'+source.replace(old,new))
