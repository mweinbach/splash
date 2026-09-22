"""Compose qualified FAST HC scheduling onto a new guard-only source tree."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[3]
PRIVATE='dev/benchmarks/guard_hc_fast_composite_sep22'
HC_PARENT=ROOT/'build/hc-pad-compact-r4-verify-teacher-sep22-worker-v3'
CHANGED={'runtime/flash/FlashForward.cpp','runtime/flash/FlashWorker.mm'}
def once(text,before,after):
 if text.count(before)!=1:raise ValueError('Composite source anchor drift:'+before[:100])
 return text.replace(before,after)
def transform(relative,text,hc_overlay=None):
 if relative not in CHANGED:return text
 if hc_overlay is None:
  spec=importlib.util.spec_from_file_location('registered_hc_overlay',HC_PARENT/'source/dev/benchmarks/hc_pad_verify_worker_sep22/overlay.py');hc_overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(hc_overlay)
 text=hc_overlay.transform(relative,text)
 text=f'#include "{PRIVATE}/policy.hpp"\n'+text
 if relative=='runtime/flash/FlashForward.cpp':
  text=once(text,'    hc_pad_verify_sep22::validateDependencies(hcUpF32,fuseHC,cacheFloat);', '''    guard_hc_fast_composite_sep22::validate(compact_native_r4_verify_sep22::requested(),
        compact_r4_preflight_sep22::requested(),hc_pad_verify_sep22::requested());
    hc_pad_verify_sep22::validateDependencies(hcUpF32,fuseHC,cacheFloat);''')
  text=once(text,'      compact_r4_preflight_sep22::marker() +','      compact_r4_preflight_sep22::marker() +\n      guard_hc_fast_composite_sep22::marker() +')
 else:
  text=once(text,'      compact_r4_preflight_sep22::validateDependencies(compact_native_r4_verify_sep22::requested());', '''      guard_hc_fast_composite_sep22::validate(compact_native_r4_verify_sep22::requested(),
          compact_r4_preflight_sep22::requested(),hc_pad_verify_sep22::requested());
      compact_r4_preflight_sep22::validateDependencies(compact_native_r4_verify_sep22::requested());''')
  text=once(text,'      << R"(,"gdn_verification_storage":{"lazy_enabled":)"', '''      << R"(,"guard_hc_fast_composite":{"schema":"C1-Bundle1-HCfastV8-VerifyR4-registered-composite-v1","requested":)"
      <<(guard_hc_fast_composite_sep22::requested()?"true":"false")<<R"(,"source_identity_sha256":)"<<json::quote(guard_hc_fast_composite_sep22::kCompositeSourceIdentitySha256)
      <<R"(,"scope":"singleton VerifyR4 only; original guard13views and HC97roles retained","HC_fast_leaf_manifest_sha256":"92680570764ecfe797fe7db566146d1be1f6e51246a73b197a423a722d1feb61","HC_down_AIR_sha256":"7cc3d642e22bbf90fb95f6c524af232a75fe6c8d99d7385a5e51f7ee572eeefa","new_GPU_allocation_bytes":0,"whole_composite_state_qualified":false})"
      << R"(,"gdn_verification_storage":{"lazy_enabled":)"''')
 return text
