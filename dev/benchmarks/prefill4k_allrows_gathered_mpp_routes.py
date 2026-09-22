"""Independent gathered-MPP executor routes on the original all-row control."""
from __future__ import annotations

from prefill4k_allrows_qmv_routes import transform as source_route_transform


def transform(relative: str, text: str) -> str:
    if relative not in ["runtime/flash/FlashForward.cpp", "runtime/flash/FlashBatchForward.cpp",
                        "runtime/flash/FlashBatchVerify.cpp", "runtime/flash/FlashBatchPrefill.cpp",
                        "runtime/flash/FlashWorker.mm"]:
        return text
    # The original control is the input; canonical buffers stay fixed.
    result = source_route_transform(relative, text)
    for before, after in [
        ("FlashGatheredI8QMV.hpp", "FlashGatheredMPP.hpp"),
        ("addGatheredQMV", "addGatheredMPP"),
        ("gatheredQMV", "gatheredMPP"),
        ("gathered_i8_qmv", "gathered_mpp"),
        ("gathered_qmv", "gathered_mpp"),
        ("distinct lane-strided F32 reduction", "direct M16/N64/dynamic-K MPP validRows1; exact original producer parity pending"),
    ]:
        result = result.replace(before, after)
    if relative == "runtime/flash/FlashForward.cpp":
        marker = '      (impl_->allRowsInt8Target ? ";private-allrows-full512-target-m16-below256-v1" : "") +'
        if result.count(marker) != 1:
            raise RuntimeError("Direct gathered MPP route identity anchor drift")
        result = result.replace(marker, marker + """
      (impl_->int8ExpertStore && impl_->int8ExpertStore->gatheredMPPEnabled()
          ? std::string(";") + std::string(gathered_mpp::kPolicy) : "") +""")
    return result
