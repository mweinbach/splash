#!/usr/bin/env python3
"""Bounded integer ownership/permutation proof; no tensor or device reads."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
WORDS = 2048 * 2 * 256

def run():
    # Integer addresses only. No BF16/model/capture/token values are created/read.
    seen_source = bytearray(WORDS)
    seen_packed = bytearray(WORDS)
    seen_tg_reads = 0
    for kv in range(2):
        for by in range(8):
            for bx in range(64):
                tile_owners = {}
                for tid in range(256):
                    lane, warp = tid % 32, tid // 32
                    for j in range(0, 32, 8):
                        token, dimension = bx * 32 + warp + j, by * 32 + lane
                        source = (token * 2 + kv) * 256 + dimension
                        tile = (warp + j) * 33 + lane
                        assert 0 <= source < WORDS
                        assert tile not in tile_owners
                        assert not seen_source[source]
                        seen_source[source] = 1
                        tile_owners[tile] = source
                for tid in range(256):
                    lane, warp = tid % 32, tid // 32
                    for j in range(0, 32, 8):
                        token, dimension = bx * 32 + lane, by * 32 + warp + j
                        packed = (kv * 256 + dimension) * 2048 + token
                        tile = lane * 33 + warp + j
                        source = tile_owners[tile]
                        assert source == (token * 2 + kv) * 256 + dimension
                        assert 0 <= packed < WORDS and not seen_packed[packed]
                        seen_packed[packed] = 1
                        seen_tg_reads += 1
    assert all(seen_source) and all(seen_packed) and seen_tg_reads == WORDS
    source = (HERE / "pack.metal").read_text()
    assert "device const ushort *source" in source and "device ushort *packed" in source
    assert source.count("threadgroup_barrier(mem_flags::mem_threadgroup)") == 1
    assert re.search(r"\b(?:float|bfloat|half|double)\b", source) is None
    assert source.index("return;") < source.index("threadgroup ushort tile")
    return {"schema": "online-packed-V-integer-permutation-proof-v1", "pass": True,
            "GPU_work": False, "tensor_payload_reads": False,
            "words": WORDS, "logical_bytes": WORDS * 2,
            "grid": [64, 8, 2], "threads": [256, 1, 1],
            "uniform_barriers": 1, "each_source_read_once": True,
            "each_packed_word_written_once": True,
            "each_store_reads_its_exact_source_word": True,
            "pack_source_sha256": hashlib.sha256((HERE / "pack.metal").read_bytes()).hexdigest(),
            "float_conversions_or_math": False,
            "consumer_map": "W[(kv*256+d)*2048+token] = V[(token*2+kv)*256+d]",
            "valid_original_gather_implies_token_in_0_2047":
                "Host fresh begin0/rows2048, original token<=globalQueryRow guard"}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
