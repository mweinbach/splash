#!/usr/bin/env python3
"""Generate private MPP shaders for persistent, exact, unscaled BF16 I8 codes.

Reads source code only. Runtime conversion and allocation belong to the
controlled one-layer oracle; this generator does not read model payloads,
create a Metal device, or modify production sources.
"""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'dev/benchmarks/prefill_moe_sep21/memory.metal'
DESTINATION = Path(__file__).with_name('candidate.metal')
HEADER = '''// Private experiment: unchanged BF16 A and persistent BF16 B containing
// exact UNSCALED signed-I8 codes. The runtime oracle converts each original
// persisted I8 code exactly, without incorporating its F32 scale. This is
// distinct from the old BF16 cache of already-scaled source-Q4 coefficients.
// F32 post-dot row scales, BF16 projections/SwiGLU and output scatter retain
// their original boundaries. B tensors retain the original K-major logical
// shape and strides, now expressed in BF16 elements. No Q4 misses or source
// mutation. Whole-K variants use F32 multiply; fixed K128 accumulates F32.
'''
VARIANTS = (
    ('m32_n64_sg4', 32, 4, 0, False),
    ('m32_n64_sg2', 32, 2, 0, False),
    ('m64_n64_sg8', 64, 8, 0, False),
    ('m32_n64_k128_sg2', 32, 2, 128, True),
)


def generate(destination: Path = DESTINATION) -> str:
    source = SOURCE.read_text()
    start = source.index('#if __METAL_VERSION__ >= 410')
    declarations = source.index('\nPREFILL4K_INT8TILES_GATE(')
    # Keep the native-compatible parameter validation and numerical pipeline
    # verbatim except for naming and the stored coefficient element type.
    body = source[start:declarations]
    if body.count('device int8_t *') != 6:
        raise RuntimeError('Signed-I8 operand declaration inventory drifted')
    if 'p.job_capacity != (p.route_capacity + M - 1) / M + 511' not in body:
        raise RuntimeError('Native M32/M64 job-capacity validation drifted')
    body = body.replace('prefill_moe_sep21_memory_', 'prefill_moe_sep21_bf16_codes_')
    body = body.replace('device int8_t *', 'device bfloat *')
    body = body.replace('PREFILL4K_INT8TILES_GATE', 'BF16_CODES_GATE')
    body = body.replace('PREFILL4K_INT8TILES_DOWN', 'BF16_CODES_DOWN')
    kernels = []
    for suffix, rows, simdgroups, reduction, static in VARIANTS:
        flag = 'true' if static else 'false'
        for macro, producer in (('BF16_CODES_GATE', 'gate_up'),
                                ('BF16_CODES_DOWN', 'down_scatter')):
            name = f'prefill_moe_sep21_bf16_codes_{producer}_{suffix}'
            kernels.append(f'{macro}({name}, {rows}, {simdgroups}, {reduction}, {flag})')
    result = (HEADER + body + '\n' + '\n'.join(kernels) +
              '\n#undef BF16_CODES_GATE\n#undef BF16_CODES_DOWN\n#endif\n')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, nargs='?', default=DESTINATION)
    generate(parser.parse_args().destination)
