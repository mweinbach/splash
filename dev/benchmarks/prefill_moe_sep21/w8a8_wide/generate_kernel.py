#!/usr/bin/env python3
"""Generate private N128 W8A8 numerical-activation-alternative producers.

Reads source code only. Original W8A8 quantizers and frozen artifacts remain
unchanged; wide AIR defines producers only and reuses the original quantizer
AIR. No model payload, activation, route-ID or GPU reads.
"""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'dev/benchmarks/prefill_moe_sep21/w8a8/candidate.metal'
DESTINATION = Path(__file__).with_name('candidate.metal')
HEADER = '''// Private N128 W8A8 geometry experiment. This remains the same numerical
// activation alternative as W8A8, not original-BF16-A math. Original signed-I8
// B, F32 late weight scales, BF16 boundaries and output scatter are unchanged.
// Reuse ORIGINAL W8A8 gate_t256/down_t128 quantizer kernels and ABI from the
// original W8A8 AIR; this AIR defines no quantizer kernels. Include the two
// original quantizer commands in complete-chain timings.
// Producers: M32N128 SG4 and M16N128 SG2; gate gridX5, down gridX20. Native
// M32/M16 jobs retain original ordering, tile_rows=M and ceil(routes/M)+511
// capacity. Width640/2560 are divisible by128, so no new column tail exists.
// Native gate/down ABI uses I8 A at buffer0 and appends activation scales at
// gate12/down11. Timed producers bind no audits. Separate _audit producers
// append scaled F32 G13/U14, D12 and raw exact I32 G15/U16, D13. G/U remain
// packed [route,640]; D remains canonical scattered [route,2560].
// I8 x original-I8 accumulates I32. Epilogue arithmetic remains explicitly
// (float(intdot)*activationScale)*weightScale with contraction/reassociation
// disabled, followed by the original BF16/SwiGLU/output stages.
'''
VARIANTS = (('m32_n128_sg4', 32, 4), ('m16_n128_sg2', 16, 2))


def replace_checked(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError(f'W8A8 wide shader source drift: {before!r}')
    return source.replace(before, after)


def generate(destination: Path = DESTINATION) -> str:
    source = SOURCE.read_text()
    start = source.index('#if __METAL_VERSION__ >= 410')
    quantizers = source.index('// BF16 nonfinite values are sanitized to zero')
    macros = source.index('#define W8A8_GATE_COMMON')
    declarations = source.index('\nW8A8_GATE(prefill_moe_sep21_w8a8_gate_up_')
    # Exclude both quantizer helpers and kernel definitions entirely. Quantizer
    # code and all activation arithmetic continue to come from the base AIR.
    body = source[start:quantizers] + source[macros:declarations]
    body = replace_checked(body, 'constexpr ushort N = 64;', 'constexpr ushort N = 128;', 2)
    body = replace_checked(body, 'if (group.x >= 10)', 'if (group.x >= 5)')
    body = replace_checked(body, 'if (group.x >= 40)', 'if (group.x >= 20)')
    body = body.replace('prefill_moe_sep21_w8a8_', 'prefill_moe_sep21_w8a8_wide_')
    if 'quantize<' in body or 'kernel void prefill_moe_sep21_w8a8_quantize_' in body:
        raise RuntimeError('Wide AIR must not duplicate original quantizers')
    kernels = []
    for suffix, rows, groups in VARIANTS:
        for producer, macro in (('gate_up', 'W8A8_GATE'), ('down_scatter', 'W8A8_DOWN')):
            name = f'prefill_moe_sep21_w8a8_wide_{producer}_{suffix}'
            kernels.append(f'{macro}({name}, {rows}, {groups})')
            kernels.append(f'{macro}_AUDIT({name}_audit, {rows}, {groups})')
    result = HEADER + body + '\n' + '\n'.join(kernels)
    result += ('\n#undef W8A8_GATE_COMMON\n#undef W8A8_DOWN_COMMON\n#undef W8A8_POSITION\n'
               '#undef W8A8_GATE\n#undef W8A8_DOWN\n#undef W8A8_GATE_AUDIT\n#undef W8A8_DOWN_AUDIT\n#endif\n')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, nargs='?', default=DESTINATION)
    generate(parser.parse_args().destination)
