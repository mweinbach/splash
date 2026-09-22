from pathlib import Path
import argparse


def source(groups):
    text = Path('dev/benchmarks/prefill4k_allrows_gathered_mpp.metal').read_text()
    text = text.replace('gathered_mpp_', f'prefill_moe_sep21_gathered_sg{groups}_')
    text = text.replace('flash_prefill_moe_sep21_gathered_', 'prefill_moe_sep21_gathered_')
    # Both well formed and malformed inputs retain the same barriers and scan ownership.
    text = text.replace('k += 128', f'k += {groups *32}')
    text = text.replace('n += 128', f'n += {groups *32}')
    text = text.replace('threads.x != 128', f'threads.x != {groups *32}')
    text = text.replace('execution_simdgroups<4>', 'execution_simdgroup' if groups ==1 else 'execution_simdgroups<2>')
    text = text.replace('_m16_n64_sg4', f'_m16_n64_sg{groups}')
    text = text.replace('// Private direct gather with the ORIGINAL MPP descriptor, validRows1.',
                        '// Private low-SIMD direct gathered MPP variant, validRows1. Original coefficient/scale/BF16 boundaries.')
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for groups in (1,2):
        (args.destination / f'gathered_sg{groups}.metal').write_text(source(groups))
